"""Frames on the three streams, and the segments a row is cut from. Pure: no torch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dual.constants import CODEBOOKS, GAP_FRAMES, MIMI_SILENCE, PAD, SILENCE, SINE

Codes = tuple[tuple[int, ...], ...]


def _codes(rows: Sequence[Sequence[int]]) -> Codes:
    return tuple(tuple(int(c) for c in row) for row in rows)


@dataclass(frozen=True)
class Segment:
    text: tuple[int, ...]
    own: Codes
    pupil: Codes
    scored: bool

    def __post_init__(self) -> None:
        if not len(self.text) == len(self.own) == len(self.pupil):
            raise ValueError(
                f"frames disagree: text {len(self.text)}, own {len(self.own)}, "
                f"pupil {len(self.pupil)}"
            )
        for name, codes in (("own", self.own), ("pupil", self.pupil)):
            if any(len(frame) != CODEBOOKS for frame in codes):
                raise ValueError(f"{name}: every frame carries {CODEBOOKS} codebooks")

    @property
    def frames(self) -> int:
        return len(self.text)


def typed_segment(tokens: Sequence[int]) -> Segment:
    if not tokens:
        raise ValueError("a typed turn with no tokens")
    n = len(tokens) + GAP_FRAMES
    return Segment(
        text=tuple(int(t) for t in tokens) + (PAD,) * GAP_FRAMES,
        own=(SILENCE,) * n,
        pupil=(SINE,) * n,
        scored=False,
    )


def spoken_segment(pupil_codes: Sequence[Sequence[int]]) -> Segment:
    if not pupil_codes:
        raise ValueError("a spoken turn with no frames")
    n = len(pupil_codes) + GAP_FRAMES
    return Segment(
        text=(PAD,) * n,
        own=(SILENCE,) * n,
        pupil=_codes(pupil_codes) + (MIMI_SILENCE,) * GAP_FRAMES,
        scored=False,
    )


def tape_segment(
    text: Sequence[int],
    own: Sequence[Sequence[int]],
    pupil: Sequence[Sequence[int]],
    scored: bool,
) -> Segment:
    return Segment(tuple(int(t) for t in text), _codes(own), _codes(pupil), scored)


@dataclass(frozen=True)
class Span:
    party: str  # "pupil" | "teacher"
    rendering: str  # "tape" | "typed" | a voice label ("A", "B", "C")
    start: int
    end: int  # half-open


@dataclass(frozen=True)
class Row:
    id: str
    text: tuple[int, ...]
    own: Codes
    pupil: Codes
    scored: tuple[int, ...]
    spans: tuple[Span, ...]

    @property
    def frames(self) -> int:
        return len(self.text)


def concat(stem: str, parts: Sequence[tuple[str, str, Segment]]) -> Row:
    text: list[int] = []
    own: list[tuple[int, ...]] = []
    pupil: list[tuple[int, ...]] = []
    scored: list[int] = []
    spans: list[Span] = []
    for party, rendering, seg in parts:
        start = len(text)
        text.extend(seg.text)
        own.extend(seg.own)
        pupil.extend(seg.pupil)
        scored.extend([int(seg.scored)] * seg.frames)
        spans.append(Span(party, rendering, start, start + seg.frames))
    return Row(stem, tuple(text), tuple(own), tuple(pupil), tuple(scored), tuple(spans))


def to_record(row: Row) -> dict[str, Any]:
    return {
        "id": row.id,
        "frames": row.frames,
        "text": list(row.text),
        "own": [list(f) for f in row.own],
        "pupil": [list(f) for f in row.pupil],
        "scored": list(row.scored),
        "spans": [
            {"party": s.party, "rendering": s.rendering, "start": s.start, "end": s.end}
            for s in row.spans
        ],
    }


def from_record(record: dict[str, Any]) -> Row:
    return Row(
        id=str(record["id"]),
        text=tuple(int(t) for t in record["text"]),
        own=_codes(record["own"]),
        pupil=_codes(record["pupil"]),
        scored=tuple(int(s) for s in record["scored"]),
        spans=tuple(
            Span(s["party"], s["rendering"], int(s["start"]), int(s["end"]))
            for s in record["spans"]
        ),
    )
