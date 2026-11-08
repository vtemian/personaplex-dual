"""Rows from a taped dialogue: the mask row, its complement, the all-spoken tape, and
the holdout record with every pupil turn in every rendering."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dual.constants import CONTEXT, PAD
from dual.frames import (
    Row,
    Segment,
    Span,
    concat,
    from_record,
    spoken_segment,
    tape_segment,
    to_record,
    typed_segment,
)
from dual.holdout import mask_bits
from dual.turns import TurnSpan

TypedTokens = Callable[[Sequence[str]], Sequence[int]]
# (turn words, voice name) -> the pupil's Mimi codes [L][8]
Voiced = Callable[[Sequence[str], str], Sequence[Sequence[int]]]

MASK_KINDS = ("mask", "complement", "spoken")


def tape_segments(
    spans: Sequence[TurnSpan], text: Sequence[int], record: dict[str, Any]
) -> list[Segment]:
    own, pupil = record["audio"], record["user_audio"]
    for s in spans:
        if s.party == "pupil" and any(t != PAD for t in text[s.start : s.end]):
            name = record.get("id", record.get("stem", "?"))
            raise ValueError(
                f"{name}: text on pupil frames {s.start}..{s.end} (turn {s.index})"
            )
    return [
        tape_segment(
            text[s.start : s.end],
            own[s.start : s.end],
            pupil[s.start : s.end],
            scored=s.party == "teacher",
        )
        for s in spans
    ]


def pupil_indices(spans: Sequence[TurnSpan]) -> list[int]:
    return [s.index for s in spans if s.party == "pupil"]


def assemble(
    stem: str,
    spans: Sequence[TurnSpan],
    base: Sequence[Segment],
    swaps: dict[int, tuple[str, Segment]],
) -> Row:
    parts: list[tuple[str, str, Segment]] = []
    for s, seg in zip(spans, base, strict=True):
        if s.index in swaps:
            rendering, swapped = swaps[s.index]
            parts.append((s.party, rendering, swapped))
        else:
            parts.append((s.party, "tape", seg))
    row = concat(stem, parts)
    if row.frames > CONTEXT:
        raise ValueError(f"{stem}: {row.frames} frames, over moshi's {CONTEXT}")
    return row


def training_rows(
    stem: str,
    spans: Sequence[TurnSpan],
    base: Sequence[Segment],
    typed: TypedTokens,
) -> list[dict[str, Any]]:
    pupils = pupil_indices(spans)
    bits = mask_bits(stem, len(pupils))
    typed_segs = {k: typed_segment(typed(spans[k].words)) for k in pupils}
    rows: list[dict[str, Any]] = []
    for kind in MASK_KINDS:
        if kind == "mask":
            chosen = [k for k, b in zip(pupils, bits, strict=True) if b == 1]
        elif kind == "complement":
            chosen = [k for k, b in zip(pupils, bits, strict=True) if b == 0]
        else:
            chosen = []
        swaps = {k: ("typed", typed_segs[k]) for k in chosen}
        record = to_record(assemble(f"{stem}#{kind}", spans, base, swaps))
        record["kind"] = kind
        record["stem"] = stem
        record["typed_turns"] = chosen
        rows.append(record)
    return rows


VOICED_KIND = "voiced"
TYPED_KINDS = ("mask", "complement")


def voiced_rows(
    stem: str,
    spans: Sequence[TurnSpan],
    base: Sequence[Segment],
    voiced: Voiced,
    label: str,
    voice: str,
) -> list[dict[str, Any]]:
    """The consistency teacher rows of B15: for each typed kind (mask, complement),
    the tape with exactly that kind's typed turns spoken by `voice` (rendering
    `label`); every other pupil turn stays the tape. The same mask as training_rows."""
    pupils = pupil_indices(spans)
    bits = mask_bits(stem, len(pupils))
    segs = {k: spoken_segment(voiced(spans[k].words, voice)) for k in pupils}
    rows: list[dict[str, Any]] = []
    for kind, want in zip(TYPED_KINDS, (1, 0), strict=True):
        chosen = [k for k, b in zip(pupils, bits, strict=True) if b == want]
        swaps = {k: (label, segs[k]) for k in chosen}
        record = to_record(assemble(f"{stem}#{kind}-{label}", spans, base, swaps))
        record["kind"] = VOICED_KIND
        record["stem"] = stem
        record["typed_turns"] = []
        record["pairs_with"] = kind
        record["voice"] = label
        record["voiced_turns"] = chosen
        rows.append(record)
    return rows


def _seg_record(seg: Segment) -> dict[str, Any]:
    return {
        "text": list(seg.text),
        "own": [list(f) for f in seg.own],
        "pupil": [list(f) for f in seg.pupil],
    }


def holdout_record(
    stem: str,
    spans: Sequence[TurnSpan],
    base: Sequence[Segment],
    typed: TypedTokens,
    voiced: Voiced,
    voices: dict[str, str],  # label -> voice name, e.g. {"A": "vctk/p225_023.wav"}
) -> dict[str, Any]:
    pupils = pupil_indices(spans)
    renderings: dict[str, dict[str, Any]] = {}
    for k in pupils:
        words = spans[k].words
        per_turn: dict[str, Any] = {"typed": _seg_record(typed_segment(typed(words)))}
        for label, voice in voices.items():
            per_turn[label] = _seg_record(spoken_segment(voiced(words, voice)))
        renderings[str(k)] = per_turn
    tape = to_record(assemble(stem, spans, base, {}))
    return {"id": stem, "base": tape, "pupil_turns": pupils, "renderings": renderings}


def add_rendering(
    record: dict[str, Any],
    spans: Sequence[TurnSpan],
    voiced: Voiced,
    voices: dict[str, str],  # new label -> voice name
) -> dict[str, Any]:
    """A copy of a holdout record with every pupil turn also in each new voice; the
    renderings it had (typed, A, B, C) are kept as they are, never re-voiced."""
    pupils = pupil_indices(spans)
    if list(record["pupil_turns"]) != pupils:
        raise ValueError(
            f"{record['id']}: pupil turns {record['pupil_turns']}, spans say {pupils}"
        )
    renderings: dict[str, dict[str, Any]] = {}
    for k in pupils:
        per_turn = dict(record["renderings"][str(k)])
        for label, voice in voices.items():
            if label in per_turn or label in ("typed", "tape"):
                raise ValueError(f"{record['id']}: turn {k} already has {label!r}")
            seg = spoken_segment(voiced(spans[k].words, voice))
            per_turn[label] = _seg_record(seg)
        renderings[str(k)] = per_turn
    return {**record, "renderings": renderings}


TAPE_FIELDS = ("frames", "text", "own", "pupil", "scored", "spans")


def check_same_tape(rebuilt: dict[str, Any], stored: Mapping[str, Any]) -> None:
    """Refuses a tape row rebuilt from the data that differs from the one an earlier
    render stored, in any frame field (compared after a JSON round trip)."""
    again = json.loads(json.dumps({k: rebuilt[k] for k in TAPE_FIELDS}))
    for k in TAPE_FIELDS:
        if again[k] != stored[k]:
            raise ValueError(
                f"{stored.get('id', '?')}: the rebuilt tape differs in {k!r}"
            )


def segment_from(rec: dict[str, Any]) -> Segment:
    return tape_segment(rec["text"], rec["own"], rec["pupil"], scored=False)


def _base_segment(row: Row, span: Span) -> Segment:
    return tape_segment(
        row.text[span.start : span.end],
        row.own[span.start : span.end],
        row.pupil[span.start : span.end],
        scored=span.party == "teacher",
    )


def holdout_row(
    record: dict[str, Any], turn: int, rendering: str
) -> tuple[dict[str, Any], tuple[int, int]]:
    """The base row with pupil turn `turn` swapped for `rendering` ("tape" keeps it),
    and the half-open frame window of the teacher's answer after it, after the shift."""
    base = from_record(record["base"])
    if not 0 <= turn < len(base.spans) or base.spans[turn].party != "pupil":
        raise ValueError(f"{record['id']}: turn {turn} is not a pupil turn")
    parts: list[tuple[str, str, Segment]] = []
    for i, s in enumerate(base.spans):
        if i == turn and rendering != "tape":
            seg = segment_from(record["renderings"][str(turn)][rendering])
            parts.append((s.party, rendering, seg))
        elif i == turn:
            parts.append((s.party, rendering, _base_segment(base, s)))
        else:
            parts.append((s.party, s.rendering, _base_segment(base, s)))
    row = concat(f"{record['id']}#{turn}#{rendering}", parts)
    if row.frames > CONTEXT:
        raise ValueError(f"{row.id}: {row.frames} frames, over moshi's {CONTEXT}")
    if turn + 1 >= len(row.spans) or row.spans[turn + 1].party != "teacher":
        raise ValueError(
            f"{record['id']}: pupil turn {turn} has no teacher answer after it"
        )
    answer = row.spans[turn + 1]
    return to_record(row), (answer.start, answer.end)
