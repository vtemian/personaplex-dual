"""The typed answer and the spoken answer it is trained toward (B9). Pure: no torch.

A training row with typed pupil turns (`kind` mask or complement) is paired with its
stem's `spoken` row, where every pupil turn is the tape. After typed pupil turn k the
tutor's answer is span k+1 in both rows: the same text, the same length, shifted by the
difference in pupil-turn lengths before it. The consistency loss compares the two text
distributions frame by frame on the answer's content frames (target text not PAD), the
frames the content bar scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dual.constants import PAD

Row = Mapping[str, Any]


@dataclass(frozen=True)
class AnswerPair:
    turn: int  # the typed pupil turn
    typed: tuple[int, int]  # the answer span in the typed row, half-open
    spoken: tuple[int, int]  # the same answer in the spoken row, half-open
    content: tuple[int, ...]  # offsets inside the span whose target text is not PAD


def answer_pairs(typed_row: Row, spoken_row: Row) -> list[AnswerPair]:
    """One pair per typed pupil turn of `typed_row` that has a tutor answer after it."""
    if typed_row["stem"] != spoken_row["stem"]:
        raise ValueError(
            f"stem {typed_row['stem']!r} paired with {spoken_row['stem']!r}"
        )
    if spoken_row.get("kind") != "spoken":
        raise ValueError(f"{spoken_row['id']}: the teacher row is not the spoken row")
    t_spans, s_spans = typed_row["spans"], spoken_row["spans"]
    if len(t_spans) != len(s_spans):
        raise ValueError(
            f"{typed_row['id']}: {len(t_spans)} spans, spoken row {len(s_spans)}"
        )
    out: list[AnswerPair] = []
    for k in sorted(typed_row["typed_turns"]):
        if not 0 <= k < len(t_spans):
            raise ValueError(f"{typed_row['id']}: typed turn {k} out of range")
        if t_spans[k]["party"] != "pupil" or t_spans[k]["rendering"] != "typed":
            raise ValueError(f"{typed_row['id']}: turn {k} is not typed in this row")
        if s_spans[k]["rendering"] != "tape":
            raise ValueError(f"{spoken_row['id']}: turn {k} is not the tape")
        if k + 1 >= len(t_spans):
            continue  # the dialogue ends on this turn: no answer to compare
        ta, sa = t_spans[k + 1], s_spans[k + 1]
        if ta["party"] != "teacher" or sa["party"] != "teacher":
            raise ValueError(f"{typed_row['id']}: no tutor answer after turn {k}")
        t_text = list(typed_row["text"][ta["start"] : ta["end"]])
        s_text = list(spoken_row["text"][sa["start"] : sa["end"]])
        if t_text != s_text:
            raise ValueError(f"{typed_row['id']}: answer text differs after turn {k}")
        out.append(
            AnswerPair(
                turn=k,
                typed=(int(ta["start"]), int(ta["end"])),
                spoken=(int(sa["start"]), int(sa["end"])),
                content=tuple(i for i, t in enumerate(t_text) if t != PAD),
            )
        )
    return out


def flat_indices(pairs: Sequence[AnswerPair]) -> tuple[list[int], list[int]]:
    """(typed-row frame indices, spoken-row frame indices) of every content frame."""
    student: list[int] = []
    teacher: list[int] = []
    for p in pairs:
        student += [p.typed[0] + i for i in p.content]
        teacher += [p.spoken[0] + i for i in p.content]
    return student, teacher


def spoken_by_stem(rows: Sequence[Row]) -> dict[str, Row]:
    """Each stem's `spoken` row; refuses a stem with typed rows but no spoken row."""
    out: dict[str, Row] = {}
    for r in rows:
        if r.get("kind") == "spoken":
            if r["stem"] in out:
                raise ValueError(f"{r['stem']}: two spoken rows")
            out[r["stem"]] = r
    lacking = sorted(
        {r["stem"] for r in rows if r.get("typed_turns") and r["stem"] not in out}
    )
    if lacking:
        raise ValueError(f"no spoken row for {lacking[:3]}")
    return out


def longest_stem(rows: Sequence[Row]) -> str:
    """The stem of the longest row; on a tie, the first stem by name."""
    if not rows:
        raise ValueError("no rows")
    return str(min(rows, key=lambda r: (-int(r["frames"]), str(r["stem"])))["stem"])
