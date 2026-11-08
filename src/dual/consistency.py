"""The typed answer and the spoken answer it is trained toward (B9). Pure: no torch.

A training row with typed pupil turns (`kind` mask or complement) is paired with its
stem's `spoken` row, where every pupil turn is the tape. After typed pupil turn k the
tutor's answer is span k+1 in both rows: the same text, the same length, shifted by the
difference in pupil-turn lengths before it. The consistency loss compares the two text
distributions frame by frame on the answer's content frames (target text not PAD), the
frames the content bar scores.

B15: the teacher may instead be a `voiced` row, the same stem with exactly the typed
row's typed turns in a TTS voice (`voice` label) and the rest the tape; it pairs with
the typed row of its `pairs_with` kind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dual.constants import PAD

Row = Mapping[str, Any]
TEACHER_KINDS = ("spoken", "voiced")


@dataclass(frozen=True)
class AnswerPair:
    turn: int  # the typed pupil turn
    typed: tuple[int, int]  # the answer span in the typed row, half-open
    spoken: tuple[int, int]  # the same answer in the spoken row, half-open
    content: tuple[int, ...]  # offsets inside the span whose target text is not PAD


def answer_pairs(typed_row: Row, spoken_row: Row) -> list[AnswerPair]:
    """One pair per typed pupil turn of `typed_row` that has a tutor answer after it.
    `spoken_row` is the teacher: the stem's all-tape row, or its voiced row (B15)."""
    if typed_row["stem"] != spoken_row["stem"]:
        raise ValueError(
            f"stem {typed_row['stem']!r} paired with {spoken_row['stem']!r}"
        )
    kind = spoken_row.get("kind")
    if kind not in TEACHER_KINDS:
        raise ValueError(
            f"{spoken_row['id']}: the teacher row is not a spoken or voiced row"
        )
    expect = "tape"
    if kind == "voiced":
        expect = str(spoken_row["voice"])
        if sorted(spoken_row["voiced_turns"]) != sorted(typed_row["typed_turns"]):
            raise ValueError(
                f"{spoken_row['id']} voices turns {spoken_row['voiced_turns']}, "
                f"{typed_row['id']} types {typed_row['typed_turns']}"
            )
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
        if s_spans[k]["rendering"] != expect:
            raise ValueError(
                f"{spoken_row['id']}: turn {k} is {s_spans[k]['rendering']}, "
                f"not {expect}"
            )
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


def tape_teachers(rows: Sequence[Row]) -> dict[str, Row]:
    """Typed row id -> its stem's spoken (all-tape) row: the B9 teacher."""
    spoken = spoken_by_stem(rows)
    return {str(r["id"]): spoken[r["stem"]] for r in rows if r.get("typed_turns")}


def voiced_teachers(rows: Sequence[Row], teacher_rows: Sequence[Row]) -> dict[str, Row]:
    """Typed row id -> the voiced row of its stem and kind (B15). Refuses a typed row
    without one, two for one (stem, kind), rows in more than one voice, and a voiced
    row that voices other turns than its typed row types."""
    by_key: dict[tuple[str, str], Row] = {}
    for t in teacher_rows:
        if t.get("kind") != "voiced":
            raise ValueError(f"{t['id']}: not a voiced row")
        key = (str(t["stem"]), str(t["pairs_with"]))
        if key in by_key:
            raise ValueError(f"{key[0]}: two voiced rows for {key[1]}")
        by_key[key] = t
    voices = sorted({str(t["voice"]) for t in teacher_rows})
    if len(voices) > 1:
        raise ValueError(f"voiced rows in {voices}: one teacher voice only")
    out: dict[str, Row] = {}
    lacking: list[str] = []
    for r in rows:
        if not r.get("typed_turns"):
            continue
        v = by_key.get((str(r["stem"]), str(r["kind"])))
        if v is None:
            lacking.append(str(r["id"]))
            continue
        if sorted(v["voiced_turns"]) != sorted(r["typed_turns"]):
            raise ValueError(
                f"{v['id']} voices {v['voiced_turns']}, {r['id']} types "
                f"{r['typed_turns']}"
            )
        out[str(r["id"])] = v
    if lacking:
        raise ValueError(f"no voiced row for {lacking[:3]}")
    return out


def longest_stem(rows: Sequence[Row]) -> str:
    """The stem of the longest row; on a tie, the first stem by name."""
    if not rows:
        raise ValueError("no rows")
    return str(min(rows, key=lambda r: (-int(r["frames"]), str(r["stem"])))["stem"])
