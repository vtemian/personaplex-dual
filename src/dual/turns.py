"""Turns of a taped dialogue: who speaks when, and the text row the teacher lays."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from chalkboard.parser import Ask, Say, Statement, parse
from chalkboard.retime import lay_turns

from dual.constants import CUE, END, LEAD, PAD, TURN

Onsets = list[list[tuple[str, int]]]
Pieces = Callable[[str], Sequence[str]]
PieceId = Callable[[str], int]
EncodeIds = Callable[[str], Sequence[int]]

SPECIAL_IDS = {"<pad>": PAD, "<unk>": CUE, "<s>": TURN, "</s>": END}


def grouped(stmts: Sequence[Statement]) -> list[list[Statement]]:
    # chalkboard training/core/corpus/dialogue_fault.py:85: an `ask` is one turn, the
    # say/pause/draw run after it is the teacher's.
    out: list[list[Statement]] = []
    for stmt in stmts:
        if isinstance(stmt, Ask) or not out or isinstance(out[-1][0], Ask):
            out.append([stmt])
        else:
            out[-1].append(stmt)
    return out


def spoken_lines(stmts: Sequence[Statement]) -> list[Say | Ask]:
    # chalkboard training/moshi/grids.py:195: positional against the record's `words`.
    return [st for st in stmts if isinstance(st, Say | Ask)]


@dataclass(frozen=True)
class TurnSpan:
    index: int  # turn number in the dialogue, 0-based
    party: str  # "pupil" | "teacher"
    words: tuple[str, ...]  # the DSL words of the turn
    onset: int  # first measured word frame
    start: int
    end: int  # half-open; each span starts where the previous one ends


def onsets_of(record: dict[str, Any]) -> Onsets:
    return [[(str(w), int(f)) for w, f in line] for line in record["words"]]


def teacher_turns(
    stmts: Sequence[Statement], onsets: Onsets
) -> list[list[tuple[str, int]]]:
    # chalkboard training/moshi/grids.py:201, verbatim in behaviour.
    turns: list[list[tuple[str, int]]] = []
    open_turn = False
    for st, words in zip(spoken_lines(stmts), onsets, strict=True):
        if isinstance(st, Ask):
            open_turn = False
            continue
        if not open_turn:
            turns.append([])
            open_turn = True
        turns[-1].extend(words)
    return [turn for turn in turns if turn]


def spans(src: str, record: dict[str, Any]) -> list[TurnSpan]:
    """Spans tile [0, frames): span k starts LEAD frames before turn k's first word
    (turn 0 at 0) and ends where span k+1 starts."""
    stmts = parse(src)
    onsets = onsets_of(record)
    lines = spoken_lines(stmts)
    if len(lines) != len(onsets):
        raise ValueError(
            f"{record['id']}: {len(lines)} spoken lines, {len(onsets)} measured"
        )
    frames = int(record["frames"])
    groups = grouped(stmts)
    # Each group's measured words: the onsets of the spoken lines inside it, in order.
    cursor = 0
    firsts: list[int] = []
    words_of: list[tuple[str, ...]] = []
    for group in groups:
        measured: list[int] = []
        words: list[str] = []
        for st in group:
            if isinstance(st, Say | Ask):
                measured.extend(f for _w, f in onsets[cursor])
                words.extend(st.words)
                cursor += 1
        if not measured:
            raise ValueError(f"{record['id']}: turn {len(firsts)} has no measured word")
        firsts.append(min(measured))
        words_of.append(tuple(words))
    out: list[TurnSpan] = []
    start = 0
    for k, group in enumerate(groups):
        end = frames if k == len(groups) - 1 else firsts[k + 1] - LEAD
        if end > frames:
            raise ValueError(f"{record['id']}: turn {k} runs past the tape")
        if end <= start:
            raise ValueError(f"{record['id']}: turn {k} is empty ({start}..{end})")
        party = "pupil" if isinstance(group[0], Ask) else "teacher"
        out.append(TurnSpan(k, party, words_of[k], firsts[k], start, end))
        start = end
    return out


def text_row(
    src: str, record: dict[str, Any], pieces: Pieces, piece_id: PieceId
) -> list[int]:
    """One text id per tape frame: the teacher's words as moshi's interleave lays them,
    `<pad>` everywhere else including every pupil frame."""
    stmts = parse(src)
    frames = int(record["frames"])
    laid = lay_turns(teacher_turns(stmts, onsets_of(record)), pieces, frames)
    ids = [SPECIAL_IDS[p] if p in SPECIAL_IDS else piece_id(p) for p in laid]
    if len(ids) > frames:
        raise ValueError(
            f"{record['id']}: text runs {len(ids) - frames} frames past the tape"
        )
    return ids + [PAD] * (frames - len(ids))


def typed_tokens(words: Sequence[str], encode_ids: EncodeIds) -> list[int]:
    # chalkboard training/moshi/tts_corpus.py:32: the last word punctuated. One
    # sentence, encoded whole, as PersonaPlex encodes its text prompt.
    spoken = list(words)
    if spoken and not spoken[-1].endswith((".", "?", "!")):
        spoken[-1] += "."
    ids = [int(i) for i in encode_ids(" ".join(spoken))]
    if not ids:
        raise ValueError("no tokens")
    return ids
