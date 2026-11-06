"""Side-by-side greedy decoding: the schedule, the packing and the per-step inputs.

Pure: no torch. The GPU loop (moshi's LMGen) forces every frame up to the tutor's
answer and lets the model generate the answer window.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dual.constants import CODEBOOKS, DELAYS
from dual.probe_text import content_ids, words_of

RENDERINGS = ("typed", "A", "B")
SWAP_RENDERINGS = ("typed", "A")


@dataclass(frozen=True)
class Item:
    stem: str
    turn: int
    rendering: str
    prefix: int  # frames forced = answer_start
    window: int  # frames generated = answer_end - answer_start

    @property
    def total(self) -> int:
        return self.prefix + self.window


def item_of(stem: str, turn: int, rendering: str, answer: tuple[int, int]) -> Item:
    start, end = answer
    if start < 1 or end <= start:
        raise ValueError(f"bad answer window {answer} for {stem} turn {turn}")
    return Item(stem, turn, rendering, start, end - start)


def batch_cap(batch: Sequence[Item], align: int = 64) -> int:
    """The longest total in the batch, rounded up to a multiple of `align`."""
    if not batch:
        raise ValueError("empty batch")
    longest = max(i.total for i in batch)
    return -(-longest // align) * align


def pack(
    items: Sequence[Item], budget: int, max_batch: int, align: int = 64
) -> list[list[Item]]:
    """Greedy batches over items sorted by (total, stem, turn, rendering).

    An item joins the current batch while the batch stays within `max_batch` and
    `len(batch) * cap <= budget`, where cap is the batch's aligned longest total.
    """
    if budget < 1 or max_batch < 1:
        raise ValueError("budget and max_batch must be at least 1")
    ordered = sorted(items, key=lambda i: (i.total, i.stem, i.turn, i.rendering))
    for it in ordered:
        if batch_cap([it], align) > budget:
            raise ValueError(f"{it} alone exceeds the budget {budget}")
    out: list[list[Item]] = []
    current: list[Item] = []
    for it in ordered:
        n = len(current) + 1
        if (
            current
            and n <= max_batch
            and n * batch_cap([*current, it], align) <= budget
        ):
            current.append(it)
        else:
            if current:
                out.append(current)
            current = [it]
    if current:
        out.append(current)
    return out


@dataclass(frozen=True)
class StepInput:
    pupil: tuple[int, ...]  # CODEBOOKS codes fed as the user stream this step
    text: int | None  # forced text token, None = model generates it
    own: tuple[int | None, ...]  # per codebook: forced code, None = generated


def step_input(row: dict[str, Any], prefix: int, step: int) -> StepInput:
    """What to feed and force at LMGen step `step`, the first `prefix` frames forced.

    `row` is an undelayed frames.to_record. At step s LMGen reads delayed frame s-1 as
    input and predicts text frame s and own codebook k at frame s - d_k, with
    d_k = DELAYS[1 + k]. It writes the user stream with its own delays, so it is
    handed the undelayed pupil frame s-1 (frame 0 at s == 0, which moshi ignores).
    Text is forced while s < prefix. Own codebook k is forced while s - d_k < prefix,
    to row["own"][s - d_k] (frame 0 when that is negative; moshi ignores it). So at
    s == prefix the text and own[0] are free but own[1..] are still forced with
    own frame prefix - 1.
    """
    total = len(row["text"])
    if step < 0 or step >= total or prefix < 1 or prefix > total:
        raise ValueError(f"bad step {step} / prefix {prefix} for {total} frames")
    pupil = tuple(int(c) for c in row["pupil"][max(step - 1, 0)])
    text = int(row["text"][step]) if step < prefix else None
    own: list[int | None] = []
    for k in range(CODEBOOKS):
        f = step - DELAYS[1 + k]
        if f < prefix:
            own.append(int(row["own"][max(f, 0)][k]))
        else:
            own.append(None)
    return StepInput(pupil, text, tuple(own))


def answer_words(ids: Sequence[int], decode: Callable[[list[int]], str]) -> str:
    content = content_ids(ids)
    return decode(content).strip() if content else ""


def same_words(a: str, b: str) -> bool:
    return words_of(a) == words_of(b)


def answer_head(text: str) -> str:
    """The normalised words before the first comma."""
    return " ".join(words_of(text.split(",", 1)[0]))


def pick_donors(
    order: Sequence[str],
    heads: Mapping[tuple[str, int], str],
    turns: Sequence[int],
) -> dict[tuple[str, int], str]:
    """For each (stem, turn) in `heads` with turn in `turns`: the first stem after it in
    `order`, cyclically, that has that turn and a different head."""
    out: dict[tuple[str, int], str] = {}
    for i, stem in enumerate(order):
        for turn in turns:
            if (stem, turn) not in heads:
                continue
            for step in range(1, len(order)):
                other = order[(i + step) % len(order)]
                if (other, turn) in heads and heads[(other, turn)] != heads[
                    (stem, turn)
                ]:
                    out[(stem, turn)] = other
                    break
            else:
                raise ValueError(
                    f"no donor with a different head for {stem} turn {turn}"
                )
    return out


def swap_record(
    record: dict[str, Any], turn: int, donor: dict[str, Any]
) -> dict[str, Any]:
    """A copy of `record` whose renderings of `turn` are the donor's."""
    for r in (record, donor):
        if turn not in r["pupil_turns"]:
            raise ValueError(f"{r['id']}: turn {turn} is not a pupil turn")
    out = dict(record)
    out["renderings"] = {
        **record["renderings"],
        str(turn): copy.deepcopy(donor["renderings"][str(turn)]),
    }
    return out


def schedule(
    true_items: Sequence[Item], swap_items: Sequence[Item], budget: int, max_batch: int
) -> list[list[Item]]:
    """The true items' batches exactly as `pack` makes them, then the swap items'."""
    return pack(true_items, budget, max_batch) + pack(swap_items, budget, max_batch)
