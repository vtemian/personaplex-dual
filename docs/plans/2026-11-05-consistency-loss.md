# Consistency Loss on the Typed Pupil Turn: Implementation Plan (B9)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan batch by batch.
> In `docs/plans/2026-10-26-dual-input-batches.md` this plan is batch **B9**. Its batches are
> C1 to C4 below.

**Goal:** a typed-to-spoken consistency term in the train job, with a smoke and a 3-epoch pilot, then a
GATE on the pilot's gap/floor ratio before any sweep.

**Architecture:** the train job keeps its length-packed batches and its LM loss on every row. For each
typed training row in a batch, it first runs a `no_grad` forward of the same stem's all-tape
`spoken` row with the current adapter (a stop-gradient teacher). It then adds
λ · KL(typed answer ‖ spoken answer) over the text distributions of the tutor-answer content frames
after each typed pupil turn. Frames are aligned by span index and offset inside the span. The pure
alignment lives in `src/dual/consistency.py` (no torch), the loss in `src/dual/kl.py`, and the
wiring in `src/dual/jobs/train.py`. With λ = 0 the code path is the baseline's, bit for bit.

**Tech stack:** Python 3, PyTorch (bf16 autocast, fp32 logits), moshi 0.2.13 (on the box only), pytest,
ruff, mypy, uv, sparks queue, POSIX sh.

---

## 0. Read first (the executor has only this file, memory and the repo)

- `CLAUDE.md`:
  - `make check` before every commit; work on a branch, never on main.
  - Pure modules (`src/dual/*.py`) never import torch at module level, except `kl.py`.
  - GPU code is `src/dual/model.py` and `src/dual/jobs/*.py`, and it imports `moshi` inside functions.
  - Never edit or commit `vendor/`; never commit `data/` or `.env`.
- Memory: `/Users/whitemonk/.claude/projects/-Users-whitemonk-projects-ai-personaplex-dual/memory/MEMORY.md`,
  then `ladder-2-result.md`, `b6-ladder-2-run.md`, `box-outages.md`, `drama3-memory-hog.md`.
- Results this builds on: `docs/results/ladder-1.md`, `ladder-2.md`, `side-by-side-1.md`,
  `bar-by-position-1.md`.
- Skills for the GPU part (C4): `authoring-a-sparks-job`, `operating-the-sparks-queue`.
- Branch for all code batches: `feat/b9-consistency` off `main` (1081cb5).

### 0.1 Facts this plan rests on (measured; do not re-derive)

| fact | value | source |
|---|---|---|
| train rows | 351 = 117 stems × 3 rows (`kind` mask, complement, spoken) | `src/dual/render.py:76-102`, render-1 |
| the `spoken` row | all pupil turns are the **tape** (recorded audio); no TTS voice in training | `render.py:89-93` |
| typed turns | each pupil turn is typed in exactly one of mask/complement; `typed_turns` lists them | `render.py:85-101` |
| row fields | `id, frames, text, own, pupil, scored, spans[{party, rendering, start, end}], kind, stem, typed_turns` | `frames.py:111-123` |
| answer text | identical in every rendering (measure asserts it); the answer span is `spans[k+1]` after pupil turn `k` | `jobs/measure.py:150-154`, `render.py:162-167` |
| lead | every answer opens with 3 PAD frames; the content bar skips PAD targets | `bar-by-position-1.md` |
| longest train row | 2,765 frames (CONTEXT 3,000) | render-1 log |
| batch packing | rows sorted by length, `rows × longest ≤ 4000` frames, batches shuffled by the seed | `jobs/train.py:93-105` |
| LM loss | `2.0·text CE + 0.5·audio CE` over `scored & mask` frames | `jobs/train.py:61-90, 228` |
| peak memory | PersonaPlex train 49.3 to 49.4 GiB compiled (59.6 eager); cap 64 GiB via `--mem-cap-gib` (a job arg, default 64) | memory `b4-bare-bar.md`, `model.py:66-95` |
| epoch time, 117 stems, compiled | **3,220 s** (~12 s/step). The "~450 s/epoch" in the brief is the 15-dialogue slice (452 s). | ladder-1 |
| measure time | ~25 min per checkpoint epoch on the holdout | ladder-1, B8 |
| holdout | 13 dialogues, 65 pupil turns, renderings typed/A/B/C/tape | render-1 |
| R0 | R_gap seed 0 `ladder-personaplex-1` **3.04**, seed 1 `ladder2-pp-s1` **2.96**; d = 0.08, T = max(2d, 0.5) = **0.5** | `docs/results/ladder-2.md`, `src/dual/noise.py` |
| moved | \|R_gap − R0\| ≥ T **and** both paired stem-bootstrap 95% CIs exclude 0 with that sign; a "down" whose median gap is not below the seeds' mean gap is "down, floor inflated" and does not count | `src/dual/noise.py` |
| epoch-2 content KL | gap ~0.018 to 0.025, floor ~0.006 to 0.008 nats/frame (ratio ~3; 3.30 at epoch 2) | ladder-1, B8 |
| B8 | from position 26: typed‖tape 0.0174 = typed‖A 0.0176; tape‖A 0.0069 vs floor 0.0051. Over all content frames tape‖A 0.0136 vs floor 0.0077 (**1.8×**) | `bar-by-position-1.md` |
| bare | silent after every spoken turn (65/65, B7); content ratio 3.93; position-0 ratio 82 | B7, B8 |
| held text loss, ladder-personaplex-1 | 0.551, 0.540, 0.611 at epochs 1 to 3 | ladder-1 |
| LoRA off switch | `vendor/training/moshi/trunk_lora.py:169-183` `lora_shut(model)` sets every wrapper's `scaling` to 0 inside a `with` (unused so far) | vendor |
| sparks metrics | a declared metric may take a dict of groups; `loss={"total","text","audio","consistency"}` is legal | `authoring-a-sparks-job` |

### 0.2 Decisions (settled here; the executor does not reopen them)

**D1. Teacher = the live student on the stem's `spoken` (all-tape) row, under `no_grad`.** It uses the
current adapter (LoRA on) and is recomputed every step. This is stop-gradient self-distillation: the typed
answer is pulled toward what *this* model says after the spoken turn, and the spoken side is never pulled
toward typed. The bar measures exactly this, the same model's typed vs spoken distributions.
Rejected:
- *Frozen base / adapter disabled* (`lora_shut`, cheap). Bare PersonaPlex is silent after 65/65 spoken
  turns (B7), and its content ratio (3.93) is worse than the trained model's. Distilling toward it would
  pull the typed answer toward a model that does not answer, and would fight the LM loss.
- *EMA of the student.* An extra 34M-param copy and a decay hyperparameter. It lags the model the bar
  measures, and stop-grad already gives a stable target.
- *Offline cache of teacher distributions.* 32,000 fp32 logits are 128 KB per frame, so tens of GB
  across the corpus. The cache would be a fixed checkpoint's distribution (stale by epoch 2), and it
  needs another job plus storage on the box.

**D2. The spoken rendering is the tape.** It is the only spoken rendering of training stems (render-1
voices only the holdout). B8 showed the gap is "typed vs any speech": typed‖tape = typed‖A. That makes
the tape a fair target.
- **Known ceiling:** over all content frames tape‖A is 1.8× the A‖B floor. A perfect typed→tape match
  is therefore expected to leave the ratio near **~1.8**, not ≤ 1.
- Closing the rest would need A-voiced training rows (a render-2). That is left to the GATE, not built here.

**D3. Frames:** the tutor-answer span right after each typed pupil turn in a typed row (`spans[k+1]` for
`k` in `typed_turns`).
- Only *content* frames count: target text ≠ PAD, and moshi's `text_mask` is true on both sides. These
  are the frames the content bar scores. The 3 lead PAD frames are out, and positions 0..end are in.
- Typed frame `typed_start + i` is paired with spoken frame `spoken_start + i`. The two spans carry
  identical text (asserted) and equal length.
- A typed turn that ends the dialogue has no answer and is skipped.
- Later answers (`k+2`, …) are not covered, matching the bar, which scores only the answer right after
  the swapped turn.

**D4. Text stream only.** The bar is a text-distribution KL. The 8 audio codebooks keep only their LM loss.

**D5. Direction = the bar's: KL(student_typed ‖ sg(teacher_spoken)).** The measured gap is KL(typed‖A).
The loss is pooled as a mean over every content frame in the batch, like the CE.
`--consistency-dir forward` (KL(teacher‖student)) exists for a later sweep only. The pilot does not use it.

**D6. Weight:** `loss = 2.0·text + 0.5·audio + λ·cons`, **λ = 10** for the pilot.
- At epoch 1–2 the content KL is ~0.02 nats/frame against a text CE of ~0.5. λ = 10 makes the term
  ~0.2, about 15% of the total at the start.
- A KL near zero has small gradients, and a 2% term (λ ≤ 2) is unlikely to move a ratio that rank, LR,
  data and base could not.
- A larger or smaller λ is a GATE option, not a pilot arm.

**D7. Every row keeps its LM loss** (mask, complement, and the spoken row). λ = 0 must take the
baseline's exact code path (no teacher forward, the same loss expression). That keeps R0 from ladder 1/2
valid as the comparison.

**D8. Batching is unchanged:** the same `batches()` and the same seed give the same steps.
- The teacher is a separate `no_grad` forward, one per distinct stem with a typed row in the batch,
  run *before* the student forward.
- Rejected: packing a stem's three rows into one batch so the spoken row's own (graph-carrying) forward
  serves as teacher. The longest row is 2,765 frames, so three rows reach ~8,000 frames, twice the
  4,000 that already peak at 49.4 GiB. It would also change batch composition and void R0.

**D9. Memory budget (target ≤ 56 GiB peak allocated under the 64 GiB cap):**
- The weights are already resident.
- The teacher pass keeps no activations (`no_grad`) and takes one spoken row at a time (≤ 2,765
  frames). Its transient logits are text fp32 2,765×32,000×4 B = 0.35 GiB and audio
  8×2,765×2,048×4 B = 0.18 GiB, freed before the student forward.
- The kept teacher log-probs cover at most the batch's content frames (≤ 4,000): 4,000×32,000×4 B = 0.49 GiB.
- Expected peak ≈ 49.4 + ≤ 1 ≈ **50.5 GiB**.
- The smoke measures the longest stem and the widest batches before any multi-epoch run.

**D10. Pilot: from scratch, seed 0, 3 epochs, 117 stems (`--slice 1`), λ = 10, compiled.** This is the
recipe whose R_gap (lowest-median-gap epoch of 1..3) defines R0 = 3.04/2.96, so `dual.noise` judges it
unchanged.
- Rejected: warm-starting from `ladder-personaplex-1` epoch 2. Three more epochs land in the overfit
  region (held text loss rises from epoch 3), and no R0 exists for that schedule.

**D11. Cost:**
- The teacher forward is roughly a third of a train step, and 2 of every 3 rows are typed, so expect
  **+20–25%: ~4,000 s/epoch**. The smoke prints the teacher seconds per step to confirm this.
- Pilot: ~3.4 h train + ~1.25 h measure (3 epochs). Smokes: ~0.5 h. **Total ≈ 5.2 GPU-hours.**

### 0.3 Batches

| batch | model | files | depends on | gate |
|---|---|---|---|---|
| C1 pure parts | sonnet | `src/dual/consistency.py`, `tests/test_consistency.py`, `src/dual/kl.py`, `tests/test_kl.py` | none | `make check` |
| C2 train wiring | opus | `src/dual/jobs/train.py`, `tests/test_train.py`, `tests/test_train_consistency.py` | C1 merged into the branch | `make check` |
| C3 orchestration | sonnet | `scripts/ladder.sh`, `tests/test_ladder_sh.py`, `Makefile` | none (parallel with C1) | `make check` |
| C4 run it | opus (orchestrator) | `docs/results/consistency-1.md`, this plan's "as run", `docs/plans/2026-10-26-dual-input-batches.md`, memory | C1–C3 merged to `main` | **GATE-0** (smoke), **GATE-1** (pilot ratio) |

C1 and C3 run in parallel. C2 starts after C1 lands on `feat/b9-consistency`. Merge to `main` after C2
and C3 pass review and `make check`. C4 runs from `main`.

---

## Batch C1: the pure parts (model: sonnet)

Files: `src/dual/consistency.py` (new), `tests/test_consistency.py` (new), `src/dual/kl.py`, `tests/test_kl.py`.

### Task C1.1: `src/dual/consistency.py` (TDD, pure, no torch)

**Files:**
- Create: `src/dual/consistency.py`
- Create: `tests/test_consistency.py`

**Step 1: Write the failing tests.** Create `tests/test_consistency.py`:

```python
import pytest

from dual.consistency import (
    AnswerPair,
    answer_pairs,
    flat_indices,
    longest_stem,
    spoken_by_stem,
)
from dual.constants import PAD
from dual.frames import tape_segment, to_record, typed_segment
from dual.render import assemble
from dual.turns import TurnSpan

OWN = [[7] * 8]
PUP = [[9] * 8]
# teacher, pupil (5 frames on tape), teacher, pupil (7), teacher
PARTS = (
    ("teacher", [PAD, PAD, 1, 40, 41, 2]),
    ("pupil", [PAD] * 5),
    ("teacher", [PAD, PAD, PAD, 1, 50, 51, 52, 2]),
    ("pupil", [PAD] * 7),
    ("teacher", [PAD, PAD, PAD, 1, 60, 2]),
)
TYPED = typed_segment([11, 12, 13])  # 3 tokens + 6 gap frames = 9 frames


def make_rows(typed_turns, parts=PARTS, stem="s"):
    spans, base, start = [], [], 0
    for i, (party, text) in enumerate(parts):
        n = len(text)
        spans.append(TurnSpan(i, party, (f"w{i}",), start, start, start + n))
        base.append(tape_segment(text, OWN * n, PUP * n, scored=party == "teacher"))
        start += n
    swaps = {k: ("typed", TYPED) for k in typed_turns}
    typed = to_record(assemble(f"{stem}#mask", spans, base, swaps))
    typed |= {"kind": "mask", "stem": stem, "typed_turns": list(typed_turns)}
    spoken = to_record(assemble(f"{stem}#spoken", spans, base, {}))
    spoken |= {"kind": "spoken", "stem": stem, "typed_turns": []}
    return typed, spoken


def test_one_typed_turn_pairs_its_answer_by_offset():
    typed, spoken = make_rows([1])
    # typed spans: [0,6) [6,15) [15,23) ...; spoken: [0,6) [6,11) [11,19) ...
    assert answer_pairs(typed, spoken) == [
        AnswerPair(turn=1, typed=(15, 23), spoken=(11, 19), content=(3, 4, 5, 6, 7))
    ]


def test_two_typed_turns_shift_the_later_answer_by_both():
    typed, spoken = make_rows([1, 3])
    pairs = answer_pairs(typed, spoken)
    assert [p.turn for p in pairs] == [1, 3]
    assert pairs[1] == AnswerPair(
        turn=3, typed=(32, 38), spoken=(26, 32), content=(3, 4, 5)
    )


def test_paired_frames_carry_the_same_target_text():
    typed, spoken = make_rows([1, 3])
    student, teacher = flat_indices(answer_pairs(typed, spoken))
    assert student == [18, 19, 20, 21, 22, 35, 36, 37]
    assert teacher == [14, 15, 16, 17, 18, 29, 30, 31]
    assert [typed["text"][i] for i in student] == [spoken["text"][i] for i in teacher]
    assert PAD not in [typed["text"][i] for i in student]


def test_no_typed_turn_no_pairs():
    typed, spoken = make_rows([])
    assert answer_pairs(typed, spoken) == []
    assert flat_indices([]) == ([], [])


def test_a_typed_turn_that_ends_the_dialogue_has_no_answer():
    parts = (*PARTS, ("pupil", [PAD] * 4))
    typed, spoken = make_rows([1, 5], parts=parts)
    assert [p.turn for p in answer_pairs(typed, spoken)] == [1]


def test_refuses_a_row_of_another_stem():
    typed, _ = make_rows([1])
    _, other = make_rows([], stem="t")
    with pytest.raises(ValueError, match="stem"):
        answer_pairs(typed, other)


def test_refuses_a_teacher_row_that_is_not_the_spoken_kind():
    typed, _ = make_rows([1])
    with pytest.raises(ValueError, match="spoken"):
        answer_pairs(typed, typed)


def test_refuses_a_typed_turn_the_row_does_not_type():
    typed, spoken = make_rows([1])
    typed["typed_turns"] = [3]  # span 3 is tape in this row
    with pytest.raises(ValueError, match="not typed"):
        answer_pairs(typed, spoken)


def test_refuses_answers_whose_text_differs():
    typed, spoken = make_rows([1])
    spoken["text"][14] = 99  # inside the spoken answer to turn 1
    with pytest.raises(ValueError, match="text differs"):
        answer_pairs(typed, spoken)


def test_refuses_rows_with_different_turn_counts():
    typed, spoken = make_rows([1])
    spoken["spans"] = spoken["spans"][:-1]
    with pytest.raises(ValueError, match="spans"):
        answer_pairs(typed, spoken)


def test_spoken_by_stem_maps_each_stem_to_its_spoken_row():
    a_typed, a_spoken = make_rows([1], stem="a")
    b_typed, b_spoken = make_rows([3], stem="b")
    got = spoken_by_stem([a_typed, a_spoken, b_typed, b_spoken])
    assert got == {"a": a_spoken, "b": b_spoken}


def test_spoken_by_stem_refuses_a_typed_stem_without_its_spoken_row():
    a_typed, _ = make_rows([1], stem="a")
    with pytest.raises(ValueError, match="no spoken row"):
        spoken_by_stem([a_typed])


def test_spoken_by_stem_refuses_two_spoken_rows_for_one_stem():
    _, a_spoken = make_rows([], stem="a")
    with pytest.raises(ValueError, match="two spoken rows"):
        spoken_by_stem([a_spoken, dict(a_spoken)])


def test_longest_stem_is_the_stem_of_the_longest_row_first_by_name_on_ties():
    rows = [
        {"stem": "b", "frames": 10},
        {"stem": "a", "frames": 30},
        {"stem": "c", "frames": 30},
    ]
    assert longest_stem(rows) == "a"
    with pytest.raises(ValueError):
        longest_stem([])
```

**Step 2: Run them and watch them fail.**
Run: `uv run pytest tests/test_consistency.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'dual.consistency'`.

**Step 3: Implement.** Create `src/dual/consistency.py`:

```python
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
```

**Step 4: Run the tests.**
Run: `uv run pytest tests/test_consistency.py -q`
Expected: `14 passed`.
- If `test_two_typed_turns_...` fails on spans, print `typed["spans"]`. The expected spans are
  `[0,6) [6,15) [15,23) [23,32) [32,38)` for the typed row and `[0,6) [6,11) [11,19) [19,26) [26,32)`
  for the spoken row.
- Fix the code, not the numbers.

**Step 5: Commit.**
```bash
make check
git add src/dual/consistency.py tests/test_consistency.py
git commit -m "feat(b9): dual.consistency pairs each typed answer with its spoken answer"
```

### Task C1.2: `consistency_loss` in `src/dual/kl.py` (TDD; torch on CPU)

**Files:**
- Modify: `src/dual/kl.py` (append after `kl_per_frame`, around line 24)
- Modify: `tests/test_kl.py` (append)

**Step 1: Write the failing tests.** Append to `tests/test_kl.py`:

```python
from dual.kl import DIRECTIONS, consistency_loss


def _logits(seed: int, n: int = 5, v: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, v, generator=g)


def test_consistency_is_zero_when_student_equals_teacher():
    x = _logits(0)
    assert float(consistency_loss(x, log_probs(x))) == 0.0


def test_consistency_bar_direction_is_the_bars_kl():
    s, t = _logits(1), _logits(2)
    want = kl_per_frame(log_probs(s), log_probs(t)).mean()
    assert torch.allclose(consistency_loss(s, log_probs(t), "bar"), want)


def test_consistency_forward_direction_swaps_the_arguments():
    s, t = _logits(1), _logits(2)
    want = kl_per_frame(log_probs(t), log_probs(s)).mean()
    assert torch.allclose(consistency_loss(s, log_probs(t), "forward"), want)


def test_consistency_gradient_reaches_the_student_only():
    s = _logits(3).requires_grad_(True)
    t_logits = _logits(4).requires_grad_(True)
    consistency_loss(s, log_probs(t_logits)).backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    assert t_logits.grad is None


def test_consistency_on_no_frames_is_zero_and_backpropagates():
    s = torch.zeros(0, 7, requires_grad=True)
    out = consistency_loss(s, torch.zeros(0, 7))
    out.backward()
    assert float(out.detach()) == 0.0


def test_consistency_refuses_bad_shapes_and_directions():
    with pytest.raises(ValueError, match="shapes"):
        consistency_loss(_logits(0, n=3), log_probs(_logits(1, n=4)))
    with pytest.raises(ValueError, match="direction"):
        consistency_loss(_logits(0), log_probs(_logits(1)), "reverse")
    assert DIRECTIONS == ("bar", "forward")
```

Make sure `tests/test_kl.py` already imports `torch`, `pytest`, `log_probs` and `kl_per_frame`. Add any
of them that are missing to its top imports. Ruff will refuse an import in the middle of the module, so
put the new `from dual.kl import DIRECTIONS, consistency_loss` at the top as well (merge it into the
existing `from dual.kl import ...` line).

**Step 2: Run and watch them fail.**
Run: `uv run pytest tests/test_kl.py -q`
Expected: `ImportError: cannot import name 'DIRECTIONS'`.

**Step 3: Implement.** In `src/dual/kl.py`, right after `kl_per_frame`:

```python
DIRECTIONS = ("bar", "forward")


def consistency_loss(
    student_logits: torch.Tensor, teacher_lp: torch.Tensor, direction: str = "bar"
) -> torch.Tensor:
    """Mean per-frame KL between the student's text distribution and a fixed teacher's.

    student_logits [N, V] (gradient flows); teacher_lp [N, V] log-probs, detached here.
    "bar": KL(student || teacher), the bar's own direction (gap = KL(typed || A)).
    "forward": KL(teacher || student). No frames: a zero that still backpropagates."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}: one of {DIRECTIONS}")
    if student_logits.shape != teacher_lp.shape:
        raise ValueError(
            f"shapes {tuple(student_logits.shape)} vs {tuple(teacher_lp.shape)}"
        )
    if student_logits.shape[0] == 0:
        return student_logits.float().sum() * 0.0
    lp_s = log_probs(student_logits)
    lp_t = teacher_lp.detach().float()
    if direction == "bar":
        return kl_per_frame(lp_s, lp_t).mean()
    return kl_per_frame(lp_t, lp_s).mean()
```

**Step 4: Run the tests.**
Run: `uv run pytest tests/test_kl.py -q`
Expected: all pass (the existing ones plus 6 new).

**Step 5: Commit.**
```bash
make check
git add src/dual/kl.py tests/test_kl.py
git commit -m "feat(b9): kl.consistency_loss, the bar-direction KL toward a detached teacher"
```

### Task C1.3: batch close

Run `make check` (expected: ruff clean, mypy `Success`, every test passing). Push the branch:
`git push -u origin feat/b9-consistency`. Report the two commit SHAs.

---

## Batch C2: the train job learns the term (model: opus)

Files: `src/dual/jobs/train.py`, `tests/test_train.py`, `tests/test_train_consistency.py` (new).
Starts after C1 is on `feat/b9-consistency` (it imports `dual.consistency` and `kl.consistency_loss`).

### Task C2.1: flags, the resume check and the smoke selectors (TDD, CPU)

**Files:**
- Modify: `src/dual/jobs/train.py` (`parse_args`, new pure helpers after `check_resume_lr`)
- Modify: `tests/test_train.py`

**Step 1: Failing tests.** Append to `tests/test_train.py`, and extend its import line to
`from dual.jobs.train import (batches, check_resume_consistency, check_resume_lr, parse_args, select_rows, widest_first)`:

```python
def test_consistency_flags_default_to_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--name", "n", "--base", "b"])
    a = parse_args()
    assert (a.consistency, a.consistency_dir) == (0.0, "bar")
    assert (a.only_stem, a.max_steps, a.widest_first) == (None, 0, False)


def test_consistency_flags_are_parsed(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        "train --name n --base b --consistency 10 --consistency-dir forward "
        "--only-stem longest --max-steps 3 --widest-first".split(),
    )
    a = parse_args()
    assert (a.consistency, a.consistency_dir) == (10.0, "forward")
    assert (a.only_stem, a.max_steps, a.widest_first) == ("longest", 3, True)


def test_a_negative_weight_is_refused(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", "train --name n --base b --consistency -1".split()
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_resume_accepts_the_same_consistency_and_an_older_checkpoint():
    check_resume_consistency({"consistency": 10.0, "consistency_dir": "bar"}, 10.0, "bar")
    check_resume_consistency({"epoch": 1}, 0.0, "bar")  # predates the flag: off


def test_resume_refuses_another_consistency_setup():
    with pytest.raises(ValueError, match="consistency"):
        check_resume_consistency({"consistency": 10.0}, 3.0, "bar")
    with pytest.raises(ValueError, match="consistency"):
        check_resume_consistency({"consistency": 10.0}, 10.0, "forward")


def test_select_rows_keeps_one_stem_or_the_longest():
    rs = [
        {"stem": "a", "frames": 5},
        {"stem": "b", "frames": 9},
        {"stem": "b", "frames": 2},
    ]
    assert select_rows(rs, None) == rs
    assert select_rows(rs, "a") == [rs[0]]
    assert select_rows(rs, "longest") == [rs[1], rs[2]]
    with pytest.raises(ValueError, match="no rows"):
        select_rows(rs, "zzz")


def test_widest_first_orders_batches_by_padded_frames():
    bs = [[{"frames": 10}] * 3, [{"frames": 100}], [{"frames": 40}] * 2]
    assert [len(b) * max(r["frames"] for r in b) for b in widest_first(bs)] == [
        100,
        80,
        30,
    ]
```

**Step 2: Run and watch them fail.** `uv run pytest tests/test_train.py -q` fails with `ImportError`.

**Step 3: Implement** in `src/dual/jobs/train.py`.

In `parse_args`, before `mode = ...`:

```python
    p.add_argument(
        "--consistency",
        type=float,
        default=0.0,
        help="weight of KL(typed answer || spoken answer); 0 = the baseline recipe",
    )
    p.add_argument("--consistency-dir", choices=DIRECTIONS, default="bar")
    p.add_argument(
        "--only-stem",
        default=None,
        help="train on one stem's rows only ('longest' = the stem of the longest row)",
    )
    p.add_argument(
        "--max-steps", type=int, default=0, help="stop after N steps, no checkpoint"
    )
    p.add_argument(
        "--widest-first",
        action="store_true",
        help="run the batches with the most padded frames first (smoke)",
    )
```

At the end of `parse_args`, replace `return p.parse_args()` with:

```python
    args = p.parse_args()
    if args.consistency < 0:
        p.error(f"--consistency {args.consistency}: must be >= 0")
    return args
```

Add `from dual.kl import DIRECTIONS, consistency_loss, log_probs` to the module imports (`train.py` is a
GPU job, so a torch import is fine). After `check_resume_lr`, add:

```python
def check_resume_consistency(
    run_meta: dict[str, Any], consistency: float, direction: str
) -> None:
    """--resume keeps the recipe: the same weight and direction. A checkpoint without
    the keys predates B9 and ran without the term."""
    had = float(run_meta.get("consistency", 0.0))
    had_dir = str(run_meta.get("consistency_dir", "bar"))
    if (had, had_dir) != (consistency, direction):
        raise ValueError(
            f"--resume: checkpoint ran at consistency {had} ({had_dir}), "
            f"flags say {consistency} ({direction})"
        )


def select_rows(rows: list[dict[str, Any]], only: str | None) -> list[dict[str, Any]]:
    """All rows, or one stem's ('longest': the stem of the longest row)."""
    from dual.consistency import longest_stem

    if only is None:
        return rows
    stem = longest_stem(rows) if only == "longest" else only
    kept = [r for r in rows if r["stem"] == stem]
    if not kept:
        raise ValueError(f"--only-stem {only}: no rows")
    return kept


def widest_first(bs: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Batches by padded frames (rows x longest), largest first; stable on ties."""
    return sorted(bs, key=lambda b: -len(b) * max(r["frames"] for r in b))
```

**Step 4:** `uv run pytest tests/test_train.py -q` is all green.
**Step 5:** `make check`, then commit `feat(b9): train flags for the consistency term and the smoke selectors`.

### Task C2.2: `build_teachers` and the consistency term in `losses` (TDD on CPU with a fake forward)

**Files:**
- Modify: `src/dual/jobs/train.py` (`losses`, new `build_teachers`)
- Create: `tests/test_train_consistency.py`

**Step 1: Failing tests.** Create `tests/test_train_consistency.py`. It reuses C1's fixture rows and
replaces `dual.model.forward` and `dual.model.codes_from_row` with CPU fakes. `losses` and
`build_teachers` import those names *inside* the function, so `monkeypatch.setattr(dual.model, ...)`
reaches them.

```python
import functools

import pytest
import torch
import torch.nn.functional as F

import dual.model
from dual.constants import SINE
from dual.jobs.train import build_teachers, losses
from dual.model import Forward
from test_consistency import make_rows

V = 64


class FakeLM(torch.nn.Module):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(c))


def fake_forward(lm, codes):
    """Text logits peaked on each frame's own text id, tilted by how many typed (SINE)
    pupil frames came before: typed and spoken renderings differ iff lm.c != 0."""
    text = codes[:, 0]
    b, t = text.shape
    seen = torch.cumsum((codes[:, 9] == SINE[0]).float(), dim=-1)
    tilt = torch.arange(V, dtype=torch.float32) / V
    logits = 3.0 * F.one_hot(text % V, V).float() + lm.c * seen[..., None] * tilt
    return Forward(
        text_logits=logits,
        text_mask=torch.ones(b, t, dtype=torch.bool),
        audio_logits=torch.zeros(b, 8, t, 2048),
        audio_mask=torch.ones(b, 8, t, dtype=torch.bool),
    )


@pytest.fixture(autouse=True)
def cpu_model(monkeypatch):
    monkeypatch.setattr(dual.model, "forward", fake_forward)
    real = dual.model.codes_from_row
    monkeypatch.setattr(
        dual.model, "codes_from_row", functools.partial(real, device="cpu")
    )


def test_teachers_cover_the_typed_row_only_at_its_answer_content_frames():
    typed, spoken = make_rows([1])
    teachers = build_teachers(FakeLM(1.0), [spoken, typed], {"s": spoken})
    assert list(teachers) == [1]
    idx, lp = teachers[1]
    assert idx.tolist() == [18, 19, 20, 21, 22]
    assert lp.shape == (5, V)
    assert not lp.requires_grad


def test_zero_when_typed_and_spoken_agree():
    typed, spoken = make_rows([1, 3])
    lm = FakeLM(0.0)
    teachers = build_teachers(lm, [typed], {"s": spoken})
    _, _, cons = losses(lm, [typed], teachers, "bar")
    assert float(cons) == 0.0


def test_positive_and_trainable_when_they_differ():
    typed, spoken = make_rows([1, 3])
    lm = FakeLM(1.0)
    teachers = build_teachers(lm, [typed], {"s": spoken})
    _, _, cons = losses(lm, [typed], teachers, "bar")
    assert float(cons) > 0
    cons.backward()
    assert lm.c.grad is not None and float(lm.c.grad) != 0.0


def test_the_lm_losses_do_not_change_with_the_term():
    typed, spoken = make_rows([1])
    lm = FakeLM(1.0)
    base = losses(lm, [typed, spoken])
    with_term = losses(lm, [typed, spoken], build_teachers(lm, [typed, spoken], {"s": spoken}))
    assert torch.equal(base[0], with_term[0]) and torch.equal(base[1], with_term[1])
    assert float(base[2]) == 0.0


def test_teacher_frames_line_up_inside_a_padded_batch():
    # the short typed row is padded to the spoken row's width; indices stay the row's own
    typed, spoken = make_rows([1])
    long_typed, long_spoken = make_rows([1, 3], stem="t")
    lm = FakeLM(0.0)
    batch = [typed, long_typed]
    teachers = build_teachers(lm, batch, {"s": spoken, "t": long_spoken})
    _, _, cons = losses(lm, batch, teachers, "bar")
    assert float(cons) == 0.0
```

`tests/` is on the pytest rootdir path (`pythonpath = ["src", "vendor"]` in `pyproject.toml`), but
`from test_consistency import make_rows` needs `tests/` importable.
- If it is not (pytest's default `rootdir`-relative import mode usually allows it), move `make_rows`,
  `PARTS`, `OWN`, `PUP` and `TYPED` into a new `tests/rows_fixture.py` and import from there in both
  test files. That changes C1's test file, so do it in this task and commit both files together.
- The stem `"t"` in the last test must be a key of the spoken map. `make_rows(..., stem="t")` sets it.

**Step 2: Run and watch them fail** (`ImportError: build_teachers`).

**Step 3: Implement** in `src/dual/jobs/train.py`. Replace `losses` with:

```python
Teacher = tuple[torch.Tensor, torch.Tensor]  # typed-row frame index [N], log-probs [N, V]


def losses(
    lm: Any,
    batch: list[dict[str, Any]],
    teachers: dict[int, Teacher] | None = None,
    direction: str = "bar",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Text and audio CE over scored frames only, and the consistency KL of each typed
    answer toward its teacher (zero without teachers). Rows are padded to the longest;
    padding is unscored. NaN logits at moshi's masked frames are excluded by index,
    never multiplied."""
    from dual.model import codes_from_row, forward

    width = max(r["frames"] for r in batch)
    codes = torch.cat(
        [
            torch.nn.functional.pad(
                codes_from_row(r), (0, width - r["frames"]), value=3
            )
            for r in batch
        ]
    )
    scored = torch.zeros(len(batch), width, dtype=torch.bool, device=codes.device)
    for i, r in enumerate(batch):
        scored[i, : r["frames"]] = torch.tensor(r["scored"], dtype=torch.bool)
    out = forward(lm, codes)
    text_target = codes[:, 0]
    text_keep = scored & out.text_mask
    text = torch.nn.functional.cross_entropy(
        out.text_logits[text_keep], text_target[text_keep]
    )
    audio_target = codes[:, 1:9]
    audio_keep = scored[:, None, :] & out.audio_mask
    audio = torch.nn.functional.cross_entropy(
        out.audio_logits[audio_keep], audio_target[audio_keep]
    )
    students: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for i, (idx, lp) in (teachers or {}).items():
        keep = out.text_mask[i, idx]
        students.append(out.text_logits[i, idx[keep]])
        targets.append(lp[keep])
    if students:
        cons = consistency_loss(torch.cat(students), torch.cat(targets), direction)
    else:
        cons = out.text_logits.new_zeros(())
    return text, audio, cons


def build_teachers(
    lm: Any, batch: list[dict[str, Any]], spoken: dict[str, dict[str, Any]]
) -> dict[int, Teacher]:
    """For each typed row of the batch (by position), the frame indices of its answer
    content frames and the model's own text log-probs at the same frames after the
    spoken (all-tape) rendering of its stem: a no_grad forward with the current adapter,
    one stem at a time, so a single spoken forward is alive at once."""
    from dual.consistency import answer_pairs, flat_indices
    from dual.model import codes_from_row, forward

    out: dict[int, Teacher] = {}
    have: tuple[str, Any] | None = None
    with torch.no_grad():
        for i in sorted(range(len(batch)), key=lambda j: str(batch[j]["stem"])):
            row = batch[i]
            if not row.get("typed_turns"):
                continue
            stem = row["stem"]
            student, teacher = flat_indices(answer_pairs(row, spoken[stem]))
            if not student:
                continue
            if have is None or have[0] != stem:
                have = None  # drop the previous stem's logits first
                have = (stem, forward(lm, codes_from_row(spoken[stem])))
            t = have[1]
            s_idx = torch.tensor(student, device=t.text_logits.device)
            t_idx = torch.tensor(teacher, device=t.text_logits.device)
            keep = t.text_mask[0, t_idx]
            out[i] = (s_idx[keep], log_probs(t.text_logits[0, t_idx[keep]]))
    return out
```

The held-out evaluation line `held_text = [float(losses(lm, [r])[0]) for r in held]` works unchanged:
it takes element 0 of a 3-tuple.

**Step 4:** `uv run pytest tests/test_train_consistency.py tests/test_train.py -q` is green.
**Step 5:** `make check`, then commit `feat(b9): build_teachers and the consistency KL in train.losses`.

### Task C2.3: wire it into `main` (the step loop, logs, checkpoint meta, smoke stop)

**Files:** Modify `src/dual/jobs/train.py` (`main`). No new test: this is box-only glue, and C4's smoke
checks it. Keep the λ = 0 path identical to today's.

1. After `rows = [...]` and before `held = ...`:
   ```python
   spoken = spoken_by_stem(rows) if args.consistency > 0 else {}
   rows = select_rows(rows, args.only_stem)
   ```
   Import `spoken_by_stem` from `dual.consistency` inside `main`, next to the other in-function imports.
   `spoken` is built from the whole rung, so a selected stem still finds its spoken row.

2. Extend the startup print with
   `f"consistency {args.consistency} ({args.consistency_dir}); typed rows {sum(bool(r['typed_turns']) for r in rows)}"`.

3. In the `--resume` branch, after `check_resume_lr(...)`, inside the same `try`:
   `check_resume_consistency(blob.get("run", {}), args.consistency, args.consistency_dir)`.

4. Replace the step loop body. Keep the λ = 0 loss expression literally as before:
   ```python
            plan = batches(rows, rng, limit)
            if args.widest_first:
                plan = widest_first(plan)
            for i, batch in enumerate(plan):
                t_step = time.monotonic()
                teachers = (
                    build_teachers(lm, batch, spoken) if args.consistency > 0 else None
                )
                t_teacher = time.monotonic() - t_step
                text, audio, cons = losses(lm, batch, teachers, args.consistency_dir)
                loss = TEXT_WEIGHT * text + AUDIO_WEIGHT * audio
                if args.consistency > 0:
                    loss = loss + args.consistency * cons
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                norm = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
                opt.step()
                vals = (
                    float(loss.detach()),
                    float(text.detach()),
                    float(audio.detach()),
                    norm,
                    float(cons.detach()),
                )
                if not all(map(math.isfinite, vals)):
                    raise RuntimeError(f"epoch {epoch} step {i}: non-finite {vals}")
                norms.append(norm)
                totals.append(vals[0])
                conses.append(vals[4])
                frames_t = sum(len(idx) for idx, _ in (teachers or {}).values())
                run.step(
                    loss={
                        "total": vals[0],
                        "text": vals[1],
                        "audio": vals[2],
                        "consistency": vals[4],
                    },
                    grad_norm=norm,
                    learning_rate=args.lr,
                    epoch=epoch - 1 + (i + 1) / steps_per_epoch,
                )
                done += 1
                if i % 10 == 0 or args.max_steps:
                    print(
                        f"epoch {epoch} step {i}: loss {vals[0]:.4f} text "
                        f"{vals[1]:.4f} audio {vals[2]:.4f} cons {vals[4]:.5f} "
                        f"({frames_t} frames) grad {norm:.3f} "
                        f"rows {len(batch)}x{max(r['frames'] for r in batch)} "
                        f"teacher {t_teacher:.1f} s step "
                        f"{time.monotonic() - t_step:.1f} s"
                    )
                if args.max_steps and done >= args.max_steps:
                    print(
                        f"smoke: {done} steps, peak mem "
                        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; "
                        "no checkpoint written"
                    )
                    return
   ```
   - Initialise `done = 0` before the `with track(...)`, and `conses: list[float] = []` next to `norms`/`totals`.
   - `steps_per_epoch` stays `len(batches(rows, random.Random(0), limit))`, computed on the selected rows.
   - `return` inside `with track(...)` closes the run cleanly. Do not `sys.exit` there.

5. Epoch summary print: add `f"mean cons {sum(conses) / len(conses):.5f} "`. In the checkpoint meta
   dict, add `"consistency": args.consistency, "consistency_dir": args.consistency_dir,
   "only_stem": args.only_stem`.

6. Do **not** change `TEXT_WEIGHT`, `AUDIO_WEIGHT`, `LR`, `batches()` or the held-out eval.

**Verify:**
- `make check` is green.
- `uv run python -c "import dual.jobs.train"` works on the laptop (torch only, no moshi at import).
- Read the diff once against D7: with `--consistency 0`, `build_teachers` is never called and `loss`
  is `TEXT_WEIGHT * text + AUDIO_WEIGHT * audio` exactly.

Commit `feat(b9): train applies the consistency term, logs it, and stops early for a smoke`.

### Task C2.4: batch close

- `make check`.
- Request a code review (superpowers:requesting-code-review) of `git diff main...feat/b9-consistency
  -- src tests`. Point the reviewer at D1–D9, and ask specifically:
  - Is the λ = 0 path bit-identical to the baseline?
  - Do the student and teacher indices align inside padded batches?
  - Can NaN logits ever be indexed?
  - Is there any torch import at module level in `dual/consistency.py`?
- Fix the findings, then push.

---

## Batch C3: ladder 3 and the noise variants (model: sonnet)

Files: `scripts/ladder.sh`, `tests/test_ladder_sh.py`, `Makefile`. Independent of C1/C2. Run it in
parallel on the same branch, in a separate worktree, then rebase.

### Task C3.1: `LADDER=3` runs the pilot (TDD on `tests/test_ladder_sh.py`)

**Step 1: Failing tests.** In `tests/test_ladder_sh.py`:
- Change `test_unknown_ladder_exits_2` to use `box(LADDER="9")`, since 3 now exists.
- Append:

```python
def test_ladder_3_submits_the_pilot_then_its_measure(box):
    r, log = box(LADDER="3")
    assert r.returncode == 0, r.stderr
    assert submits(log) == [
        "make IMAGE=img:tag -s train NAME=cons1-pp-l10 BASE=personaplex SLICE=1"
        " RENDER=render-1 EPOCHS=3 ARGS=--consistency 10",
        "make IMAGE=img:tag -s measure NAME=cons1-pp-l10 BASE=personaplex"
        " RUN=cons1-pp-l10 RENDER=render-1",
    ]
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_3_train_carries_exactly_one_args_word(box, tmp_path):
    r, _ = box(LADDER="3")
    assert r.returncode == 0, r.stderr
    (train,) = [c for c in argv_calls(tmp_path) if "train" in c]
    assert [w for w in train if w.startswith("ARGS=")] == ["ARGS=--consistency 10"]


def test_ladder_3_rejects_a_ladder_2_run_name(box):
    r, log = box(LADDER="3", FROM="ladder2-pp-s1")
    assert r.returncode == 2
    assert "cons1-pp-l10" in r.stdout + r.stderr
    assert submits(log) == []
```

**Step 2:** `uv run pytest tests/test_ladder_sh.py -q`. The 3 new tests fail.

**Step 3: Implement** in `scripts/ladder.sh`:
- In the header comment, after the LADDER=2 sentence, add: "LADDER=3: the B9 consistency pilot, one
  personaplex SLICE=1 EPOCHS=3 run, cons1-pp-l10 (--consistency 10)."
- In the `case "$LADDER"`, before `*)`:
  ```sh
    3) RUNS="cons1-pp-l10,personaplex,1,3,--consistency,10" ;;
  ```
- In the `*)` message, change `(1 or 2)` to `(1, 2 or 3)`.

**Step 4:** `uv run pytest tests/test_ladder_sh.py -q` is all green.

### Task C3.2: `make noise VARIANTS=...`

In `Makefile`, replace the `noise` recipe with:

```make
.PHONY: noise
NOISE_SEEDS ?= personaplex-ladder-personaplex-1 personaplex-ladder2-pp-s1
VARIANTS ?= personaplex-ladder2-pp-lr5e-6 personaplex-ladder2-pp-lr1e-4 personaplex-ladder2-pp-r64 personaplex-ladder2-pp-r4
noise:
	PYTHONPATH=src uv run python -m dual.noise runs/ --seeds $(NOISE_SEEDS) --variants $(VARIANTS)
```

Check that `make -n noise` prints the same command as before:
`make -n noise | grep -c 'ladder2-pp-r4'` prints 1. Then check
`make -n noise VARIANTS=personaplex-cons1-pp-l10` ends in `--variants personaplex-cons1-pp-l10`.

### Task C3.3: batch close

Run `make check` and commit `feat(b9): ladder 3 runs the consistency pilot; make noise takes VARIANTS`.
Push, then rebase onto C2's head if C2 landed first. Neither batch touches the other's files.

---

## Batch C4: run it (model: opus; the orchestrator)

Preconditions:
- C1–C3 are merged to `main` with `make check` green.
- Read the `operating-the-sparks-queue` and `authoring-a-sparks-job` skills.
- Vlad's standing permissions:
  - drama3: the orchestrator may run `cd ~/drama3 && docker compose stop` on the box before GPU runs.
    Never remove or edit its container, compose file or crontab.
  - HF_TOKEN: it lives in the untracked `.env`. Never commit it, echo it, log it, or pass it on a
    command line. No step here needs it, because the weights are already on the box.

### Task C4.1: image and box

1. `SHA=$(git rev-parse --short HEAD)` on `main`.
2. `rm -rf docker/src docker/vendor && cp -RL src docker/src && cp -RL vendor docker/vendor`.
3. `DOCKER_BUILDKIT=0 docker build -t 100.114.56.75:5000/vlad/cons1:$SHA docker && docker push 100.114.56.75:5000/vlad/cons1:$SHA`.
4. Set `IMAGE` to the registry form earlier submits used. Copy the exact prefix from
   `sparks status job-20261105-141338-vlad-measure-b8-pos-pp`; per B6.4 it is
   `spark.local:5000/vlad/<name>:<sha>`.
5. `ssh -o ConnectTimeout=15 vlad@100.114.56.75 'cd ~/drama3 && docker compose stop; free -g'`.
   Expect about 110 to 118 GiB available.
6. `sparks queue --json` must be empty before any submit. Run one job at a time.

### Task C4.2: smoke and **GATE-0**

Two short jobs, one after the other. Each loads the 7B model, so each takes ~10–15 min.

- `IMAGE=... make train NAME=cons-smoke-1 EPOCHS=1 ARGS='--consistency 10 --only-stem longest --max-steps 3'`
  covers one dialogue, the longest: its 3 rows, one step each, with the teacher on a ~2,765-frame row.
- `IMAGE=... make train NAME=cons-smoke-wide EPOCHS=1 ARGS='--consistency 10 --widest-first --max-steps 4'`
  covers the 4 widest batches of the full train set, the worst case for memory.

Read the full log (`/srv/spark/runs/run-*-cons-smoke-*/output.log`; `sparks logs` truncates). Record:
- the `smoke: ... peak mem` line;
- `cons` (finite; > 0 on steps with typed rows; expected 0.005–0.1);
- the teacher frames per step;
- `teacher s / step s`. Ignore step 0, which includes compile.

**GATE-0, decided by the orchestrator. Ask Vlad only on a fail.**
- Pass: peak ≤ 56 GiB in both smokes, every `cons` finite, `cons > 0` on some step, and teacher
  share ≤ 40% of step time. Go to C4.3.
- Peak 56–62 GiB, or a teacher share > 40%: stop and ask Vlad.
  - Options: (a) run anyway, with ~X h per epoch stated; (b) store the teacher log-probs in bf16 and
    re-smoke; (c) lower `--frames-per-batch` to 3000. That changes the batches and voids R0, so it
    needs a λ = 0 control run.
- OOM, a non-finite value, or `cons == 0` everywhere: stop and debug (superpowers:systematic-debugging)
  before anything else.
- Also check that the startup line of `cons-smoke-wide` prints the same `steps/epoch` as
  `ladder-personaplex-1`'s output.log. This confirms the batches are unchanged (D8).

### Task C4.3: the pilot

1. Stop drama3 again (someone has restarted it during runs). Optionally keep the Mac-side watchdog
   from `drama3-memory-hog.md` running, which stops it every 120 s.
2. Launch detached:
   `LADDER=3 IMAGE=<tag> SPARKS_HOST=vlad@100.114.56.75 nohup caffeinate -i sh scripts/ladder.sh > /tmp/ladder3.log 2>&1 &`
   This runs train `cons1-pp-l10` (3 epochs, ~3.4 h), then `measure-cons1-pp-l10` (~75 min), then
   `make bar`.
3. Watch it with bounded ssh (`ConnectTimeout=15`) and `sparks status --json`.
   - Epoch time well above ~4,000 s, or ballooning, preceded the earlier freezes. If the box drops
     off tailscale, follow `box-outages.md`.
   - A hung train resumes with `make train NAME=cons1-pp-l10 ... ARGS='--consistency 10 --resume'`,
     then a manual measure, as in `b6-ladder-2-run.md`.
4. Write a memory note `b9-cons-pilot-run.md` (run names, image tag, job ids, how to resume) and add
   it to `MEMORY.md`.

### Task C4.4: read the pilot

On the laptop, after `make bar`:
1. `make noise VARIANTS=personaplex-cons1-pp-l10`: R_gap and its epoch, R, median gap, floor, p90
   ratio, both paired CIs, and the verdict (moved down / up / not moved / down, floor inflated).
2. `make position FILES="runs/measure/personaplex-cons1-pp-l10/epoch-0E.jsonl runs/measure/personaplex-ladder-personaplex-1-pos/epoch-02.jsonl"`,
   where E is the R_gap epoch. This compares by-position ratios with B8's epoch 2 (0–10 vs 11–25
   vs 26+).
3. From the train log: held text loss per epoch against ladder-personaplex-1 (0.551 / 0.540 / 0.611),
   and mean `cons` per epoch. Did the training-side KL fall?
4. From `summary.json`: `tape_gap_median` and `tape_floor_median` at E. The ceiling check: did
   typed‖tape fall toward tape‖A?

### Task C4.5: `docs/results/consistency-1.md` and the record

- **The answer first:** did the term move the content ratio (the noise verdict and numbers), and at
  what cost to held text loss and to the floor.
- **Tables:** the noise row next to R0; bar-by-position against B8 epoch 2; per-epoch held loss and
  mean `cons`; typed‖tape against tape‖A (the D2 ceiling).
- **What was run:** commits, image, job ids, times, peak memory (smoke and pilot).
- **The GATE (GATE-1)** with the options below and a recommendation.

Then:
- Append `### B9 as run (date)` to this plan file (Code, Jobs, Result, Reviews bullets).
- Add a short `## Batch B9: consistency loss (date)` section to
  `docs/plans/2026-10-26-dual-input-batches.md`: Vlad's quote, a pointer to this plan, and the
  result line.
- `make check`, then commit on a branch `docs/b9-consistency-1` and merge.
- Update the memory notes (`b9-cons-pilot-run.md` marked DONE; new `cons-pilot-result.md`).

**GATE-1: "the consistency pilot moved the ratio by X; what next?"** Present it to Vlad and do not
start another GPU run before his answer. Options:
- (a) **Moved down and still well above ~1.8:** a λ step (30), plus a second seed of λ = 10 to confirm.
  2 runs, ~10 GPU-hours.
- (b) **Moved down to near the tape ceiling (~1.8–2.2):** render A-voiced spoken training rows
  (render-2, the TTS on 117 stems) and distil toward voice A, which the bar's floor is built on.
  This needs a new render job and its own plan.
- (c) **Not moved, with the training-side `cons` falling:** the term fits the training rows but does
  not generalise. Try `--consistency-dir forward`, or cover every answer after a typed turn (not only
  the next). Or stop, and take B8's option 3 (the tape‖A yardstick) or 4 (go to the engine on B7's
  evidence).
- (d) **Not moved, with `cons` flat:** λ is too small to bite. Run λ = 30–100 once and then decide.
- (e) **Held text loss up > 5% at the R_gap epoch, or "down, floor inflated":** the term hurts the
  answer. Lower λ (3), or stop.

---

## Appendix: what this plan could not settle (for the GATE, not for the executor)

- **λ = 10** is reasoned, not measured (D6). The pilot is the measurement.
- **The tape ceiling (~1.8)** is from B8's epoch-2 tape‖A. A model trained with the term may move
  tape‖A as well.
- Whether `torch.compile` recompiles for the teacher's B = 1 shapes enough to matter. The smoke's
  per-step timings will show it, and D11's +20–25% is an estimate.
- A single pilot seed. The noise threshold T = 0.5 comes from the baseline seeds, and a "moved" verdict
  is confirmed by a second seed only at GATE-1 (a).

---

### B9 as run (2026-11-05 to 11-06)

- **Code:**
  - C1 (`dual.consistency`, `kl.consistency_loss`), C2 (train flags, `build_teachers`, the term in `losses`, the `main()` wiring) and C3 (`LADDER=3`, `make noise VARIANTS=`) as specified. Merged to `main` as 9690aa5, 269 tests.
  - Deviations:
    - The plan's smoke stop wrote a checkpoint when `--max-steps` exceeded the steps run. A smoke now always skips the eval and checkpoint.
    - `main()` validates every typed row's pairs at startup.
    - `main()` is exercised on CPU with fakes (4 tests), which retired the "written but not run" risk.
  - Real render-1 slice 1: 227 typed rows, 585 answer pairs, 94,464 content frames.
- **Reviews:**
  - Sonnet reviewed C1+C3 (nothing serious) and C2 (the four plan questions all yes).
  - Opus reviewed before the merge: nothing blocking.
  - Fixed after review:
    - `build_teachers` held the previous stem's logits through the next teacher forward.
    - The smoke prints reserved memory and dynamo counters.
    - The zero-frame KL test asserts the gradient.
- **Jobs:**
  - Image `cons1:9690aa5`.
  - GATE-0 passed:
    - `cons-smoke-1`: peak 39.2 GiB, teacher 32%.
    - `cons-smoke-wide`: peak 50.1 GiB, teacher 33%, 287 steps/epoch = baseline, 12 dynamo graphs.
  - Pilot `cons1-pp-l10`: 3 × ~4,300 s (+33%), peak 56.0 GiB. Its measure took 4,506 s. drama3 was stopped throughout.
- **Result** (`docs/results/consistency-1.md`):
  - **Down.** R_gap 1.90 (epoch 1) against R0 3.00; −1.09; CIs [−1.50, −0.65] and [−1.83, −0.44]. Median gap 0.0061 against 0.018. R 1.67 (epoch 2).
  - typed‖A reached tape‖A at epoch 2 (0.0061 vs 0.0069): the D2 tape ceiling. typed‖tape is still 0.0067.
  - Held text: +7.7% at epoch 1, +1.0% at epoch 2, −10.6% at epoch 3.
  - The floor also fell (A‖B 0.0032 vs 0.0058).
  - The bar still fails: ratio 1.7-1.9, p90 ratio 1.3. Every KL shrank (floor ×0.36 per turn); whether answers still depend on the pupil is unchecked (B7 side-by-side). Held text +7.7% at the R_gap epoch triggers option (e) as written.
- **GATE-1:** pending Vlad's choice; options (a)-(e) are in the results doc.

---

## Batch B10: side by side 2, the pilot's epoch 2 (2026-11-06)

Vlad, at GATE-1, to "do step one of option (b) regardless": **"do step by step"**. So this batch is step one only. It runs no training and renders nothing. It ends at a GATE before step two (TTS-voiced training rows).

**The question.** After the consistency term, does the tutor still respond to what the pupil said, in each rendering? B9 found every KL shrank (floor ×0.36 per turn). That fits two readings:
- "less sensitive to how the turn arrived" (healthy);
- "less sensitive to the pupil turn itself" (bad).

Every bar pair compares renderings of the same content, so the bar cannot tell them apart. B10 decodes and reads the answers.

**Comparable to B7 by construction.** The run uses B7's job, holdout (65 turns), renderings (typed, A, B), greedy LMGen with `dep_q = 8`, budget 48,000 and max batch 64. The pilot's true-turn items are packed by the same `pack` call as B7's, so they land in identical batch shapes. B7 found greedy decoding is shape-unstable (6/30 differ in substance). B7's baseline answers (`ladder-personaplex-1` epoch 2) are reused and not re-decoded. B7's batch-shape control is not repeated, and its 6/30 rate stands as the numerics noise. Repeating it costs 14 GPU-minutes and answers no new question.

**The direct test of the bad reading: a judged swap, not a KL.** For pupil turns 2 and 6 of every holdout dialogue (26 turns), the pupil turn's content is swapped for a donor dialogue's turn at the same index. The swap covers typed and A, and everything else is the tape. Both checkpoints are decoded on the swapped rows. A blind judge then sees the previous tutor turn, the two candidate pupil turns (true and donor, in random order) and the answer, and says which one the opening takes up.
- **Healthy:** the answer follows the turn that was actually played, for the pilot as often as for the baseline.
- **The bad reading predicts:** the pilot's answers after a swapped turn still follow the dialogue's own pupil turn, or neither.
- **Why not the KL alternative:** KL(answer after the true turn ‖ after the swapped turn), teacher-forced, for both checkpoints, needs a full forward per row. That is about 4.6 s each, so ~16 GPU-minutes more, on top of decoding. It scores the tape's words, which belong to the true turn, so it measures surprise at the tape answer rather than whether the tutor follows the pupil. The swap decode reuses the B7 job unchanged, costs ~10 GPU-minutes per model, and answers the behavioural question directly.

**Donors.** Holdout order is `holdout.jsonl`'s. For each (dialogue i, turn t), the donor is the first dialogue j after i, cyclically, whose typed pupil turn t has a different **head**. The head is the normalised words before the first comma of the decoded typed text. This avoids swapping "minus three" for "minus three".

### Task B10.1: swap rows in the side-by-side job (model: sonnet; TDD)

Files: `src/dual/sidebyside.py`, `src/dual/jobs/sidebyside.py`, `tests/test_sidebyside.py`, `Makefile` (nothing else).
- Pure, in `dual.sidebyside`:
  - `SWAP_RENDERINGS = ("typed", "A")`.
  - `answer_head(text) -> str`: `words_of` of the text before the first comma, joined by spaces.
  - `pick_donors(order: Sequence[str], heads: Mapping[tuple[str, int], str], turns) -> dict[(stem, turn), donor_stem]`. Each (stem, turn) present in `heads` gets the first stem after it, cyclically, that has that turn and a different head. It raises if there is none.
  - `swap_record(record, turn, donor) -> dict`: a copy of `record` whose `renderings[str(turn)]` is the donor's. The input is not mutated. It raises if `turn` is not a pupil turn of both.
  - `schedule(true_items, swap_items, budget, max_batch) -> list[list[Item]]` = `pack(true_items) + pack(swap_items)`. A test pins that the true batches equal `pack(true_items)` and that no batch mixes the two.
- Job:
  - `--swap-turns 2,6` (default none) and `--swap-only` (decode only the swap items; needs `--swap-turns`).
  - The swap items' rendering is `typed-swap` / `A-swap`, built with `holdout_row(swap_record(...), t, "typed" | "A")`. The answer window's text must equal the true row's. Their answers.jsonl rows add `donor`, and `true_pupil` (the dialogue's own typed text); `pupil` is the donor's typed text. `meta.json` records the swap turns and the donor map.
  - `build_items` and the true-item path stay byte-for-byte as B7's.
- `make check` green. Commit on the given branch only.

### Task B10.2: answer quality in `sbs_report` (model: sonnet; TDD)

Files: `src/dual/sbs_report.py`, `tests/test_sbs_report.py` (nothing else).
- `onset(ids) -> int | None`: the first answer-window frame whose text id is not `PAD` (`<s>` counts, as in B7); None if silent.
- `quality_table(rows, models)`, one line per (model, rendering) with:
  - n, silent;
  - onset frames 0/1/2/3/other;
  - median words;
  - **repetitive**: the answer contains a word 4-gram 3 or more times;
  - **unfinished**: the last word is not followed by `.`, `?` or `!`.
- `first_word_table(rows, models)`: per model and pair, (typed, A), (A, B) and (typed, B), the number of turns whose first content word is the same.
- The CLI gains `--quality`, which prints both. Unknown renderings (`typed-swap`, `A-swap`) are ignored by the B7 pair tables, not refused.

### Task B10.3: run it (the orchestrator)

- One image from the batch branch head.
- drama3 stays stopped. The watchdog runs while a job runs.
- `--mem-cap-gib 64`, one job at a time.
- Jobs:
  1. `sbs-2`: `--run cons1-pp-l10 --epoch 2 --no-bare --swap-turns 2,6`. That is 195 true rows and 52 swap rows, ~45 min.
  2. `sbs-2-base`: `--run ladder-personaplex-1 --epoch 2 --no-bare --swap-turns 2,6 --swap-only`. That is 52 rows, ~10 min.

### Task B10.4: blind judging (opus judges, xhigh) and `docs/results/side-by-side-2.md`

- **B7's protocol, verbatim:** pair and single-answer rubrics from `side-by-side-1/judge-prompts.txt`. One key is added to the answer rubric: `well_formed` (no garbled or broken text). Both models' true-row answers go into one shuffled pool, so baseline and pilot get the same judges. B7's own baseline labels are kept for comparison.
- **The swap judge:** for every true and swapped answer at turns 2 and 6, typed and A, both models (208 items), shuffled and blind to model and condition. Output: `takes_up` ∈ {P, Q, both, neither}.
- **The doc:**
  - counts next to B7's;
  - onset and quality tables;
  - first-word agreement;
  - the swap table;
  - a few example turns;
  - a one-paragraph verdict: healthy (the tutor still listens) or not.
- Then a sonnet review of the batch diff and an opus review before the merge. Merge `--no-ff`. The plan records the run, and a memory note is written. End at a GATE before step two.

### B10 as run (2026-11-06)

- **Code:** B10.1 and B10.2 as specified.
  - `decode_all` takes pre-packed batches.
  - The first-word table compares `words_of(answer)[0]`.
  - Review follow-ups: a swap-row characterisation test, and `quality_table` refuses duplicates. 286 tests.
- **Jobs (image `sbs2:72868db`, drama3 stopped):**
  - smoke (6 min);
  - `sbs-2` (`cons1-pp-l10` epoch 2, 195 true + 52 swap rows, 2,719 s; the true rows ran in B7's exact batch shapes);
  - `sbs-2-base` (`ladder-personaplex-1` epoch 2, 52 swap rows, 653 s);
  - ~68 GPU-minutes.
- **Judging:** nine blind opus judges over 260 pairs, 390 answers and 208 swap items. Re-judging B7's baseline reproduced its counts within 1-2.
- **Result** (`docs/results/side-by-side-2.md`):
  - **Mid-lesson the tutor still listens.** Uptake on turns 2-8 is 50/50/49 of 52 (baseline 52/50/50), and typed≠A is 18/52 (baseline 17/52).
  - **The swap is followed:** 46/52 answers take up the swapped turn, and only 1 takes up the dialogue's own.
  - **Tracking slipped:** 94/104 against 103/104 (p ≈ 0.01).
  - **The lesson opening collapsed:**
    - After the spoken first question the pilot is silent in 13/13 (A) and 10/13 (B).
    - After the typed first question it starts on frame 8-9 instead of 3, and takes the question up in 8/13.
  - No mid-lesson first-word divergence at epoch 2.
  - **Verdict:** not healthy as a checkpoint. GATE: λ 3, the term without turn 0, an epoch-1 side-by-side, or step two anyway.
- **Reviews:**
  - Sonnet reviewed the code diff: no blockers. Its follow-ups are the swap-row test and the duplicate refusal.
  - Opus reviewed before the merge and recomputed every number. The corrections:
    - "neither" counts toward the bad reading, so the healthy bar (as often as the baseline) is not met;
    - there is no mid-lesson behavioural gain, only converged wording;
    - the turn-0 collapse is not B9's pooled position-0 divergence, which is mid-lesson. At turn 0, typed and A agree more under teacher forcing (lead-frame gap 0.095 against 0.736), so the bar is blind to it;
    - step two would use a new voice D, not A or B (the bar's voices) and not C (the door).

## Batch B11: side by side 3, the pilot's epoch 1 (2026-11-06)

Vlad, at B10's GATE ("The pilot listens mid-lesson but goes silent after the spoken opening question. Which fix next?"): **"Check epoch 1 first"**. This batch runs no training, renders nothing and changes no code. It ends at a GATE.

**The question.** Is the lesson opening already broken at epoch 1 (R_gap 1.90, held text +7.7%)?
- **Broken at epoch 1 too:** the collapse comes with the term from the start, so the lever is λ or the opening's place in the term, not training length.
- **Healthy at epoch 1, broken at epoch 2:** it grows with training, so fewer steps (or early stop) is a lever, and λ 3 a likely one.

**Comparable to B10 by construction.** Same image (`sbs2:72868db`), job, holdout (65 turns), renderings (typed, A, B), greedy decode, budget 48,000, max batch 64, `--swap-turns 2,6`. Only `--epoch 1` changes, so the true rows land in B7's and B10's batch shapes. The baseline's answers (B7's true rows, B10's swap rows) and the epoch-2 answers are reused, not re-decoded.

### Task B11.1: run it (the orchestrator)

- drama3 stays stopped; the watchdog runs while the job runs. `--mem-cap-gib 64`, one job.
- `sbs-3`: `--run cons1-pp-l10 --epoch 1 --no-bare --swap-turns 2,6`. That is 195 true and 52 swap rows, ~57 GPU-minutes.

### Task B11.2: GPU-free side reads (model: sonnet)

- From the pilot's train log: the consistency term per step and epoch, and per pupil-turn index if logged (the opening's share).
- From B9's measure outputs: turn-0 lead and first-content-frame KL(typed‖A) at epoch 1, next to epoch 2 and the baseline; and, if the outputs hold the distributions, whether the pilot puts more mass on `<pad>` at the opening after spoken turns than the baseline does (B10's untested reading).

### Task B11.3: blind judging (opus judges) and `docs/results/side-by-side-3.md`

- B10's protocol verbatim (pair, answer and swap rubrics from `side-by-side-2/judge-prompts.txt`). Epoch-1 answers are pooled with epoch-2's and the baseline's, so the three share judges. B10's labels are kept for comparison.
- **The doc:** epoch 1 next to epoch 2 and the baseline, for every B10 table: substance, uptake (all, turns 2-8, turn 0), swap, onset and quality, first word. Plus the side reads and a one-paragraph answer: is the opening broken at epoch 1, and what that says about the lever.
- Sonnet review, opus review, merge `--no-ff`, the plan records the run, a memory note. End at a GATE: (i) λ 10 without the opening answer in the term, (ii) λ 3, (iii) fewer epochs / early stop, (iv) TTS-voiced teacher rows in voice D.

### B11 as run (2026-11-06)

- **No code.** The job already takes `--epoch`.
- **Job (image `sbs2:72868db`, drama3 stopped, watchdog on):** `job-20261106-121325-vlad-sbs-3`, `cons1-pp-l10` epoch 1.
  - 195 true and 52 swap rows, in B10's exact batch shapes and donor map.
  - 2,720 s of decoding (~48 GPU-minutes with load, against the ~57 estimated), peak 38.9 GiB.
- **Judging:** 13 blind opus judges over 390 pairs, 585 answers and 312 swap items, all three checkpoints in one pool. On the shared items they agree with B10's labels in 238/260, 382/390 and 202/208.
- **Result** (`docs/results/side-by-side-3.md`):
  - **The opening is not silent at epoch 1.** It is silent in 0/13 after A and 0/13 after B, against 13/13 and 10/13 at epoch 2, so the silence grows with training. The turn-0 lead-frame KL(typed‖A) tightens over epochs 1-3: 0.54, 0.095, 0.061 (baseline 0.74).
  - **But epoch 1 is not healthy.**
    - The typed opening matches the tape's move in 1/13 and is well formed in 5/13.
    - Mid-lesson, answers break off as fragments. They stop mid-sentence with ≥ 20% of the window left in 20/11/17 of 52 (typed/A/B), against 1/2/0 for the baseline and 0/1/1 at epoch 2. Most stop within the first tenth of the window, and median words on turns 2-8 are 41/57/38.
    - Uptake holds (51/49/50). The swap count is 95/104, but all 9 misses are fragments or a loop.
  - **Side reads:**
    - The term is logged per step only, never per turn.
    - It covers content frames only, so the lead frames are outside it.
    - Turn 0 had the largest bare-model gap, 2-3× the other turns.
    - P(`<pad>`) is not stored, so B10's reading stays untested.
  - **Verdict:**
    - At λ = 10 no decoded epoch is healthy, so an early stop is out.
    - The epoch-1 fragments fit either the term or ordinary under-training. The baseline's epoch 1 was never decoded.
    - The loss weight is the untested candidate.
  - **GATE:** (ii) λ 3 is recommended, with a baseline epoch-1 decode (~45 GPU-minutes). The other options are (i) λ 10 without turn 0 (untested, since the term never sees the lead frames), (iii) early stop (not supported) and (iv) voice D.
- **Reviews:**
  - Sonnet: no blockers. Its fixes were the turns 2-8 medians, the lead-frame attribution and the table labels.
  - Opus recomputed every number. Its corrections:
    - hedge the lever claim (λ was never varied);
    - epoch 1's swap misses are truncation;
    - argue against (i) from epoch 2's swap misses, not epoch 1's truncation;
    - the side-by-side costs ~1.6 GPU-hours, so (ii) is ~6.5.
