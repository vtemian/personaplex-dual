# personaplex-dual: the typed-or-spoken door, batched implementation plan

> **For Claude:** REQUIRED SUB-SKILL: use superpowers:executing-plans. One fresh agent per
> batch, batches on non-overlapping file sets, TDD for every pure module, the quality gate
> (`make check`) before every commit. The design this implements is
> `docs/plans/2026-10-26-dual-input-design.md`; where the two disagree, this plan wins and
> section 0 says why.

**Goal:** measure, deterministically and on one checkpoint at a time, whether a trunk LoRA
on PersonaPlex (and on moshiko, as the control base) makes a typed pupil turn teach the same
as the same words spoken, with the "same" defined by the KL between two spoken voices.

**Architecture:** a small Python package `dual` beside chalkboard's vendored source. Pure
modules (frame builders, turn spans, row rendering, KL maths, holdout and ladder slicing)
are unit-tested on the laptop. Five GPU jobs (smoke, fetch-weights, probe, render, train,
measure) run on the sparks box through `sparks submit`, read `/data` and the box's shared
`runs/` and `models/` directories, and report through `sparks.emit`. The model code reuses
kyutai `moshi==0.2.13` (its `LMModel.forward` applies the acoustic delays and returns
text and audio logits aligned to the input) and chalkboard's `attach_trunk_lora`.

**Tech stack:** Python 3.12, uv, pytest, ruff, mypy; torch 2.13 cu130 on the box;
`moshi==0.2.13`; sentencepiece; `sparks-dgx==0.1.1`; chalkboard at a pinned commit, vendored.

---

## 0. Read first: corrections to the design doc, and the decisions this plan takes

### 0.1 Corrections (measured 2026-10-26 in chalkboard, `~/projects/ai/bbm-v2`)

1. **The round-2 set is 130 dialogues with 650 pupil turns (5 each), not 5,501 dialogues
   and 27,000 turns.** The 5,501 / 3.64 M frames in the design are `scenes.jsonl`, draw-only
   teacher scenes with no pupil (`docs/results/derivatives-runs.md:307-325` in chalkboard).
   Everything downstream shrinks: 13 dialogues held out, 117 train, **351 training rows**
   (three per dialogue), about 585 typed-turn instances and 1,170 spoken; ladder rungs of
   15, 30, 59 and 117 dialogues. Training a run takes minutes, not hours, so the whole
   ladder is cheap. Growing the corpus is chalkboard's generator (`make dialogues` there)
   and is **not** in this plan; B5 ends at a human gate that asks whether to grow it.
2. **The tape's pupil voice is moshika, not a TTS.** Only each dialogue's opening `ask` is
   TTS (ex03 for 4/10 of lessons, a VCTK/Expresso voice for 4/10, moshika for 2/10); the
   other four pupil turns are moshika forced through the text (`dialogue_tape.py:85-96`,
   `pupil_voices.py:55-73`). So "spoken" in the training rows means the tape's rendition.
   The holdout's voices A and B are re-rendered with the Kyutai TTS (decision D1).
3. **A typed turn costs `len(tokens) + 6` frames**, PersonaPlex's gap being
   `int(0.5 * 12.5) = 6` frames, so a 12-word turn is about 1.5 s of frames, not 1 s.
4. **Rows are whole dialogues** (median 674 frames, tape budget 2,900), one row per
   rendering, never per exchange. A typed rendering may lengthen a row; the renderer refuses
   any row over 3,000 frames (moshi's context).
5. **PersonaPlex's fork is moshi 0.1.x; chalkboard runs moshi 0.2.13.** Nothing from the
   fork is vendored. Its prompt mechanism is reproduced on 0.2.13 by (a) `LMModel.forward`
   over explicit codes for training and measurement, and (b) for the probe, `LMGen` with
   `on_text_hook` writing the forced text token in place and `depformer_replace_tokens`
   forcing moshi's own audio, after setting `lm.dep_q = 8` so the user streams come from
   the input and the depformer stops at moshi's 8 codebooks (its first 8 per-step weights).

### 0.2 Decisions (the model's; Vlad can override any of them before `/ship-it`)

- **D1 voices.** A = `vctk/p225_023.wav`, B = `vctk/p248_023.wav`, C (the conversion-door
  control, typed-to-TTS) = `vctk/p302_023.wav`; all are `kyutai/tts-1.6b-en_fr` embeddings
  shipped in chalkboard's `data/derivative-r2/voices/`. A fourth rendering, `tape`, is the
  tape's own moshika/opening rendition, the voice the LoRA trained on. It is measured
  beside, never in the bar.
- **D2 which frames the KL reads.** The answer is the teacher span that follows pupil turn
  k: from that span's start (first word onset minus 4 frames, like chalkboard's `LEAD`) to
  the next pupil span's start or the row's end. **The bar reads the content frames**, those
  whose target text token is not `<pad>` (word pieces, `<s>`, `<unk>` cue, `</s>`). The
  all-frames mean is written beside it.
- **D3 KL direction.** gap = KL(typed ‖ A), floor = KL(A ‖ B), door = KL(C ‖ A), per frame
  `sum p (log p - log q)` over the 32,000-way text softmax in fp32, averaged over the frames
  of D2. The bar: `median(gap) <= median(floor)` and `p90(gap) <= p90(floor)`.
- **D4 re-rendered spoken segments.** Pupil stream = the TTS's Mimi codes then 6 frames of
  chalkboard's `MIMI_SILENCE`; moshi's own stream = PersonaPlex's `SILENCE` on every frame;
  text = `<pad>`. The tape's own spoken turns keep the tape's codes on both streams.
- **D5 the typed text.** The pupil's words, last word punctuated with `.` as chalkboard's
  tape does, encoded as one sentence with the SPM (`EncodeAsIds`), one id per frame, then 6
  gap frames of `<pad>`; own stream `SILENCE`, pupil stream `SINE` throughout. No
  `<system>` tags. The probe (B1) tries both with and without the tags, as a datum.
- **D6 no voice prompt** anywhere: training rows, measurement rows and the bare-model
  controls all start at frame 0 with the dialogue. PersonaPlex's voice prompt is the design's
  open point and stays open.
- **D7 loss.** `2.0 * text_ce + 0.5 * audio_ce` over moshi's 8 audio streams on scored
  frames only (every teacher span); zero on every pupil span of either kind; chalkboard's
  dv3a ratio.
- **D8 holdout and ladder order.** Stems sorted by `sha256(f"{stem}:holdout")`; the first
  `ceil(0.1 * n)` are held out. Train stems sorted by `sha256(f"{stem}:ladder")`; rung
  `1/d` takes the first `ceil(len/d)`, so rungs nest.
- **D9 the mask.** Bit k of turn k is bit k of `sha256(f"{stem}:mask")`; row 1 types the
  turns with bit 1, row 2 the complement, row 3 nothing.
- **D10 epochs.** 8 epochs, constant LR 2e-5, AdamW(0.9, 0.95), clip 1.0, a checkpoint
  every epoch, no early stopping; the measure job reads the bar at every epoch and the
  report shows the best epoch per run.
- **D11 weights.** PersonaPlex is gated and 16.7 GB: it is **not** baked into the image.
  One sparks job (`make fetch-weights`, needs `HF_TOKEN` in the submitting shell) downloads
  it into the box's shared directory at `models/personaplex-7b-v1/`; every job loads it from
  `Path.cwd() / "models"`. moshiko, Mimi and the TTS are baked, with the Dockerfile's first
  layers byte-identical to chalkboard's so Docker's cache serves them.
- **D12 LoRA.** Rank 16, scaling 2.0, on the four trunk linears of all 32 layers, fp32
  adapters, B zeroed; exactly chalkboard's `attach_trunk_lora`, 34,078,720 parameters.

### 0.3 What only Vlad can provide (the orchestrator reports `BLOCKED` on these)

- `HF_TOKEN` exported in the shell that runs `/ship-it` (B1, `make fetch-weights`). Vlad has
  accepted the `nvidia/personaplex-7b-v1` terms; no token is on the Mac or the box today.
- Nothing else: ssh to `vlad@spark.local`, the `sparks` CLI, Docker, chalkboard at
  `~/projects/ai/bbm-v2` and the tape at `/srv/spark/runs/dv3a-tape/tape/speech.jsonl` were
  all verified on 2026-10-26.

### 0.4 Batches

| batch | model | files (non-overlapping) | gate |
|---|---|---|---|
| B0 scaffold | sonnet | `pyproject.toml`, `Makefile`, `.gitignore`, `CLAUDE.md`, `scripts/vendor.sh`, `scripts/data.sh`, `src/dual/{__init__,constants,frames,holdout}.py`, `tests/test_{frames,holdout}.py` | `make check`, `make vendor`, `make data` |
| B1 box | opus | `docker/Dockerfile`, `docker/.dockerignore`, `scripts/submit.sh`, `src/dual/model.py`, `src/dual/jobs/{__init__,smoke,fetch_weights,probe}.py`, Makefile targets `smoke fetch-weights probe` | smoke job passes; weights load; probe prints its classification |
| B2 pure rendering + KL | sonnet | `src/dual/{turns,render,kl,ladder}.py`, `tests/test_{turns,render,kl,ladder}.py` | `make check` |
| B3 render job | sonnet | `src/dual/jobs/render.py`, Makefile target `render`; runs it | counts: 351 train rows, 13 holdout records x 5 turns x 4 renderings |
| B4 train + measure | opus | `src/dual/jobs/{train,measure}.py`, `src/dual/report.py`, `tests/test_report.py`, Makefile targets `train measure bar`; one 1-epoch smoke run; the bare controls | the bare-model bar table prints |
| B5 ladder | sonnet | `scripts/ladder.sh`, `docs/results/ladder-1.md` | eight runs measured; `GATE` with the table |
| B6 ladder 2 | sonnet | `src/dual/{model,noise}.py`, `src/dual/jobs/{train,measure}.py`, `scripts/ladder.sh`, `Makefile` (`noise`), `tests/test_{train,model_checks,measure,ladder_sh,noise}.py`, `docs/results/ladder-2.md` | five 3-epoch runs measured; `make noise` verdict; `GATE` |

B1 and B2 run in parallel after B0. B3 needs B1 and B2. B4 needs B3. B5 needs B4. B6 needs B5 and Vlad's ladder-1 gate answer.

---

## 1. Repo layout after B5

```
personaplex-dual/
  pyproject.toml  Makefile  .gitignore  CLAUDE.md
  src/dual/
    __init__.py constants.py frames.py holdout.py turns.py render.py kl.py ladder.py
    model.py report.py
    jobs/__init__.py smoke.py fetch_weights.py probe.py render.py train.py measure.py
  docker/Dockerfile  docker/.dockerignore         # docker/src and docker/vendor are assembled, gitignored
  scripts/vendor.sh  scripts/data.sh  scripts/submit.sh  scripts/ladder.sh
  vendor/            # gitignored: chalkboard's src/ at CHALKBOARD_SHA (chalkboard/, training/, harness/)
  data/              # gitignored: dialogues/*.dsl (130), voices/{vctk,expresso}/*.safetensors (16), tape/speech.jsonl
  tests/
  docs/plans/  docs/results/
```

On the box (shared dir `/srv/spark`, the job's working directory): `models/personaplex-7b-v1/`,
`runs/<name>/{render,train,measure,probe}/`.

---

## Batch B0: scaffold (model: sonnet)

### Task B0.1: project files

**Files:** create `pyproject.toml`, `.gitignore`, `Makefile`, `CLAUDE.md`, `src/dual/__init__.py` (empty).

`pyproject.toml`:

```toml
[project]
name = "personaplex-dual"
version = "0.1.0"
description = "One speech model, every pupil turn typed or spoken: the door and its bar"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.24,<1.0",
    "safetensors>=0.4",
    "sparks-dgx>=0.1.1",
    "torch>=2.13.0",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-xdist>=3.8", "ruff>=0.8", "mypy>=1.13"]

[tool.pytest.ini_options]
addopts = "-n auto -q"
testpaths = ["tests"]
pythonpath = ["src", "vendor"]

[tool.ruff]
line-length = 88
target-version = "py312"
src = ["src", "tests"]
extend-exclude = ["vendor", "docker"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN", "RUF"]
ignore = ["ANN401"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["ANN"]

[tool.mypy]
strict = true
mypy_path = "src:vendor"
files = ["src/dual"]
exclude = ["vendor"]

[[tool.mypy.overrides]]
module = ["chalkboard.*", "training.*", "harness.*", "moshi.*", "sentencepiece", "sparks.*", "numpy.*"]
ignore_missing_imports = true
follow_imports = "skip"
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
vendor/
data/
docker/src/
docker/vendor/
runs/
.mypy_cache/ .ruff_cache/ .pytest_cache/
```

`Makefile` (tabs, not spaces):

```make
CHALKBOARD ?= $(HOME)/projects/ai/bbm-v2
CHALKBOARD_SHA ?= b58c21ac
BOX ?= vlad@spark.local
TAPE ?= /srv/spark/runs/dv3a-tape/tape/speech.jsonl

.PHONY: check lint typecheck test vendor data

check: lint typecheck test

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy

test:
	uv run pytest

vendor:
	CHALKBOARD=$(CHALKBOARD) CHALKBOARD_SHA=$(CHALKBOARD_SHA) sh scripts/vendor.sh

data:
	CHALKBOARD=$(CHALKBOARD) BOX=$(BOX) TAPE=$(TAPE) sh scripts/data.sh
```

`CLAUDE.md`:

```markdown
# personaplex-dual

What this is: `docs/plans/2026-10-26-dual-input-design.md` (the question) and
`docs/plans/2026-10-26-dual-input-batches.md` (the plan, its corrections and decisions).

- `make check` before every commit; work on a branch, never on main.
- GPU work goes through sparks only (`authoring-a-sparks-job`, `operating-the-sparks-queue`).
  Nothing starts by hand on the box.
- `vendor/` is chalkboard's `src/` at `CHALKBOARD_SHA` (`make vendor`); never edit it, never
  commit it. `data/` is fetched by `make data`; never commit it.
- Pure modules in `src/dual/*.py` never import torch at module level except `kl.py`; the
  GPU-only code is `src/dual/model.py` and `src/dual/jobs/*.py`, and imports `moshi` inside
  functions, as chalkboard does.
```

**Step 1:** write the four files. **Step 2:** `uv sync` then `uv run pytest` (no tests yet:
exit code 5 is fine). **Step 3:** commit `chore: project scaffold`.

### Task B0.2: `scripts/vendor.sh` and `scripts/data.sh`

**Files:** create `scripts/vendor.sh`, `scripts/data.sh`.

```sh
#!/bin/sh
# scripts/vendor.sh: chalkboard's src/ at one commit, extracted from git, never from the
# working tree, so a render is reproducible whatever bbm-v2 has checked out.
set -eu
: "${CHALKBOARD:?}" "${CHALKBOARD_SHA:?}"
rm -rf vendor && mkdir -p vendor
git -C "$CHALKBOARD" archive "$CHALKBOARD_SHA" src | tar -x -C vendor --strip-components=1
rm -rf vendor/harness/turns
git -C "$CHALKBOARD" rev-parse "$CHALKBOARD_SHA" > vendor/SHA
uv run python -c "import chalkboard.parser, chalkboard.retime, training.moshi.trunk_lora, harness.engine.audio; print('vendor ok')"
```

```sh
#!/bin/sh
# scripts/data.sh: the 130 round-2 dialogues, the 16 TTS voices and the dv3a tape.
set -eu
: "${CHALKBOARD:?}" "${BOX:?}" "${TAPE:?}"
mkdir -p data/dialogues data/voices data/tape
cp "$CHALKBOARD"/data/derivative-r2/dlg-*.dsl data/dialogues/
cp -R "$CHALKBOARD"/data/derivative-r2/voices/. data/voices/
[ -f data/tape/speech.jsonl ] || scp "$BOX:$TAPE" data/tape/speech.jsonl
n_dsl=$(ls data/dialogues/*.dsl | wc -l | tr -d ' ')
n_tape=$(wc -l < data/tape/speech.jsonl | tr -d ' ')
n_voice=$(find data/voices -name '*.safetensors' | wc -l | tr -d ' ')
echo "dialogues=$n_dsl tape=$n_tape voices=$n_voice"
[ "$n_dsl" = 130 ] && [ "$n_tape" = 130 ] && [ "$n_voice" = 16 ] || { echo "data counts off"; exit 1; }
```

**Step 1:** write both. **Step 2:** `make vendor` must print `vendor ok` (if an import fails,
the missing package is a chalkboard runtime dep: add it to `dependencies` in
`pyproject.toml`, `uv sync`, retry; `pydantic` and `pillow` are the likely ones).
**Step 3:** `make data` must print `dialogues=130 tape=130 voices=16`. **Step 4:** commit
`chore: vendor chalkboard and fetch the round-2 data`.

### Task B0.3: `src/dual/constants.py`

```python
"""Every number the three streams agree on. All measured, none guessed; the source of each
is named so a later reader can re-measure it."""

FRAME_RATE = 12.5
CODEBOOKS = 8
STREAMS = 1 + 2 * CODEBOOKS  # text, moshi's 8, the pupil's 8
CONTEXT = 3000  # moshi's temporal context, frames

# moshi's acoustic delays, moshi/models/loaders.py:118 (0.2.13): text, own 8, pupil 8.
DELAYS = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)

# The four reserved SPM ids (chalkboard/grid.py:11-31).
CUE = 0  # <unk>: the frame before a later word
TURN = 1  # <s>: opens a teacher turn
END = 2  # </s>: after the turn's last piece
PAD = 3  # <pad>: silence on the text stream
TEXT_CARD = 32000
AUDIO_CARD = 2048

# PersonaPlex (NVIDIA/personaplex moshi/models/lm.py): moshi's own stream while it reads a
# prompt, and the pupil stream while text is typed. Never swap them.
SILENCE = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
SINE = (430, 1268, 381, 1611, 1095, 1495, 56, 472)

# chalkboard training/moshi/stream.py:35, 2 s of silence through Mimi: the pupil stream
# after a spoken turn, which is real silence and not the typed marker.
MIMI_SILENCE = (1049, 1700, 1626, 546, 306, 1443, 1871, 2008)

# PersonaPlex's server: int(0.5 * 12.5) frames of pad after a text prompt.
GAP_FRAMES = 6

# Frames kept ahead of a turn's first word, chalkboard training/moshi/cut_turns.py:33.
LEAD = 4

IGNORE = -100

PERSONAPLEX = "personaplex"
MOSHIKO = "moshiko"
BASES = (PERSONAPLEX, MOSHIKO)
PERSONAPLEX_FILES = {
    "moshi": "model.safetensors",
    "mimi": "tokenizer-e351c8d8-checkpoint125.safetensors",
    "spm": "tokenizer_spm_32k_3.model",
}
MOSHIKO_REPO = "kyutai/moshiko-pytorch-bf16"
```

No test; `make check` covers it. Commit `feat: the stream constants`.

### Task B0.4: `src/dual/frames.py` (TDD)

**Files:** create `tests/test_frames.py`, `src/dual/frames.py`.

**Step 1: the failing tests**

```python
import pytest

from dual.constants import GAP_FRAMES, MIMI_SILENCE, PAD, SILENCE, SINE
from dual.frames import Row, Segment, Span, concat, from_record, spoken_segment, tape_segment, to_record, typed_segment


def test_typed_segment_is_tokens_then_gap_on_silence_and_sine():
    seg = typed_segment([10, 11, 12])
    assert seg.frames == 3 + GAP_FRAMES
    assert seg.text == (10, 11, 12) + (PAD,) * GAP_FRAMES
    assert set(seg.own) == {SILENCE}
    assert set(seg.pupil) == {SINE}
    assert seg.scored is False


def test_typed_segment_refuses_no_tokens():
    with pytest.raises(ValueError, match="no tokens"):
        typed_segment([])


def test_spoken_segment_is_codes_then_mimi_silence_under_pad():
    codes = [[1] * 8, [2] * 8]
    seg = spoken_segment(codes)
    assert seg.frames == 2 + GAP_FRAMES
    assert seg.text == (PAD,) * (2 + GAP_FRAMES)
    assert seg.pupil[:2] == ((1,) * 8, (2,) * 8)
    assert set(seg.pupil[2:]) == {MIMI_SILENCE}
    assert set(seg.own) == {SILENCE}


def test_segment_refuses_ragged_streams():
    with pytest.raises(ValueError, match="frames"):
        Segment(text=(PAD, PAD), own=(SILENCE,), pupil=(SINE, SINE), scored=False)
    with pytest.raises(ValueError, match="codebooks"):
        Segment(text=(PAD,), own=((1, 2),), pupil=(SINE,), scored=False)


def test_concat_records_spans_and_scored_mask():
    a = tape_segment([PAD, 5], [SILENCE, SILENCE], [SINE, SINE], scored=True)
    b = typed_segment([7])
    row = concat("dlg-1", [("teacher", "tape", a), ("pupil", "typed", b)])
    assert row.frames == 2 + 1 + GAP_FRAMES
    assert row.scored == (1, 1) + (0,) * (1 + GAP_FRAMES)
    assert row.spans == (
        Span("teacher", "tape", 0, 2),
        Span("pupil", "typed", 2, 2 + 1 + GAP_FRAMES),
    )
    assert row.text[:3] == (PAD, 5, 7)


def test_record_round_trip():
    row = concat("dlg-1", [("pupil", "typed", typed_segment([7, 8]))])
    assert from_record(to_record(row)) == row
    assert to_record(row)["frames"] == row.frames
```

**Step 2:** `uv run pytest tests/test_frames.py -n 0` fails with `ModuleNotFoundError`.

**Step 3: the implementation**

```python
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
    rendering: str  # "tape" | "typed" | "voice:<name>"
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
        spans=tuple(Span(s["party"], s["rendering"], int(s["start"]), int(s["end"])) for s in record["spans"]),
    )
```

**Step 4:** tests pass. **Step 5:** `make check`; commit `feat: typed and spoken segments and the row they concatenate into`.

### Task B0.5: `src/dual/holdout.py` (TDD)

**Files:** create `tests/test_holdout.py`, `src/dual/holdout.py`.

Tests:

```python
from dual.holdout import mask_bits, ordered, split


def test_split_holds_out_a_tenth_rounded_up_and_is_stable():
    stems = [f"dlg-{i:03d}" for i in range(130)]
    held, train = split(stems)
    assert len(held) == 13 and len(train) == 117
    assert set(held) | set(train) == set(stems) and not set(held) & set(train)
    assert split(list(reversed(stems))) == (held, train)


def test_split_refuses_duplicates():
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        split(["a", "a"])


def test_ordered_is_a_permutation_keyed_by_salt():
    stems = ["a", "b", "c", "d"]
    assert sorted(ordered(stems, "x")) == stems
    assert ordered(stems, "x") == ordered(stems, "x")


def test_mask_bits_is_stable_and_bit_per_turn():
    bits = mask_bits("dlg-00005-000", 5)
    assert len(bits) == 5 and set(bits) <= {0, 1}
    assert bits == mask_bits("dlg-00005-000", 5)
    assert mask_bits("dlg-00005-000", 3) == bits[:3]
```

Implementation:

```python
"""Which dialogues are held out, in what order the ladder cuts the rest, and which turns a
row types. All by sha256 of the stem with a salt, never by file order or run order."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

HOLDOUT_FRACTION = 0.1


def _digest(stem: str, salt: str) -> int:
    return int(hashlib.sha256(f"{stem}:{salt}".encode()).hexdigest(), 16)


def ordered(stems: Sequence[str], salt: str) -> list[str]:
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate stems")
    return sorted(stems, key=lambda stem: (_digest(stem, salt), stem))


def split(stems: Sequence[str]) -> tuple[list[str], list[str]]:
    by_hash = ordered(stems, "holdout")
    held = math.ceil(HOLDOUT_FRACTION * len(by_hash))
    return by_hash[:held], sorted(by_hash[held:])


def mask_bits(stem: str, turns: int) -> tuple[int, ...]:
    digest = _digest(stem, "mask")
    return tuple((digest >> k) & 1 for k in range(turns))
```

Commit `feat: holdout split, ladder order and the per-turn mask`. Run `make check` and report.

---

## Batch B1: PersonaPlex on the box (model: opus)

Read the `authoring-a-sparks-job` skill before writing anything here. The box: DGX Spark
GB10, CUDA 13.0, 128 GB unified, shared dir `/srv/spark` is the job's cwd, `/data` read-only.

### Task B1.1: the image and the submit wrapper

**Files:** create `docker/Dockerfile`, `docker/.dockerignore`, `scripts/submit.sh`; append to `Makefile`.

`docker/Dockerfile`: copy chalkboard's `docker/moshi/Dockerfile` (`~/projects/ai/bbm-v2/docker/moshi/Dockerfile`)
**verbatim from `FROM` through the `sparks-dgx==0.1.1` layer, with two changes**: drop the
`tts-0.75b-en-public` layer and the `moshika` layer (neither is needed), and replace the
final `COPY` block with:

```dockerfile
# Last, so a code change re-pushes ~1 MB. vendor/ is chalkboard's src at CHALKBOARD_SHA.
COPY vendor /app/vendor
COPY src /app/src
ENV PYTHONPATH=/app/src:/app/vendor:/app
```

(Keeping the torch, apt, moshi and moshiko/TTS bake layers byte-identical means the local
Docker cache already holds them from chalkboard's builds.) Set `ENV HF_HUB_OFFLINE=1` as
chalkboard does; the PersonaPlex weights never come from the hub inside a job.

`docker/.dockerignore`: `__pycache__`, `*.pyc`.

`scripts/submit.sh`:

```sh
#!/bin/sh
# scripts/submit.sh NAME JOB [args...]: assemble the context, submit, print the job id.
# JOB is a module under dual.jobs. $SECRETS (space-separated names) become --secret flags.
set -eu
name="$1"; job="$2"; shift 2
[ -f vendor/SHA ] || { echo "run make vendor first"; exit 1; }
if [ -z "${IMAGE:-}" ] && [ -z "${DUAL_ALLOW_DIRTY:-}" ] && [ -n "$(git status --porcelain -- src)" ]; then
  echo "uncommitted changes under src/: the image would carry the previous commit's tag"; exit 1
fi
rm -rf docker/src docker/vendor && cp -R src docker/src && cp -R vendor docker/vendor
set -- python -m "dual.jobs.$job" "$@"
secrets=""; for s in ${SECRETS:-}; do secrets="$secrets --secret $s"; done
if [ -n "${IMAGE:-}" ]; then
  sparks submit --image "$IMAGE" --data "${DATA:-data}" --name "$name" $secrets -- "$@"
else
  sparks submit --context docker --data "${DATA:-data}" --name "$name" $secrets -- "$@"
fi
```

Makefile additions:

```make
.PHONY: smoke fetch-weights probe
NAME ?= dev

smoke:
	sh scripts/submit.sh smoke-$(NAME) smoke

fetch-weights:
	SECRETS=HF_TOKEN sh scripts/submit.sh fetch-personaplex fetch_weights

probe:
	sh scripts/submit.sh probe-$(NAME) probe --base personaplex $(ARGS)
```

### Task B1.2: `src/dual/jobs/smoke.py`

```python
"""Proves the path: sparks import, /data, the shared dir, the baked moshiko and Mimi, the
GPU. No PersonaPlex, no training."""

import os
import pathlib
import time

import torch
from sparks.emit import track


def main() -> None:
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    print("data:", sorted(p.name for p in data.iterdir()))
    print("cwd:", pathlib.Path.cwd(), "models:", sorted(p.name for p in (pathlib.Path.cwd() / "models").iterdir()))
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    from moshi.models.loaders import CheckpointInfo

    info = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    spm = info.get_text_tokenizer()
    print("spm pieces:", spm.GetPieceSize(), "hello ->", spm.EncodeAsIds("hello there."))
    mimi = info.get_mimi(device="cuda")
    with torch.no_grad(), mimi.streaming(1):
        codes = mimi.encode(torch.zeros(1, 1, 1920, device="cuda"))
    print("mimi silence frame:", [int(c) for c in codes[0, :, 0]])
    with track(total=10) as run:
        for step in range(10):
            run.step(loss=1.0 / (step + 1))
            time.sleep(0.2)
    print("smoke ok")


if __name__ == "__main__":
    main()
```

Run `make smoke`, then `sparks wait "$JOB" && sparks logs "$JOB"`. The log must end in
`smoke ok`. Anything else is plumbing: fix the Dockerfile, resubmit. Commit
`feat: the image, the submit wrapper and the smoke job`.

### Task B1.3: `src/dual/jobs/fetch_weights.py`

```python
"""Downloads nvidia/personaplex-7b-v1 once into the shared dir. Needs HF_TOKEN (a --secret)."""

import os
import pathlib

from huggingface_hub import snapshot_download

from dual.constants import PERSONAPLEX_FILES

REPO = "nvidia/personaplex-7b-v1"


def main() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    target = pathlib.Path.cwd() / "models" / "personaplex-7b-v1"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        local_dir=str(target),
        allow_patterns=list(PERSONAPLEX_FILES.values()) + ["config.json", "README.md"],
        token=os.environ["HF_TOKEN"],
    )
    for name in PERSONAPLEX_FILES.values():
        path = target / name
        print(name, path.stat().st_size if path.exists() else "MISSING")
    print("fetched", target)


if __name__ == "__main__":
    main()
```

Run with `HF_TOKEN` exported in the shell: `make fetch-weights`. If the submit fails naming
`HF_TOKEN`, stop and report `BLOCKED: export HF_TOKEN in the shell running /ship-it`.
Expected log: `model.safetensors 16.7e9`-ish, the Mimi file 385 MB, the SPM 553 kB.
Verify from the Mac: `ssh vlad@spark.local ls -la /srv/spark/models/personaplex-7b-v1`.
Commit `feat: the one-time PersonaPlex fetch job`.

### Task B1.4: `src/dual/model.py`

The loader for both bases, LoRA attach, adapter state round-trip, and the one forward used
by train and measure. `moshi` is imported inside functions.

```python
"""GPU-only: the two bases, the trunk LoRA and the teacher-forced forward."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import torch

from dual.constants import BASES, CODEBOOKS, MOSHIKO, MOSHIKO_REPO, PERSONAPLEX, PERSONAPLEX_FILES, STREAMS

RANK = 16
SCALING = 2.0
ADAPTER_PARAMS = 34_078_720


@dataclass
class Base:
    name: str
    lm: Any  # moshi LMModel
    spm: Any  # sentencepiece processor
    info: Any  # CheckpointInfo, for get_mimi


def personaplex_dir() -> pathlib.Path:
    return pathlib.Path.cwd() / "models" / "personaplex-7b-v1"


def load_base(name: str, device: str = "cuda") -> Base:
    from moshi.models.loaders import CheckpointInfo

    if name not in BASES:
        raise ValueError(f"base {name!r}: one of {BASES}")
    if name == MOSHIKO:
        info = CheckpointInfo.from_hf_repo(MOSHIKO_REPO)
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    else:
        root = personaplex_dir()
        info = CheckpointInfo(
            moshi_weights=root / PERSONAPLEX_FILES["moshi"],
            mimi_weights=root / PERSONAPLEX_FILES["mimi"],
            tokenizer=root / PERSONAPLEX_FILES["spm"],
            lm_config=None,  # moshi 7B defaults, dep_q overridden below
        )
        # PersonaPlex's depformer predicts 16 streams: moshi's 8 then the pupil's 8.
        lm = info.get_moshi(device=device, dtype=torch.bfloat16, lm_kwargs_overrides={"dep_q": 16})
    lm.eval()
    spm = info.get_text_tokenizer()
    return Base(name, lm, spm, info)


def describe(base: Base) -> dict[str, int]:
    return {
        "params": sum(p.numel() for p in base.lm.parameters()),
        "dep_q": int(base.lm.dep_q),
        "n_q": int(base.lm.num_audio_codebooks),
        "layers": len(base.lm.transformer.layers),
    }


def attach_lora(base: Base) -> int:
    from training.moshi.trunk_lora import attach_measured

    modules, drift = attach_measured(base.lm, RANK, SCALING)
    if drift != 0.0:
        raise RuntimeError(f"a fresh adapter moved the trunk by {drift}")
    for name, param in base.lm.named_parameters():
        param.requires_grad_(".lora_" in name)
    trainable = sum(p.numel() for p in base.lm.parameters() if p.requires_grad)
    if trainable != ADAPTER_PARAMS:
        raise RuntimeError(f"{trainable} trainable parameters, expected {ADAPTER_PARAMS}")
    return modules


def adapter_state(base: Base) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in base.lm.named_parameters() if ".lora_" in n}


def load_adapters(base: Base, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = base.lm.load_state_dict(state, strict=False)
    unexpected_l = list(unexpected)
    if unexpected_l or any(".lora_" in m for m in missing):
        raise RuntimeError(f"adapter keys off: unexpected {unexpected_l[:3]}, missing lora {[m for m in missing if '.lora_' in m][:3]}")


def codes_from_row(record: dict[str, Any], device: str = "cuda") -> torch.Tensor:
    """[1, 17, T] long: text, own 8, pupil 8. Undelayed; LMModel.forward applies DELAYS."""
    text = torch.tensor(record["text"], dtype=torch.long)
    own = torch.tensor(record["own"], dtype=torch.long).t()
    pupil = torch.tensor(record["pupil"], dtype=torch.long).t()
    codes = torch.cat([text[None], own, pupil], dim=0)
    assert codes.shape == (STREAMS, len(record["text"])), codes.shape
    return codes[None].to(device)


@dataclass
class Forward:
    text_logits: torch.Tensor  # [B, T, 32000] fp32, NaN where moshi masks
    text_mask: torch.Tensor  # [B, T] bool
    audio_logits: torch.Tensor  # [B, 8, T, 2048] (moshi's own streams only)
    audio_mask: torch.Tensor  # [B, 8, T] bool


def forward(lm: Any, codes: torch.Tensor) -> Forward:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm(codes)
    return Forward(
        text_logits=out.text_logits[:, 0].float(),
        text_mask=out.text_mask[:, 0],
        audio_logits=out.logits[:, :CODEBOOKS].float(),
        audio_mask=out.mask[:, :CODEBOOKS],
    )
```

**Known risks for this task, in order of likelihood; each is yours to resolve in-session:**

1. `CheckpointInfo.get_moshi(..., lm_kwargs_overrides=...)`: `get_moshi` forwards `**kwargs`
   to `get_moshi_lm`, which takes `lm_kwargs_overrides` (loaders.py:366-388 in 0.2.13).
   If not, build `LMModel` yourself from `loaders._lm_kwargs | {"dep_q": 16}` and
   `load_state_dict(load_file(...), assign=True)` as `get_moshi_lm` does.
2. The PersonaPlex state dict against the 0.2.13 module names. moshiko's own 0.1-era file
   loads through 0.2.13's load hook (chalkboard's `trunk_lora.py` comment on
   `modules/transformer.py:407-446`), so PersonaPlex's should too, with 16 per-step depformer
   weights matching `dep_q=16`. If `load_state_dict` raises, print the first ten missing and
   unexpected keys and map them; do not `strict=False` past a real mismatch.
3. `get_mimi` with `lm_config=None` builds an 8-codebook Mimi: right.
4. Memory: 16.7 GB bf16 plus the TTS and Mimi; fine on 128 GB.

Add to `smoke.py` a `--base` flag that, when given, calls `load_base`, prints `describe`,
runs `forward` on a 40-frame row of `PAD`/`SILENCE`/`SINE` codes and prints the text
logits' shape and whether any NaN sits outside the masked frames. Run
`make smoke NAME=load ARGS=...` (add `$(ARGS)` to the smoke target) for both bases.
Expected: `dep_q 16, n_q 16, layers 32` for personaplex; `dep_q 8` for moshiko; text
logits `[1, 40, 32000]`. Commit `feat: load both bases, attach the trunk LoRA, one forward`.

### Task B1.5: the zero-training probe, `src/dual/jobs/probe.py`

What it answers: on bare PersonaPlex, after a spoken exchange, does a typed turn forced
mid-dialogue with the triple get **answered**, **echoed** or **ignored**? Greedy decoding
(`temp=0`, `temp_text=0`), so a rerun reproduces it.

Design on moshi 0.2.13's `LMGen`:

- Load PersonaPlex via `load_base`, then `lm.dep_q = 8` **for this job only**: the depformer
  then runs moshi's 8 steps with its first 8 per-step weights, and `LMGen._step` reads the
  pupil's 8 codes from `input_tokens` (`needed_tokens = 17 - 8 - 1 = 8`).
- `lm_gen = LMGen(lm, temp=0.0, temp_text=0.0, on_text_hook=hook)` where `hook(text_token)`
  does `text_token.copy_(forced)` when a forced id is pending (a one-slot box the loop sets
  before each `step`), else leaves the sample. `lm.eval()` first; `LMGen` asserts it.
- Each frame: `out = lm_gen._step(input_tokens[B=1, 8, 1], depformer_replace_tokens=own[1, 8, 1] or None)`.
  Out is `[1, 9, 1]` (text, own 8) after `max_delay` warm-up frames, or `None`.
- Phases of one probe, per holdout dialogue (take the first 3 by `split` order, cheap):
  1. **spoken opening**: pupil turn 1 as TTS voice A (D1) through `voice_many` + `encode_wav`
     (vendored `harness.engine.audio`), fed as `input_tokens`, text free, own audio free;
     then let moshi answer freely for up to 150 frames with `input_tokens = MIMI_SILENCE`,
     stopping 12 frames after its text emits `</s>` (id 2) or at the cap.
  2. **typed second turn**: the DSL's second `ask`, encoded per D5; for each token one frame:
     `input_tokens = SINE`, forced text = token, `depformer_replace_tokens = SILENCE`; then 6
     frames with forced `PAD`. Then free run for up to 150 frames with `MIMI_SILENCE` on the
     pupil stream, recording the text tokens moshi emits and its audio codes.
  3. Repeat 2 with the tokens wrapped `<system> ... <system>` (the string, SPM-encoded).
- Classify the free-run text after the typed turn, decoded with the SPM, pieces joined:
  - `echo` if ≥ 50% of the typed words (alphanumeric-normalised) appear in the first 40
    emitted non-pad tokens in order;
  - `ignore` if fewer than 4 non-pad tokens are emitted in 150 frames;
  - `answer` otherwise.
- Write `runs/<name>/probe/probe.jsonl`: per dialogue and variant, the typed words, the
  emitted text, the class, frame counts; and `summary.json` with counts per class per
  variant. Decode the answer audio to wav with Mimi (`mimi.decode`) into
  `runs/<name>/probe/<stem>-<variant>.wav` so a human can listen if the class is odd.
- `track(total=...)` is not needed; print, the log is the record.

Run `make probe NAME=bare`. Read the log and `summary.json` (scp it back). Report the counts.
Any exception in `LMGen` with `dep_q` tampering is a real finding: fix it here (the
`_init_streaming_state` cache sizes by `num_codebooks`, not `dep_q`, so it should hold).
Commit `feat: the zero-training probe on bare PersonaPlex`. Report with the probe table.
This is a datum, not a gate: the ladder runs whatever it says.

---

## Batch B2: turn spans, rendering, KL and the ladder (model: sonnet; pure, laptop-tested)

Needs `vendor/` (B0). Uses `chalkboard.parser.parse`, `chalkboard.retime.lay_turns` and the
SPM. For tests, a fake tokenizer splits a word into characters with `▁` on the first; the
one real-SPM test is `skipif` the file is not in the HF cache
(`huggingface_hub.try_to_load_from_cache("kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model")`).

### Task B2.1: `src/dual/turns.py` (TDD)

What it does: from a DSL source and its `speech.jsonl` record (`frames`, `words` = one list
of `[word, frame]` per spoken line, `audio` and `user_audio` = `[T][8]`), compute the turn
groups, each turn's first onset, the span boundaries, and the text row over the whole tape.

Record shape and the functions copied from chalkboard (cite them in comments; they are short
and importing their modules drags in torch and the lint tree):

```python
"""Turns of a taped dialogue: who speaks when, and the text row the teacher's words lay on."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from chalkboard.parser import Ask, Pause, Say, Statement, parse
from chalkboard.retime import lay_turns

from dual.constants import CUE, END, LEAD, PAD, TURN

Onsets = list[list[tuple[str, int]]]
Pieces = Callable[[str], Sequence[str]]
PieceId = Callable[[str], int]

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
    end: int


def onsets_of(record: dict[str, Any]) -> Onsets:
    return [[(str(w), int(f)) for w, f in line] for line in record["words"]]


def teacher_turns(stmts: Sequence[Statement], onsets: Onsets) -> list[list[tuple[str, int]]]:
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
    stmts = parse(src)
    onsets = onsets_of(record)
    lines = spoken_lines(stmts)
    if len(lines) != len(onsets):
        raise ValueError(f"{record['id']}: {len(lines)} spoken lines, {len(onsets)} measured")
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
    prev_end = 0
    for k, group in enumerate(groups):
        start = 0 if k == 0 else max(firsts[k] - LEAD, prev_end + 1)
        end = frames if k == len(groups) - 1 else max(firsts[k + 1] - LEAD, start + 1)
        if end > frames:
            raise ValueError(f"{record['id']}: turn {k} runs past the tape")
        party = "pupil" if isinstance(group[0], Ask) else "teacher"
        out.append(TurnSpan(k, party, words_of[k], firsts[k], start, end))
        prev_end = end
    return out


def text_row(src: str, record: dict[str, Any], pieces: Pieces, piece_id: PieceId) -> list[int]:
    """One text id per tape frame: the teacher's words as moshi's interleave lays them,
    `<pad>` everywhere else including every pupil frame."""
    stmts = parse(src)
    laid = lay_turns(teacher_turns(stmts, onsets_of(record)), pieces, int(record["frames"]))
    ids = [SPECIAL_IDS[p] if p in SPECIAL_IDS else piece_id(p) for p in laid]
    if len(ids) > int(record["frames"]):
        raise ValueError(f"{record['id']}: text runs {len(ids) - int(record['frames'])} frames past the tape")
    return ids + [PAD] * (int(record["frames"]) - len(ids))


def typed_tokens(words: Sequence[str], encode_ids: Callable[[str], Sequence[int]]) -> list[int]:
    # chalkboard training/moshi/tts_corpus.py:32: the last word punctuated. One sentence,
    # encoded whole, as PersonaPlex encodes its text prompt.
    spoken = list(words)
    if spoken and not spoken[-1].endswith((".", "?", "!")):
        spoken[-1] += "."
    ids = [int(i) for i in encode_ids(" ".join(spoken))]
    if not ids:
        raise ValueError("no tokens")
    return ids
```

Check `chalkboard.parser.Say`/`Ask` have `.words` and `Pause` exists (they do:
`parser.py:8-17`). Check `lay_turns(turns, tokenize, frames)` returns piece strings with
`<pad>`, `<unk>`, `<s>`, `</s>` (retime.py:140-212); `grow` may lengthen it past `frames`,
hence the check.

Tests (`tests/test_turns.py`): a 2-ask / 2-teacher DSL
(`ask a b\nsay c d\npause 2\nask e\nsay f g\n`) with a hand-written record of 60 frames and
onsets `[[("a",3),("b",5)],[("c",14),("d",17)],[("e",32)],[("f",41),("g",44)]]`:

- `spans` gives 4 spans, parties pupil/teacher/pupil/teacher, starts `0, 10, 28, 37`,
  ends `10, 28, 37, 60`, `words` as in the DSL, onsets `3, 14, 32, 41`.
- a record with a sentence count off by one raises `ValueError("spoken lines")`.
- `text_row` with the fake tokenizer: frame 13 is `TURN` (1), 14.. the pieces of `c`, every
  pupil frame is `PAD`, the row length equals `frames`, and the last piece of `g` is
  followed by `END` (2).
- `typed_tokens(["so", "what"], fake_ids)` encodes `"so what."` and refuses an empty list.
- the real-SPM test (skipif): `typed_tokens(["two", "since", "the", "curve"], spm.EncodeAsIds)`
  has between 4 and 10 ids and none of them is 0..3.

Commit `feat: turn spans and the text row of a taped dialogue`.

### Task B2.2: `src/dual/render.py` (TDD)

Pure assembly: given a dialogue's spans, its tape streams and text row, and a rendering per
pupil turn, build the three training rows and the holdout record. The GPU job (B3) supplies
the typed tokens and the TTS codes through callables.

```python
"""Rows from a taped dialogue: the mask row, its complement, the all-spoken tape, and the
holdout record with every pupil turn in every rendering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dual.constants import CONTEXT
from dual.frames import Row, Segment, concat, spoken_segment, tape_segment, to_record, typed_segment
from dual.holdout import mask_bits
from dual.turns import TurnSpan

TypedTokens = Callable[[Sequence[str]], Sequence[int]]
# (turn words, voice name) -> the pupil's Mimi codes [L][8]
Voiced = Callable[[Sequence[str], str], Sequence[Sequence[int]]]

MASK_KINDS = ("mask", "complement", "spoken")


def tape_segments(spans: Sequence[TurnSpan], text: Sequence[int], record: dict[str, Any]) -> list[Segment]:
    own, pupil = record["audio"], record["user_audio"]
    return [
        tape_segment(text[s.start : s.end], own[s.start : s.end], pupil[s.start : s.end], scored=s.party == "teacher")
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
        row = assemble(f"{stem}#{kind}", spans, base, {k: ("typed", typed_segs[k]) for k in chosen})
        record = to_record(row)
        record["kind"] = kind
        record["stem"] = stem
        record["typed_turns"] = chosen
        rows.append(record)
    return rows


def holdout_record(
    stem: str,
    spans: Sequence[TurnSpan],
    base: Sequence[Segment],
    typed: TypedTokens,
    voiced: Voiced,
    voices: dict[str, str],  # label -> voice name, e.g. {"A": "vctk/p225_023.wav", ...}
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


def _seg_record(seg: Segment) -> dict[str, Any]:
    return {"text": list(seg.text), "own": [list(f) for f in seg.own], "pupil": [list(f) for f in seg.pupil]}


def segment_from(rec: dict[str, Any]) -> Segment:
    return tape_segment(rec["text"], rec["own"], rec["pupil"], scored=False)


def holdout_row(record: dict[str, Any], turn: int, rendering: str) -> tuple[dict[str, Any], tuple[int, int]]:
    """The base row with pupil turn `turn` swapped for `rendering` ("tape" keeps it), and the
    half-open frame window of the teacher's answer that follows it, after the shift."""
    from dual.frames import from_record

    base = from_record(record["base"])
    spans = list(base.spans)
    k = turn
    if rendering == "tape":
        seg = tape_segment(base.text[spans[k].start : spans[k].end], base.own[spans[k].start : spans[k].end], base.pupil[spans[k].start : spans[k].end], scored=False)
    else:
        seg = segment_from(record["renderings"][str(k)][rendering])
    parts: list[tuple[str, str, Segment]] = []
    for i, s in enumerate(spans):
        if i == k:
            parts.append((s.party, rendering, seg))
        else:
            parts.append((s.party, s.rendering, tape_segment(base.text[s.start : s.end], base.own[s.start : s.end], base.pupil[s.start : s.end], scored=s.party == "teacher")))
    row = concat(f"{record['id']}#{k}#{rendering}", parts)
    if row.frames > CONTEXT:
        raise ValueError(f"{row.id}: {row.frames} frames, over moshi's {CONTEXT}")
    if k + 1 >= len(row.spans) or row.spans[k + 1].party != "teacher":
        raise ValueError(f"{record['id']}: pupil turn {k} has no teacher answer after it")
    answer = row.spans[k + 1]
    return to_record(row), (answer.start, answer.end)
```

Tests (`tests/test_render.py`), with the two-ask spans from B2.1 and a 60-frame fake record
(own/pupil frames filled with a constant 8-tuple), `typed = lambda words: [100 + len(w) for w in words]`,
`voiced = lambda words, voice: [[len(voice)] * 8] * (2 * len(words))`:

- `training_rows` returns 3 records with kinds mask/complement/spoken; `typed_turns` of the
  mask and the complement partition `[0, 2]`; the spoken row equals the tape (`text`, `own`,
  `pupil` unchanged, `frames == 60`); a typed row is `60 - (end-start) + len(tokens) + 6`
  frames for each typed turn; `scored` is 0 on every pupil span and 1 on every teacher span.
- `holdout_record` has `pupil_turns == [0, 2]` and, per turn, keys `typed, A, B, C`.
- `holdout_row(rec, 2, "A")`: the row's span 2 has rendering `A`, the answer window equals
  span 3's `(start, end)`, and the answer's text equals the base's span-3 text (same tokens,
  shifted). `holdout_row(rec, 2, "tape")` reproduces the base row exactly.
- `assemble` raises past 3,000 frames (build a 2,999-frame base and swap in a 10-frame segment).

Commit `feat: training rows and the holdout record from one taped dialogue`.

### Task B2.3: `src/dual/kl.py` (TDD; torch on CPU)

```python
"""The number: per-frame KL between two renderings' text distributions, and the bar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from dual.constants import PAD


def log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def kl_per_frame(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per frame; both [T, V] log-probs over the same frames."""
    if lp_p.shape != lp_q.shape:
        raise ValueError(f"shapes {tuple(lp_p.shape)} vs {tuple(lp_q.shape)}")
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


@dataclass(frozen=True)
class AnswerKL:
    content: float  # mean over frames whose target text is not PAD
    all_frames: float
    content_frames: int
    frames: int


def answer_kl(lp_p: torch.Tensor, lp_q: torch.Tensor, targets: Sequence[int]) -> AnswerKL:
    """lp_* are the answer window's log-probs [W, V]; targets the window's text ids."""
    per = kl_per_frame(lp_p, lp_q)
    if per.shape[0] != len(targets):
        raise ValueError(f"{per.shape[0]} frames of KL, {len(targets)} targets")
    content = torch.tensor([t != PAD for t in targets])
    if not bool(content.any()):
        raise ValueError("an answer with no content frame")
    return AnswerKL(
        content=float(per[content].mean()),
        all_frames=float(per.mean()),
        content_frames=int(content.sum()),
        frames=int(per.shape[0]),
    )


@dataclass(frozen=True)
class Bar:
    median_gap: float
    median_floor: float
    p90_gap: float
    p90_floor: float
    turns: int

    @property
    def passes(self) -> bool:
        return self.median_gap <= self.median_floor and self.p90_gap <= self.p90_floor


def bar(gaps: Sequence[float], floors: Sequence[float]) -> Bar:
    if len(gaps) != len(floors) or not gaps:
        raise ValueError("gaps and floors are one number per holdout turn")
    g, f = np.asarray(gaps, dtype=float), np.asarray(floors, dtype=float)
    return Bar(
        median_gap=float(np.median(g)),
        median_floor=float(np.median(f)),
        p90_gap=float(np.percentile(g, 90)),
        p90_floor=float(np.percentile(f, 90)),
        turns=len(gaps),
    )
```

Tests: `kl_per_frame` of identical log-probs is 0 and of `[0.9, 0.1]` vs `[0.5, 0.5]` is
`0.9*ln(1.8) + 0.1*ln(0.2)` to 1e-6; `answer_kl` averages content frames only (targets
`[PAD, 7, PAD, 8]`, per-frame KLs made distinct by construction) and refuses an all-PAD
window; `bar` passes when gaps are below floors on both statistics and fails when the p90
alone is above; mismatched lengths raise.

Commit `feat: the per-frame KL and the bar`.

### Task B2.4: `src/dual/ladder.py` (TDD)

```python
"""Nested slices of the training stems for the data ladder."""

from __future__ import annotations

import math
from collections.abc import Sequence

from dual.holdout import ordered

RUNGS = (8, 4, 2, 1)  # denominators: an eighth, a quarter, a half, all


def rung(train_stems: Sequence[str], denominator: int) -> list[str]:
    if denominator not in RUNGS:
        raise ValueError(f"denominator {denominator}: one of {RUNGS}")
    by_hash = ordered(train_stems, "ladder")
    return sorted(by_hash[: math.ceil(len(by_hash) / denominator)])
```

Tests: 117 stems give rungs of 15, 30, 59, 117 stems, each a subset of the next; a bad
denominator raises. Commit `feat: the ladder's nested rungs`. Run `make check`, report.

---

## Batch B3: the render job (model: sonnet; runs on the box)

Needs B1 (image, `model.py`) and B2.

### Task B3.1: `src/dual/jobs/render.py` and `make render`

Inputs from `/data`: `dialogues/*.dsl`, `tape/speech.jsonl`, `voices/`. Output under the
shared dir `runs/<name>/render/`: `rows-train.jsonl` (351 rows), `holdout.jsonl` (13
records), `stems.json` (`{"held": [...], "train": [...], "chalkboard_sha": ..., "voices": {...}}`),
`render.log` lines in stdout.

```python
"""Renders every training row and the holdout records once. GPU for the TTS and Mimi."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

import torch
from harness.engine.audio import encode_wav, load_tts, voice_many

from dual.holdout import split
from dual.model import load_base
from dual.render import holdout_record, tape_segments, training_rows
from dual.turns import spans, text_row, typed_tokens

VOICES = {"A": "vctk/p225_023.wav", "B": "vctk/p248_023.wav", "C": "vctk/p302_023.wav"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=0, help="dialogues, for a smoke render")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    out = pathlib.Path.cwd() / "runs" / args.name / "render"
    out.mkdir(parents=True, exist_ok=True)
    records = {r["id"]: r for r in map(json.loads, (data / "tape" / "speech.jsonl").read_text().splitlines())}
    sources = {p.stem: p.read_text() for p in sorted((data / "dialogues").glob("*.dsl"))}
    stems = sorted(set(records) & set(sources))
    if len(stems) != len(sources):
        raise SystemExit(f"{len(sources) - len(stems)} dialogues have no tape record")
    held, train = split(stems)
    if args.limit:
        held, train = held[: max(1, args.limit // 10)], train[: args.limit]

    base = load_base("moshiko")  # the SPM and Mimi; the weights are not used here
    spm = base.spm
    mimi = base.info.get_mimi(device="cuda")
    tts = load_tts("cuda")
    voice_cache: dict[tuple[str, str], list[list[int]]] = {}

    def pieces(word: str) -> Sequence[str]:
        return list(spm.EncodeAsPieces(word))

    def typed(words: Sequence[str]) -> Sequence[int]:
        return typed_tokens(words, spm.EncodeAsIds)

    def voiced(words: Sequence[str], voice: str) -> list[list[int]]:
        key = (" ".join(words), voice)
        if key not in voice_cache:
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            text = " ".join(words)
            if not text.endswith((".", "?", "!")):
                text += "."
            (v,) = voice_many(tts, [text], [str(data / "voices" / f"{voice}.1e68beda@240.safetensors")])
            voice_cache[key] = encode_wav(mimi, v.pcm, "cuda")
        return voice_cache[key]

    n_rows = 0
    with (out / "rows-train.jsonl").open("w") as rows_f, (out / "holdout.jsonl").open("w") as hold_f:
        for stem in train + held:
            rec, src = records[stem], sources[stem]
            sp = spans(src, rec)
            text = text_row(src, rec, pieces, spm.PieceToId)
            segs = tape_segments(sp, text, rec)
            if stem in held:
                hold_f.write(json.dumps(holdout_record(stem, sp, segs, typed, voiced, VOICES)) + "\n")
                print("held", stem, "pupil turns", sum(s.party == "pupil" for s in sp))
            else:
                for row in training_rows(stem, sp, segs, typed):
                    rows_f.write(json.dumps(row) + "\n")
                    n_rows += 1
                print("train", stem, "rows", 3)
    # Determinism: the same words in the same voice encode to the same codes twice.
    probe = voiced(("the", "same", "words"), VOICES["A"])
    voice_cache.clear()
    again = voiced(("the", "same", "words"), VOICES["A"])
    print("tts deterministic:", probe == again)
    (out / "stems.json").write_text(json.dumps({
        "held": held, "train": train, "voices": VOICES,
        "chalkboard_sha": pathlib.Path("/app/vendor/SHA").read_text().strip(),
    }, indent=1))
    print(f"rendered train rows {n_rows} holdout {len(held)}")


if __name__ == "__main__":
    main()
```

Makefile:

```make
.PHONY: render
render:
	sh scripts/submit.sh render-$(NAME) render --name render-$(NAME) $(ARGS)
```

**Steps.** (1) `make render NAME=smoke ARGS='--limit 10'`; wait; the log must say
`rendered train rows 30 holdout 1` and `tts deterministic: True`. If the TTS is not
deterministic under `manual_seed`, the TTS samples on CUDA with its own generator: find the
sampling call in vendored `moshi/models/tts.py` (`tts.generate`) and seed
`torch.cuda.manual_seed_all` too; if still not, report it as an open issue and keep the
cache (one render is the holdout; it is never rendered twice). (2) `make render NAME=1`
on everything; expect `rendered train rows 351 holdout 13`. (3) Copy `stems.json` back to
`docs/results/render-1-stems.json` and commit it with the job: `feat: the render job and
render-1's stems`. The `.1e68beda@240.safetensors` suffix is chalkboard's
`pupil_voices.VOICE_SUFFIX`; `voice_many` reads a `.safetensors` path as a local embedding
(`harness/engine/audio.py:349`).

Report: counts, the determinism line, max row frames (print it), and the run name.

---

## Batch B4: train and measure (model: opus)

Needs B3's `runs/render-1`.

### Task B4.1: `src/dual/jobs/train.py` and `make train`

```make
.PHONY: train measure bar
BASE ?= personaplex
SLICE ?= 1
RENDER ?= render-1
EPOCHS ?= 8

train:
	sh scripts/submit.sh $(NAME) train --name $(NAME) --base $(BASE) --slice $(SLICE) --render $(RENDER) --epochs $(EPOCHS) $(ARGS)
```

The loop, in full:

```python
"""The trunk LoRA on one base over one rung of the rendered rows."""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import torch
from sparks.emit import track

from dual.constants import IGNORE
from dual.ladder import rung
from dual.model import adapter_state, attach_lora, codes_from_row, forward, load_base, RANK, SCALING

LR = 2e-5
BETAS = (0.9, 0.95)
CLIP = 1.0
TEXT_WEIGHT = 2.0
AUDIO_WEIGHT = 0.5
FRAMES_PER_BATCH = 4000  # rows are packed by length into batches under this many frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--slice", type=int, default=1)
    p.add_argument("--render", default="render-1")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def losses(lm, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Text and audio CE over scored frames only. Rows are padded to the longest; padding is
    unscored. NaN logits at moshi's masked frames are excluded by index, never multiplied."""
    width = max(r["frames"] for r in batch)
    codes = torch.cat([torch.nn.functional.pad(codes_from_row(r), (0, width - r["frames"]), value=3) for r in batch])
    scored = torch.zeros(len(batch), width, dtype=torch.bool, device=codes.device)
    for i, r in enumerate(batch):
        scored[i, : r["frames"]] = torch.tensor(r["scored"], dtype=torch.bool)
    out = forward(lm, codes)
    text_target = codes[:, 0]
    text_keep = scored & out.text_mask
    text = torch.nn.functional.cross_entropy(out.text_logits[text_keep], text_target[text_keep])
    audio_target = codes[:, 1:9]
    audio_keep = scored[:, None, :] & out.audio_mask
    audio = torch.nn.functional.cross_entropy(out.audio_logits[audio_keep], audio_target[audio_keep])
    return text, audio


def batches(rows: list[dict], rng: random.Random) -> list[list[dict]]:
    by_len = sorted(rows, key=lambda r: r["frames"])
    out: list[list[dict]] = [[]]
    for r in by_len:
        if out[-1] and (len(out[-1]) + 1) * max(x["frames"] for x in out[-1] + [r]) > FRAMES_PER_BATCH:
            out.append([])
        out[-1].append(r)
    rng.shuffle(out)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    render = pathlib.Path.cwd() / "runs" / args.render / "render"
    stems = json.loads((render / "stems.json").read_text())
    keep = set(rung(stems["train"], args.slice))
    rows = [r for r in map(json.loads, (render / "rows-train.jsonl").read_text().splitlines()) if r["stem"] in keep]
    held = [json.loads(line)["base"] for line in (render / "holdout.jsonl").read_text().splitlines()]
    out = pathlib.Path.cwd() / "runs" / args.name / "train"
    out.mkdir(parents=True, exist_ok=True)
    print(f"base {args.base} slice 1/{args.slice}: {len(keep)} dialogues, {len(rows)} rows, {len(held)} held")

    base = load_base(args.base)
    attach_lora(base)
    lm = base.lm
    params = [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS)
    steps_per_epoch = len(batches(rows, random.Random(0)))

    with track(total=args.epochs * steps_per_epoch, arm=f"{args.base}-{args.slice}") as run:
        for epoch in range(1, args.epochs + 1):
            lm.train()
            for batch in batches(rows, rng):
                text, audio = losses(lm, batch)
                loss = TEXT_WEIGHT * text + AUDIO_WEIGHT * audio
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, CLIP)
                opt.step()
                run.step(loss={"total": float(loss), "text": float(text), "audio": float(audio)}, grad_norm=float(norm), learning_rate=LR, epoch=epoch)
            lm.eval()
            with torch.no_grad():
                held_text = [float(losses(lm, [r])[0]) for r in held]
            run.log(eval_loss=sum(held_text) / len(held_text), epoch=epoch)
            blob = {"state": adapter_state(base), "base": args.base, "rank": RANK, "scaling": SCALING,
                    "epoch": epoch, "slice": args.slice, "render": args.render, "seed": args.seed}
            tmp = out / f"model-epoch-{epoch:02d}.pt.tmp"
            torch.save(blob, tmp)
            tmp.rename(out / f"model-epoch-{epoch:02d}.pt")
            print(f"epoch {epoch} held text {sum(held_text) / len(held_text):.4f}")
    print("trained", out)


if __name__ == "__main__":
    main()
```

Notes for the agent: the mixed-length batch pads with `3` on every stream, which is `<pad>`
on text and an arbitrary code on audio; both are unscored, fine. `out.text_mask` and
`out.audio_mask` carry moshi's delay-invalid frames. `losses` on a held row in eval mode is
the per-row text NLL on the teacher spans of the all-spoken rendering, the `eval_loss`
series. `unmoved(lm)` from `training.moshi.trunk_lora` after epoch 1 should be empty; print it.

**Steps.** (1) `make train NAME=smoke-train BASE=personaplex SLICE=8 EPOCHS=1`; must finish,
print `trained`, and leave `model-epoch-01.pt` (~136 MB). Read the log: loss finite, grad
norm finite, `unmoved: []`. (2) Same for `BASE=moshiko`. Commit `feat: the train job`.

### Task B4.2: `src/dual/jobs/measure.py`, `src/dual/report.py`, `make measure`, `make bar`

Measure takes a base and either no checkpoint (the bare control) or a run name (every
`model-epoch-*.pt` in it). Per holdout record, per pupil turn `k`, it builds five rows with
`holdout_row` (`typed`, `A`, `B`, `C`, `tape`), runs `forward` on each (batch of one; T
differs), takes `log_probs` of the text logits over each row's answer window, and writes
one line per turn:

```json
{"stem": ..., "turn": k, "epoch": e, "gap": AnswerKL(typed,A), "floor": AnswerKL(A,B),
 "door": AnswerKL(C,A), "tape_gap": AnswerKL(typed,tape), "tape_floor": AnswerKL(tape,A),
 "answer_frames": W, "content_frames": c}
```

to `runs/<name>/measure/<base>[-<run>]/epoch-NN.jsonl` (epoch 00 for bare), and a
`summary.json` with `bar(...)` on the `content` numbers and on `all_frames`, per epoch:
`{"epoch": e, "content": Bar, "all": Bar, "door_median": ..., "tape_gap_median": ...}`.
The answer window's targets are the row's `text[start:end]`. Assert every window has at
least one content frame (the renderer guarantees a teacher span has words). Print the
summary table.

`report.py` (pure, tested): `table(summaries: list[dict]) -> str` renders a markdown table
with columns `run, base, slice, epoch, median gap, median floor, p90 gap, p90 floor, pass,
door, tape gap`, and `best(summaries)` picks per run the epoch with the smallest
`median gap / median floor`. Test on two hand-written summaries.

Makefile:

```make
measure:
	sh scripts/submit.sh measure-$(NAME) measure --base $(BASE) --render $(RENDER) $(if $(RUN),--run $(RUN),) $(ARGS)

bar:
	scp -r $(BOX):/srv/spark/runs/measure-*/measure runs/ 2>/dev/null || true
	uv run python -m dual.report runs/ > docs/results/bar.md && cat docs/results/bar.md
```

**Steps.** (1) `make measure NAME=bare-pp BASE=personaplex` and `make measure NAME=bare-mk BASE=moshiko`:
the starting gaps. (2) `make measure NAME=smoke-pp BASE=personaplex RUN=smoke-train`. (3)
`make bar`; commit `feat: the measure job and the bar report` with `docs/results/bar.md`.
Report the bare-model table. Expected shape, not a gate: `door` near `floor` on both bases
(else the harness or the TTS is at fault: stop and report); `gap` far above `floor` on
moshiko; PersonaPlex's gap is the datum.

---

## Batch B5: the ladder (model: sonnet)

`scripts/ladder.sh`:

```sh
#!/bin/sh
# Eight runs, one queue: both bases over the four rungs of render-1, then one measure each.
set -eu
IMAGE="${IMAGE:?set IMAGE to the tag smoke-train pushed (sparks status <job> names it)}"
for base in personaplex moshiko; do
  for slice in 8 4 2 1; do
    name="ladder-$base-$slice"
    JOB=$(IMAGE=$IMAGE make -s train NAME=$name BASE=$base SLICE=$slice RENDER=render-1 EPOCHS=8)
    sparks wait "$JOB"
    MJOB=$(IMAGE=$IMAGE make -s measure NAME=$name BASE=$base RUN=$name RENDER=render-1)
    sparks wait "$MJOB"
  done
done
make bar
```

Run it (`sh scripts/ladder.sh`), then write `docs/results/ladder-1.md`: the bar table from
`make bar`, the eval_loss per epoch per run (from the logs), the probe's summary from B1,
the render counts, and three sentences reading the curve against the design's three
outcomes (still falling, flat above the floor, inside the floor). Commit
`docs: ladder 1, two bases over four rungs of 117 dialogues`.

Finish with `GATE: read docs/results/ladder-1.md. Options: (a) grow the corpus with
chalkboard's generator and rerun the ladder; (b) change rank/LR/base and rerun; (c) the door
passes, move to the engine (the open points: 16-step depformer per frame, voice prompt,
typing while moshi speaks).`

---

## Appendix: facts the agents will need and should not re-derive

- `speech.jsonl` record: `id`, `frames`, `words` (list per spoken DSL line of `[word, frame]`),
  `audio` `[T][8]` (teacher = moshi's own stream), `user_audio` `[T][8]` (pupil), `voice`.
  Undelayed. 130 records, every one has a `dlg-00005-NNN.dsl`.
- DSL: `ask <words>` is the pupil, `say <words>` the teacher, `pause <n>` and `draw ...` are
  the teacher's; strict alternation opening on `ask`; 5 asks per dialogue; words are plain,
  numbers spelled out.
- SPM: `kyutai/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model`; PersonaPlex ships the same
  file. Ids 0..3 are `<unk> <s> </s> <pad>`.
- Mimi: one checkpoint for both bases (same `tokenizer-e351c8d8-checkpoint125`), 24 kHz,
  1,920 samples per frame, 8 codebooks of 2,048.
- `moshi==0.2.13`: `LMModel.forward(codes[B,17,T]) -> LMOutput(logits[B,dep_q,T,2048], mask,
  text_logits[B,1,T,32000], text_mask)`, delays applied inside, logits aligned to the input
  frames, NaN at masked frames. `LMGen(lm, temp, temp_text, on_text_hook, ...)`,
  `_step(input_tokens[B,8,1], depformer_replace_tokens[B,dep_q,1] | None)`.
- chalkboard pins: torch 2.13.0 cu130, moshi 0.2.13 `--no-deps`, sphn 0.2.1, sparks-dgx 0.1.1,
  `HF_HOME=/opt/hf`, `HF_HUB_OFFLINE=1`, `HOME=/tmp`, `TRITON_CACHE_DIR=/tmp/triton`.
- sparks: `--data` is required; the image tag is `<registry>/vlad/<name>:<git sha>`; two
  submits from one commit under one name overwrite; `--secret HF_TOKEN` reads the value
  from the submitting shell; the first `run.step` reports no rate.
- Box paths: shared dir `/srv/spark` (cwd in a job); `runs/<name>/...`; `models/`.
- chalkboard's dv3a tape run: `job-20261023-172331-vlad-dv3a-tape`, image `deriv:f70c0143`.

---

## Batch B6: ladder 2, the recipe sweep (model: sonnet for code, the orchestrator runs it)

Vlad's answer at the ladder-1 gate (2026-11-03): **"(b) change rank / LR / base, rerun."**
Ladder 1 left two constraints: run **3 epochs or fewer**, and **repeat one rung with a
second seed** so noise is measured. B6 is that sweep: one rung, one base, five new runs,
about a day of the single queue.

### B6.0 What the sweep asks, and what it can and cannot tell

The question: **does any rank or LR change move the median gap/floor ratio (content frames)
of PersonaPlex on all 117 dialogues away from ladder 1's ~3?**

- **Base: PersonaPlex only.** The two bases tracked each other within noise at every rung of
  ladder 1, so a moshiko run is the least informative use of four hours.
- **Rung: 1/1 (117 dialogues), 3 epochs.** Held-out text loss bottomed at epoch 2 in every
  117-dialogue run, and the lowest gap was at epoch 1.
- **Seed 0 is already measured.** Ladder 1's `ladder-personaplex-1` ran this recipe: rank 16,
  scaling 2.0, constant LR 2e-5 with no schedule, seed 0. Its epochs 1 to 3 are a 3-epoch run
  of the same recipe, step for step, and they are measured. So the noise repeat costs one run,
  seed 1, not two.
- **Holdout noise is already large.** A stem-level bootstrap (resampling the 13 holdout
  dialogues with their 5 turns, 2,000 draws) on ladder 1's per-turn lines gives:
  - for one run's median ratio, a 95% interval of about ±0.45 (`ladder-personaplex-1`
    epoch 1: 3.04, [2.57, 3.44]);
  - for the paired difference of two runs on the same stems, about ±0.55
    (`ladder-personaplex-1` minus `ladder-personaplex-2` at epoch 1: -0.17, [-0.71, 0.38]).

  So with 65 turns the sweep can detect a move of about 0.5 or more, and nothing smaller.
  That is enough for the decision: the bar needs the ratio at 1, so a lever that is worth a
  ladder 3 has to move it by much more than 0.5. A move under 0.5 is reported as "not moved",
  never as "small".
- **Rank uses matched alpha.** chalkboard's `attach_trunk_lora` takes the multiplier itself,
  2.0 at rank 16, which is alpha 32 at the usual alpha/r. Rank 4 and rank 64 keep alpha at 32
  (scaling 8.0 and 0.5), the parametrisation under which "LoRA Without Regret" finds the best
  LR roughly independent of rank. So the rank runs change capacity, not the early step size.
  Holding the multiplier at 2.0 instead would make rank 64 also a roughly 4x larger effective
  rate, and the result could not be read as capacity. (moshi's own note is to hold 2.0 and
  sweep the rate; the LR runs below cover the rate.)

### B6.0.1 The runs (one at a time, in this order)

| run | rank | scaling | LR | seed | what it asks |
|---|---|---|---|---|---|
| `ladder-personaplex-1` (ladder 1, epochs 1-3) | 16 | 2.0 | 2e-5 | 0 | the reference, already measured |
| `ladder2-pp-s1` | 16 | 2.0 | 2e-5 | 1 | noise: the same recipe, another seed |
| `ladder2-pp-lr5e-6` | 16 | 2.0 | 5e-6 | 0 | a slower rate: does a less-overfit start do better than epoch 1? |
| `ladder2-pp-lr1e-4` | 16 | 2.0 | 1e-4 | 0 | a faster rate: is the ratio a function of progress only? |
| `ladder2-pp-r64` | 64 | 0.5 | 2e-5 | 0 | capacity up 4x at matched alpha |
| `ladder2-pp-r4` | 4 | 8.0 | 2e-5 | 0 | capacity down 4x at matched alpha |

Every run: `BASE=personaplex SLICE=1 RENDER=render-1 EPOCHS=3`, compile engine, measured at
epochs 1, 2 and 3. The fifth variant is rank 4 rather than moshiko or regularisation. The
bases were already equal, and AdamW's 0.01 weight decay moves a LoRA by lr x 0.01 per step,
which is nothing at 2e-5. Rank 4 with rank 64 gives a three-point rank axis (4, 16, 64)
through the reference, beside the three-point LR axis (5e-6, 2e-5, 1e-4).

Cost: a PersonaPlex 117-dialogue train is ~54 min per epoch (3,220 s), so ~2.7 h for 3
epochs, plus ~25 min per measured epoch, so ~1.3 h. That is ~4 h a run and ~20 h for the five.

### B6.0.2 What counts as "the recipe moved the ratio"

Per run, read on content frames over epochs 1 to 3 only:
- **R_gap decides.** It is the median gap/floor at the run's lowest-median-gap epoch
  (`lowest_gap()`); that is the run's **decision epoch**. **R** is the best median gap/floor
  of the three epochs, `report.best`'s pick. R is still reported, but it no longer decides:
  as a run overfits, the floor inflates faster than the gap and the ratio falls while the gap
  triples (ladder 1, personaplex, per epoch: 3.04, 3.24, 2.87, 2.43 at epoch 4), so a worse
  fast variant could be called "down". This rule was changed before any ladder-2 result
  existed.
- **Seed noise:** `d = |R_gap(ladder-personaplex-1) - R_gap(ladder2-pp-s1)|`. The reference
  **R0** is the mean of the two seeds' R_gap.
- **Threshold:** `T = max(2 d, 0.5)`. The 0.5 is the holdout's own paired noise. It keeps a
  seed pair that agrees by chance from making the threshold tiny.

A variant **moved the ratio** only if both hold:
1. `|R_gap - R0| >= T`, and
2. the paired stem-bootstrap 95% interval (2,000 draws, seed 0, the same stems on both sides,
   each run at its own R_gap epoch) of R_gap minus R_gap(seed 0) and of R_gap minus
   R_gap(seed 1) both exclude 0, with the same sign, which is also the sign of R_gap - R0.

It moved it **down** (better) or **up**. Anything else is **not moved**.

**Floor-inflation guard.** A "down" is plain only if the run's median content gap at its
decision epoch is at most the mean of the two seeds' median content gaps at their decision
epochs. Otherwise the verdict becomes **"down, floor inflated"**: the ratio fell without the
gap falling, so it fell through the floor. The bar is gap <= floor, so a recipe whose ratio
falls only because the floor grows does not move toward it. (A floor test such as 1.5 x the
seeds' floor is too loose: with R0 estimated at about 3.12 before the results; it came out at 3.00 and T = 0.5, a variant with an unchanged gap
clears T on about 1.19 x floor growth.) The gate does not count it as moved. A p90 ratio
(read at the decision epoch) that contradicts the median verdict is reported beside it, but
it does not change the verdict.

The table gives, per run: R (epoch), R_gap (epoch), the median gap and median floor at the
decision epoch, the p90 ratio there, R_gap - R0, the two intervals and the verdict. Seed rows
show "-" for the verdict and for their own self-interval.

How the gate reads the answer ("moved down" means a plain "down", never "down, floor
inflated"):
- **No variant moved down:** rank and LR are not the lever at this corpus size. The gate
  goes back to Vlad as (a) grow the corpus, (c) the engine, or a new recipe axis outside
  rank/LR (for example a consistency term on the typed rendering's answer).
- **One or more moved down:** ladder 3 reruns the data rungs at the best variant's recipe.

`ladder-2.md` must say that the bootstrap intervals are optimistic (the epoch is picked on
the same 65 turns, and it is a percentile bootstrap over 13 stems), and that seed noise d is
measured only at r16/2e-5.

### Task B6.1: `--rank`, `--scaling`, `--lr` on the train job; measure attaches the checkpoint's own rank (TDD)

Files: `src/dual/model.py`, `src/dual/jobs/train.py`, `src/dual/jobs/measure.py`,
`tests/test_train.py`, `tests/test_model_checks.py`, `tests/test_measure.py`.

- `model.py`:
  - `ADAPTER_PARAMS_PER_RANK = 2_129_920`, which is 66,560 a unit of rank per layer x 32
    layers, and `adapter_params(rank) -> int`. Rank 16 must stay 34,078,720.
  - `attach_lora(base, rank=RANK, scaling=SCALING)` records the rank and scaling on the
    `Base` (new optional fields), and checks trainable == `adapter_params(rank)`.
  - `adapter_meta(base)` reads them from the `Base`, and raises if no adapter is attached.
  - New pure helper `adapter_setup(meta) -> (rank, scaling)` that refuses a meta missing
    either key.
  - Defaults are unchanged, so every ladder-1 checkpoint (meta rank 16, scaling 2.0, params
    34,078,720) still loads.
- `train.py`:
  - `--lr` (default `LR`), `--rank` (default `RANK`), `--scaling` (default `SCALING`).
  - The optimiser and `run.step(learning_rate=...)` use `args.lr`. The checkpoint's `run`
    dict gains `rank`, `scaling` and `lr`, and the start line prints the recipe.
  - `--resume` refuses a checkpoint whose `run.lr` differs from `--lr`. A missing key means
    the old default 2e-5. Rank and scaling are already refused by `check_adapter_meta`.
- `measure.py`: read the first checkpoint's blob, then `attach_lora(base, *adapter_setup(blob["meta"]))`.
  `load_blob_adapters` still checks every checkpoint against that setup. **Nothing about what
  is measured changes**: no change to rows, KL, windows, summaries or output paths.
- Tests (laptop, CPU, no moshi):
  - argparse defaults and flags;
  - `adapter_params(4/16/64)`;
  - `adapter_setup` on a ladder-1 meta and on a broken one;
  - `adapter_meta` on a fake `Base` with rank 64;
  - `check_adapter_meta` refusing a rank mismatch;
  - the resume LR check as a pure helper `check_resume_lr(run_meta, lr)`.

`make check`, commit `feat: the train job takes rank, scaling and LR; measure attaches the checkpoint's setup`.

### Task B6.2: `scripts/ladder.sh` runs ladder 2 (TDD on `tests/test_ladder_sh.py`)

Files: `scripts/ladder.sh`, `tests/test_ladder_sh.py`.

- The run list moves out of the nested loops into a `LADDER` selector. `LADDER=1`, the
  default, is ladder 1's eight runs at `EPOCHS=8`, exactly as before. `LADDER=2` is B6.0.1's
  five runs, each `BASE=personaplex SLICE=1 RENDER=render-1 EPOCHS=3` with its own
  `ARGS='--seed 1'`, `'--lr 5e-6'`, `'--lr 1e-4'`, `'--rank 64 --scaling 0.5'` or
  `'--rank 4 --scaling 8.0'`.
- Everything else stays as it is: one run at a time, the memory gate, the ssh-retrying
  submit, wait and lookup, and `make bar` at the end.
- `FROM` is validated against the selected list.
- Measure gets no ARGS: it reads the setup from the checkpoint.
- **Trap:** the list must not be fed to a `while read` loop on stdin. `ssh` in `gate` reads
  stdin and would eat the remaining runs. Iterate over words, for example
  `name,epochs,arg,arg` split with `IFS=,` inside a function, or give ssh `-n`.
- Tests with the existing fakes:
  - `LADDER=2` submits the five trains in order, each with its ARGS and `EPOCHS=3`, and each
    followed by its measure;
  - `LADDER=2 FROM=ladder2-pp-r64` starts there;
  - an unknown `FROM` exits 2;
  - the ladder-1 tests still pass untouched.

`make check`, commit `feat: the ladder script runs ladder 2's recipe sweep`.

### Task B6.3: `src/dual/noise.py` and `make noise`, the seed and bootstrap verdict (TDD, pure)

Files: `src/dual/noise.py`, `tests/test_noise.py`, `Makefile` (target `noise`).

- **Input:** the per-turn lines in `runs/measure/<label>/epoch-NN.jsonl`, the files `make bar`
  copies. Each line holds `stem`, `turn`, and `gap`/`floor` as `{"content", "all_frames", ...}`.
- **Imports:** numpy and the standard library only, no torch. The median and p90 must equal
  `kl.bar`'s (`np.median`, `np.percentile(..., 90)`).
- **Functions:**
  - `ratio(lines)`, the median content gap over the median content floor, and `p90_ratio(lines)`;
  - `pick(by_epoch, max_epoch=3) -> (epoch, R)`, `report.best`'s rule (a passing epoch first,
    then the smallest ratio) restricted to epochs `1..max_epoch`;
  - `lowest_gap(by_epoch, max_epoch=3) -> (epoch, ratio)`;
  - `paired_ci(a, b, draws=2000, seed=0) -> (delta, lo, hi)`. It resamples stems with
    replacement and uses the same draw on both sides. A stem missing from either side raises.
  - `verdict(r_gap, r_s0, r_s1, ci_s0, ci_s1, threshold) -> "down" | "up" | "not moved"`,
    per B6.0.2, and `gap_guard(v, gap, gap_s0, gap_s1)`, which turns a "down" whose
    median gap is above the seeds' mean into "down, floor inflated".
- **CLI:** `python -m dual.noise ROOT --seeds LABEL LABEL --variants LABEL...` prints a
  markdown table and the noise line. The table columns are run, R (epoch), R_gap (epoch),
  median gap and median floor at the decision epoch, p90 ratio there, R_gap - R0, the 95%
  intervals against each seed, and the verdict (B6.0.2, with the gap guard). The
  noise line gives d, R0 and T, all on R_gap.
- **Makefile:**
  ```make
  noise:
  	PYTHONPATH=src uv run python -m dual.noise runs/ --seeds personaplex-ladder-personaplex-1 personaplex-ladder2-pp-s1 --variants personaplex-ladder2-pp-lr5e-6 personaplex-ladder2-pp-lr1e-4 personaplex-ladder2-pp-r64 personaplex-ladder2-pp-r4
  ```
- **Tests:** hand-built lines.
  - A ratio and p90 checked against numpy.
  - `pick` ignores epoch 4 or later.
  - `paired_ci` of a run against itself is (0, 0, 0).
  - A shifted run gives an interval that excludes 0.
  - `verdict` covers all three outcomes, including |Δ| >= T with an interval that includes 0,
    which is "not moved".
  - Plus one test on ladder 1's real numbers, if a fixture is small enough to commit
    (a handful of lines).

`make check`, commit `feat: the noise verdict, seeds and a paired stem bootstrap`.

B6.1, B6.2 and B6.3 touch disjoint files and run in parallel; one sonnet review on their
merged diff, then an opus review before main.

### Task B6.4: run it (the orchestrator)

1. Build once from the merged B6 head and push to the registry the box can reach. Off the
   LAN, use the B1 workaround:
   ```sh
   SHA=$(git rev-parse --short HEAD)
   rm -rf docker/src docker/vendor && cp -RL src docker/src && cp -RL vendor docker/vendor
   DOCKER_BUILDKIT=0 docker build -t 100.114.56.75:5000/vlad/ladder2:$SHA docker
   docker push 100.114.56.75:5000/vlad/ladder2:$SHA
   ```
   Then confirm the new flags reach the job: `make train NAME=ladder2-smoke SLICE=8 EPOCHS=1
   ARGS='--rank 4 --scaling 8.0'` with `IMAGE=spark.local:5000/vlad/ladder2:$SHA`. Read
   `unmoved: []`, the recipe line and the param count (8,519,680), then `sparks cancel` it
   after epoch 1's first steps. This is optional when the B6.1 tests cover the parse.
2. Stop drama3 (Vlad-authorised; never remove or edit it):
   `ssh vlad@<box> 'cd ~/drama3 && docker compose stop'`. Then `free -g`, and expect about
   110 GiB available.
3. Launch detached from the B6 worktree:
   ```sh
   LADDER=2 IMAGE=spark.local:5000/vlad/ladder2:$SHA SPARKS_HOST=vlad@<box> \
     nohup caffeinate -i sh scripts/ladder.sh > $SCRATCH/ladder2.log 2>&1 &
   ```
   Poll the log and `sparks queue`, and check `free -g` every ~10 min with bounded ssh
   (`-o ConnectTimeout=15`). After a reboot, re-stop drama3, resubmit the hung train with
   `ARGS='<its args> --resume'`, wait for it, measure it by hand, then relaunch with
   `FROM=<next run>`. If ssh is dead for more than 15 min, stop with `BLOCKED: power-cycle
   the box` and name the running job.
4. `make bar BOX=vlad@<box>`, which regenerates `docs/results/bar.md` with the ladder-2 rows,
   then `make noise`.

### Task B6.5: `docs/results/ladder-2.md`

Contents, in ladder-1.md's style:
- the runs and the image;
- the content-frame table per run and epoch (from `bar.md`);
- held-out text loss per epoch (from the train logs);
- `make noise`'s table and noise line;
- the verdict per variant;
- the answer, in the terms of B6.0.2.

Commit `docs: ladder 2, rank and LR at 117 dialogues, three epochs, two seeds`, merge
`--no-ff`, then finish with `GATE` (B6.0.2's two branches).

### B6 as run (2026-11-03 to 11-04)

- **Images:** s1, lr5e-6, lr1e-4 and r64 ran on `ladder2:5a2d32e`. r4 ran on `ladder2:a800c4c`, which is the same train and measure code plus `--mem-cap-gib 64` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, in `cap_gpu_memory` in `src/dual/model.py`.
- **Smoke:** `ladder2-smoke-r4` (slice 8, 1 epoch, rank 4) ran to the end and was measured, to prove that the attach and the measure work at a non-default rank.
- **The freeze:** the first r4 train froze the box (`NVRM NV_ERR_NO_MEMORY`) on 11-04 at 14:52. drama3 had been restarted by someone and was running beside it. Vlad power-cycled the box, and r4 was rerun from scratch with the cap. The cap bounds PyTorch's allocator only, so GPU runs still need drama3 down. A Mac-side watchdog stopped drama3 during the rerun if it reappeared.
- **The verdict rule** was changed twice before any result: R_gap decides, then the gap guard. See B6.0.2. The results are in `docs/results/ladder-2.md`.

## Batch B7: side by side (2026-11-05)

Vlad, at the ladder-2 gate: "ok, do the side by side check first". The request: before picking among an alignment loss, the bar, the engine or more data, read what a trained model says after a typed pupil turn and after the same turn spoken. No bar change, no training.

- **Code:**
  - `src/dual/sidebyside.py`, pure: the LMGen step alignment against `LMModel.forward`'s layout, and batch packing.
  - `src/dual/sbs_report.py`, pure: word-level statistics and the per-turn tables. Two silent answers are counted apart.
  - `src/dual/jobs/sidebyside.py` and `make sidebyside`: greedy decoding of every holdout pupil turn as typed, A and B, for bare and one checkpoint. Batched with LMGen's per-item exec mask, and each batch's KV ring cut to its longest item.
- **As run:**
  - `sbs-1`: bare plus `ladder-personaplex-1` epoch 2, 65 turns x 3 renderings.
  - `sbs-1-ctl`: the same rows for 10 turns, in other batch shapes.
  - About 84 GPU-minutes in all.
  - Blind substance labels by model judges.
- **Results:** `docs/results/side-by-side-1.md`. Typed vs A differs in substance in 21/65 turns and A vs B in 23/65. The same row in another batch shape differs in 6/30. The tutor takes up the pupil's turn after 65/65 typed turns and 61/65 for each voice. The one systematic difference is onset: after A or B the trained model starts 2-3 frames early (frame 0 in 52/65), while after typed it starts on the tape's frame 3 in 65/65. That shows on PAD frames, which the content bar skips (all-frame gap/floor ~19). Untested: how much of the content-frame ~3 is this onset.

## Batch B8: the bar by position in the answer (2026-11-05)

Vlad, at the side-by-side gate: "Re-check the bar (Recommended). Cheap first: one measure pass on epoch 2 (no training) reporting KL by position in the answer, to see if the gap sits in the opening frames. Then decide whether to change the bar."

The question: B7 found that after A or B the trained model starts 2-3 frames earlier than after typed (or tape). The bar's content mean skips PAD targets, so most of that onset is hidden from it (all-frame gap/floor ~19, content ~3). How much of the content ratio of ~3 is still the opening? The measure job keeps only per-turn means, so B8 adds per-frame output and one pass. No training, no bar change.

**Position.** In the answer window `[start, end)` from `holdout_row`, let `f0` be the first frame whose target text is not PAD. A frame's position is `i - f0`. Buckets: `lead` (position < 0, the silence before the tutor's first word, where the onset shows), `0`, `1`, `2`, `3-5`, `6-10`, `11-25`, `26+`. The targets are the tape's teacher text in every rendering (measure asserts this), so positions are the same for every pair.

### Task B8.1: `src/dual/position.py` and `make position` (TDD, pure: numpy, no torch)

- `BUCKETS`, `bucket(pos) -> str`, `positions(content_mask) -> list[int]`. A mask with no content frame raises.
- `turn_means(per_frame, content_mask, *, content_only) -> dict[bucket, float | None]`: one turn's mean KL per bucket, or None if the bucket has no frames. With `content_only`, PAD frames are dropped. `lead` is all PAD, so it is None there.
- `bucket_table(lines, mode)`: one row per bucket. Columns: turns with frames, frames, median over turns of the per-turn bucket mean for gap, floor and door, the ratio median gap / median floor, and each bucket's share of the pooled gap and floor KL mass (sum over frames in the bucket over the sum over all frames in that mode).
- `bar_from(lines, min_pos, mode)`: the bar (median/p90 gap and floor, ratio) on per-turn means over frames with position >= `min_pos`, for `min_pos` in (all, 0, 1, 2, 3, 6, 11). Content mode with `min_pos` 0 equals the existing content bar, and all mode with no cut equals the all-frame bar. Tests pin both identities.
- `python -m dual.position FILE...` prints markdown for each `epoch-NN.jsonl`. Make target: `make position FILES=...`.

### Task B8.2: the measure job writes per-frame KL (TDD on the pure parts)

- Each line gains `content_mask` (0/1 per answer frame) and `per_frame`: `{pair: [KL per frame, 6 significant digits]}` for the five pairs. The existing fields are unchanged.
- `--epochs N [N ...]` measures only those checkpoints. It is ignored when bare.
- `--tag T` appends `-T` to the output label, so a partial pass never overwrites a full run's `summary.json` under `runs/measure/` (`make bar` merges every `measure-*` dir by label).

### Task B8.3: run it (the orchestrator)

One image from the branch head. Jobs:

- `make measure NAME=b8-pos-pp RUN=ladder-personaplex-1 ARGS='--epochs 2 --tag pos'`
- `make measure NAME=b8-pos-bare ARGS='--tag pos'` (control)

Each takes about 24 min, with drama3 stopped. Check that the epoch-2 content and all bars equal ladder-1's epoch-2 numbers (same code path, same checkpoint), and that `bar_from` reproduces them.

### Task B8.4: `docs/results/bar-by-position-1.md`

The bucket tables (content and all frames) and the bar-from-position table, for epoch 2 and bare. Then one paragraph answering "does the gap sit in the opening frames?", and what follows for the bar: which frames a fair bar scores, and whether the TTS onset should be aligned to the recorded pupil onset instead. Finish with a GATE.

### B8 as run (2026-11-05)

- **Code:** B8.1 (`dual.position`, `make position`) and B8.2 (measure writes `content_mask` and `per_frame`, and takes `--epochs` and `--tag`) as specified. Review found that `dual.report.read` refused the tagged passes as duplicate (run, base, epoch) summaries. It now skips a `<base>[-<run>]-<tag>` dir.
- **Jobs:**
  - Image `b8pos:d80113b`.
  - `measure-b8-pos-pp` (epoch 2, 1,456 s) and `measure-b8-pos-bare` (231 s), with drama3 stopped throughout.
  - Epoch 2 reproduced ladder 1's numbers to bf16 drift.
- **Result** (`docs/results/bar-by-position-1.md`):
  - The all-frame ratio of ~18 is the 3 lead PAD frames before the first word: about 7 nats per frame of typed‖A, 81% of the all-frame gap mass. Without them the all-frame ratio is 3.2.
  - On content frames the opening is the closest to the floor: ratio 3.7, 2.5, 1.3, 2.0 and 1.8 at positions 0 to 10, and 1.30 frame-weighted. The ~3 sits in the body of the answer: 3.4 at 26+.
  - The floor is front-loaded, so cutting the opening makes p90 worse. No cut passes.
  - From position 26 on, typed‖tape (0.0174) equals typed‖A (0.0176), while tape‖A (0.0069) is near the floor (0.0051). The gap is typed against any speech.
  - Trimming the TTS trailing silence is predicted not to lower the gap. Its effect on the floor is unmeasured.
  - The opus review corrected the first reading ("flat", "onset alignment").
- **Reviews:**
  - Sonnet reviewed the measure diff and then the batch.
  - Opus reviewed before the merge and recomputed every number from the jsonl.
  - Follow-ups:
    - `dual.position` refuses empty input.
    - The measure job's epoch check raises instead of asserting.
    - A test pins that the lead is in the no-cut all-frame bar and out from position 0.

## Batch B9: consistency loss (2026-11-05)

Vlad, at the bar-by-position gate: "Add a consistency loss (Recommended). Keep the content bar. Plan a distillation/consistency loss: typed rendering trained toward the frozen model's distribution after the spoken rendering. Needs a /plan-it style plan first, then training."

Plan: `docs/plans/2026-11-05-consistency-loss.md`. The teacher is the live student with stop-gradient on the stem's all-tape spoken row. The term is λ = 10 · KL(typed answer ‖ spoken answer) on the answer content frames after each typed turn. The pilot is the ladder recipe otherwise unchanged.

Result (`docs/results/consistency-1.md`): the first change to move the ratio. R_gap went from 3.00 to 1.90 ("down", both CIs exclude 0), and R is 1.67 at epoch 2. The gap, typed‖A, reached tape‖A, the tape ceiling. The bar still does not pass. Every KL shrank, so a side-by-side check comes first. GATE-1 is pending.

## Batch B10: side by side 2 (2026-11-06)

Vlad, at GATE-1: "do step by step". Step one of option (b): B7's side-by-side on the consistency pilot (`cons1-pp-l10`) epoch 2, plus a judged pupil-turn swap. Does the tutor still respond to what the pupil said, or did the term make it deaf to the pupil turn? No training, no render. The spec is in `docs/plans/2026-11-05-consistency-loss.md`, Batch B10.

Result (`docs/results/side-by-side-2.md`): mid-lesson the tutor still listens. Uptake on turns 2-8 is 50/50/49 of 52, and a swapped pupil turn is followed in 46/52. But after the lesson's spoken first question the pilot is silent in 23/26, so epoch 2 is not healthy as a checkpoint. A GATE is pending.

## Batch B11: side by side 3 (2026-11-06)

Vlad, at B10's GATE: "Check epoch 1 first". This batch is B10's side-by-side on the pilot's epoch 1. Result (`docs/results/side-by-side-3.md`): epoch 1 is not silent at the opening, but it answers in fragments (mid-sentence stops 20/11/17 of 52). So no λ = 10 epoch is healthy.

## Batch B12: λ = 3 pilot (2026-11-06)

Vlad, at B11's GATE: "Weight 3 + baseline epoch 1 decode (Recommended)". The run is `cons1-pp-l3`, identical to the λ = 10 pilot except the weight. The spec is in `docs/plans/2026-11-05-consistency-loss.md`, Batch B12.

Result (`docs/results/consistency-2.md`): no λ = 3 epoch is both healthy and better on the bar. The rule says "not moved" (R_gap 2.51).
- Epoch 3 restores the opening at R 1.87. But it affirms what the pupil did not say more often: 94/104 against 101/104, p ≈ 0.09, with no epoch-3 λ = 0 control.
- Epoch 2's typed opening starts late.
- The baseline's own epoch 1 is silent at the spoken opening and fragmented, so those failures also occur in plain early training.

GATE pending; recommended first: the baseline's epoch 3 and λ3's epoch 1 as controls (~1.6 GPU-hours).

## Batch B15: voice-D teacher rows (2026-11-08)

Vlad, at B14's GATE (B13 and B14 are recorded in `docs/plans/2026-11-05-consistency-loss.md`): "Plan voice-D teacher rows (Recommended)". The spec is `docs/plans/2026-11-08-voice-d-teacher.md`: a TTS voice D voices each typed row's typed turns and replaces the tape as the λ = 3 consistency teacher (run `voiced1-pp-l3`, render `render-2`).

Result (`docs/results/consistency-5.md`): **NO BETTER** (the pre-registered D9 verdict); R failed, behaviour passed. R 2.28 / 1.87 / 1.53 (λ3-tape 2.51 / 1.95 / 1.87); the e3 CI against λ3-tape is −0.34 [−0.66, +0.03]. The ceiling did move (D‖A 0.75 × floor), but typed‖D stayed as large as the gap, and the drop in R is mostly a higher floor. Epoch 3 is healthy (swap 91, at the limit); the rule's epoch 1 is silent at the spoken opening. GATE-2 pending.
