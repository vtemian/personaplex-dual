# Voice-D Teacher Rows for the Consistency Loss: Implementation Plan (B15)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan batch by batch.
> In `docs/plans/2026-10-26-dual-input-batches.md` this plan is batch **B15**. Its batches are
> V1 to V6 below. It continues `docs/plans/2026-11-05-consistency-loss.md` (B9 to B14).

**Goal:** a new TTS voice D voices the typed pupil turns of the 117 training dialogues, and those
rows replace the tape as the consistency teacher at λ = 3. One seed-0 run from scratch, 3 epochs,
then the bar, the behaviour checks and a GATE. This is the only lever left on the ~1.8 tape ceiling.

**Architecture:**
- A render job extends render-1 without re-rendering it:
  - `--from render-1` copies render-1's rows, stems and holdout byte for byte;
  - it voices only the added labels, using the same TTS, voice embeddings, seeding and splice as render-1;
  - it writes `rows-teacher.jsonl`: per train stem, one voiced row for each typed row (`mask`, `complement`);
  - in each voiced row, exactly that row's typed turns are in voice D and every other pupil turn is the tape.
- The train job takes `--teacher voiced`. It pairs each typed row with its voiced row by (stem, kind)
  instead of with the stem's all-tape row. Batches, the LM loss and the alignment are unchanged.
- The measure job reads the render's `stems.json` voices. Any label past A/B/C adds pairs
  KL(typed‖L), KL(L‖A) and KL(L‖B), so typed‖D and the new ceiling D‖A come out of the same pass
  as the bar.

**Tech stack:** Python 3.12, PyTorch (bf16 autocast, fp32 logits), moshi 0.2.13 and the kyutai DSM TTS
(on the box only, via `vendor/harness/engine/audio.py`), pytest (`-n auto`), ruff, mypy strict, uv,
sparks queue, POSIX sh.

---

## 0. Read first (the executor has only this file, memory and the repo)

- `CLAUDE.md`:
  - `make check` before every commit; work on a branch, never on main.
  - Pure modules (`src/dual/*.py`) never import torch at module level, except `kl.py`.
  - GPU code is `src/dual/model.py` and `src/dual/jobs/*.py`, and it imports `moshi`/`harness.engine`
    inside functions.
  - Never edit or commit `vendor/`; never commit `data/` or `.env`.
- Memory: `/Users/whitemonk/.claude/projects/-Users-whitemonk-projects-ai-personaplex-dual/memory/MEMORY.md`,
  then `b14-gate-decision.md`, `l3-seed1-result.md`, `b14-l3-s1-run.md`, `controls-result.md`,
  `render-1-run.md`, `slim-image-weights-layout.md`, `loader-facts.md` (off-LAN registry push),
  `box-outages.md`, `drama3-memory-hog.md`.
- Results this builds on: `docs/results/consistency-4.md` (B14, the comparison base),
  `consistency-2.md` (B12, `cons1-pp-l3`), `bar-by-position-1.md` (B8, the tape ceiling).
- Skills for the GPU part (V5): `authoring-a-sparks-job`, `operating-the-sparks-queue`.
- Branch for all code batches: `feat/b15-voice-d` off `main` (da91925).

### 0.1 Facts this plan rests on (measured or read in code; do not re-derive)

| fact | value | source |
|---|---|---|
| train rows | 351 = 117 stems × (`mask`, `complement`, `spoken`); `spoken` is the all-tape row | `src/dual/render.py:75-98` |
| typed turns | `mask_bits(stem, n_pupils)`: bit 1 → typed in `mask`, 0 → typed in `complement` | `render.py:81-92` |
| TTS | `harness.engine.audio.load_tts("cuda")` (moshi's `DEFAULT_DSM_TTS_REPO`, `kyutai/tts-1.6b-en_fr`), `voice_many(tts, [text], [path])`, cfg 2.0 | `src/dual/jobs/render.py:89-101`, `vendor/harness/engine/audio.py:307-349` |
| a voice | a speaker-embedding file `data/voices/<name>.1e68beda@240.safetensors`; seed = `stable_seed(words, voice)`, sha256 of `"<words>|<voice>"` | `jobs/render.py:18-24, 92-98` |
| bar voices | A = `vctk/p225_023.wav`, B = `vctk/p248_023.wav` (floor pair), C = `vctk/p302_023.wav` (door) | `jobs/render.py:17` |
| other voices on disk | vctk p226, p237, p245, p251, p280, p294, p311, p326, p376 (`_023.wav`); 4 expresso | `data/voices/`, `scripts/data.sh:17` |
| splice | a voiced turn = TTS Mimi codes + `GAP_FRAMES` = 6 frames of `MIMI_SILENCE` on the pupil stream, text PAD, own SILENCE; later spans shift; no padding to the tape length | `frames.py:52-61, 95-108` |
| row limit | `CONTEXT` = 3,000 frames; `assemble` raises past it; train max row 2,765, holdout max 2,284 | `constants.py`, `render.py:70`, `render-1-run.md` |
| alignment | `answer_pairs(typed_row, teacher_row)`: answer = span k+1 in each row, each row's own span start, offset i ↔ i; refuses differing answer text, differing span counts | `src/dual/consistency.py:30-68` |
| the "3 lead frames" | the 3 PAD frames that open every tutor answer, inside the answer span. They are not TTS padding, and content frames exclude them. TTS trailing silence sits in the pupil span. | `bar-by-position-1.md`, `consistency.py:65` |
| teacher pass | `build_teachers` (`jobs/train.py:138-167`): no_grad, one forward per distinct teacher row, keeps only content-frame log-probs | B9 |
| cost with teacher | ~4,300 s/epoch (3 × ~4,305 in B14), peak **56.0 GiB** under `--mem-cap-gib 64` (default 64); teacher ~33% of a step | B12, B14 |
| measure | `RENDERINGS = (typed, A, B, C, tape)`; one forward per rendering per turn; 4,551 s for 3 epochs | `jobs/measure.py:32-40`, B14 |
| R | median per-turn content gap / median per-turn content floor; the bar passes iff median and p90 gap ≤ floor | `noise.py:41-52`, `report.py:31-32` |
| B14 base, seed 0 | `cons1-pp-l3` (λ3, tape teacher): R 2.51 / 1.95 / **1.87**, gap .0094/.0106/.0146, floor .0037/.0054/.0078; λ0 `ladder-personaplex-1` 3.04/3.24/2.87 | `consistency-4.md` |
| ceiling | tape‖A ≈ 1.7-1.8 × floor; matching the tape cannot take R below ~1.8 | B8, B12 |
| λ3 e3 behaviour, seed 0 (B14 pool) | speaks 13/13 A and B, typed onset frame 3 13/13, stops 0/1/0, swap 93/104 (λ0 s0 e3 94) | `consistency-4.md` |
| render time | render-1 voiced 195 TTS segments in ~8 min (~2.5 s each, unbatched) | `side-by-side-2.md:219` |
| side-by-side | image `sbs2:72868db`, `--no-bare --swap-turns 2,6 --mem-cap-gib 64`, ~2,900 s a checkpoint | B14 |
| run dirs | a job reads and writes `runs/<name>/...` relative to its cwd = `/srv/spark/runs/` on the box | `jobs/train.py:287`, `jobs/render.py:59` |

### 0.2 Decisions (settled here; the executor does not reopen them)

**D1. The teacher row is a voice-D rendering of exactly the typed row's typed turns. It replaces the
tape teacher; the two are not mixed.**
- For each typed row (`mask` or `complement`) there is one voiced row of the same stem. In it, every
  turn that the typed row types is spoken by voice D, and every other pupil turn is the tape.
- The pair (typed row, voiced row) therefore differs only in the modality of the typed turns, as the
  B9 pair (typed row, all-tape row) does. The bar's pair (typed vs A, one swapped turn, the rest tape)
  has the same shape.
- A row with every pupil turn in D was rejected. It would also change the history at the untyped turns,
  which is a mismatch the bar never measures.
- Keeping both teachers was rejected, both as an average and alternating. The tape is what caps R at
  ~1.8, so averaging it in keeps half the ceiling. Two teacher forwards would also double the 33% teacher
  share (over B9's 40% stop rule). Alternating halves the D signal for no gain the GATE needs.

**D2. The voiced rows are teacher-only. They do not enter the LM loss.**
- `rows-train.jsonl` is copied byte for byte, so `batches()` and `--seed 0` give exactly
  `cons1-pp-l3`'s steps, init and batch order.
- The new run is then paired with B12's λ3 run by construction: the only change is the teacher row.
- Adding voiced rows to the LM loss changes batching, voids the pairing, and adds a 4th row per stem
  (+33% LM time). It is a GATE option, not this run.

**D3. Voice D is chosen by a pre-registered screen (V5.2, GATE-0) among three VCTK speakers not used
for A, B or C:**
- D1 = `vctk/p226_023.wav`, D2 = `vctk/p280_023.wav`, D3 = `vctk/p294_023.wav`. They are VCTK like A, B
  and C, so in-distribution for the TTS prompt style, and different speakers by construction.
- Genders and accents are not asserted here; the numbers decide.
- The screen measures `cons1-pp-l3` epoch 3 on the holdout with D1-D3 added. It reads, per candidate L:
  - `r_L` = median KL(L‖A) / median floor;
  - `r_L_b` = median KL(L‖B) / median floor;
  - `s_L` = median KL(typed‖L) / median KL(typed‖A).
- It also reads the reference ratios on the same pass: tape‖A / floor (the old ceiling) and C‖A / floor
  (the door, the best prior for a VCTK voice).
- **Admissible:** `max(r_L, r_L_b) ≤ 1.3` and `0.7 ≤ s_L ≤ 1.3`. The second condition means the model
  treats L as speech, as it treats A.
- **Pick** the admissible candidate with the smallest `r_L`. On a tie within 0.05, take the smaller `r_L_b`.
- **Lever check:** `r_D ≤ tape ratio − 0.3`. Otherwise voice D cannot lower the ceiling by a margin the
  noise rule (T = 0.5) could see in R.
- The chosen voice is rendered under the label **D** in render-2. Its audio is identical to the screen's,
  because the seed and cache keys use the voice name, not the label.

**D4. Alignment is unchanged and checked twice.**
- `answer_pairs` already anchors each row's answer at its own span start. A D pupil turn of a different
  length, with its 6 gap frames and any TTS trailing silence, only moves where the answer starts. The
  answer span is the tape's tutor segment in both rows (same text, own and pupil audio, length), so the
  lead PAD frames and the content offsets are identical.
- Both requirements are already enforced: same text, and `party == "teacher"` on both sides.
- The new check is on the teacher side's pupil span. On a voiced teacher it must carry the voice label
  and voice exactly the typed row's typed turns.
- **Check 1 (render-2 job):** for every train stem, the number of answer pairs against the voiced rows
  equals the number against the tape row. The rebuilt tape row equals render-1's `spoken` row in every
  frame field (`check_same_tape`).
- **Check 2 (the train smoke):** the startup line's `answer pairs N, content frames M` equals
  `cons1-pp-l3`'s own startup line, because the content frames are the tutor's and do not depend on the
  teacher.

**D5. Memory and time: the budget is unchanged; the smoke confirms it.**
- A voiced row has the same shape as the tape row. Only the typed turns' pupil spans change length.
  The teacher forward is one row ≤ 3,000 frames, and its transient logits are ≤ 0.4 GiB text + 0.2 GiB
  audio at 3,000 frames.
- One teacher forward per distinct teacher row: `mask` and `complement` of one stem have different
  lengths, so they rarely share a length-packed batch.
- Expect ~4,300 s/epoch and a peak of ~56 GiB.
- GATE-1 checks the stem with the longest voiced row, the widest batches, and the teacher share.

**D6. The recipe is B12's, with only the teacher changed:**
- `--consistency 3 --teacher voiced`, seed 0, from scratch, 3 epochs, 117 stems (`--slice 1`);
- rank 16, LR 2e-5, compiled, per-epoch checkpoints, `--mem-cap-gib 64` (default).
- Run name **`voiced1-pp-l3`**, render **`render-2`**, ladder **`LADDER=6`**. `LADDER=5` was B14's
  scratchpad-only copy, so 6 avoids confusion.

**D7. The holdout gets voice D too. The bar's A/B/C/typed/tape renderings are copied, never re-voiced.**
- The measure on render-2 therefore reproduces render-1's gap and floor exactly. It adds `gap_D`
  (typed‖D, the term the run minimises) and `floor_D` (D‖A, the new ceiling).

**D8. Eval is mandatory and complete before any claim:**
- the bar: `make bar`; `make noise` (rule verdict, R0 3.00, T 0.5); R per epoch;
- the paired same-epoch stem-bootstrap CI against `cons1-pp-l3` and against `ladder-personaplex-1`;
- `make position`;
- held text loss, mean `cons` per epoch, turn-0 lead KL;
- typed‖D and D‖A per epoch;
- the behaviour checks next to every bar number: side-by-side decodes of epoch 3 and of the rule's
  epoch if it is not 3; opening silence, onset frame, stops; swap-follow of 104 and substance from one
  fresh blind opus pool that contains B14's eleven checkpoints plus the new ones.

**D9. Pre-registered verdicts (this file is committed before any voice-D number exists):**
- **PASS:** at epoch 3, or at the rule's epoch if it is decoded, `content.passes` is true (median and p90
  gap ≤ floor, i.e. R ≤ 1 at median and p90), **and** that epoch is healthy against `ladder-personaplex-1`
  e3 in the new pool. Healthy means all of:
  - speaks after the spoken opening in ≥ 12/13 for A and for B;
  - typed-opening onset by frame 4 in ≥ 11/13;
  - mid-lesson stops ≤ 3 of 52 per rendering;
  - swap-follow ≥ (λ0 s0 e3's count in the new pool) − 5.
- **LOWERS THE CEILING:** not PASS, but R at epoch 3 ≤ 1.37 (1.87 − T) with the paired e3 CI against
  `cons1-pp-l3` excluding 0, and epoch 3 healthy.
- **NO BETTER:** otherwise. Say which part failed: R, behaviour, or both.

**D10. Cost (GPU-hours):**

| step | estimate |
|---|---|
| screen render (195 segments) | 0.25 h |
| screen measure (1 epoch, 8 renderings) | 0.7 h |
| render-2 smoke | 0.15 h |
| render-2 (~585 train + 65 holdout segments) | 0.6 h |
| two train smokes | 0.5 h |
| train | 3 × ~4,300 s = 3.6 h |
| measure (3 epochs, 6 renderings) | ~1.5 h |
| side-by-side, 1-2 checkpoints | 0.8-1.6 h |
| **total** | **≈ 8.1-8.9 GPU-hours** |

### 0.3 Batches

| batch | model | files | depends on | gate |
|---|---|---|---|---|
| V1 pure parts | sonnet | `src/dual/render.py`, `tests/test_render.py`, `src/dual/consistency.py`, `tests/test_consistency.py` | none | targeted tests per task |
| V2 render job | sonnet | `src/dual/jobs/render.py`, `tests/test_render_job.py` | V1 on the branch | targeted tests |
| V3 train wiring | sonnet | `src/dual/jobs/train.py`, `tests/test_train.py`, `tests/test_train_consistency.py` | V1 on the branch | targeted tests |
| V4 measure + ladder | sonnet | `src/dual/jobs/measure.py`, `tests/test_measure.py`, `scripts/ladder.sh`, `tests/test_ladder_sh.py` | none (parallel with V1) | targeted tests |
| V5 run it | opus (orchestrator) | memory notes only, no repo code | V1-V4 merged to `main` | **GATE-0** (voice), **GATE-1** (render-2 + smoke) |
| V6 read, judge, record | opus | `docs/results/consistency-5.md`, `docs/results/consistency-5/`, this plan's "as run", `docs/plans/2026-10-26-dual-input-batches.md`, memory | V5 done | **GATE-2** (Vlad) |

- V1 and V4 run in parallel. V2 and V3 run in parallel after V1 lands on `feat/b15-voice-d`.
- Tasks carry no commit step and no full-gate step. Each names one targeted command, and each batch ends
  with one commit of its paths.
- `/ship-it` runs the project's full quality gate (`make check`: ruff format and check, mypy strict,
  pytest) on the branch right before it merges to `main`.
- V5 runs from `main` after the merge; push `main` after each merge.

---

## Batch V1: the pure parts (model: sonnet)

Files: `src/dual/render.py`, `tests/test_render.py`, `src/dual/consistency.py`, `tests/test_consistency.py`.
No torch anywhere in this batch.

### Task V1.1: `voiced_rows` in `src/dual/render.py` (TDD)

**Step 1: failing tests.** Append to `tests/test_render.py`. Add `voiced_rows` to the
`from dual.render import (...)` list, and add `spoken_segment` to the `from dual.frames import` line.

```python
def test_voiced_rows_voice_exactly_each_typed_rows_typed_turns():
    rows = training_rows("stem", two_ask_spans(), base(), typed)
    vrows = voiced_rows("stem", two_ask_spans(), base(), voiced, "D", "vdddd")
    assert [v["pairs_with"] for v in vrows] == ["mask", "complement"]
    assert [v["id"] for v in vrows] == ["stem#mask-D", "stem#complement-D"]
    for v, r in zip(vrows, rows[:2], strict=True):
        assert v["kind"] == "voiced" and v["stem"] == "stem" and v["voice"] == "D"
        assert v["typed_turns"] == []
        assert v["voiced_turns"] == r["typed_turns"]
        for k, s in enumerate(v["spans"]):
            if k in v["voiced_turns"]:
                assert s["rendering"] == "D"
            else:
                assert s["rendering"] == "tape"


def test_voiced_turns_have_the_tts_length_and_answers_keep_the_tape():
    spoken = training_rows("stem", two_ask_spans(), base(), typed)[2]
    for v in voiced_rows("stem", two_ask_spans(), base(), voiced, "D", "vdddd"):
        for k, (s, t) in enumerate(zip(v["spans"], spoken["spans"], strict=True)):
            if k in v["voiced_turns"]:
                words = two_ask_spans()[k].words
                assert s["end"] - s["start"] == 2 * len(words) + GAP_FRAMES
                assert list(v["pupil"][s["start"]]) == [5] * 8  # len("vdddd")
            if s["party"] == "teacher":
                assert v["text"][s["start"] : s["end"]] == spoken["text"][t["start"] : t["end"]]
                assert v["own"][s["start"] : s["end"]] == spoken["own"][t["start"] : t["end"]]


def test_voiced_rows_refuse_past_the_context():
    def long_voiced(words, voice):
        return [[1] * 8] * 4000

    with pytest.raises(ValueError, match="over moshi"):
        voiced_rows("stem", two_ask_spans(), base(), long_voiced, "D", "v")
```

Note: `mask_bits("stem", 2)` decides which kind voices which turn. A kind with no typed turns still gets
a voiced row (identical to the tape row, `voiced_turns == []`), so the test loops over both.

**Step 2:** `uv run pytest tests/test_render.py -q -k voiced` → FAIL (`ImportError: voiced_rows`).

**Step 3: implement.** In `src/dual/render.py`, after `training_rows`:

```python
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
```

**Step 4:** `uv run pytest tests/test_render.py -q` → all pass.

### Task V1.2: `add_rendering` and `check_same_tape` in `src/dual/render.py` (TDD)

**Step 1: failing tests.** Append to `tests/test_render.py` and import `add_rendering` and `check_same_tape`:

```python
def test_add_rendering_adds_a_label_to_every_turn_and_keeps_the_rest():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    out = add_rendering(rec, two_ask_spans(), voiced, {"D": "vdddd"})
    assert out["base"] == rec["base"] and out["pupil_turns"] == rec["pupil_turns"]
    for k in ("0", "2"):
        assert set(out["renderings"][k]) == {"typed", "A", "B", "C", "D"}
        for lab in ("typed", "A", "B", "C"):
            assert out["renderings"][k][lab] == rec["renderings"][k][lab]
        assert out["renderings"][k]["D"]["pupil"][0] == [5] * 8
    assert set(rec["renderings"]["0"]) == {"typed", "A", "B", "C"}  # input untouched
    row, _ = holdout_row(out, 2, "D")
    assert row["spans"][2]["rendering"] == "D"


@pytest.mark.parametrize("label", ["A", "typed", "tape"])
def test_add_rendering_refuses_a_label_that_exists(label):
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    with pytest.raises(ValueError, match="already"):
        add_rendering(rec, two_ask_spans(), voiced, {label: "vx"})


def test_add_rendering_refuses_other_spans():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    with pytest.raises(ValueError, match="pupil turns"):
        add_rendering(rec, two_ask_spans()[:2], voiced, {"D": "vx"})


def test_check_same_tape_accepts_the_round_trip_and_refuses_a_change():
    import json

    rebuilt = training_rows("stem", two_ask_spans(), base(), typed)[2]
    stored = json.loads(json.dumps(rebuilt))
    check_same_tape(rebuilt, stored)
    stored["pupil"][0] = [0] * 8
    with pytest.raises(ValueError, match="pupil"):
        check_same_tape(rebuilt, stored)
```

**Step 2:** `uv run pytest tests/test_render.py -q -k "add_rendering or same_tape"` → FAIL (ImportError).

**Step 3: implement.** Add `import json` and `from collections.abc import Callable, Mapping, Sequence`
to `src/dual/render.py`, then after `holdout_record`:

```python
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
```

**Step 4:** `uv run pytest tests/test_render.py -q` → all pass.

### Task V1.3: voiced teachers in `src/dual/consistency.py` (TDD)

**Step 1: failing tests.** In `tests/test_consistency.py`:
- Add `spoken_segment` to the `dual.frames` import.
- Add `tape_teachers` and `voiced_teachers` to the `dual.consistency` import.
- Factor the start of `make_rows` into a helper and add `make_voiced`, as below. The existing tests keep
  calling `make_rows` unchanged.

```python
VOICED = spoken_segment([[5] * 8] * 4)  # 4 TTS frames + 6 gap = 10 frames


def tape_parts(parts=PARTS):
    spans, base, start = [], [], 0
    for i, (party, text) in enumerate(parts):
        n = len(text)
        spans.append(TurnSpan(i, party, (f"w{i}",), start, start, start + n))
        base.append(tape_segment(text, OWN * n, PUP * n, scored=party == "teacher"))
        start += n
    return spans, base


def make_rows(typed_turns, parts=PARTS, stem="s"):
    spans, base = tape_parts(parts)
    swaps = {k: ("typed", TYPED) for k in typed_turns}
    typed = to_record(assemble(f"{stem}#mask", spans, base, swaps))
    typed |= {"kind": "mask", "stem": stem, "typed_turns": list(typed_turns)}
    spoken = to_record(assemble(f"{stem}#spoken", spans, base, {}))
    spoken |= {"kind": "spoken", "stem": stem, "typed_turns": []}
    return typed, spoken


def make_voiced(voiced_turns, parts=PARTS, stem="s", kind="mask", label="D", seg=VOICED):
    spans, base = tape_parts(parts)
    swaps = {k: (label, seg) for k in voiced_turns}
    row = to_record(assemble(f"{stem}#{kind}-{label}", spans, base, swaps))
    row |= {
        "kind": "voiced",
        "stem": stem,
        "typed_turns": [],
        "pairs_with": kind,
        "voice": label,
        "voiced_turns": list(voiced_turns),
    }
    return row
```

New tests (append):

```python
def test_a_voiced_teacher_pairs_by_offset_after_a_longer_pupil_turn():
    typed, _ = make_rows([1])
    voiced = make_voiced([1])  # turn 1: 10 frames (tape 5, typed 9)
    assert answer_pairs(typed, voiced) == [
        AnswerPair(turn=1, typed=(15, 23), spoken=(16, 24), content=(3, 4, 5, 6, 7))
    ]


def test_a_voiced_teacher_pairs_like_the_tape_teacher_frame_for_frame():
    typed, spoken = make_rows([1, 3])
    voiced = make_voiced([1, 3])
    by_tape, by_voice = answer_pairs(typed, spoken), answer_pairs(typed, voiced)
    assert [(p.turn, p.content) for p in by_tape] == [
        (p.turn, p.content) for p in by_voice
    ]
    s_idx, t_idx = flat_indices(by_voice)
    assert [typed["text"][i] for i in s_idx] == [voiced["text"][i] for i in t_idx]


def test_refuses_a_voiced_teacher_of_other_turns():
    typed, _ = make_rows([1])
    with pytest.raises(ValueError, match="voices"):
        answer_pairs(typed, make_voiced([3]))


def test_refuses_a_voiced_teacher_whose_turn_is_the_tape():
    typed, _ = make_rows([1])
    bad = make_voiced([1]) | {"voice": "E"}  # spans say D
    with pytest.raises(ValueError, match="not E"):
        answer_pairs(typed, bad)


def test_tape_teachers_map_each_typed_row_to_its_stems_spoken_row():
    s_typed, s_spoken = make_rows([1])
    t_typed, t_spoken = make_rows([1, 3], stem="t")
    out = tape_teachers([s_typed, s_spoken, t_typed, t_spoken])
    assert out == {"s#mask": s_spoken, "t#mask": t_spoken}


def test_voiced_teachers_map_each_typed_row_to_its_kinds_voiced_row():
    s_typed, s_spoken = make_rows([1])
    v = make_voiced([1])
    other = make_voiced([3], kind="complement")
    assert voiced_teachers([s_typed, s_spoken], [v, other]) == {"s#mask": v}


def test_voiced_teachers_refuse_a_missing_row_two_rows_and_two_voices():
    s_typed, s_spoken = make_rows([1])
    with pytest.raises(ValueError, match="no voiced row"):
        voiced_teachers([s_typed, s_spoken], [make_voiced([3], kind="complement")])
    with pytest.raises(ValueError, match="two voiced rows"):
        voiced_teachers([s_typed], [make_voiced([1]), make_voiced([1])])
    with pytest.raises(ValueError, match="one teacher voice"):
        voiced_teachers(
            [s_typed],
            [make_voiced([1]), make_voiced([3], kind="complement", label="E")],
        )
    with pytest.raises(ValueError, match="not a voiced row"):
        voiced_teachers([s_typed], [s_spoken])
    with pytest.raises(ValueError, match="voices"):
        voiced_teachers([s_typed], [make_voiced([3])])
```

**Step 2:** `uv run pytest tests/test_consistency.py -q` → the new tests FAIL. The existing tests still pass.

**Step 3: implement** in `src/dual/consistency.py`:
1. Update the module docstring. Append: "B15: the teacher may instead be a `voiced` row, the same stem with
   exactly the typed row's typed turns in a TTS voice (`voice` label) and the rest the tape; it pairs with
   the typed row of its `pairs_with` kind."
2. Add `TEACHER_KINDS = ("spoken", "voiced")` under `Row = ...`.
3. Replace the head of `answer_pairs` up to the loop, and the tape check inside the loop:

```python
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
    ...  # unchanged down to the rendering check, which becomes:
        if s_spans[k]["rendering"] != expect:
            raise ValueError(
                f"{spoken_row['id']}: turn {k} is {s_spans[k]['rendering']}, "
                f"not {expect}"
            )
```

The existing test `test_refuses_a_teacher_row_that_is_not_the_spoken_kind` matches `"spoken"`, and the
new message keeps that word.

4. Add after `spoken_by_stem`:

```python
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
        t = by_key.get((str(r["stem"]), str(r["kind"])))
        if t is None:
            lacking.append(str(r["id"]))
            continue
        if sorted(t["voiced_turns"]) != sorted(r["typed_turns"]):
            raise ValueError(
                f"{t['id']} voices {t['voiced_turns']}, {r['id']} types "
                f"{r['typed_turns']}"
            )
        out[str(r["id"])] = t
    if lacking:
        raise ValueError(f"no voiced row for {lacking[:3]}")
    return out
```

**Step 4:** `uv run pytest tests/test_consistency.py tests/test_train_consistency.py -q` → all pass.
`test_train_consistency.py` imports `make_rows`, so this confirms the refactor kept it.

### Task V1.4: batch close

`uv run ruff format src tests && uv run ruff check src tests && uv run mypy`, then one commit:

```bash
git add src/dual/render.py tests/test_render.py src/dual/consistency.py tests/test_consistency.py
git commit -m "feat(b15): voiced teacher rows, add_rendering, check_same_tape; answer_pairs takes a voiced teacher"
```

---

## Batch V2: the render job extends a render (model: sonnet)

Files: `src/dual/jobs/render.py`, `tests/test_render_job.py`. It needs V1.

### Task V2.1: `parse_voices` and the flags (TDD, CPU)

**Step 1: failing tests**, appended to `tests/test_render_job.py`. The two import lines go at the top
of the file with the existing imports (ruff E402). This holds for every test block in this plan.

```python
import pytest

from dual.jobs.render import VOICES, parse_args, parse_voices


def test_parse_voices_reads_labels_and_refuses_bar_voices_and_clashes() -> None:
    assert parse_voices(["D1=vctk/p226_023.wav", "D2=vctk/p280_023.wav"], VOICES) == {
        "D1": "vctk/p226_023.wav",
        "D2": "vctk/p280_023.wav",
    }
    for bad, why in [
        (["A=vctk/p226_023.wav"], "already"),
        (["D=vctk/p225_023.wav"], "already voices"),
        (["typed=x"], "label"),
        (["d=x"], "label"),
        (["D"], "LABEL=VOICE"),
        (["D=x", "D=y"], "twice"),
        (["D=x", "E=x"], "twice"),
    ]:
        with pytest.raises(ValueError, match=why):
            parse_voices(bad, VOICES)


def test_flags_default_to_render_1s_behaviour(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["render", "--name", "render-x"])
    a = parse_args()
    assert (a.source, a.add_voice, a.teacher_voice, a.limit) == (None, [], None, 0)


def test_teacher_voice_must_be_an_added_label_and_needs_from(monkeypatch) -> None:
    base = ["render", "--name", "r"]
    for extra in (
        ["--add-voice", "D=x"],  # no --from
        ["--from", "render-1", "--teacher-voice", "D"],  # D not added
    ):
        monkeypatch.setattr("sys.argv", [*base, *extra])
        with pytest.raises(SystemExit):
            parse_args()
    monkeypatch.setattr(
        "sys.argv",
        [*base, "--from", "render-1", "--add-voice", "D=x", "--teacher-voice", "D"],
    )
    a = parse_args()
    assert (a.source, a.add_voice, a.teacher_voice) == ("render-1", ["D=x"], "D")
```

**Step 2:** `uv run pytest tests/test_render_job.py -q` → FAIL (ImportError `parse_voices`).

**Step 3: implement** in `src/dual/jobs/render.py`. Add `import re` and
`from collections.abc import Mapping, Sequence`.

```python
LABEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")  # a measure pair key suffix: gap_<label>


def parse_voices(specs: Sequence[str], taken: Mapping[str, str]) -> dict[str, str]:
    """--add-voice LABEL=VOICE words -> {label: voice}. Refuses a label or a voice the
    render already has (A, B, C: the bar and the door are never a teacher), a
    lower-case or reserved label, and a label or voice given twice."""
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--add-voice {spec!r}: want LABEL=VOICE")
        label, voice = spec.split("=", 1)
        if not LABEL.match(label):
            raise ValueError(f"--add-voice {spec!r}: label must match {LABEL.pattern}")
        if label in taken:
            raise ValueError(f"--add-voice {spec!r}: label {label} already exists")
        if voice in taken.values():
            raise ValueError(f"--add-voice {spec!r}: {voice} already voices a label")
        if label in out or voice in out.values():
            raise ValueError(f"--add-voice {spec!r}: label or voice given twice")
        out[label] = voice
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=0, help="dialogues, for a smoke render")
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="an earlier render: copy its rows, stems and holdout; voice only the "
        "added labels (B15)",
    )
    p.add_argument(
        "--add-voice",
        action="append",
        default=[],
        metavar="LABEL=VOICE",
        help="a holdout rendering to add, e.g. D=vctk/p280_023.wav (with --from)",
    )
    p.add_argument(
        "--teacher-voice",
        default=None,
        help="an added label: also write rows-teacher.jsonl in that voice",
    )
    args = p.parse_args()
    if (args.add_voice or args.teacher_voice) and not args.source:
        p.error("--add-voice and --teacher-voice need --from")
    labels = [s.split("=", 1)[0] for s in args.add_voice]
    if args.teacher_voice and args.teacher_voice not in labels:
        p.error(f"--teacher-voice {args.teacher_voice}: not an --add-voice label")
    return args
```

**Step 4:** `uv run pytest tests/test_render_job.py -q` → all pass.

### Task V2.2: factor the loading out of `main` (no behaviour change)

**Files:** `src/dual/jobs/render.py`.

1. Add `import types` and `from collections.abc import Callable` (with `Mapping` and `Sequence`).
2. Move lines 61-72 of `main`, which read the tape records and the `.dsl` sources, into:

```python
def load_dialogues(
    data: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """The tape records by id, the dialogue sources by stem, and the stems with both."""
    records = {
        r["id"]: r
        for r in map(
            json.loads, (data / "tape" / "speech.jsonl").read_text().splitlines()
        )
    }
    sources = {
        p.stem: p.read_text() for p in sorted((data / "dialogues").glob("*.dsl"))
    }
    stems = sorted(set(records) & set(sources))
    if len(stems) != len(sources):
        raise SystemExit(f"{len(sources) - len(stems)} dialogues have no tape record")
    return records, sources, stems
```

3. Move lines 77-101 (the SPM, Mimi, TTS and the `pieces`/`typed`/`voiced` closures) into:

```python
def load_engines(data: pathlib.Path) -> types.SimpleNamespace:
    """The SPM, Mimi and the TTS on cuda; pieces, typed and the cached, seeded voiced
    exactly as render-1 ran them. GPU; imports torch and harness inside."""
    import torch
    from harness.engine.audio import encode_wav, load_tts, voice_many

    from dual.model import load_base

    base = load_base("moshiko")  # the SPM and Mimi; the weights are not used here
    spm = base.spm
    mimi = base.info.get_mimi(device="cuda")
    tts = load_tts("cuda")
    cache: dict[tuple[str, str], list[list[int]]] = {}

    def pieces(word: str) -> Sequence[str]:
        return list(spm.EncodeAsPieces(word))

    def typed(words: Sequence[str]) -> Sequence[int]:
        return typed_tokens(words, spm.EncodeAsIds)

    def voiced(words: Sequence[str], voice: str) -> list[list[int]]:
        key = (" ".join(words), voice)
        if key not in cache:
            seed = stable_seed(words, voice)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            text = " ".join(words)
            if not text.endswith((".", "?", "!")):
                text += "."
            path = str(data / "voices" / f"{voice}{VOICE_SUFFIX}")
            (v,) = voice_many(tts, [text], [path])
            cache[key] = encode_wav(mimi, v.pcm, "cuda")
        return cache[key]

    return types.SimpleNamespace(
        spm=spm, pieces=pieces, typed=typed, voiced=voiced, cache=cache
    )
```

4. In `main`:
   - Use `records, sources, stems = load_dialogues(data)` and `eng = load_engines(data)`.
   - Replace `pieces` → `eng.pieces`, `typed` → `eng.typed`, `voiced` → `eng.voiced`,
     `spm.PieceToId` → `eng.spm.PieceToId` and `voice_cache.clear()` → `eng.cache.clear()`.
   - Right after `out.mkdir(...)`, add `if args.source: return extend_render(args, data)`. Task V2.3
     defines `extend_render`, and `main` keeps its `-> None` return.
   - Keep the order of the TTS calls, the loading and the prints exactly as before: render-1 must stay
     reproducible.
   - Do not move `out.mkdir`: `extend_render` makes its own dir from `args.name`.

**Check:** `uv run pytest tests/test_render_job.py tests/test_render.py -q` passes, and `uv run mypy`
is clean.

### Task V2.3: `extend_render` (GPU path; its pure checks are V1's)

**Files:** `src/dual/jobs/render.py`. Add to the imports:
`from dual.consistency import answer_pairs`,
`from dual.frames import to_record` and
`from dual.render import add_rendering, assemble, check_same_tape, voiced_rows`.
Add these to the existing `dual.render` import line.

```python
def extend_render(args: argparse.Namespace, data: pathlib.Path) -> None:
    """--from (B15): copy an earlier render's rows-train.jsonl byte for byte and its
    stems and holdout; voice only the added labels. Every holdout record gains each
    added rendering; with --teacher-voice, rows-teacher.jsonl gets each train stem's
    two voiced rows (mask, complement). The tape is rebuilt from the data and must equal
    the stored rows, so the voiced rows sit on exactly the rows the train job reads."""
    runs = pathlib.Path.cwd() / "runs"
    src = runs / args.source / "render"
    out = runs / args.name / "render"
    if out.resolve() == src.resolve():
        raise SystemExit("--from names this render itself")
    stems = json.loads((src / "stems.json").read_text())
    sha = pathlib.Path("/app/vendor/SHA").read_text().strip()
    if stems["chalkboard_sha"] != sha:
        raise SystemExit(
            f"{args.source} was rendered at chalkboard {stems['chalkboard_sha']}, "
            f"this image has {sha}"
        )
    added = parse_voices(args.add_voice, stems["voices"])
    rows_text = (src / "rows-train.jsonl").read_text()
    by_stem: dict[str, dict[str, dict[str, Any]]] = {}
    for r in map(json.loads, rows_text.splitlines()):
        by_stem.setdefault(r["stem"], {})[r["kind"]] = r
    held_in = [
        json.loads(line) for line in (src / "holdout.jsonl").read_text().splitlines()
    ]
    records, sources, _ = load_dialogues(data)
    train, held = list(stems["train"]), list(stems["held"])
    if args.limit:
        held, train = held[: max(1, args.limit // 10)], train[: args.limit]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows-train.jsonl").write_text(rows_text)  # the same rows, the same batches
    (out / "stems.json").write_text(
        json.dumps(
            {
                **stems,
                "voices": {**stems["voices"], **added},
                "from": args.source,
                "teacher_voice": args.teacher_voice,
            },
            indent=1,
        )
    )
    eng = load_engines(data)

    def tape_of(stem: str) -> tuple[Any, Any]:
        rec, source = records[stem], sources[stem]
        sp = spans(source, rec)
        text = text_row(source, rec, eng.pieces, eng.spm.PieceToId)
        return sp, tape_segments(sp, text, rec)

    if args.teacher_voice:
        label = args.teacher_voice
        n_rows = pairs_tape = pairs_voiced = pupil_tape = pupil_voiced = 0
        longest = ("", 0)
        with (out / "rows-teacher.jsonl").open("w") as f:
            for stem in train:
                sp, segs = tape_of(stem)
                kinds = by_stem[stem]
                spoken = kinds["spoken"]
                check_same_tape(
                    to_record(assemble(f"{stem}#spoken", sp, segs, {})), spoken
                )
                for row in voiced_rows(stem, sp, segs, eng.voiced, label, added[label]):
                    typed_row = kinds[row["pairs_with"]]
                    if sorted(row["voiced_turns"]) != sorted(typed_row["typed_turns"]):
                        raise SystemExit(f"{row['id']}: voices other turns than typed")
                    pairs_tape += len(answer_pairs(typed_row, spoken))
                    pairs_voiced += len(answer_pairs(typed_row, row))
                    for k in row["voiced_turns"]:
                        v, t = row["spans"][k], spoken["spans"][k]
                        pupil_voiced += v["end"] - v["start"]
                        pupil_tape += t["end"] - t["start"]
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
                    if row["frames"] > longest[1]:
                        longest = (stem, row["frames"])
                print("teacher", stem, "rows 2")
        if pairs_voiced != pairs_tape:
            raise SystemExit(f"answer pairs: voiced {pairs_voiced}, tape {pairs_tape}")
        print(
            f"teacher rows {n_rows} voice {label}={added[label]}; answer pairs "
            f"{pairs_voiced} (tape {pairs_tape}); typed-turn pupil frames voiced "
            f"{pupil_voiced} vs tape {pupil_tape}; longest teacher row "
            f"{longest[1]} frames, stem {longest[0]}"
        )

    held_out: list[dict[str, Any]] = []
    keep = set(held)
    with (out / "holdout.jsonl").open("w") as f:
        for h in held_in:
            if h["id"] not in keep:
                continue
            sp, segs = tape_of(h["id"])
            check_same_tape(to_record(assemble(h["id"], sp, segs, {})), h["base"])
            h2 = add_rendering(h, sp, eng.voiced, added)
            f.write(json.dumps(h2) + "\n")
            held_out.append(h2)
            print("held", h["id"], "added", ",".join(added))
    words = ("the", "same", "words")
    for lab, voice in added.items():
        first = eng.voiced(words, voice)
        eng.cache.clear()
        print(f"tts deterministic {lab}:", first == eng.voiced(words, voice))
    labels = (*stems["voices"], *added)
    n_hold, hold_max, min_window = validate_holdout(held_out, tuple(labels))
    print(f"holdout rows {n_hold} max frames {hold_max} min answer window {min_window}")
    print(f"extended {args.source} -> {args.name}: {len(train)} train, {len(held)} held")
```

**Check:** `uv run pytest tests/test_render_job.py -q` passes and `uv run mypy` is clean. On mypy: the
`SimpleNamespace` attributes are `Any`. If mypy strict flags `tape_of`'s return, type it
`tuple[list[TurnSpan], list[Segment]]` and import both names (`dual.turns.TurnSpan`,
`dual.frames.Segment`). No CPU test of `extend_render` exists, because it needs the TTS. Its logic is
V1's pure functions, and V5's render smoke exercises it.

### Task V2.4: batch close

Update the module docstring's first line to:
`"""Renders every training row and the holdout records once, or (--from) extends an earlier render with
new voices (B15). GPU for the TTS and Mimi."""`.
Run ruff format, ruff check and mypy, then commit:

```bash
git add src/dual/jobs/render.py tests/test_render_job.py
git commit -m "feat(b15): render --from/--add-voice/--teacher-voice extends render-1 with voice D"
```

---

## Batch V3: the train job takes a voiced teacher (model: sonnet)

Files: `src/dual/jobs/train.py`, `tests/test_train.py`, `tests/test_train_consistency.py`. It needs V1
(`tape_teachers`, `voiced_teachers`, `make_voiced`).

### Task V3.1: `--teacher` and the resume check (TDD)

**Step 1: failing tests** in `tests/test_train.py`, next to `test_consistency_flags_are_parsed`. Copy that
test's `sys.argv` monkeypatch pattern; `run_args` below is its argv list:

```python
def test_teacher_defaults_to_the_tape(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--name", "n", "--base", "b"])
    assert parse_args().teacher == "tape"


def test_teacher_voiced_is_parsed_and_needs_the_term(monkeypatch):
    argv = ["train", "--name", "n", "--base", "b", "--teacher", "voiced"]
    monkeypatch.setattr("sys.argv", [*argv, "--consistency", "3"])
    assert parse_args().teacher == "voiced"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_resume_refuses_another_teacher():
    meta = {"consistency": 3.0, "consistency_dir": "bar"}
    check_resume_consistency(meta, 3.0, "bar", "tape")  # predates B15: the tape
    with pytest.raises(ValueError, match="teacher"):
        check_resume_consistency(meta, 3.0, "bar", "voiced")
    check_resume_consistency(meta | {"teacher": "voiced"}, 3.0, "bar", "voiced")
```

**Step 2:** `uv run pytest tests/test_train.py -q -k "teacher"` → FAIL.

**Step 3: implement** in `src/dual/jobs/train.py`:
- Next to `DIRECTIONS`' import or definition: `TEACHERS = ("tape", "voiced")`.
- In `parse_args`, after `--consistency-dir`:

```python
    p.add_argument(
        "--teacher",
        choices=TEACHERS,
        default="tape",
        help="the consistency teacher row: the stem's all-tape row (B9), or the voiced "
        "row of the typed row's kind from the render's rows-teacher.jsonl (B15)",
    )
```

  and after the negative-weight check:

```python
    if args.teacher != "tape" and args.consistency == 0:
        p.error(f"--teacher {args.teacher} needs --consistency > 0")
```

- Replace `check_resume_consistency` with:

```python
def check_resume_consistency(
    run_meta: dict[str, Any], consistency: float, direction: str, teacher: str = "tape"
) -> None:
    """--resume keeps the recipe: the same weight, direction and teacher. A checkpoint
    without the keys predates B9 (no term) or B15 (the tape teacher)."""
    had = float(run_meta.get("consistency", 0.0))
    had_dir = str(run_meta.get("consistency_dir", "bar"))
    had_teacher = str(run_meta.get("teacher", "tape"))
    if (had, had_dir, had_teacher) != (consistency, direction, teacher):
        raise ValueError(
            f"--resume: checkpoint ran at consistency {had} ({had_dir}, teacher "
            f"{had_teacher}), flags say {consistency} ({direction}, teacher {teacher})"
        )
```

**Step 4:** `uv run pytest tests/test_train.py -q` → all pass, including the existing
`test_resume_refuses_another_consistency_setup`, which matches `"consistency"`.

### Task V3.2: `build_teachers` takes a typed-row-id map (TDD on CPU)

**Step 1: tests.** In `tests/test_train_consistency.py`:
- Import `make_voiced` from `test_consistency` and `tape_teachers` from `dual.consistency`.
- Replace the teacher-map argument at the existing call sites as follows. The behaviour is unchanged,
  only the key.
  - `build_teachers(FakeLM(1.0), [spoken, typed], {"s": spoken})` → `..., {typed["id"]: spoken})`
  - `build_teachers(lm, [typed], {"s": spoken})` (twice) → `..., {typed["id"]: spoken})`
  - `build_teachers(lm, [typed, spoken], {"s": spoken})` → `..., {typed["id"]: spoken})`
  - `build_teachers(lm, batch, {"s": spoken, "t": long_spoken})` → `..., tape_teachers([typed, spoken, long_typed, long_spoken]))`
- Append:

```python
def test_a_voiced_teacher_lines_up_after_a_longer_pupil_turn():
    typed, _ = make_rows([1])
    voiced = make_voiced([1])
    teachers = build_teachers(FakeLM(1.0), [typed], {typed["id"]: voiced})
    idx, lp = teachers[0]
    assert idx.tolist() == [18, 19, 20, 21, 22]  # the typed row's own frames
    assert lp.shape == (5, V)


def test_zero_when_typed_and_voiced_agree_positive_when_they_differ():
    typed, _ = make_rows([1, 3])
    voiced = make_voiced([1, 3])
    for c, positive in ((0.0, False), (1.0, True)):
        lm = FakeLM(c)
        _, _, cons = losses(lm, [typed], build_teachers(lm, [typed], {typed["id"]: voiced}))
        assert (float(cons.detach()) > 0) is positive


def test_one_teacher_forward_per_distinct_teacher_row(monkeypatch):
    typed, spoken = make_rows([1])
    calls = []

    def counting(lm, codes):
        calls.append(codes.shape[-1])
        return fake_forward(lm, codes)

    monkeypatch.setattr(dual.model, "forward", counting)
    other = dict(typed, id="s#complement", kind="complement")
    build_teachers(FakeLM(1.0), [typed, other], {typed["id"]: spoken, other["id"]: spoken})
    assert len(calls) == 1  # both typed rows share the stem's tape row
```

**Step 2:** `uv run pytest tests/test_train_consistency.py -q` → the new tests FAIL with `KeyError`.

**Step 3: implement.** Replace `build_teachers` in `src/dual/jobs/train.py`. Add `Mapping` to the
`collections.abc` import.

```python
def build_teachers(
    lm: Any, batch: list[dict[str, Any]], teachers: Mapping[str, Mapping[str, Any]]
) -> dict[int, Teacher]:
    """For each typed row of the batch (by position), the frame indices of its answer
    content frames and the model's own text log-probs at the same frames after its
    teacher row (`teachers`: typed row id -> the stem's all-tape row, or the voiced row
    of its kind): a no_grad forward with the current adapter, one teacher row at a time,
    so a single teacher forward is alive at once."""
    from dual.consistency import answer_pairs, flat_indices
    from dual.model import codes_from_row, forward

    out: dict[int, Teacher] = {}
    have: tuple[str, Any] | None = None
    typed_at = [i for i in range(len(batch)) if batch[i].get("typed_turns")]
    order = sorted(
        typed_at,
        key=lambda j: (str(batch[j]["stem"]), str(teachers[batch[j]["id"]]["id"])),
    )
    with torch.no_grad():
        for i in order:
            row = batch[i]
            t_row = teachers[str(row["id"])]
            student, teacher = flat_indices(answer_pairs(row, t_row))
            if not student:
                continue
            if have is None or have[0] != t_row["id"]:
                t = have = None  # drop the previous teacher's logits first
                have = (str(t_row["id"]), forward(lm, codes_from_row(dict(t_row))))
            t = have[1]
            s_idx = torch.tensor(student, device=t.text_logits.device)
            t_idx = torch.tensor(teacher, device=t.text_logits.device)
            keep = t.text_mask[0, t_idx]
            out[i] = (s_idx[keep], log_probs(t.text_logits[0, t_idx[keep]]))
    return out
```

The tape path is unchanged in values: the same forwards, one per stem, because every typed row of a stem
maps to the same `#spoken` row id. Only their order within a stem can differ, and these are `no_grad`
and deterministic. (`dict(t_row)` is there because `codes_from_row` is typed for a dict.)

**Step 4:** `uv run pytest tests/test_train_consistency.py -q` → all pass, except the four `main` smoke
tests, which fail until V3.3. Run `-k "not main"` to see the rest green.

### Task V3.3: wire it into `main` (TDD with the existing `job` fixture)

**Step 1: tests.** In `tests/test_train_consistency.py`, extend the `job` fixture so the render dir also
holds a teacher file:

```python
    vrows = [
        make_voiced([1]),
        make_voiced([], kind="complement"),
        make_voiced([1, 3], stem="t"),
        make_voiced([], stem="t", kind="complement"),
    ]
    (render / "rows-teacher.jsonl").write_text(
        "\n".join(json.dumps(r) for r in vrows) + "\n"
    )
```

Then append:

```python
def test_main_with_the_voiced_teacher_pairs_the_same_frames_and_saves_it(job, capsys):
    out = job.run("--consistency", "3", "--teacher", "voiced", "--epochs", "1")
    log = capsys.readouterr().out
    assert "teacher voiced D" in log
    assert "typed rows 2, answer pairs 3, content frames 13" in log  # = the tape's
    blob = torch.load(out / "model-epoch-01.pt", weights_only=True)
    assert blob["run"]["teacher"] == "voiced"
    with pytest.raises(SystemExit, match="teacher"):
        job.run("--consistency", "3", "--epochs", "2", "--resume")


def test_main_with_the_tape_teacher_logs_as_before(job, capsys):
    out = job.run("--consistency", "10", "--epochs", "1")
    assert "teacher voiced" not in capsys.readouterr().out
    blob = torch.load(out / "model-epoch-01.pt", weights_only=True)
    assert blob["run"]["teacher"] == "tape"
```

**Step 2:** `uv run pytest tests/test_train_consistency.py -q -k main` → the new tests FAIL.

**Step 3: implement** in `main` of `src/dual/jobs/train.py`:
- Import `tape_teachers` and `voiced_teachers` from `dual.consistency`. Drop the now-unused
  `spoken_by_stem` import.
- Replace the `spoken = cast(...)` block and the `consistency` log lines (current lines 295-310) with:

```python
    # the rows are json dicts; the helpers type them as read-only Mappings
    teachers: dict[str, dict[str, Any]] = {}
    voice = ""
    if args.consistency > 0 and args.teacher == "voiced":
        teacher_rows = [
            r
            for r in map(
                json.loads, (render / "rows-teacher.jsonl").read_text().splitlines()
            )
            if r["stem"] in keep
        ]
        teachers = cast(dict[str, dict[str, Any]], voiced_teachers(rows, teacher_rows))
        voice = str(teacher_rows[0]["voice"]) if teacher_rows else "?"
    elif args.consistency > 0:
        teachers = cast(dict[str, dict[str, Any]], tape_teachers(rows))
    rows = select_rows(rows, args.only_stem)
    typed_rows = [r for r in rows if r.get("typed_turns")]
    how = args.consistency_dir + (f", teacher voiced {voice}" if voice else "")
    consistency = f"consistency {args.consistency} ({how}); "
    consistency += f"typed rows {len(typed_rows)}"
    if args.consistency > 0:
        # every typed row pairs with its teacher row now, not mid-epoch
        pairs = [answer_pairs(r, teachers[r["id"]]) for r in typed_rows]
        content = sum(len(flat_indices(p)[0]) for p in pairs)
        consistency += (
            f", answer pairs {sum(map(len, pairs))}, content frames {content}"
        )
```

- In the resume block: `check_resume_consistency(<meta>, args.consistency, args.consistency_dir, args.teacher)`.
- In the step loop: `build_teachers(lm, batch, teachers) if args.consistency > 0 else None`.
- In the checkpoint meta dict, after `"consistency_dir"`: `"teacher": args.teacher,`.
- The log for the tape teacher stays exactly `consistency 10.0 (bar); typed rows 2, ...`. The existing
  `test_main_at_zero_never_builds_a_teacher` asserts the λ = 0 form `consistency 0.0 (bar); typed rows 2\n`.

**Step 4:** `uv run pytest tests/test_train_consistency.py tests/test_train.py tests/test_mem_cap.py -q`
→ all pass.

### Task V3.4: batch close

Update the `--consistency` help to `"weight of KL(typed answer || teacher answer); 0 = the baseline
recipe"`. Run ruff format, ruff check and mypy, then commit:

```bash
git add src/dual/jobs/train.py tests/test_train.py tests/test_train_consistency.py
git commit -m "feat(b15): train --teacher voiced pairs each typed row with its voiced row"
```

---

## Batch V4: the measure reports the added voices; ladder 6 (model: sonnet)

Files: `src/dual/jobs/measure.py`, `tests/test_measure.py`, `scripts/ladder.sh`, `tests/test_ladder_sh.py`.
It is independent of V1-V3.

### Task V4.1: `extra_labels`, `plan_pairs` and the summary (TDD)

**Step 1: failing tests**, appended to `tests/test_measure.py`. Extend the import to
`from dual.jobs.measure import PAIRS, RENDERINGS, extra_labels, label, pair_fields, plan_pairs, select_ckpts, summarise`.

```python
def test_render_1s_voices_add_nothing():
    assert extra_labels({"A": "a", "B": "b", "C": "c"}) == ()
    assert plan_pairs(()) == (RENDERINGS, PAIRS)


def test_an_added_voice_adds_its_gap_and_two_floors():
    assert extra_labels({"A": "a", "D2": "x", "B": "b", "C": "c", "D1": "y"}) == (
        "D1",
        "D2",
    )
    renderings, pairs = plan_pairs(("D",))
    assert renderings == (*RENDERINGS, "D")
    assert pairs[: len(PAIRS)] == PAIRS
    assert pairs[len(PAIRS) :] == (
        ("gap_D", "typed", "D"),
        ("floor_D", "D", "A"),
        ("floor_D_b", "D", "B"),
    )


def test_summarise_reports_the_added_pairs_and_leaves_the_bar_alone():
    lines = [line(1.0, 0.5, 0.4, 2.0), line(3.0, 0.7, 0.6, 4.0), line(2.0, 0.6, 0.5, 3.0)]
    plain = summarise(lines, epoch=3, base="b", run="r", slice_=1)
    for x, d in zip(lines, (0.2, 0.4, 0.3), strict=True):
        x["gap_D"] = kl(d, d)
        x["floor_D"] = kl(d / 2, d / 2)
    s = summarise(lines, epoch=3, base="b", run="r", slice_=1)
    assert s["content"] == plain["content"]
    assert s["gap_D_median"] == pytest.approx(0.3)
    assert s["floor_D_median"] == pytest.approx(0.15)
    assert "gap_D_median" not in plain


def test_pair_fields_takes_the_planned_pairs():
    g = torch.Generator().manual_seed(0)
    lps = {r: log_probs(torch.randn(4, 16, generator=g)) for r in (*RENDERINGS, "D")}
    _, pairs = plan_pairs(("D",))
    out = pair_fields(lps, [PAD, 3, 4, 5], pairs)
    assert set(out["per_frame"]) == {k for k, _, _ in pairs}
```

**Step 2:** `uv run pytest tests/test_measure.py -q` → FAIL (ImportError).

**Step 3: implement** in `src/dual/jobs/measure.py`. Add `Mapping` to the `collections.abc` import.

```python
BAR_VOICES = ("A", "B", "C")
Pair = tuple[str, str, str]


def extra_labels(voices: Mapping[str, str]) -> tuple[str, ...]:
    """Rendering labels a render added past the bar's A, B and C (B15: the teacher's)."""
    return tuple(sorted(v for v in voices if v not in BAR_VOICES))


def plan_pairs(extra: Sequence[str]) -> tuple[tuple[str, ...], tuple[Pair, ...]]:
    """The renderings and pairs to measure: the bar's, plus per added label L
    gap_L = KL(typed || L), floor_L = KL(L || A) and floor_L_b = KL(L || B)."""
    pairs = list(PAIRS)
    for lab in extra:
        pairs += [
            (f"gap_{lab}", "typed", lab),
            (f"floor_{lab}", lab, "A"),
            (f"floor_{lab}_b", lab, "B"),
        ]
    return (*RENDERINGS, *extra), tuple(pairs)
```

- `pair_fields(lps, targets, pairs: Sequence[Pair] = PAIRS)`: loop over `pairs`, not `PAIRS`.
- `summarise`: replace `for key in ("door", "tape_gap", "tape_floor"):` with

```python
    added = [k for k in (lines[0] if lines else {}) if k.startswith(("gap_", "floor_"))]
    for key in ("door", "tape_gap", "tape_floor", *added):
```

- `turn_lines(lm, records, epoch, renderings: Sequence[str] = RENDERINGS, pairs: Sequence[Pair] = PAIRS)`:
  - loop `for rendering in renderings:`;
  - call `fields = pair_fields(lps, targets, pairs)`;
  - write the line with `for key, _, _ in pairs:`.
- `main`: after reading `records`, add

```python
    voices = json.loads((render / "stems.json").read_text()).get("voices", {})
    renderings, pairs = plan_pairs(extra_labels(voices))
    print("renderings", ",".join(renderings), "pairs", ",".join(k for k, _, _ in pairs))
```

  and pass `renderings, pairs` to every `turn_lines(...)` call (find them with
  `grep -n "turn_lines(" src/dual/jobs/measure.py`).
- Docstring: add "Labels a render added past A/B/C (stems.json `voices`, B15) add gap_L = KL(typed || L),
  floor_L = KL(L || A), floor_L_b = KL(L || B); render-1 adds none."

**Step 4:** `uv run pytest tests/test_measure.py tests/test_report.py tests/test_noise.py tests/test_position.py -q`
→ all pass.

### Task V4.2: `LADDER=6` on render-2 (TDD on `tests/test_ladder_sh.py`)

**Step 1: failing tests**, appended:

```python
def test_ladder_6_runs_the_voiced_teacher_on_render_2(box):
    r, log = box(LADDER="6")
    assert r.returncode == 0, r.stderr
    assert submits(log) == [
        "make IMAGE=img:tag -s train NAME=voiced1-pp-l3 BASE=personaplex SLICE=1"
        " RENDER=render-2 EPOCHS=3 ARGS=--consistency 3 --teacher voiced",
        "make IMAGE=img:tag -s measure NAME=voiced1-pp-l3 BASE=personaplex"
        " RUN=voiced1-pp-l3 RENDER=render-2",
    ]
    assert " bar BOX=vlad@box" in log[-1]


def test_ladder_6_train_carries_exactly_one_args_word(box, tmp_path):
    r, _ = box(LADDER="6")
    assert r.returncode == 0, r.stderr
    (train,) = [c for c in argv_calls(tmp_path) if "train" in c]
    assert [w for w in train if w.startswith("ARGS=")] == [
        "ARGS=--consistency 3 --teacher voiced"
    ]


def test_ladder_6_rejects_a_ladder_4_run_name(box):
    r, log = box(LADDER="6", FROM="cons1-pp-l3")
    assert r.returncode == 2
    assert "voiced1-pp-l3" in r.stdout + r.stderr
    assert submits(log) == []
```

**Step 2:** `uv run pytest tests/test_ladder_sh.py -q -k ladder_6` → FAIL (exit 2, "not a ladder").

**Step 3: implement** in `scripts/ladder.sh`:
- Before `case "$LADDER" in`, add `RENDER_RUN=render-1   # the render every train and measure of the ladder reads`.
- Add a case: `6) RUNS="voiced1-pp-l3,personaplex,1,3,--consistency,3,--teacher,voiced"; RENDER_RUN=render-2 ;;`
- Change the `*)` message to `(1, 2, 3, 4 or 6)`.
- In `run_one`, replace the three `RENDER=render-1` with `RENDER="$RENDER_RUN"`.
- Header comment: after the LADDER=4 sentence, add "LADDER=6: the B15 voice-D teacher, voiced1-pp-l3
  (--consistency 3 --teacher voiced), its train and measure on render-2 (render-1 plus voice D). (LADDER=5
  was B14's scratchpad-only copy.)"

**Step 4:** `uv run pytest tests/test_ladder_sh.py -q` → all pass. The existing ladder 1-4 tests prove
that `RENDER=render-1` is unchanged.

### Task V4.3: batch close

Run ruff format, ruff check and mypy, then commit:

```bash
git add src/dual/jobs/measure.py tests/test_measure.py scripts/ladder.sh tests/test_ladder_sh.py
git commit -m "feat(b15): measure adds gap/floor pairs for render-added voices; LADDER=6 voiced teacher on render-2"
```

---

## Batch V5: run it (model: opus; the orchestrator)

Preconditions:
- V1-V4 are merged to `main` (full gate green) and `main` is pushed.
- Read the `operating-the-sparks-queue` and `authoring-a-sparks-job` skills.
- Vlad's standing permissions:
  - **drama3:** run `cd ~/drama3 && docker compose stop` on the box before GPU runs. Never remove or
    edit its container, compose file or crontab, and never restart it. It is currently stopped. The
    Mac-side watchdog from `drama3-memory-hog.md` may run.
  - **HF_TOKEN:** it is in the untracked, gitignored `.env`. Never commit, echo or log it, or pass it on
    a command line. No step needs it: the weights are on the box (`/srv/spark/hf/hub`,
    `/srv/spark/models/personaplex-7b-v1`).
- Write the memory note `b15-voice-d-run.md` (image, job ids, run names, how to resume) at the first
  submit, and add it to `MEMORY.md`.

### Task V5.1: image and box

1. On `main`: `SHA=$(git rev-parse --short HEAD)`.
2. `rm -rf docker/src docker/vendor && cp -RL src docker/src && cp -RL vendor docker/vendor`.
3. `DOCKER_BUILDKIT=0 docker build -t 100.114.56.75:5000/vlad/voiced1:$SHA docker && docker push 100.114.56.75:5000/vlad/voiced1:$SHA`.
   If the push fails off-LAN, see `loader-facts.md`.
4. `IMAGE=spark.local:5000/vlad/voiced1:$SHA`, the registry prefix B12-B14 submits used.
5. `ssh -o ConnectTimeout=15 vlad@100.114.56.75 'cd ~/drama3 && docker compose stop; free -g; ls /srv/spark/runs/render-1/render /srv/spark/runs/cons1-pp-l3/train'`.
   Expect ~110-118 GiB available, render-1's three files, and `model-epoch-03.pt`.
6. Locally, `ls data/voices/vctk/ | grep -E 'p226|p280|p294'` must show the three
   `_023.wav.1e68beda@240.safetensors` files (`make data` puts them there; the job reads `data/` from
   the submit).
7. `sparks queue --json` must be empty before every submit. Run one job at a time, all with
   `SPARKS_HOST=vlad@100.114.56.75`.

### Task V5.2: the voice screen and **GATE-0** (pre-registered in D3)

1. `IMAGE=$IMAGE make render NAME=d-screen ARGS='--from render-1 --add-voice D1=vctk/p226_023.wav --add-voice D2=vctk/p280_023.wav --add-voice D3=vctk/p294_023.wav'` (~0.25 h).
   The log must show `tts deterministic D1/D2/D3: True` and `holdout rows 520` (13 × 5 × 8). It holds no
   teacher rows.
2. `IMAGE=$IMAGE make measure NAME=d-screen BASE=personaplex RENDER=render-d-screen RUN=cons1-pp-l3 ARGS='--epochs 3 --tag d-screen'`
   (~0.7 h). It must print `renderings typed,A,B,C,tape,D1,D2,D3`.
3. Read `/srv/spark/runs/measure-d-screen/measure/personaplex-cons1-pp-l3-d-screen/summary.json`
   (`scp` it into `runs/`).
   - **Reproduction:** `content.median_gap` / `median_floor` must equal B12's `cons1-pp-l3` epoch 3
     within 2% (.0146 / .0078; R 1.87 ± 0.03). Otherwise stop and debug: the A/B/typed rows must be
     byte copies.
   - Compute `r_L`, `r_L_b` and `s_L` for D1-D3, plus tape‖A/floor and C‖A/floor, as in D3. Use the
     `*_median` keys and `content.median_floor`; `s_L` = `gap_L_median` / `content.median_gap`.

**GATE-0, decided by the orchestrator. Ask Vlad only on a fail.**
- **Pass:** some candidate is admissible and passes the lever check. Pick it by D3's rule and record
  every number in the memory note.
- **Fail** (none admissible, or the lever check fails): stop and ask Vlad, with the table. The options:
  - (a) screen the other six VCTK voices (p237, p245, p251, p311, p326, p376), ~1 GPU-h;
  - (b) take the best candidate anyway, with its expected ceiling stated;
  - (c) stop the line: λ3-tape e3 to the engine.

### Task V5.3: render-2, the smokes and **GATE-1**

1. Smoke render: `IMAGE=$IMAGE make render NAME=2-smoke ARGS='--from render-1 --add-voice D=<chosen> --teacher-voice D --limit 10'` (~0.15 h).
   It must print `teacher rows 20`, `answer pairs N (tape N)` and `tts deterministic D: True`, with no
   `check_same_tape` error.
2. Full render: `IMAGE=$IMAGE make render NAME=2 ARGS='--from render-1 --add-voice D=<chosen> --teacher-voice D'` (~0.6 h).
   Record from its log:
   - `teacher rows 234`;
   - the answer pairs (voiced = tape);
   - the typed-turn pupil frames, voiced vs tape;
   - the longest teacher row and its stem (must be ≤ 3,000; the job would have raised otherwise);
   - `holdout rows 390` (13 × 5 × 6).
3. Train smokes, one after the other (~0.25 h each):
   - `IMAGE=$IMAGE make train NAME=voiced-smoke-1 RENDER=render-2 EPOCHS=1 ARGS='--consistency 3 --teacher voiced --only-stem <stem of the longest teacher row> --max-steps 3'`
   - `IMAGE=$IMAGE make train NAME=voiced-smoke-wide RENDER=render-2 EPOCHS=1 ARGS='--consistency 3 --teacher voiced --widest-first --max-steps 4'`
4. Read the full logs (`/srv/spark/runs/run-*-voiced-smoke-*/output.log`; `sparks logs` truncates).

**GATE-1, decided by the orchestrator. Ask Vlad only on a fail.**
- **Pass:** all of the following.
  - Peak ≤ 57 GiB in both smokes (B12/B14: 56.0).
  - Every `cons` is finite, and `cons > 0` on some step.
  - The teacher share is ≤ 40% of step time, ignoring step 0.
  - `voiced-smoke-wide`'s startup line prints the same `steps/epoch` and the same `typed rows …, answer
    pairs …, content frames …` as `cons1-pp-l3`'s output.log (D4 check 2), plus `teacher voiced D`.
- **Peak 57-62 GiB or teacher share > 40%:** stop and ask Vlad, with the per-epoch hours stated.
- **OOM, non-finite values, or a pair/content count mismatch:** stop and debug with
  superpowers:systematic-debugging.

### Task V5.4: the run (`LADDER=6`)

1. Stop drama3 again, then launch detached:
   `LADDER=6 IMAGE=$IMAGE SPARKS_HOST=vlad@100.114.56.75 nohup caffeinate -i sh scripts/ladder.sh > /tmp/ladder6.log 2>&1 &`.
   It runs:
   - train `voiced1-pp-l3` (3 × ~4,300 s);
   - `measure-voiced1-pp-l3` (~1.5 h; `renderings typed,A,B,C,tape,D`);
   - `make bar`.
2. Watch with bounded ssh (`ConnectTimeout=15`) and `sparks status --json`.
   - An epoch well above ~4,500 s, or a ballooning one, preceded earlier freezes. If the box drops off
     tailscale, follow `box-outages.md`.
   - A hung train resumes with
     `IMAGE=$IMAGE make train NAME=voiced1-pp-l3 BASE=personaplex SLICE=1 RENDER=render-2 EPOCHS=3 ARGS='--consistency 3 --teacher voiced --resume'`,
     then the measure by hand:
     `make measure NAME=voiced1-pp-l3 BASE=personaplex RUN=voiced1-pp-l3 RENDER=render-2`.
3. When `make bar` has run: `make noise VARIANTS=personaplex-voiced1-pp-l3`. Note the rule's epoch E.
4. Side-by-side decodes, using B14's image and flags for comparability, on render-1 (A/B/typed only):
   - `IMAGE=spark.local:5000/vlad/sbs2:72868db make sidebyside NAME=sbs-v1-e3 BASE=personaplex RENDER=render-1 RUN=voiced1-pp-l3 EPOCH=3 ARGS='--no-bare --swap-turns 2,6 --mem-cap-gib 64'`
     (~0.8 h);
   - if E ≠ 3, also `NAME=sbs-v1-eE ... EPOCH=E`;
   - if sbs2's loader refuses the new `teacher` meta key, use `$IMAGE` instead and say so in the results.
5. Check each meta (batch shapes 53/35/29/26/23/21/8 and 28/24, peak memory), then copy `answers.jsonl`
   and `meta.json` into `docs/results/consistency-5/` (V6).

---

## Batch V6: read it, blind judging, record (model: opus)

GPU-free. Files: `docs/results/consistency-5.md`, `docs/results/consistency-5/` (answers, metas,
`pool/`), this plan's "B15 as run", a pointer line under B15 in
`docs/plans/2026-10-26-dual-input-batches.md`, and memory.

### Task V6.1: the bar and the ceiling

- **The noise rule:** the verdict, R_gap and its epoch (R0 3.00, T 0.5).
- **R per epoch** for `voiced1-pp-l3`, next to `cons1-pp-l3` and `ladder-personaplex-1`.
- **Paired same-epoch CIs** against both, from a copy of `docs/results/consistency-4/pool/epoch_ci.py`
  extended to the new run. It must reproduce consistency-4's seed-0 rows exactly.
- **`content.passes` and the p90 ratio** per epoch.
- **`make position FILES="runs/measure/personaplex-voiced1-pp-l3/epoch-03.jsonl runs/measure/personaplex-cons1-pp-l3/epoch-03.jsonl"`:**
  the ratio at positions 26+.
- **The ceiling**, per epoch, from `summary.json`:
  - `gap_D_median` (typed‖D, what the term minimises);
  - `floor_D_median` / `content.median_floor` (D‖A over the floor: the new ceiling);
  - `floor_D_b_median`;
  - `tape_gap_median` and `tape_floor_median` (did typed drift from the tape?).
- **From the train log:** held text loss per epoch against `cons1-pp-l3` (B12) and `ladder-personaplex-1`
  (0.551 / 0.540 / 0.611); mean `cons`; max grad; s/epoch; peak.
- **Turn-0 lead-frame KL(typed‖A)** per epoch, as B14 did.

### Task V6.2: the blind pool (opus judges, xhigh)

- One fresh blind pool: B14's eleven checkpoints plus the new one or two.
  - `pool/combine13.py` is a copy of `consistency-4/pool/combine8.py` extended to the new answers. With
    the new specs removed it must reproduce B14's eleven-checkpoint input byte for byte.
  - `pool/blind5.py` is a copy of `blind4.py`, seed 13.
- B10's rubrics verbatim (`consistency-4/pool/rubric-{pair,answer,swap}.txt`).
- One opus judge per ~98-item chunk, reading only that chunk's `items.jsonl` and `rubric.txt`; ~55 judges.
- Unblind with copies of `score4.py` and `swap6.py`. Report agreement with B14's labels on the shared items.
- Text counts with `consistency-4/pool/qual4.py`: speaks after the spoken opening, onset frame, stops,
  median words.

### Task V6.3: `docs/results/consistency-5.md`, the verdict, and the record

- Lead with the D9 verdict (PASS / LOWERS THE CEILING / NO BETTER) and the pieces it rests on.
- Every bar number sits next to the behaviour table for the same checkpoints: `voiced1` e3 (and eE),
  `cons1-pp-l3` e3, and `ladder-personaplex-1` e3.
- The voice screen table (GATE-0) and the render-2 facts (pupil frames voiced vs tape, longest row).
- GPU-hours per job against D10.
- `make check`, a sonnet review, then an opus review that recomputes every number. Merge `--no-ff`.
- This plan gets "B15 as run" and `dual-input-batches.md` a B15 line.
- Memory: a result note plus a `MEMORY.md` line. Push `main`.

### **GATE-2** (Vlad)

Ask with the verdict and these options:
- **PASS:** (a) a seed-1 replication (`--seed 1`, paired with `ladder2-pp-s1`, ~6.5 GPU-h), then the engine;
  (b) straight to the engine.
- **LOWERS THE CEILING:**
  - (a) a second seed;
  - (b) λ 5, between λ3's health and λ10's breakage;
  - (c) voiced rows also in the LM loss. This changes batches and needs its own λ = 0 control.
  - (d) two teacher voices, D and E, alternating by epoch.
- **NO BETTER:** (a) stop the consistency line and send λ3-tape e3 to the engine; (b) revisit the bar.

---

## Appendix: what this plan could not settle (for the GATEs, not for the executor)

- **Whether any VCTK voice sits near the floor against A.**
  - C‖A (the door) is the only prior, and it is not quoted in the results read for this plan. The screen
    settles it at ~1 GPU-h before render-2 is paid for.
  - If D‖A is itself ~1.7× the floor, a voice-D teacher cannot pass the bar.
- **The genders and accents of p226/p280/p294** are not checked here. The screen judges in-distribution
  by the model's own KL, not by metadata.
- **Whether the model, trained on tape pupil audio only, answers after a TTS turn as it does after a tape
  turn.** It does for A and B at the bar (λ3 e3 speaks 13/13), so the teacher on D should too. The
  side-by-side does not decode a D rendering. If GATE-2 needs it, a decode of render-2's D is a
  follow-up (~0.8 h).
- **Voice-D render time** is an extrapolation from render-1 (~2.5 s per segment, unbatched).

---

## B15 as run (2026-11-08 / 09)

- **Code:** V1-V4 as planned, plus two review fixes. `answer_pairs` checks every teacher pupil span's rendering,
  not the `voiced_turns` metadata. `voiced_rows` refuses reserved labels. `render --from` refuses a missing
  voice file before the TTS and writes `stems.json` last, with the stems it covers. Also a V3 deviation: the
  plan's snippet let the per-batch `teachers` variable shadow the teacher map; it is renamed `targets`.
  Merged to main at 80d7766 (334 tests); image `voiced1:80d7766`.
- **GATE-0:** all three candidates were admissible. D = D3 = `vctk/p294_023.wav` (r 0.755, r_b 0.937, s 0.954;
  D1 had the smallest r, 0.730, but a larger r_b). The screen reproduced B12 e3 exactly.
- **GATE-1:** passed (peaks 39.2 / 50.1 GiB, teacher 33%, startup = `cons1-pp-l3`'s).
- **Run:** `LADDER=6` (13,227 s train, 5,366 s measure); decodes sbs-v1-e3 and sbs-v1-e1 (the rule's epoch)
  on `sbs2:72868db`, which read the new checkpoint. 8.07 GPU-hours in all.
- **Verdict:** NO BETTER; R failed, behaviour passed (`docs/results/consistency-5.md`).
