# Bar by position 1: where in the tutor's answer the gap sits

Vlad, 2026-11-05, at the side-by-side gate: "Re-check the bar (Recommended). Cheap first: one measure pass on epoch 2 (no training) reporting KL by position in the answer, to see if the gap sits in the opening frames. Then decide whether to change the bar."

## The answer

**No. The gap does not sit in the opening frames. On the frames the bar scores, the opening is where typed and spoken agree best. The ~3 sits in the body of the answer, and it is a typed-vs-speech gap: it is just as large against the dialogue's original recorded pupil turn as against the TTS voices.**

- **The all-frame ~18 is the 3 lead frames.**
  - Every answer window opens with exactly 3 PAD frames before the tutor's first word (65/65 turns). The tape's text always opens `<pad><pad><pad><s>`, so this comes from the corpus, not from the window.
  - On those frames KL(typed‖A) has a median of **7.4 nats per frame**, against a floor of 0.003. They hold **81% of the all-frame gap mass**.
  - Dropping them takes the all-frame ratio from **18.2 to 3.2**, the content bar's value. The content bar already leaves them out.
- **On content frames the opening is closest to the floor.**
  - By bucket, the ratio is **3.7, 2.5, 1.3, 2.0 and 1.8** at positions 0, 1, 2, 3-5 and 6-10. It is **3.4** at 11-25 and **3.4** at 26+.
  - Frame-weighted, the ratio is **1.30** over positions 0-10 and **3.45** from 26 on.
  - Per turn, the ratio before position 26 is below the ratio from 26 on in 42/65 turns.
  - Bucket 2 is a near-deterministic second token: every pair is ~0.0001 nats, in the 36 turns where it is a content frame.
  - The bar is a median of per-turn means, and 91% of content frames are at 26+. So starting it later barely moves it: from position 0, 1, 2, 3, 6 and 11 it is 3.30, 3.25, 3.12, 3.10, 3.18 and 3.29.
  - A stem bootstrap gives a 5-95% band of 2.49-3.47 for the content ratio, and -0.15 to +0.95 for the 26+ ratio minus the full ratio.
- **The floor is front-loaded.**
  - Positions 0-25 are 9% of content frames, but hold 59% of the content floor mass (A‖B) and only 40% of the gap mass.
  - So cutting the opening makes the p90 test harder. p90 gap / p90 floor is 1.62 with no cut, 2.98 from position 11 and 3.59 from 26.
  - No cut makes epoch 2 pass.
- **The remaining gap is typed against any speech.**
  - From position 26 on, typed‖tape is **0.0174** and typed‖A is **0.0176**. tape‖A is **0.0069**, near the floor of **0.0051**.
  - The two spoken renderings, recorded and TTS, nearly agree with each other. Typed differs from both by about 3x.
- **The lead artifact is the TTS renderings'.**
  - On the lead frames, typed‖tape is 0.02 nats and tape‖A is 7.0, the same as typed‖A.
  - B7 traced this to the 6 frames of Mimi silence the TTS renderings add after the voice. That is the pupil turn's end, not the tutor's onset.
- **Bare is not a reference here.**
  - Bare PersonaPlex's ratio is 82 at position 0 and 15 at position 1: the teacher-forced probability of the first word token.
  - From position 3 on, its ratio is 3.4-3.9, and its content bar is 3.93.
  - At epoch 2, training brought positions 0-1 down to the level of the rest of the answer (3.7 and 2.5). It left the body at ~3.4.

**What follows for the bar.**

- The content-frame bar is the fairer of the two existing bars on this question. It skips the lead, where the TTS artifact lives.
- No starting position helps: the median ratio stays ~3, and p90 gets worse.
- Trimming the TTS renderings' trailing silence to match the tape would clean up the lead frames and the all-frame number, which the bar does not use.
- Trimming is **predicted not to lower the gap**: typed‖tape already equals typed‖A on content frames.
- Its effect on the **floor** (A‖B, TTS vs TTS, which sits mostly in the opening content frames) is not measured. tape‖A over all content frames is 0.0136 against a floor of 0.0077, so the voices do differ from the tape, mostly early.
- B7's greedy answers did not show the ~3 as a difference in substance (typed vs A 21/65, A vs B 23/65).

## Options for the bar (the gate)

1. **Keep the content bar as it is** and treat the ~3 as a real typed-vs-speech gap in the body of the answer. If the target stays "typed is as close to spoken as two voices are", a training change (a consistency or distillation term on the typed answer) has to close it. Ladder 2 showed that rank and LR do not.
2. **Trim the TTS trailing silence to the tape and remeasure.** A re-render of the holdout plus one measure pass, about 1 GPU-hour. It is predicted to leave the gap and could only move the floor. Do it only if the floor's sensitivity matters for (1) or (3).
3. **Change the yardstick.** Use tape‖A (recorded vs TTS, 0.0136) instead of A‖B as the floor. The gap is then 1.8x, not 3x. That is defensible ("typed should be as close to speech as two kinds of speech are"), but it is a new bar, decided before any further run.
4. **Declare epoch 2 good enough and go to the engine,** on B7's behavioural evidence: same substance, uptake 65/65. That accepts a KL bar that has never passed.

## What was run

- **Code** (branch `feat/b8-bar-by-position`):
  - `src/dual/jobs/measure.py` now writes `content_mask` and `per_frame` KL for every pair, and takes `--epochs` and `--tag`.
  - `src/dual/position.py` and `make position FILES=...` build the tables below. The module is pure.
  - `dual.report` skips `--tag` passes, so `make bar` is unchanged.
- **Jobs** (image `b8pos:d80113b`, drama3 stopped, 64 GiB cap):
  - `job-20261105-141338-vlad-measure-b8-pos-pp`: `ladder-personaplex-1` epoch 2, 1,456 s.
  - `job-20261105-141342-vlad-measure-b8-pos-bare`: bare, 231 s.
  - Outputs are in `/srv/spark/runs/measure-b8-pos-{pp,bare}/measure/` on the box.
- **Reproduction check:**
  - Epoch 2 gives content gap/floor 0.0252/0.0077 and all-frame 0.0752/0.0041. Ladder 1 measured 0.0251/0.0077 and 0.0756/0.0039.
  - The difference is bf16 run-to-run drift, which B7 also saw.
  - `bar_from` at position 0 (content) and with no cut (all) reproduces the per-turn scalars exactly.
- **Position:** a frame's position is its index in the answer window minus that of the first frame whose target text is not PAD.
  - The targets are the tape's teacher text in every rendering, so positions agree across pairs.
  - `lead` = the frames before the first word (always 3).
  - In content mode, the PAD frames inside the answer are dropped from the means but keep their positions. So bucket `2` has frames in only 36 turns.
- **Columns:**
  - Medians are over turns of each turn's mean in the bucket.
  - The share is the bucket's part of the pooled KL mass in that mode.
  - gap = KL(typed‖A), floor = KL(A‖B), door = KL(C‖A).

## Lead frames, per frame (epoch 2, medians over 65 turns, nats)

| lead frame | typed‖A (gap) | A‖B (floor) | typed‖tape | tape‖A |
|---|---|---|---|---|
| 0 | 7.10 | 0.0012 | 0.020 | 6.22 |
| 1 | 9.78 | 0.0011 | 0.011 | 9.62 |
| 2 | 6.24 | 0.0025 | 0.017 | 6.13 |

## The tape pairs by position (epoch 2, content frames, median per-turn mean)

| bucket | typed‖A (gap) | typed‖tape | tape‖A | A‖B (floor) |
|---|---|---|---|---|
| 0 | 0.0199 | 0.0122 | 0.0343 | 0.0054 |
| 1 | 0.1347 | 0.1103 | 0.0943 | 0.0530 |
| 2 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| 3-5 | 0.0709 | 0.0438 | 0.0469 | 0.0346 |
| 6-10 | 0.0319 | 0.0269 | 0.0272 | 0.0177 |
| 11-25 | 0.0629 | 0.0591 | 0.0277 | 0.0185 |
| 26+ | 0.0176 | 0.0174 | 0.0069 | 0.0051 |

Per turn, the content gap is above the content floor in 60/65 turns at epoch 2 and 64/65 bare.

## The tables (`make position`)

### personaplex-ladder-personaplex-1-pos/epoch-02.jsonl

65 turns, 13 stems. Per-turn scalars, median gap / median floor: content 0.0252 / 0.0077 = 3.30; all_frames 0.0752 / 0.0041 = 18.15.

By position, content frames:

| bucket | turns | frames | median gap | median floor | ratio | median door | gap share | floor share |
|---|---|---|---|---|---|---|---|---|
| lead | 0 | 0 | - | - | - | - | 0% | 0% |
| 0 | 65 | 65 | 0.0199 | 0.0054 | 3.71 | 0.0031 | 2% | 1% |
| 1 | 65 | 65 | 0.1347 | 0.0530 | 2.54 | 0.0527 | 5% | 8% |
| 2 | 36 | 36 | 0.0001 | 0.0001 | 1.27 | 0.0001 | 0% | 0% |
| 3-5 | 48 | 95 | 0.0709 | 0.0346 | 2.05 | 0.0359 | 5% | 10% |
| 6-10 | 54 | 146 | 0.0319 | 0.0177 | 1.80 | 0.0116 | 8% | 16% |
| 11-25 | 65 | 541 | 0.0629 | 0.0185 | 3.40 | 0.0165 | 20% | 24% |
| 26+ | 65 | 9463 | 0.0176 | 0.0051 | 3.44 | 0.0040 | 60% | 40% |

By position, all frames:

| bucket | turns | frames | median gap | median floor | ratio | median door | gap share | floor share |
|---|---|---|---|---|---|---|---|---|
| lead | 65 | 195 | 7.3737 | 0.0031 | 2408.00 | 0.0024 | 81% | 2% |
| 0 | 65 | 65 | 0.0199 | 0.0054 | 3.71 | 0.0031 | 0% | 1% |
| 1 | 65 | 65 | 0.1347 | 0.0530 | 2.54 | 0.0527 | 1% | 8% |
| 2 | 65 | 65 | 0.0003 | 0.0001 | 2.50 | 0.0002 | 0% | 0% |
| 3-5 | 65 | 195 | 0.0242 | 0.0069 | 3.50 | 0.0092 | 1% | 10% |
| 6-10 | 65 | 325 | 0.0178 | 0.0081 | 2.19 | 0.0061 | 2% | 15% |
| 11-25 | 65 | 975 | 0.0375 | 0.0107 | 3.52 | 0.0095 | 4% | 23% |
| 26+ | 65 | 22007 | 0.0081 | 0.0023 | 3.58 | 0.0019 | 11% | 41% |

Bar from position:

| mode | from position | turns | frames | median gap | median floor | ratio | p90 gap | p90 floor |
|---|---|---|---|---|---|---|---|---|
| content | all | 65 | 10411 | 0.0252 | 0.0077 | 3.30 | 0.0509 | 0.0313 |
| content | 0 | 65 | 10411 | 0.0252 | 0.0077 | 3.30 | 0.0509 | 0.0313 |
| content | 1 | 65 | 10346 | 0.0249 | 0.0077 | 3.25 | 0.0478 | 0.0315 |
| content | 2 | 65 | 10281 | 0.0229 | 0.0073 | 3.12 | 0.0471 | 0.0254 |
| content | 3 | 65 | 10245 | 0.0229 | 0.0074 | 3.10 | 0.0475 | 0.0255 |
| content | 6 | 65 | 10150 | 0.0225 | 0.0071 | 3.18 | 0.0450 | 0.0210 |
| content | 11 | 65 | 10004 | 0.0215 | 0.0065 | 3.29 | 0.0382 | 0.0128 |
| all | all | 65 | 23892 | 0.0752 | 0.0041 | 18.15 | 0.1152 | 0.0145 |
| all | 0 | 65 | 23697 | 0.0121 | 0.0038 | 3.21 | 0.0216 | 0.0146 |
| all | 1 | 65 | 23632 | 0.0121 | 0.0038 | 3.21 | 0.0216 | 0.0146 |
| all | 2 | 65 | 23567 | 0.0110 | 0.0036 | 3.06 | 0.0211 | 0.0118 |
| all | 3 | 65 | 23502 | 0.0110 | 0.0036 | 3.06 | 0.0212 | 0.0118 |
| all | 6 | 65 | 23307 | 0.0108 | 0.0033 | 3.28 | 0.0200 | 0.0098 |
| all | 11 | 65 | 22982 | 0.0106 | 0.0030 | 3.53 | 0.0182 | 0.0060 |

### personaplex-pos/epoch-00.jsonl

65 turns, 13 stems. Per-turn scalars, median gap / median floor: content 0.0461 / 0.0117 = 3.93; all_frames 0.0283 / 0.0065 = 4.37.

By position, content frames:

| bucket | turns | frames | median gap | median floor | ratio | median door | gap share | floor share |
|---|---|---|---|---|---|---|---|---|
| lead | 0 | 0 | - | - | - | - | 0% | 0% |
| 0 | 65 | 65 | 0.1486 | 0.0018 | 81.80 | 0.0019 | 2% | 0% |
| 1 | 65 | 65 | 0.5996 | 0.0394 | 15.21 | 0.0286 | 9% | 2% |
| 2 | 36 | 36 | 0.1691 | 0.0122 | 13.90 | 0.0081 | 1% | 0% |
| 3-5 | 48 | 95 | 0.1513 | 0.0388 | 3.90 | 0.0355 | 4% | 4% |
| 6-10 | 54 | 146 | 0.1281 | 0.0371 | 3.45 | 0.0277 | 6% | 9% |
| 11-25 | 65 | 541 | 0.1110 | 0.0329 | 3.37 | 0.0269 | 17% | 24% |
| 26+ | 65 | 9463 | 0.0290 | 0.0076 | 3.82 | 0.0057 | 61% | 60% |

By position, all frames:

| bucket | turns | frames | median gap | median floor | ratio | median door | gap share | floor share |
|---|---|---|---|---|---|---|---|---|
| lead | 65 | 195 | 0.1244 | 0.0032 | 38.90 | 0.0045 | 12% | 1% |
| 0 | 65 | 65 | 0.1486 | 0.0018 | 81.80 | 0.0019 | 1% | 0% |
| 1 | 65 | 65 | 0.5996 | 0.0394 | 15.21 | 0.0286 | 6% | 2% |
| 2 | 65 | 65 | 0.2476 | 0.0238 | 10.39 | 0.0248 | 3% | 2% |
| 3-5 | 65 | 195 | 0.1209 | 0.0203 | 5.94 | 0.0205 | 4% | 4% |
| 6-10 | 65 | 325 | 0.0750 | 0.0170 | 4.42 | 0.0159 | 5% | 9% |
| 11-25 | 65 | 975 | 0.0811 | 0.0218 | 3.72 | 0.0192 | 14% | 22% |
| 26+ | 65 | 22007 | 0.0171 | 0.0044 | 3.91 | 0.0035 | 54% | 60% |

Bar from position:

| mode | from position | turns | frames | median gap | median floor | ratio | p90 gap | p90 floor |
|---|---|---|---|---|---|---|---|---|
| content | all | 65 | 10411 | 0.0461 | 0.0117 | 3.93 | 0.0987 | 0.0252 |
| content | 0 | 65 | 10411 | 0.0461 | 0.0117 | 3.93 | 0.0987 | 0.0252 |
| content | 1 | 65 | 10346 | 0.0451 | 0.0118 | 3.82 | 0.0950 | 0.0254 |
| content | 2 | 65 | 10281 | 0.0406 | 0.0114 | 3.56 | 0.0893 | 0.0246 |
| content | 3 | 65 | 10245 | 0.0406 | 0.0114 | 3.55 | 0.0893 | 0.0246 |
| content | 6 | 65 | 10150 | 0.0379 | 0.0114 | 3.31 | 0.0843 | 0.0241 |
| content | 11 | 65 | 10004 | 0.0346 | 0.0098 | 3.54 | 0.0816 | 0.0226 |
| all | all | 65 | 23892 | 0.0283 | 0.0065 | 4.37 | 0.0662 | 0.0137 |
| all | 0 | 65 | 23697 | 0.0276 | 0.0065 | 4.24 | 0.0523 | 0.0138 |
| all | 1 | 65 | 23632 | 0.0268 | 0.0065 | 4.11 | 0.0516 | 0.0138 |
| all | 2 | 65 | 23567 | 0.0255 | 0.0065 | 3.94 | 0.0482 | 0.0138 |
| all | 3 | 65 | 23502 | 0.0248 | 0.0064 | 3.85 | 0.0464 | 0.0137 |
| all | 6 | 65 | 23307 | 0.0229 | 0.0065 | 3.54 | 0.0442 | 0.0135 |
| all | 11 | 65 | 22982 | 0.0203 | 0.0053 | 3.84 | 0.0425 | 0.0126 |

