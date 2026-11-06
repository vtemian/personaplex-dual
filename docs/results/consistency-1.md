# Consistency 1: a typed-to-spoken consistency term, one pilot

Vlad, 2026-11-05, at the bar-by-position gate: "Add a consistency loss (Recommended). Keep the content bar. Plan a distillation/consistency loss: typed rendering trained toward the frozen model's distribution after the spoken rendering. Needs a /plan-it style plan first, then training."

The plan is `docs/plans/2026-11-05-consistency-loss.md` (B9). The pilot is the ladder recipe:
- PersonaPlex, 117 dialogues, 3 epochs, seed 0, rank 16, LR 2e-5, the same batches;
- plus λ = 10 · KL(typed answer ‖ sg(the same model's answer after the all-tape spoken row)), over the content frames of the answer right after each typed pupil turn.

## The answer

**The term moved the content ratio from ~3.0 to 1.90 (1.67 at its best epoch), to about the tape ceiling the plan predicted. It is the first change since ladder 1 (data, rank, LR, seed) that moved it at all. The bar still does not pass. Before building on it, one question needs answering: is the answer now less sensitive to the pupil's voice, or less sensitive to the pupil turn as a whole?**

- **The noise rule says "down".**
  - R_gap is **1.90** (epoch 1) against R0 = 3.00. R_gap − R0 = **−1.09**, against a threshold of T = 0.5.
  - Both paired stem-bootstrap 95% intervals exclude 0: **[−1.50, −0.65]** against seed 0 and **[−1.83, −0.44]** against seed 1.
  - It is a real gap drop, not floor inflation. The median gap is **0.0061**, against 0.018 for both seeds.
  - The best epoch, R, is **1.67** (epoch 2), where the median gap ties epoch 1's.
- **The gap reached the tape ceiling (D2).**
  - D2's ceiling: with the tape as teacher, a perfect match leaves typed‖A ≈ tape‖A.
  - At epoch 2 the gap, typed‖A, is **0.0061**, against tape‖A at **0.0069**. Seed 0 was at 0.025 against 0.014.
  - The per-turn median of gap / tape‖A is 1.03 at epoch 2, against 1.74 for seed 0.
  - As ratios to the floor, tape‖A / A‖B is 1.67 (epoch 1) and 1.88 (epoch 2), against the predicted ~1.8.
  - typed‖tape is still 0.0067, so typed has not converged on the tape. It sits about as far from the tape as from A.
  - It is "about" the ceiling, not a bound. KL is not a metric: typed‖A is already below tape‖A on 32 of 65 turns at epoch 2 (seed 0: 14). The ceiling itself also moved, as the plan's appendix warned. tape‖A fell to 0.50× of seed 0's per turn, and is lower on 60 of 65 turns.
- **The bar still does not pass.** The median gap is 1.7 to 1.9× the floor. p90 gap / p90 floor is 1.28 (epoch 1) and 1.32 (epoch 2), against 1.68 and 2.01 for the seeds.
- **The "down" does not depend on how the medians are taken.** The median of per-turn ratios is 1.96, against 2.97 and 2.87 for the seeds.
- **The cost to the answer is small and early.**
  - Held text loss is **+7.7%** at epoch 1 (0.593 against 0.551). It is **+1.0%** at epoch 2 (0.545 against 0.540).
  - At epoch 3 it is **10.6% better** than the baseline (0.546 against 0.611). The baseline overfits at epoch 3; this run does not, or not yet.
- **Every rendering-to-rendering KL shrank, not only the gap.**
  - Per-turn medians at epoch 2, against seed 0: gap ×0.28, typed‖tape ×0.26, floor (A‖B) ×0.36, door (C‖A) ×0.38, tape‖A ×0.50.
  - The floor fell on 64 of 65 turns, and the gap on 65 of 65. The shrink is even across turn index.
  - It grows with epochs. The floor is −45% at epoch 1 and −52 to −57% at epoch 2. The door is −44 to −51%, then −64 to −70%.
  - The ratio fell even though its denominator shrank (gap ×0.28 against floor ×0.36), so this is not a floor artifact.
  - But a uniform shrink of every KL is what a model whose answer is less sensitive to the pupil audio as a whole would show. It is also what the term should do if the answer now depends on what the pupil said rather than how it sounded.
  - The measure cannot tell these apart: every pair compares renderings of the same content.
- **The term reordered which turns are hard.** The Spearman correlation of per-turn gap between this run and seed 0 is 0.64, against 0.87 between the two seeds (floor 0.75 against 0.88).
- **Cost:** 4,300 s per epoch against 3,220 (+33%). Peak memory 56.0 GiB against 49.4, under the 64 GiB cap.

## The verdict (`make noise VARIANTS=personaplex-cons1-pp-l10`, content frames, epochs 1-3)

Seed noise (R_gap): d = 0.08, R0 = 3.00, T = max(2d, 0.5) = 0.50.

| run | R (epoch) | R_gap (epoch) | median gap | median floor | p90 ratio | R_gap - R0 | 95% CI vs personaplex-ladder-personaplex-1 | 95% CI vs personaplex-ladder2-pp-s1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| personaplex-ladder-personaplex-1 | 2.87 (3) | 3.04 (1) | 0.018 | 0.006 | 2.01 | +0.04 | - | +0.08 [-0.63, +0.61] | - |
| personaplex-ladder2-pp-s1 | 2.96 (1) | 2.96 (1) | 0.018 | 0.006 | 1.68 | -0.04 | -0.08 [-0.61, +0.63] | - | - |
| personaplex-cons1-pp-l10 | 1.67 (2) | 1.90 (1) | 0.006 | 0.003 | 1.28 | -1.09 | -1.13 [-1.50, -0.65] | -1.05 [-1.83, -0.44] | down |

The rule is ladder 2's (B6.0.2), fixed before this run. A run moved iff |R_gap − R0| ≥ T and both paired intervals exclude 0 with that sign. A "down" also needs its gap below the seeds' mean gap.

## Per epoch (content frames)

| run | epoch | median gap | median floor | ratio | p90 gap | p90 floor | typed‖tape | tape‖A | typed‖tape / tape‖A | door | held text | mean cons (train) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cons1-pp-l10 | 1 | 0.0061 | 0.0032 | 1.90 | 0.0113 | 0.0088 | 0.0068 | 0.0053 | 1.29 | 0.0029 | 0.5933 | 0.0081 |
| | 2 | 0.0061 | 0.0037 | 1.67 | 0.0121 | 0.0091 | 0.0067 | 0.0069 | 0.97 | 0.0027 | 0.5449 | 0.0050 |
| | 3 | 0.0101 | 0.0047 | 2.15 | 0.0198 | 0.0168 | 0.0080 | 0.0101 | 0.79 | 0.0042 | 0.5455 | 0.0052 |
| ladder-personaplex-1 (seed 0) | 1 | 0.0177 | 0.0058 | 3.04 | 0.0333 | 0.0165 | 0.0165 | 0.0105 | 1.58 | 0.0052 | 0.5511 | - |
| | 2 | 0.0251 | 0.0077 | 3.24 | 0.0499 | 0.0316 | 0.0290 | 0.0139 | 2.09 | 0.0076 | 0.5397 | - |
| | 3 | 0.0414 | 0.0144 | 2.87 | 0.0692 | 0.0358 | 0.0424 | 0.0224 | 1.89 | 0.0133 | 0.6105 | - |
| ladder2-pp-s1 (seed 1) | 1 | 0.0182 | 0.0061 | 2.98 | 0.0343 | 0.0204 | 0.0164 | 0.0106 | 1.55 | 0.0059 | - | - |
| | 2 | 0.0261 | 0.0085 | 3.07 | 0.0430 | 0.0254 | 0.0278 | 0.0178 | 1.56 | 0.0089 | - | - |
| | 3 | 0.0455 | 0.0145 | 3.14 | 0.0698 | 0.0357 | 0.0473 | 0.0272 | 1.74 | 0.0130 | - | - |

typed‖tape and tape‖A are the measure's `tape_gap_median` and `tape_floor_median`. The gap and floor columns are KL(typed‖A) and KL(A‖B). "mean cons" is the training-side term: its teacher is an all-tape row and a typed training row types about half its pupil turns, so it is not the bar's quantity. It fell from 0.0081 to ~0.005 and then stayed flat.

All three epochs of this run are below every seed epoch 1-3 in both median gap and ratio. Epoch 3 drifts back up, gap 0.0101, but stays far below the seeds' 0.041 to 0.046 at epoch 3.

## Where in the answer (`make position`, content frames, ratio by bucket)

| position | cons1 epoch 1 | cons1 epoch 2 | B8: ladder-personaplex-1 epoch 2 |
|---|---|---|---|
| 0 | 11.2 | 5.3 | 3.7 |
| 1 | 2.41 | 2.10 | 2.5 |
| 2 | 2.29 | 1.14 | 1.3 |
| 3-5 | 1.42 | 1.82 | 2.0 |
| 6-10 | 1.57 | 1.50 | 1.8 |
| 11-25 | 2.24 | 2.65 | 3.4 |
| 26+ | 2.02 | 2.00 | 3.4 |
| content bar | 1.90 | 1.67 | 3.30 (the B8 `-pos` pass; 3.24 in ladder 1's own pass) |

- **The drop is in the body of the answer, where B8 put the gap.** At 26+, 91% of content frames, the ratio went from 3.4 to 2.0. The gap there is 0.0035 / 0.0034, against a floor of 0.0018 / 0.0017.
- **Position 0, the first word, gets worse every epoch.**
  - Its gap grows from 0.012 to 0.031 to 0.28 over epochs 1-3. Its share of the gap mass grows from 3% to 7% to 18%. At epoch 3 its ratio is 35.6.
  - At epoch 1 its floor is also very low (0.0010); at epoch 2 it is normal (0.0059).
  - Epoch 3's drift back up is partly here: the answer's onset is diverging between typed and spoken. This matters for taking a checkpoint to the engine.
- **The lead frames are unchanged in kind.** The all-frame ratio is still dominated by the 3 lead PAD frames (79 to 85% of the all-frame gap mass), the TTS-silence artifact B7/B8 traced. The content bar skips them.

## Caveats

- **One seed.** The intervals exclude 0 by a wide margin, but T and d come from the baseline seeds. Seed noise under the term is not measured.
- **The decision epoch is chosen on the holdout it is scored on**, as in ladder 2. Epochs 1 and 2 have the same median gap to 4 decimals, so the choice does not matter here.
- **Held text loss at the R_gap epoch is +7.7%.**
  - Under the plan's pre-registered GATE-1 list, that triggers option (e), "the term hurts the answer". Seed-to-seed spread in held text is about 1%, so +7.7% is real.
  - Epoch 2 has the same median gap, with held text +1.0%. By epoch 3 the run is 10.6% better than the baseline.
  - I read it as a one-epoch cost. But (e) fires under the rule as written.
- **Less sensitive to the voice, or to the pupil?**
  - Every KL shrank, uniformly, and more with each epoch (above).
  - Held text loss (teacher-forced, after the tape) is near the baseline from epoch 2, so the tutor's text is still predicted as well as before.
  - The jsonl cannot settle this: no pair contrasts different pupil content.
  - B7's side-by-side job is the check: blind judges on greedy answers, with the "answers the pupil's content" count that was 61-65/65 before. It was not run here. Anything built on this result should wait for it.
- **The ceiling is the tape's.** The teacher is the recorded tape, the only spoken rendering of training stems. tape‖A is now about equal to typed‖tape. Closing the rest of the way toward the A‖B floor needs a teacher in the floor's own voices (TTS-voiced training rows).

## What was run

- **Code:** `main` 9690aa5 (B9 C1-C3: `dual.consistency`, `kl.consistency_loss`, the train flags `--consistency`, `--consistency-dir`, `--only-stem`, `--max-steps`, `--widest-first`, ladder 3, `make noise VARIANTS=`). 269 tests.
- **Image:** `spark.local:5000/vlad/cons1:9690aa5`.
- **GATE-0 smokes:**
  - `cons-smoke-1` (job-20261105-160646): the longest stem, 3 steps. Peak 39.2 GiB, cons 0.041 / 0.036, teacher 32% of step time.
  - `cons-smoke-wide` (job-20261105-161117): the 4 widest batches. Peak 50.1 GiB allocated / 51.4 reserved, cons 0.04-0.08, teacher 8.8 s of 26.7 s (33%), 287 steps/epoch (= baseline), 12 dynamo graphs (no recompile loop). Passed on the plan's criteria.
- **Pilot:**
  - Train `cons1-pp-l10` (job-20261105-161657): 3 × ~4,300 s, peak 56.0 GiB.
  - Measure `measure-cons1-pp-l10` (job-20261105-221501): 4,506 s.
  - drama3 was stopped throughout (watchdog on).
  - The Mac lost ssh to the box from 19:23 to 22:13 on 11-05 (the box did not reboot). The train ran through it, and ladder.sh picked up again.
- **Peak memory note:** the pilot's 56.0 GiB is above the wide smoke's 50.1. The heaviest step is the batch with the most typed answer frames, not the widest one, as the opus review predicted.

## The gate (GATE-1): the consistency pilot moved the ratio by −1.09 (3.00 → 1.90, R 1.67); what next?

Under the plan's pre-registered list, two options fire:
- **(b)** "moved down to near the tape ceiling (~1.8-2.2)": R_gap is 1.90.
- **(e)** "held text up > 5% at the R_gap epoch": +7.7% at epoch 1. I read it as a one-epoch cost, because epoch 2 has the same gap at +1.0%.

The plan's (c), "not moved, cons falling", and (d), "not moved, cons flat", do not apply.

Options:
- **(b) Recommended, in two steps:**
  1. **First, a precondition (~1 GPU-hour, no training):** run the B7 side-by-side on `cons1-pp-l10` epoch 2. Confirm the tutor still answers the pupil's content (61-65/65 before) and that the shrink of every KL is "voice-insensitive", not "pupil-insensitive".
  2. **If it holds:** render A-voiced spoken training rows (render-2, the TTS on the 117 training stems) and distil toward them, in the floor's own voices. This needs a new render job and its own plan.
- **(a) Confirm and push:** a second seed of λ = 10, plus λ = 30. 2 runs, ~10 GPU-hours. The gap already sits near the tape ceiling, so more λ probably buys little beyond confidence.
- **(e) Lower λ (3):** if the epoch-1 held-loss cost matters more than its recovery by epoch 2.
- **(f) Change the yardstick** (B8's option 3): floor = tape‖A. On it, epoch 2 is at 0.89 at the median (gap 0.0061 against tape‖A 0.0069).
- **(g) Take epoch 2 of this run to the engine** as the best model so far, after the side-by-side check. Not epoch 3: its first-word gap is diverging.
