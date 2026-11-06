# Consistency 2: the λ = 3 pilot, with the baseline's epoch 1 decoded

Vlad, 2026-11-06, at B11's GATE ("Weight 10 has no healthy epoch: fragments at epoch 1, silent opening at epoch 2. Which lever next?"): **"Weight 3 + baseline epoch 1 decode (Recommended)"**. This batch (B12) trains `cons1-pp-l3`, identical to B9's `cons1-pp-l10` except λ = 3, measures all three epochs, decodes epochs 2 and 3 side by side, and decodes the λ = 0 baseline's own epoch 1. It changes no training code.

## The answer

**No λ = 3 epoch is both healthy and better on the bar. Epoch 3 comes closest:**
- its lesson opening is back to the baseline's;
- it has no fragments;
- its ratio is 1.87 against the seeds' ~3.0 at the same epoch.

**But it slips on following the pupil's turn as λ = 10's epoch 2 did:** 94/104 against 101/104, McNemar p ≈ 0.09. The bar's pre-registered rule calls λ = 3 "not moved" (R_gap 2.51 at epoch 1, −0.48 against a threshold of 0.5), and no checkpoint passes the bar. **And the baseline's epoch 1 changes the reading of B10 and B11.** The λ = 0 model is itself silent after the spoken opening question at epoch 1 (11/13 A, 10/13 B), and it fragments mid-lesson (stops 16/9/8 of 52). So B11's fragments and B10's silent opening are both stages the baseline passes through. The term did not create them. λ = 10 kept the opening in that stage, and λ = 3 lets it out by epoch 3.

- **Baseline epoch 1 (λ = 0).** It is silent the whole answer window after the spoken first question in 11/13 (A) and 10/13 (B), while the typed opening starts on frame 3 as usual. Mid-lesson it stops mid-sentence in 16/9/8 of 52, against 1/2/0 at epoch 2 and 20/11/17 for λ10 epoch 1. It still listens: the swap is followed 100/104.
  - So most of λ10 epoch 1's fragments are under-training. The term adds some on top, most after B (17 against 8).
  - λ10 epoch 1 actually spoke at the opening where the baseline's epoch 1 did not.
- **λ = 3 epoch 2.**
  - The spoken opening is taken up in 11/13 (A) and 12/13 (B), and the answers are not fragmented (stops 2/3/3).
  - The swap is followed 100/104, like the baseline.
  - **But the typed opening starts late,** on frame 7-8 in 11/13 instead of the tape's frame 3. It takes the question up in only 9/13 and makes the tape's move in 5/13 (baseline 13/13 and 13/13).
  - Bar: R 1.95.
- **λ = 3 epoch 3.**
  - The opening is the baseline's. Every opening is spoken. Uptake is 13/12/13 (typed/A/B), the typed opening makes the tape's move in 12/13, and the onsets are frame 3 typed and 2 spoken, exactly the baseline's.
  - There are no fragments (stops 0/1/0).
  - **But it follows the pupil's turn less:** 94/104 against 101/104 (paired McNemar 10 vs 3, p ≈ 0.09). The 10 misses are fluent full answers that affirm what the pupil did not say: "right, minus two" after "up, but barely"; "yes, two" after "one".
  - Its mid-lesson spoken uptake is 48/52 after A (baseline 51).
  - There is no substance gain: typed vs A differs in 18/52 mid-lesson, against 14/52 for the baseline.
  - Bar: R 1.87, the best ratio of any λ = 3 epoch.
- **The turn-0 lead-frame gap tracked the opening in every run.** In B10 and B11 a low turn-0 lead-frame KL(typed‖A) marked the silent opening. λ = 3 goes 0.062, 0.25 and 0.70 over epochs 1-3 (baseline epoch 2 0.74), and the opening is spoken at epochs 2 and 3. λ = 10 goes the other way, 0.54, 0.095 and 0.061.

## Bar and behaviour, side by side

Behaviour comes from this batch's blind pool: 26 opus judges over all six checkpoints. Text counts come from the answer stream. "Stops" means stops mid-sentence with ≥ 20% of the window left (B11's rule). Turns 2-8 are mid-lesson, 52 answers per rendering.

| | baseline e1 | baseline e2 | λ10 e1 | λ10 e2 | λ3 e2 | λ3 e3 |
|---|---|---|---|---|---|---|
| content R (bar) | 3.04 | 3.24 | **1.90** | **1.67** | 1.95 | 1.87 |
| median gap / floor | 0.0177 / 0.0058 | 0.0251 / 0.0077 | 0.0061 / 0.0032 | 0.0061 / 0.0037 | 0.0106 / 0.0054 | 0.0146 / 0.0078 |
| held text loss | 0.5511 | 0.5397 | 0.5933 | 0.5449 | 0.5366 | 0.5775 (base e3 0.6105) |
| turn-0 lead KL(typed‖A) | - | 0.74 | 0.54 | 0.095 | 0.25 | 0.70 |
| speaks after spoken opening, A / B (of 13) | **2 / 3** | 13 / 13 | 13 / 13 | **0 / 3** | 13 / 13 | 13 / 13 |
| opening uptake, typed / A / B (of 13) | 11 / 2 / 3 | 13 / 12 / 11 | 8 / 10 / 10 | 9 / 0 / 2 | 9 / 11 / 12 | 13 / 12 / 13 |
| typed opening makes the tape's move (of 13) | 7 | 13 | **2** | 5 | **5** | 12 |
| typed-opening onset frame (tape 3) | 3 (11/13) | 3 | 3-5 | **8-9** | **7-8** | 3 |
| mid-lesson uptake, typed / A / B | 52 / 51 / 51 | 52 / 51 / 50 | 51 / 50 / 51 | 51 / 50 / 48 | 52 / 51 / 50 | 51 / 48 / 51 |
| mid-lesson stops, typed / A / B | **16 / 9 / 8** | 1 / 2 / 0 | **20 / 11 / 17** | 0 / 1 / 1 | 2 / 3 / 3 | 0 / 1 / 0 |
| mid-lesson median words, typed / A / B | 57 / 64 / 58.5 | 64.5 / 66 / 67 | 41 / 57 / 38 | 62 / 61 / 61 | 65 / 64 / 62 | 63.5 / 60.5 / 64 |
| swap: played turn taken up (of 104) | 100 | 101 | 95 | 94 | 100 | **94** |
| mid-lesson substance differs, typed vs A / A vs B | 15 / 19 | 14 / 20 | 10 / 13 | 15 / 17 | 16 / 23 | 18 / 19 |

- Baseline bar columns: `ladder-personaplex-1`'s own measure (`dual.noise`), at epochs 1 and 2. Its lead-frame KL at epoch 2 is from B8's per-frame rerun. The epoch-1 measure stores no per-frame KLs, so that cell is empty.
- Speaking after the opening is a text count: the answer window has a non-`<pad>` text token. The baseline e1's spoken openings that "speak" are 2/13 and 3/13.
- In this pool the baseline e2 swap is 101/104; B10's and B11's judges gave 103/104.

**The bar.** `make noise VARIANTS=personaplex-cons1-pp-l3`: R0 3.00, threshold 0.50.

| run | R (epoch) | R_gap (epoch) | median gap | median floor | p90 ratio | R_gap − R0 | 95% CI vs seed 0 | 95% CI vs seed 1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| cons1-pp-l3 | 1.87 (3) | 2.51 (1) | 0.009 | 0.004 | 1.42 | −0.48 | −0.52 [−1.19, −0.34] | −0.44 [−1.27, −0.11] | **not moved** |
| cons1-pp-l10 | 1.67 (2) | 1.90 (1) | 0.006 | 0.003 | 1.28 | −1.09 | −1.13 [−1.50, −0.65] | −1.05 [−1.83, −0.44] | down |

**The rule's verdict for λ = 3 is "not moved".** The rule scores R_gap at the lowest-gap epoch, which for λ = 3 is epoch 1. There it misses the size threshold by 0.02, though both CIs exclude 0.

Scored at the epoch that matters for behaviour, the drop is real, but it is not a pass. These are paired stem-bootstrap tests outside the pre-registered rule:
- **Epoch 2:** R 1.95, −1.09 and −1.01 against the seeds' epoch 1; −1.30 and −1.13 against the seeds' epoch 2. Every CI excludes 0.
- **Epoch 3:** R 1.87, −1.17 and −1.09 against the seeds' epoch 1; −1.01 [−1.47, −0.41] and −1.28 [−1.71, −0.70] against the seeds' epoch 3.
- **Not a collapsed floor.** Epoch 3's floor is 0.0078, the seeds' level of 0.006-0.008, and its gap is 0.0146, below the seeds' 0.018.
- In the answer body (positions 26+), the gap is 0.0102 against a floor of 0.0047: ratio 2.18, against 3.44 for the baseline epoch 2.
- **The tape ceiling holds.** At epoch 3, typed‖tape is 0.0156 and tape‖A 0.0130. The ratio cannot go much below the ~1.8 that tape‖A over the floor allows. No checkpoint passes the bar: the ratio is 1.9-2.5 and the p90 ratio 1.42.

**Held text and the term.**
- Held text loss is 0.5574, 0.5366 and 0.5775 over epochs 1-3, against the baseline's 0.5511, 0.5397 and 0.6105: +1.1%, −0.6% and −5.4%. λ = 10 was +7.7% at epoch 1.
- Mean `cons` is 0.0099, 0.0071 and 0.0077 (λ10 0.0081, 0.0050, 0.0052). Per-turn logging of the term is not supported, so the opening's share is unknown.

## Examples (truncated)

**λ3 epoch 3, swap misses (true rows).** The answer affirms a number the pupil never said:
- The pupil says "up, but barely, since the dot sits in the bottom of the dip." The answer: "right, minus two, though the bottom of the hump is where the height comes to zero..."
- The pupil says "one, since the slope doubles x and x is zero point five." The answer: "yes, two, and the numbers can settle it..."

The baseline answers both items with the pupil's own claim. The full set is in `consistency-2/sbs-report.md`, and the labels in `judge-labels.jsonl`.

**Baseline epoch 1 at the opening** (dlg-00005-038, "hello, so what does a derivative actually tell you about a curve?"): after A and after B, silence. After typed it gives a full answer starting on frame 3.

## Verdict

λ = 3 does what λ = 10 could not: it reaches an epoch (3) whose opening, fragments and held loss look like the baseline's, at a ratio of 1.87. But the same epoch slips on following the pupil's turn (94/104, p ≈ 0.09 in this pool). That is the failure λ10 epoch 2 showed mid-lesson. The other λ = 3 epoch is no cleaner. Epoch 2 follows the pupil like the baseline, but its typed opening starts ~0.4 s late and makes the tape's move in 5/13. So no epoch is healthy on every check, and none passes the bar.

The baseline's epoch 1 adds two things:
- The early-training state is silent at the spoken opening and fragmented mid-lesson even with no term. What the term changes fits a change in how fast, and in which order, training leaves that state, rather than a new failure. One seed per λ cannot show more.
- Its swap tracking at epoch 1 (100/104) says the pupil-tracking slip is not an early-training effect. It appears only in runs with the term: λ10 epochs 1-2 and λ3 epoch 3. It is not in λ3 epoch 2 (100/104).

## What was run

- **Code:** `scripts/ladder.sh` `LADDER=4` (one run, `cons1-pp-l3`, `--consistency 3`) with two tests. 288 tests pass. No training code changed.
- **Jobs**, one at a time. drama3 was stopped and the watchdog ran for the whole GPU span. About 7.3 GPU-hours:
  - `job-20261106-132449-vlad-sbs-base-e1`: `sbs2:72868db`, `--run ladder-personaplex-1 --epoch 1 --no-bare --swap-turns 2,6`. 2,894 s, peak 38.9 GiB, adapter norm 26.83.
  - `job-20261106-132730-vlad-cons1-pp-l3`: `cons1:9690aa5`, 3 × ~4,300 s, peak 56.0 GiB. The startup line matches the λ10 pilot: 287 steps/epoch, 227 typed rows, 585 pairs, 94,464 content frames.
  - `job-20261106-175400-vlad-measure-cons1-pp-l3`: `cons1:9690aa5`.
  - `job-20261106-175416-vlad-sbs-l3-e2`, adapter norm 27.27.
  - `job-20261106-191130-vlad-sbs-l3-e3`, adapter norm 27.68.
  - All decodes ran in B7/B10's batch shapes (53/35/29/26/23/21/8 true, 28/24 swap) with B10's donor map.
- **Judging:** B10's rubrics verbatim. 26 blind opus judges over 780 pairs, 1,170 answers and 624 swap items; six checkpoints in one shuffled pool, seed 13 (scratchpad `judge4/blind4.py`, `blind3.py` generalised). On the 1,287 items shared with B11 they agree with B11's labels:
  - substance 352/390;
  - uptake 570/585 (well formed 512/585);
  - same move as the tape 561/585;
  - swap 298/312.
- **Files in `consistency-2/`:**
  - `answers.jsonl`: this batch's 741 rows (`base-e1`, `l3-e2`, `l3-e3`);
  - `meta-sbs-*.json`;
  - `judge-labels.jsonl`: all six checkpoints, unblinded;
  - `judge-prompts.txt`;
  - `sbs-report.md`: the six checkpoints side by side, with the quality tables.

## Caveats

- **One seed per λ, 13 holdout dialogues.** The turn-0 cells have 13 items each. The swap test has 104 items, and p ≈ 0.09 is not significant on its own. It matches λ10 epoch 2's slip in size and kind.
- **Greedy, bf16, one path per row.** B7's control: 6/30 rows change substance under another batch shape. The batch shapes here are identical across checkpoints.
- **The R_gap rule picks the lowest-gap epoch.** For λ = 3 that is the least-trained epoch. The per-epoch CIs above are supplementary and were not pre-registered.
- **Mid-lesson spoken onset stays at frame 0-2 in every checkpoint.** This is the TTS artefact B7 traced.

## The gate

- **(a) A second seed of λ = 3, decoded at epochs 2 and 3** (~5.5 GPU-hours). This tests whether epoch 3's swap slip (p ≈ 0.09) and epoch 2's late typed opening are real. Recommended if the consistency line continues: every λ = 3 failure is one-seed and small-n.
- **(b) The term without turn 0, at λ = 3** (a small train-side change, ~5.5 GPU-hours). This targets epoch 2's late typed opening. It does not address the swap slip.
- **(c) λ = 1-2** (~5.5 GPU-hours). This trades bar for listening. At λ = 3 the ratio already sits near the tape ceiling, so a lower λ is likely to land above R_gap 2.5, which is "not moved" (a guess, not measured).
- **(d) Voice-D TTS teacher rows** (needs a plan, ~6.5 GPU-hours plus render). This is the only lever on the tape ceiling (~1.8), which caps every run here. It should start from a healthy λ, and none is confirmed yet.
- **(e) Stop the consistency line** and take the baseline to the engine. λ = 3 epoch 3 is not a clear improvement in behaviour, and no run passes the bar.
