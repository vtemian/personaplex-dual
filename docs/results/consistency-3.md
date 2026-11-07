# Consistency 3: control decodes, the baseline's epoch 3 and λ3's epoch 1

Vlad, 2026-11-07, at B12's GATE ("Weight-3 epoch 3 looks close to healthy at ratio 1.87, with one unverified flaw. What next?"): **"Control decodes first (Recommended)"**, "Decode the baseline's epoch 3 and weight-3 epoch 1. About 1.6 GPU-hours, no training, no code. Shows whether the 94-of-104 slip is the loss or a normal epoch-3 effect, so epoch 3 can be judged fairly." This batch (B13) ran exactly those two side-by-side decodes and judged all eight decoded checkpoints in one fresh blind pool. No training, no render, no code.

## The answer

**λ3 epoch 3 cannot be told apart from the λ = 0 checkpoint at the same epoch, on 13 dialogues and 104 swap items, at bar R 1.87 (baseline epoch 3: 2.87).** It is the best λ3 candidate. It still fails the bar, and both epoch-3 checkpoints follow the pupil less than epoch 2 does.

- **About half of the swap slip is epoch 3; the term's share cannot be separated.** The baseline's own epoch 3 slips too: it takes up the turn actually played in 95/104, against 100 at its epoch 2 and 101 at its epoch 1. λ3 epoch 3 is at 91/104. Paired on the same 104 items, that is 8 against 4 discordant items (exact McNemar p 0.39). 5 of the baseline's 9 misses are also λ3's, and most are the same kind: fluent "right, X" affirmations of a claim the pupil did not make (all 13 of λ3's misses, 7 of the baseline's 9).
  - λ3 trails λ = 0 by a few items at every epoch: 2, 4 and 4 at e1-e3 (discordant 3 vs 1, 6 vs 2, 8 vs 4). Pooled, that is 17 vs 7, sign-test p 0.064. This is post hoc, and the epochs share items, so it is not an effect. It is a reason not to call the term's share zero.
- **At λ = 3, the epoch-1 breakage is the baseline's own.** λ3 epoch 1 looks like the baseline's epoch 1. It is silent after the spoken opening in 12/13 (A) and 9/13 (B), against 11/13 and 10/13. Mid-lesson it stops mid-sentence 8/14/11 times, against 16/9/8 (33 stops each), though it has more short answers after A and B: ≤ 10 words in 6/6/5, against 6/3/1. Swap is 99 against 101. Its turn-0 lead-frame KL of 0.062 predicted the silent opening, and that prediction holds.
  - One difference is the term's: λ3 epoch 1's **typed** opening is worse. It is silent in 4/13, starts on frames 4-12 instead of 3, and takes the question up in 6/13, against 1/13 silent, frame 3 and 11/13 for the baseline.
- **Per epoch, against λ = 0 at the same epoch:**
  - **e1:** neither is healthy.
  - **e2:** λ3 is not healthy. Its typed opening starts on frame 7-8 in 11/13 and makes the tape's move in 7/13, against frame 3 and 12/13 for the baseline.
  - **e3:** no difference is detectable. Most judged opening and uptake counts are at or a little below the baseline's, and substance and well-formedness are a little better.

## All eight checkpoints

Behaviour comes from this batch's blind pool: 35 opus judges over all eight checkpoints. Text counts come from the answer stream.
- **Stops:** stops mid-sentence with ≥ 20% of the window left (B11's rule).
- **Mid-lesson:** turns 2-8, 52 answers per rendering. **Opening:** turn 0, 13 per rendering.

| | base e1 | base e2 | base e3 | λ10 e1 | λ10 e2 | λ3 e1 | λ3 e2 | λ3 e3 |
|---|---|---|---|---|---|---|---|---|
| content R (bar) | 3.04 | 3.24 | 2.87 | 1.90 | 1.67 | 2.51 | 1.95 | **1.87** |
| median gap / floor | 0.0177 / 0.0058 | 0.0251 / 0.0077 | 0.041 / 0.014 | 0.0061 / 0.0032 | 0.0061 / 0.0037 | 0.0094 / 0.0037 | 0.0106 / 0.0054 | 0.0146 / 0.0078 |
| held text loss | 0.5511 | 0.5397 | 0.6105 | 0.5933 | 0.5449 | 0.5574 | 0.5366 | 0.5775 |
| turn-0 lead KL(typed‖A) | - | 0.74 | - | 0.54 | 0.095 | 0.062 | 0.25 | 0.70 |
| speaks after spoken opening, A / B (of 13) | **2 / 3** | 13 / 13 | 13 / 13 | 13 / 13 | **0 / 3** | **1 / 4** | 13 / 13 | 13 / 13 |
| opening uptake, typed / A / B | 11 / 2 / 3 | 13 / 12 / 11 | 13 / 13 / 13 | 7 / 11 / 10 | 9 / 0 / 1 | 6 / 1 / 4 | 10 / 11 / 12 | 12 / 13 / 13 |
| opening makes the tape's move, typed / A / B | 9 / 0 / 0 | 12 / 8 / 7 | 11 / 12 / 9 | 1 / 6 / 5 | 6 / 0 / 0 | 3 / 0 / 1 | 7 / 8 / 7 | 9 / 11 / 8 |
| typed-opening onset frame (tape 3) | 3 (11/13; 1 silent) | 3 (13/13) | 3 (13/13) | 3-5 | **8-9** (2 silent) | **4-12** (4 silent) | **7-8** (11/13) | 3 (13/13) |
| mid-lesson uptake, typed / A / B | 52 / 51 / 51 | 52 / 52 / 50 | 52 / 48 / 48 | 51 / 50 / 50 | 50 / 49 / 48 | 51 / 49 / 48 | 52 / 50 / 48 | 51 / 48 / 47 |
| mid-lesson stops, typed / A / B | **16 / 9 / 8** | 1 / 2 / 0 | 0 / 0 / 0 | **20 / 11 / 17** | 0 / 1 / 1 | **8 / 14 / 11** | 2 / 3 / 3 | 0 / 1 / 0 |
| mid-lesson median words, typed / A / B | 57 / 64 / 58.5 | 64.5 / 66 / 67 | 63 / 65 / 61 | 41 / 57 / 38 | 62 / 61 / 61 | 62 / 56 / 57.5 | 65 / 64 / 62 | 63.5 / 60.5 / 64 |
| swap: played turn taken up (of 104) | 101 | 100 | 95 | 95 | 94 | 99 | 96 | 91 |
| mid-lesson substance differs, typed vs A / A vs B (of 52) | 13 / 15 | 14 / 19 | 19 / 17 | 8 / 11 | 14 / 15 | 6 / 8 | 16 / 22 | 17 / 15 |

- **Bar columns.**
  - The baseline is `ladder-personaplex-1` (seed 0), from its own ladder measure (`dual.noise`; epoch 3 rounded in `bar.md`).
  - λ3 comes from `measure-cons1-pp-l3` and λ10 from `measure-cons1-pp-l10`.
  - λ3 epoch 1's R is 2.51; that is the rule's R_gap epoch ("not moved", B12).
  - Lead-frame KL: base e2 is from B8's per-frame rerun; the ladder measures of base e1 and e3 store no per-frame KLs.
- **Speaking after the opening** is a text count: the answer window has a non-`<pad>` text token.
- **This pool against B12's on the six shared checkpoints:**
  - Swap counts move by up to 4: λ3 e2 100 → 96, λ3 e3 94 → 91, base e2 101 → 100, base e1 100 → 101; λ10 unchanged.
  - The typed opening's tape move moves by up to 3: λ3 e3 12 → 9, base e1 7 → 9, λ3 e2 5 → 7.
  - Substance counts move by up to 4.
  - Uptake counts move by up to 4: λ3 e3 mid-lesson B 51 → 47; others by 1-2.
  - This is judge noise. The text counts are identical to B12's. For the six old checkpoints, this pool's judged numbers supersede consistency-2's.

## (a) Is λ3 epoch 3's swap slip the term or epoch 3?

**About half epoch 3. The rest cannot be told from noise.**

Within this pool, the baseline (λ = 0) itself falls from 100/104 at epoch 2 to 95/104 at epoch 3 (9 vs 4 discordant, p 0.27), and λ3 is at 91/104 at epoch 3. The fair comparison is the one at the same epoch: 91 against 95. Paired on the 104 identical items, the baseline follows the pupil where λ3 does not on 8 items, and the reverse happens on 4. That is a gap of 4 items, exact McNemar p 0.39, with a rough 95% interval of −11 to +3 items (Wald, on the discordant pairs).

The misses are mostly the same kind. All 13 of λ3's and 7 of the baseline's 9 open with "right" or "yes". 5 of the baseline's 9 epoch-3 misses are on rows where λ3 epoch 3 also misses, for example "right, zero" and "yes, one is right" to pupils who said something else. Both are fluent affirmations of a claim the pupil never made.

Against the baseline's epoch 2, λ3 epoch 3 looks worse in this pool than in B12's (10 vs 1, p 0.012). But about half of that 9-item drop (100 → 91) also happens to the baseline from epoch 2 to epoch 3 (100 → 95).

Plain reading: epoch 3 costs about 5 items of pupil-following with or without the term. λ3's own drop from e2 to e3 (96 → 91, 9 vs 4) is the same size. The term may cost about 4 more: λ3 trails λ = 0 by 2, 4 and 4 at e1-e3, pooled 17 vs 7 (p 0.064, post hoc, shared items). But 104 items cannot tell a 4-item gap from noise. Re-judging alone moved λ3 e2 by 4 items between pools, and 19 of the 624 shared swap labels flipped.

## (b) Does λ3 epoch 1 look like the baseline's epoch 1?

**Yes, in the two failures that matter.**

- **Spoken opening.** After the spoken opening question, λ3 epoch 1 is silent the whole answer window in 12/13 (A) and 9/13 (B); the baseline's epoch 1 is silent in 11/13 and 10/13. Opening uptake is 1/13 and 4/13 against 2/13 and 3/13.
- **Mid-lesson fragments.** λ3 epoch 1 stops mid-sentence 8/14/11 times (typed/A/B), against 16/9/8: 33 stops each, distributed differently. There are 6/6/5 answers of ≤ 10 words against 6/3/1.
- **The pupil is still followed:** 99/104 against 101/104 (3 vs 1 discordant, p 0.63).

So, at λ = 3, the silent opening and the fragments at epoch 1 are a stage plain training also passes through; the term does not create them. That is one seed per λ. It does not cover λ = 10: its epoch 1 fragments more stubbily (stops 20/11/17), and its epoch 2 goes silent at the spoken opening, where the baseline speaks 13/13.

The one term-specific difference at epoch 1 is the **typed** opening:
- silent in 4/13 against 1/13;
- the others start on frames 4-12, against frame 3 in 11/13 for the baseline;
- uptake 6/13 against 11/13;
- the tape's move 3/13 against 9/13.

λ3 epoch 2 shows the same late typed opening, on frames 7-8 in 11/13. The low turn-0 lead-frame KL (0.062) predicted the silent spoken opening, as it did for λ10 epoch 2 (0.095). Across the six checkpoints where both were measured, every low lead KL (< 0.1) came with a silent spoken opening (0/3 and 1/4 of 13 spoken), and every KL ≥ 0.25 with a spoken one. That is post hoc, n = 6, and only two of the points are low, both λ ≠ 0. The baseline's silent epoch 1 has no KL measured.

## Verdict

Against the λ = 0 checkpoint at the same epoch, **λ3 epoch 3 is the only λ3 epoch with no detectable difference**, within what 13 dialogues and 104 swap items resolve:
- **Opening:** spoken after A and B in 13/13. Typed onset is on frame 3 in 13/13. Uptake is 12/13/13 against 13/13/13, and the tape's move 9/11/8 against 11/12/9.
- **Fragments:** none (stops 0/1/0 against 0/0/0).
- **Mid-lesson:** uptake 51/48/47 against 52/48/48, and substance typed vs A 17 against 19.
- **Swap:** 91 against 95 (p 0.39).
- **Held text:** 0.5775 against 0.6105. Both are past the baseline's best (0.5397 at e2), so this is not evidence of health.
- **The bar:** R **1.87** against 2.87. B12's paired test against seed 0 at epoch 3 gives −1.01 [−1.47, −0.41]; this was not pre-registered.

λ3 epoch 2 (R 1.95) is not healthy: its typed opening is late and weak. λ3 epoch 1 (R 2.51) is broken like the baseline's epoch 1.

**What remains:**
- **Pupil-following.** Both epoch-3 checkpoints follow the pupil less than epoch 2 does (95 and 91 against 100).
- **The bar.** No checkpoint passes. R 1.87 is near the tape ceiling, tape‖A over the floor at ~1.7-1.8 (B8, B12).
- **Seeds.** There is one seed per λ.

## Examples (truncated)

**Same item, both epoch-3 checkpoints miss** (dlg-00005-074, turn 6, rendering A). The baseline's e3 answer: "right, zero, because the curve stops climbing at the hump and hasn't reached the dip yet..." λ3's e3 answer: "right, zero, and the numbers say why..." The judges read both openings as taking up the decoy turn, not the turn played.

**λ3 epoch 1 at the opening** (spoken A and B): silence in 12/13 and 9/13. The typed opening is silent in 4/13, and the typed openings that speak start on frames 4-12.

The full side-by-side for all eight checkpoints is in `consistency-3/sbs-report.md`, and the labels are in `consistency-3/judge-labels.jsonl`.

## What was run

- **Jobs**, one at a time. Both used image `sbs2:72868db` with `--no-bare --swap-turns 2,6 --mem-cap-gib 64`, B12's donor map and B7/B10's batch shapes (53/35/29/26/23/21/8 true, 28/24 swap). drama3 was stopped and the watchdog ran. 1.51 GPU-hours of decode time (1.62 h queue wall time).
  - `job-20261107-090658-vlad-sbs-base-e3`: `--run ladder-personaplex-1 --epoch 3`. 2,719 s, peak 38.9 GiB, adapter norm 27.73.
  - `job-20261107-090705-vlad-sbs-l3-e1`: `--run cons1-pp-l3 --epoch 1`. 2,720 s, peak 38.9 GiB, adapter norm 26.81.
- **Pool.** `consistency-3/pool/combine8.py` builds the eight-checkpoint input from the committed answer files (side-by-side-1/2/3, consistency-2, consistency-3). It reproduces B12's six-checkpoint input exactly (1,482 rows, 0 differences, compared against B12's scratchpad `combined6.jsonl`, which is not committed). `blind4.py` (B12's, unchanged) shuffles it with seed 13, and two builds were byte-identical.
  - The pool is 1,040 pairs, 1,560 answers and 832 swap items.
  - It went to 35 opus judges (11 pair, 16 answer and 8 swap chunks of 90-104 items). Each read only its own `items.jsonl` and `rubric.txt` (B10's rubrics verbatim, in `pool/`).
  - `score4.py` unblinds the labels and `swap5.py` gives the swap counts and McNemar tests. The blinding keys are committed in `pool/` (`*-key.jsonl`), so the labels can be checked against the shuffled items.
- **Agreement with B12's labels** on the 2,574 shared items:
  - substance 696/780;
  - uptake 1,141/1,170;
  - tape's move 1,124/1,170;
  - well formed 1,073/1,170;
  - swap followed-vs-not 605/624.
- **Files in `consistency-3/`:**
  - `answers.jsonl`: this batch's 494 rows (`base-e3`, `l3-e1`);
  - `meta-sbs-*.json`;
  - `judge-labels.jsonl`: all eight checkpoints, unblinded;
  - `judge-prompts.txt`;
  - `sbs-report.md`: `dual.sbs_report --quality` over the eight;
  - `pool/`.

## Caveats

- **One seed per λ, 13 holdout dialogues, 104 swap items.** A 4-item swap gap is below what this design resolves, and the turn-0 cells have 13 items each.
- **Judge noise is the same order as the effects:** up to 4 swap items and 3 tape-move items per checkpoint between pools.
- **Greedy, bf16, one path per row.** The batch shapes are identical across checkpoints. B7's control showed 6/30 rows change substance under another batch shape.
- **Epoch 3 is past the baseline's best held text loss** (0.6105 against 0.5397 at epoch 2), so "healthy against λ = 0 at the same epoch" compares two checkpoints that both follow the pupil a little less than epoch 2.
- **The epoch comparisons are post hoc.** The pre-registered bar rule scores λ3 at epoch 1 ("not moved"), and the per-epoch R comparisons are supplementary.

## The gate

- **(a) Accept λ3 epoch 3 as the consistency checkpoint and take it to the engine.** It is the best λ ≠ 0 checkpoint and indistinguishable from λ = 0 at the same epoch, but it does not pass the bar (R 1.87), and both epoch-3 checkpoints follow the pupil a little less than epoch 2.
- **(b) A second λ3 seed, decoded at epochs 2 and 3, with a λ = 0 epoch-3 decode of a second baseline seed** (~7 GPU-hours, no code). This tests whether "indistinguishable at epoch 3" and the same-sign 2-4-item swap deficit replicate. It doubles the swap items to 208 per arm, and it checks λ3's late typed opening at epoch 2. **Recommended.** It is the cheapest step that decides whether to stake a new render and plan on λ = 3. B12 also ordered it next if the controls left epoch 3 standing, which they did.
- **(c) Voice-D TTS teacher rows at λ = 3, 3 epochs** (needs a plan; ~6.5 GPU-hours plus a render). The teacher is the tape, so the ratio is capped at ~1.7-1.8, and new teacher rows are the only lever on that cap. Whether they would reach R ≈ 1 is untested. λ3 at 3 epochs is the best candidate recipe, but it rests on one seed. Run (c) after (b) if (b) replicates, or now if replication is judged not worth ~7 GPU-hours.
- **(d) λ = 1-2.** Weakly indicated at most. λ = 3 shows no detectable behavioural cost at epoch 3, and a lower λ would mostly give back bar. It could shrink the small same-sign swap deficit, if that is real.
- **(e) Stop the consistency line and revisit the bar.** The bar cannot pass while the teacher is the tape, at ~1.7-1.8.
