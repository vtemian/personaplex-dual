# Consistency 4: a second λ = 3 seed, with the seed-1 baseline's epoch 3

Vlad, 2026-11-07, at B13's GATE ("Weight-3 epoch 3 matches the baseline's behaviour at ratio 1.87. What next?"): **"Second weight-3 seed (Recommended)"**, "Repeat the run with another seed and decode epochs 2-3 plus a baseline epoch 3 of that seed. About 7 GPU-hours, no code. Doubles the evidence before betting a render on weight 3." This batch (B14) ran exactly that: one λ = 3 train at seed 1, its measure, three side-by-side decodes, and one fresh blind pool over all eleven decoded checkpoints. No render, no new loss variant, no repo code.

## The answer

**It replicates.** Every pre-registered criterion (plan B14) passes. Seed-1 λ3 epoch 3 has bar R **1.87**, the same as seed 0's 1.87. Against the λ = 0 checkpoint of its own seed at the same epoch, its behaviour is healthy.

- **Opening:** it speaks after the spoken opening in 13/13 for both A and B. The typed opening starts on frame 3 in 13/13.
- **Fragments:** mid-lesson stops are 1/3/1 (typed/A/B) of 52.
- **Swap:** it follows the played turn in 95/104, against 97/104 for λ = 0 seed 1.
- **The bar:** paired against λ = 0 seed 1 at epoch 3, R is −1.28 [−1.68, −0.84]. Against seed 0 at epoch 3 it is −1.01 [−1.55, −0.38].
- **The rule:** the pre-registered noise rule, decided at seed 1's lowest-gap epoch (epoch 2, R_gap 2.09), says **down** (−0.90, both CIs exclude 0). Seed 0's run was "not moved" at its lowest-gap epoch 1 (2.51).

**No consistent swap-follow cost of the term can be shown.** By the pre-registered wording this is "a lean, unresolved".
- λ3 epoch 3 trails λ = 0 epoch 3 at both seeds, by 1 item (93 vs 94) and 2 items (95 vs 97).
- Pooled over the 208 paired items: λ = 0 191, λ3 188. Discordant 14 vs 11, exact McNemar p 0.69, 95% Wald interval −7 to +13 items.
- B13's seed-0 gap was 4 items (91 vs 95). It shrinks to 1 in this pool, which is judge noise at that size.

**The bar still fails, at both seeds.** R ≈ 1.87 sits on the tape ceiling (tape‖A over the floor, ~1.7-1.8, B8/B12).

**Epoch 2 does not replicate as a stage.** Seed 0's epoch 2 had a late typed opening (frame 7-8). Seed 1's epoch 2 opens on frame 3, but follows the pupil only 90/104 (12 vs 5 against λ = 0 seed-1 e3, p 0.14). Epoch 3 is the only λ3 checkpoint that is healthy at both seeds.

## Bar R per epoch, both seeds, both λ

From `make noise` / `dual.noise` on each run's ladder measure: R = median per-turn gap / median per-turn floor over content frames. The CIs are paired stem-bootstrap, 95%, against λ = 0 at the same epoch. They are post hoc, except the rule's verdict.

| run | λ | seed | e1 R | e2 R | e3 R | e1 / e2 / e3 median gap | e1 / e2 / e3 median floor | rule (R_gap, epoch) |
|---|---|---|---|---|---|---|---|---|
| `ladder-personaplex-1` | 0 | 0 | 3.04 | 3.24 | 2.87 | .0177 / .0251 / .0414 | .0058 / .0077 / .0144 | seed (3.04, e1) |
| `ladder2-pp-s1` | 0 | 1 | 2.96 | 3.07 | 3.14 | .0182 / .0261 / .0455 | .0061 / .0085 / .0145 | seed (2.96, e1) |
| `cons1-pp-l3` | 3 | 0 | 2.51 | 1.95 | **1.87** | .0094 / .0106 / .0146 | .0037 / .0054 / .0078 | not moved (2.51, e1) |
| `cons1-pp-l3-s1` | 3 | 1 | 2.30 | 2.09 | **1.87** | .0105 / .0104 / .0138 | .0046 / .0049 / .0074 | **down** (2.09, e2) |

- **Paired against λ = 0 at the same epoch** (seed 0 / seed 1 as the reference):

  | | e1 | e2 | e3 |
  |---|---|---|---|
  | λ3 s0 | −0.52 [−1.19, −0.34] / −0.44 [−1.27, −0.11] | −1.30 [−1.57, −0.61] / −1.13 [−1.60, −0.40] | −1.01 [−1.47, −0.41] / −1.28 [−1.71, −0.70] |
  | λ3 s1 | −0.74 [−1.21, −0.27] / −0.66 [−1.38, −0.10] | −1.15 [−1.64, −0.62] / −0.98 [−1.50, −0.40] | −1.01 [−1.55, −0.38] / −1.28 [−1.68, −0.84] |

  Seed 0's rows reproduce B12's exactly (`consistency-4/pool/epoch_ci.py`).
- **The rule** (B6.0.2): R0 3.00, d 0.08, T 0.5. Moved iff |R_gap − R0| ≥ T and both paired CIs exclude 0, with R_gap read at the run's lowest-median-gap epoch.
  - The brief expected the epoch-1 verdict to be pre-registered. The rule as written picks the lowest-gap epoch, which is epoch 1 at seed 0 and epoch 2 at seed 1 (gap .0104 against .0105).
  - Read at epoch 1, seed 1 would also be "down" (2.30: −0.70, CIs [−1.21, −0.27] / [−1.38, −0.10]).
- **Held text loss**, λ3 s1 against λ = 0 s1: 0.5659 / 0.5353 / 0.5720 against 0.556 / 0.546 / 0.616 (+1.8%, −2.0%, −7.1%). Seed 0: +1.1% / −0.6% / −5.4%.
- **Mean train `cons`** at seed 1: 0.01006 / 0.00679 / 0.00767 (seed 0: 0.00989 / 0.00709 / 0.00772).
- **Turn-0 lead-frame KL(typed‖A)** at λ3 s1: 0.041 / 0.188 / 0.411. Seed 0 had 0.062 / 0.25 / 0.70, and λ = 0 seed 0 at e2 0.74. The seed-1 λ = 0 ladder measure stores no per-frame KLs.
  - B13's post-hoc pattern (KL < 0.1 means a silent spoken opening) does not predict seed 1 well.
  - Seed-1 e2 (0.19) and e3 (0.41) both open on time and speak 12-13/13.
  - The lead KL is not a reliable health signal across seeds.
- **By position** (`make position`, content ratio at positions 26+): λ3 s1 2.55 / 2.04 / 1.99 at e1-e3. Seed 0 had 2.18 at e3 (B12), against 3.44 for λ = 0 seed 0 at e2.
- **Training:** 4,308 / 4,303 / 4,305 s per epoch, peak 56.0 GiB, 64 GiB cap. Max grad norm was 5.8 / 7.8 / **36.5**: one spike in epoch 3 (logged as an epoch max, every 10 steps). Seed 0's largest was 11.5.

## All eleven checkpoints

Behaviour comes from this batch's blind pool: 48 opus judges over all eleven checkpoints. Text counts come from the answer stream.
- **Stops:** stops mid-sentence with ≥ 20% of the window left (B11's rule).
- **Mid-lesson:** turns 2-8, 52 answers per rendering. **Opening:** turn 0, 13 per rendering.

The seven checkpoints this batch is about:

| | base s0 e2 | base s0 e3 | **base s1 e3** | λ3 s0 e2 | λ3 s0 e3 | **λ3 s1 e2** | **λ3 s1 e3** |
|---|---|---|---|---|---|---|---|
| bar R | 3.24 | 2.87 | 3.14 | 1.95 | 1.87 | 2.09 | **1.87** |
| held text loss | 0.5397 | 0.6105 | 0.616 | 0.5366 | 0.5775 | 0.5353 | 0.5720 |
| speaks after spoken opening, A / B (of 13) | 13 / 13 | 13 / 13 | 13 / 13 | 13 / 13 | 13 / 13 | 12 / 13 | 13 / 13 |
| opening uptake, typed / A / B | 12 / 12 / 11 | 13 / 13 / 12 | 13 / 10 / 11 | 9 / 10 / 13 | 11 / 12 / 13 | 12 / 10 / 11 | 12 / 12 / 11 |
| opening makes the tape's move, typed / A / B | 11 / 8 / 6 | 12 / 12 / 8 | 13 / 8 / 6 | 6 / 8 / 9 | 10 / 10 / 8 | 11 / 9 / 10 | 10 / 9 / 7 |
| typed-opening onset frame (tape 3) | 3 (13/13) | 3 (13/13) | 3 (13/13) | **7-8** (11/13) | 3 (13/13) | 3 (13/13) | 3 (13/13) |
| mid-lesson uptake, typed / A / B | 52 / 51 / 50 | 52 / 50 / 48 | 52 / 50 / 49 | 52 / 51 / 50 | 51 / 49 / 50 | 52 / 50 / 49 | 51 / 50 / 48 |
| mid-lesson stops, typed / A / B | 1 / 2 / 0 | 0 / 0 / 0 | 0 / 0 / 1 | 2 / 3 / 3 | 0 / 1 / 0 | 1 / 4 / 1 | 1 / 3 / 1 |
| mid-lesson median words, typed / A / B | 64.5 / 66 / 67 | 63 / 65 / 61 | 59 / 62 / 67.5 | 65 / 64 / 62 | 63.5 / 60.5 / 64 | 64 / 63 / 61.5 | 64.5 / 63.5 / 64.5 |
| swap: played turn taken up (of 104) | 101 | 94 | 97 | 96 | 93 | **90** | 95 |
| mid-lesson substance differs, typed vs A / A vs B (of 52) | 15 / 20 | 22 / 14 | 22 / 19 | 17 / 18 | 16 / 18 | 15 / 12 | 18 / 18 |

The other four, as this pool judges them. Their text counts are B12/B13's and unchanged.

| | base s0 e1 | λ10 e1 | λ10 e2 | λ3 s0 e1 |
|---|---|---|---|---|
| swap (of 104) | 104 | 96 | 93 | 101 |
| opening uptake, typed / A / B | 11 / 2 / 3 | 8 / 11 / 10 | 8 / 0 / 1 | 7 / 1 / 4 |
| opening tape's move, typed / A / B | 9 / 0 / 0 | 2 / 7 / 5 | 4 / 0 / 0 | 3 / 0 / 1 |
| mid-lesson uptake, typed / A / B | 52 / 49 / 51 | 51 / 49 / 50 | 50 / 51 / 49 | 51 / 49 / 50 |
| mid substance differs, typed vs A / A vs B | 11 / 16 | 9 / 13 | 15 / 15 | 9 / 7 |

- **Bar columns.** Each value comes from that run's own ladder measure (`dual.noise`). Base s1 is `ladder2-pp-s1` (B6, held text from its train log, three digits).
- **Speaking after the opening, onset frame, stops and median words** are text counts (`b12-read/qual4.py`, B12's script). For the old checkpoints they are identical to B12/B13's.
- **This pool against B13's on the eight shared checkpoints:**
  - swap followed-vs-not 807/832;
  - substance 937/1,040;
  - uptake 1,526/1,560;
  - tape's move 1,509/1,560;
  - well formed 1,436/1,560.
  - Swap counts move by up to 3 (base e1 101 → 104, λ3 s0 e1 99 → 101, λ3 s0 e3 91 → 93). Opening tape-move counts move by up to 2.
  - This is judge noise. For the eight old checkpoints, this pool's judged numbers supersede consistency-3's.

## Does λ3 epoch 3 replicate?

The pre-registered criteria (plan B14, written before any seed-1 number existed), for seed-1 λ3 e3 against λ = 0 seed-1 e3:

| criterion | needed | seed-1 λ3 e3 | pass |
|---|---|---|---|
| R within 0.5 of 1.87 | ≤ 2.37 | 1.87 | yes |
| speaks after the spoken opening, A and B | ≥ 12/13 each | 13 / 13 | yes |
| typed-opening onset by frame 4 | ≥ 11/13 | 13/13 (frame 3) | yes |
| mid-lesson stops per rendering | ≤ 3 of 52 | 1 / 3 / 1 | yes |
| swap-follow within 5 of λ = 0 seed-1 e3 | ≥ 92 | 95 (λ = 0: 97) | yes |

The judged columns agree. Against base s1 e3:
- opening uptake 12/12/11 against 13/10/11;
- the tape's move 10/9/7 against 13/8/6;
- mid-lesson uptake 51/50/48 against 52/50/49;
- substance typed vs A 18 against 22.

Seed 0 shows the same picture against base s0 e3.

## Is there a consistent swap-follow cost of the term?

**Not one that 208 items can see.**

| | λ = 0 e3 | λ3 e3 | λ = 0 only / λ3 only | exact p |
|---|---|---|---|---|
| seed 0 | 94 | 93 | 6 / 5 | 1.0 |
| seed 1 | 97 | 95 | 8 / 6 | 0.79 |
| pooled (208) | 191 | 188 | 14 / 11 | 0.69 |

- **The pre-registered call:** λ3 trails at both seeds and p ≥ 0.05, so it is "a lean, unresolved". The lean is 1-2 items per seed, and the 95% interval on the pooled gap is −7 to +13 items.
- **Seeds against seeds:** λ = 0 s0 against s1 at e3 is 6 vs 9 discordant, and λ3 s0 against s1 is 5 vs 7. The seed-to-seed spread is as large as the λ effect.
- **Kinds of miss (post hoc):** of each epoch-3 checkpoint's misses, the share that open with an affirmation ("right", "yes", ...) is 7/11 and 3/9 for λ3, against 4/10 and 0/7 for λ = 0. B12 named this miss as a fluent affirmation of something the pupil never said. The pattern leans toward λ3 (10/20 against 4/17), but it was not pre-registered and the counts are small.
  - Shared misses: λ = 0 and λ3 share 5 misses at seed 0 and 1 at seed 1. λ3 s0 and s1 share 4.
- **Epoch 3 against epoch 2 for λ = 0 seed 0, re-judged here:** 101 → 94 (9 vs 2, p 0.065). That is the same epoch-3 drop B13 saw.

## Verdict

"λ3 epoch 3 is behaviourally healthy and R ≈ 1.9" replicates across seeds.
- **Bar:** seed 1 lands on R 1.87 exactly as seed 0 did. That is about 1.0-1.3 below λ = 0 at the same epoch, every paired CI excluding 0.
- **Behaviour:** it passes every pre-registered behaviour check against its own seed's λ = 0 epoch 3: spoken and typed openings on time, no fragments, and the swap within 2 items.
- **Swap cost:** the term has no swap-follow cost that the pooled 208 items can show (191 against 188, p 0.69). λ3 trails by 1-2 items at each seed, and a small post-hoc lean toward affirmation-type misses is the only hint of one.
- **Rule verdict:** under the pre-registered rule the second seed reads "down" (R_gap 2.09 at epoch 2), where seed 0 read "not moved" (2.51 at epoch 1).
- **What is not stable:** epoch 2. Seed 0's typed opening is late; seed 1's opens on time but follows the pupil less (90/104).
- **What has not changed:** no checkpoint passes the bar. Both seeds sit at R 1.87, on the tape ceiling (~1.7-1.8), so another λ or seed on this teacher cannot reach it.

## Examples (truncated)

The full side-by-side for all eleven checkpoints is in `consistency-4/sbs-report.md`, and the labels are in `consistency-4/judge-labels.jsonl`.

**Seed-1 λ3 epoch 3 misses where its own seed's λ = 0 follows** (dlg-00005-123, turn 2, rendering A, true row). λ3: "right, and since the climb is gentle, I'd say the height barely budges..." (the judge: neither pupil number is taken up). λ = 0 seed 1: "right, zero point three one holds, and the step itself shows why...". All 9 of seed-1 λ3 e3's misses are at turn 2. Most are a generic "let's test that" or "right, and ..." opening that does not take up the pupil's number.

**Seed-1 λ3 epoch 2.** Of its 14 misses, 10 are on swapped rows, and most are "neither" or "both" labels. Only 1 opens with an affirmation.

## What was run

- **Jobs**, one at a time, `--mem-cap-gib 64`. drama3 was stopped (down already at 12:54, never restarted) and the Mac-side watchdog ran the whole time without needing to stop it. GPU wall time was 26,543 s (7.37 h) against ~7 estimated.
  - `job-20261107-125420-vlad-sbs-s1-e3`: image `sbs2:72868db`, `--run ladder2-pp-s1 --epoch 3 --no-bare --swap-turns 2,6`. 2,913 s, peak 38.9 GiB, adapter norm 27.730 (seed 0 e3: 27.726; md5 differs, and 0 of 247 answers are identical).
  - `job-20261107-125429-vlad-cons1-pp-l3-s1`: image `cons1:9690aa5`, `--consistency 3 --seed 1`, 3 epochs, 117 dialogues, rank 16, LR 2e-5. 13,280 s, peak 56.0 GiB.
    - It ran from a frozen scratchpad copy of `scripts/ladder.sh` with one added run list (`LADDER=5`); the repo copy is unchanged.
  - `job-20261107-172426-vlad-measure-cons1-pp-l3-s1`: 4,551 s; then `make bar`.
  - `job-20261107-172500-vlad-sbs-l3s1-e2`: 2,890 s, adapter norm 27.281.
  - `job-20261107-172507-vlad-sbs-l3s1-e3`: 2,909 s, adapter norm 27.667.
  - All three decodes used B12's donor map and B7/B10's batch shapes (53/35/29/26/23/21/8 true, 28/24 swap), peak 38.9 GiB.
- **Pairing.** `--seed` sets `torch.manual_seed` (LoRA init) and the batch-order `random.Random`, and the term draws no randomness. So `cons1-pp-l3-s1` and `ladder2-pp-s1` share init and batch order, as `cons1-pp-l3` and `ladder-personaplex-1` do at seed 0.
- **Pool.**
  - **Input:** `pool/combine8.py` (B13's, unchanged) builds the input from the committed answer files. With B13's two extra specs it reproduces B13's `combined8.jsonl` byte for byte. Three more specs add this batch's checkpoints: `base-s1-e3`, `l3s1-e2`, `l3s1-e3`, 2,717 rows in all.
  - **Blinding:** `blind4.py` (seed 13, unchanged) shuffles it, and two builds were byte-identical. The pool is 1,430 pairs, 2,145 answers and 1,144 swap items.
  - **Judges:** 48 opus judges (15 pair, 22 answer and 11 swap chunks of 86-104 items), each reading only its own `items.jsonl` and `rubric.txt` (B10's rubrics verbatim, in `pool/`).
  - **Scoring:** `score4.py` unblinds the labels. `swap6.py` gives the swap counts, the McNemar tests and the pooled 208-item test. `epoch_ci.py` gives the per-epoch R and paired CIs from `runs/`. The blinding keys are in `pool/`.
- **Files in `consistency-4/`:**
  - `answers.jsonl`: this batch's 741 rows (`base-s1-e3`, `l3s1-e2`, `l3s1-e3`);
  - `meta-sbs-*.json`;
  - `judge-labels.jsonl`: all eleven checkpoints, unblinded;
  - `sbs-report.md`: `dual.sbs_report --quality` over the eleven;
  - `pool/`.

## Caveats

- **Two seeds, 13 holdout dialogues, 104 swap items per checkpoint.** The pooled swap test resolves about ±10 items of 208.
- **Judge noise is the same order as the λ effect on the swap:** up to 3 items per checkpoint between pools, and 25/832 shared swap labels flipped.
- **Epoch 3 is past both λ = 0 seeds' best held text loss** (0.6105 and 0.616 against ~0.54 at e2). "Healthy against λ = 0 at the same epoch" compares checkpoints that both follow the pupil a little less than epoch 2 (λ = 0 s0: 101 → 94).
- **The rule's verdict differs by seed** ("not moved" at seed 0, "down" at seed 1). Seed 0's epoch 1 (2.51) is the only λ3 checkpoint whose R is less than T below R0. At epochs 2-3 both seeds are 0.91-1.13 below R0, but the rule reads a single epoch, the lowest-gap one.
- **The λ = 0 seed-1 run came from an earlier image** (`ladder2:5a2d32e`), and λ3 from `cons1:9690aa5`. The λ = 0 code path is unchanged between them, as at seed 0.
- **Greedy, bf16, one path per row.** The batch shapes are identical across checkpoints.

## The gate

See the plan's B14 record and the orchestrator's GATE.
