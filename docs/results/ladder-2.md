# Ladder 2: rank and LR at 117 dialogues, three epochs, two seeds

Vlad's answer at the ladder-1 gate was "(b) change rank / LR / base, rerun". This is that sweep:
- **Base:** PersonaPlex only.
- **Rung:** all 117 training dialogues.
- **Epochs:** 3, every one measured on the 13-dialogue holdout.
- **Runs:** one rank-16 recipe repeated with a second seed, plus four variants.
- **Design:** plan section B6.

## The answer

**No rank or LR change moved the ratio.** All four variants come out "not moved" under B6.0.2.

At its healthiest epoch every run sits at the same place:
- median gap 0.018 to 0.022;
- median floor 0.006 to 0.007;
- gap/floor 2.96 to 3.36.

That is ladder 1's flat ~3. The recipe changes only how fast a run gets there and then overfits:
- **LR 1e-4** reaches it in one epoch and then inflates. Its floor grows 3.5x by epoch 3.
- **LR 5e-6** creeps along at a gap of ~0.021 for all three epochs and never goes below.
- **Rank 4 and rank 64** (matched alpha 32) are indistinguishable from rank 16.

**Seed noise is small; holdout noise is not.**
- The two rank-16 seeds give R_gap 3.04 and 2.96 (d = 0.08), with the same gap and floor to three decimals.
- The paired stem-bootstrap intervals are about ±0.3 to ±0.8. So the sweep could only have seen a move of ~0.5 or more, the threshold T.
- Every variant's R_gap - R0 is between +0.09 and +0.37, so none clears T.
- None moved down at all: every point estimate is at or above R0.

Read against the bar, which needs the ratio at 1: within rank 4-64 and LR 5e-6 to 1e-4, the recipe is not the lever at 117 dialogues.

## The verdict (`make noise`, content frames, epochs 1-3)

R_gap is the median gap/floor at the run's lowest-median-gap epoch, and it decides. R is the best of the three epochs, reported only. The reference seed 0 is ladder 1's `ladder-personaplex-1`, epochs 1 to 3: the same recipe step for step, with constant LR and no schedule.

| run | R (epoch) | R_gap (epoch) | median gap | median floor | p90 ratio | R_gap - R0 | 95% CI vs personaplex-ladder-personaplex-1 | 95% CI vs personaplex-ladder2-pp-s1 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| personaplex-ladder-personaplex-1 | 2.87 (3) | 3.04 (1) | 0.018 | 0.006 | 2.01 | +0.04 | - | +0.08 [-0.63, +0.61] | - |
| personaplex-ladder2-pp-s1 | 2.96 (1) | 2.96 (1) | 0.018 | 0.006 | 1.68 | -0.04 | -0.08 [-0.61, +0.63] | - | - |
| personaplex-ladder2-pp-lr5e-6 | 3.11 (1) | 3.22 (2) | 0.021 | 0.007 | 2.55 | +0.22 | +0.18 [-0.36, +0.62] | +0.26 [-0.55, +0.77] | not moved |
| personaplex-ladder2-pp-lr1e-4 | 2.10 (3) | 3.36 (1) | 0.020 | 0.006 | 1.50 | +0.37 | +0.32 [-0.46, +0.75] | +0.41 [-0.64, +1.00] | not moved |
| personaplex-ladder2-pp-r64 | 2.70 (3) | 3.08 (1) | 0.018 | 0.006 | 1.94 | +0.09 | +0.05 [-0.28, +0.31] | +0.13 [-0.53, +0.58] | not moved |
| personaplex-ladder2-pp-r4 | 2.83 (3) | 3.12 (1) | 0.019 | 0.006 | 2.13 | +0.13 | +0.09 [-0.32, +0.44] | +0.17 [-0.44, +0.69] | not moved |

The rule (B6.0.2): a variant moved only if |R_gap - R0| >= T = max(2d, 0.5) and both paired 95% intervals exclude 0 with the sign of R_gap - R0. A "down" also needs its gap no higher than the seeds' mean gap, so a ratio that falls through the floor alone does not count.

Caveats:
- **The intervals are optimistic.** The decision epoch is picked on the same 65 turns that are then resampled, and a percentile bootstrap over 13 stems is narrow.
- **d is measured only at rank 16, LR 2e-5.** Seed noise at LR 1e-4 may be larger. None of this changes the verdict: no variant's point estimate is even below R0.
- **R alone would have misled.** It is 2.10 for LR 1e-4, at epoch 3, which would have read as "down". That epoch's gap is 0.047 against the reference's 0.018: the floor inflation that ladder 1 described. Deciding on R_gap was fixed before any ladder-2 result existed, after the plan's final review.

## Per epoch (content frames, from [bar.md](bar.md))

| run | rank | scaling | LR | seed | epoch | median gap | median floor | p90 gap | p90 floor | door |
|---|---|---|---|---|---|---|---|---|---|---|
| ladder-personaplex-1 | 16 | 2.0 | 2e-5 | 0 | 1 | 0.018 | 0.006 | 0.033 | 0.017 | 0.005 |
| | | | | | 2 | 0.025 | 0.008 | 0.050 | 0.032 | 0.008 |
| | | | | | 3 | 0.041 | 0.014 | 0.069 | 0.036 | 0.013 |
| ladder2-pp-s1 | 16 | 2.0 | 2e-5 | 1 | 1 | 0.018 | 0.006 | 0.034 | 0.020 | 0.006 |
| | | | | | 2 | 0.026 | 0.009 | 0.043 | 0.025 | 0.009 |
| | | | | | 3 | 0.046 | 0.014 | 0.070 | 0.036 | 0.013 |
| ladder2-pp-lr5e-6 | 16 | 2.0 | 5e-6 | 0 | 1 | 0.022 | 0.007 | 0.047 | 0.014 | 0.006 |
| | | | | | 2 | 0.021 | 0.007 | 0.042 | 0.016 | 0.006 |
| | | | | | 3 | 0.022 | 0.007 | 0.045 | 0.017 | 0.007 |
| ladder2-pp-lr1e-4 | 16 | 2.0 | 1e-4 | 0 | 1 | 0.020 | 0.006 | 0.032 | 0.021 | 0.005 |
| | | | | | 2 | 0.038 | 0.016 | 0.061 | 0.035 | 0.012 |
| | | | | | 3 | 0.047 | 0.022 | 0.082 | 0.049 | 0.017 |
| ladder2-pp-r64 | 64 | 0.5 | 2e-5 | 0 | 1 | 0.018 | 0.006 | 0.031 | 0.016 | 0.005 |
| | | | | | 2 | 0.026 | 0.008 | 0.049 | 0.031 | 0.008 |
| | | | | | 3 | 0.041 | 0.015 | 0.064 | 0.038 | 0.013 |
| ladder2-pp-r4 | 4 | 8.0 | 2e-5 | 0 | 1 | 0.019 | 0.006 | 0.031 | 0.015 | 0.005 |
| | | | | | 2 | 0.028 | 0.009 | 0.049 | 0.029 | 0.009 |
| | | | | | 3 | 0.047 | 0.017 | 0.071 | 0.037 | 0.015 |

No run passes at any epoch. The door stays at or under the floor in every row, so the harness is sane.

On all frames, which include the pupil's own turn, r4's epoch 3 reads gap 0.113 against floor 0.008. That is the expected all-frames inflation, not the bar.

## Held-out text loss per epoch (train logs)

| run | 1 | 2 | 3 | s/epoch | peak mem |
|---|---|---|---|---|---|
| ladder-personaplex-1 (ladder 1) | 0.551 | **0.540** | 0.611 | 3,220 | 49.4 GiB |
| ladder2-pp-s1 | 0.556 | **0.546** | 0.616 | 3,245 | 49.4 GiB |
| ladder2-pp-lr5e-6 | 0.703 | 0.600 | **0.561** | 3,241 | 49.4 GiB |
| ladder2-pp-lr1e-4 | **0.527** | 0.619 | 0.783 | 3,239 | 49.4 GiB |
| ladder2-pp-r64 | 0.551 | **0.539** | 0.602 | 3,261 | 51.2 GiB |
| ladder2-pp-r4 | 0.550 | **0.539** | 0.609 | 3,228 | 48.9 GiB |

LR 1e-4 has the lowest held text loss of any run (0.527 at epoch 1), but its ratio there is 3.36. Held-out text loss and the typed-vs-spoken ratio move independently.

## Running it

- **Images:**
  - `ladder2:5a2d32e`: s1, lr5e-6, lr1e-4 and r64.
  - `ladder2:a800c4c`: r4. It is the same train and measure code plus a GPU memory cap, `--mem-cap-gib 64`, fraction 0.526 of 121.7 GiB, with `expandable_segments:True`. Neither changes what is trained or measured; r4 peaked at 48.9 GiB, under the cap.
- **Smoke:** before the sweep, `ladder2-smoke-r4` (slice 1/8, 1 epoch, rank 4) checked that train and measure attach a non-default rank.
- **Box freeze, 11-04 14:52:**
  - drama3 had been stopped at 20:20 on 11-03 and again at 13:35 on 11-04. Someone started it again both times, without a reboot.
  - It was running beside the first r4 train: dockerd was health-checking its container up to the freeze.
  - The kernel logged `NVRM ... NV_ERR_NO_MEMORY` from 14:52, and the box froze. MemAvailable was 91 GiB at 14:34 and 17 GiB at 14:44.
  - Vlad power-cycled it at 14:59. r4 was rerun from scratch; it had no checkpoint.
- **Guards added:** the memory cap, so an overrun fails the job, not the box; and a Mac-side watchdog that stopped drama3 whenever it was up during a ladder-2 job. The watchdog never had to act during the r4 rerun.
- **Note on the cap:** it bounds this job alone. drama3's ~61 GiB beside a ~49 GiB train still overcommits the 121 GiB, so GPU runs need drama3 down.
- **drama3 now:** it is up again, started by someone after the last job ended. It was left running.
