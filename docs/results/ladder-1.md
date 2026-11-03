# Ladder 1: two bases over four rungs of 117 dialogues

Eight LoRA runs (PersonaPlex and moshiko, slices 1/8, 1/4, 1/2, all of render-1's train rows,
8 epochs each, compile engine, same seed), every epoch measured on the 13-dialogue holdout.
Ran 2026-10-27 to 2026-11-03 on the box through sparks (image
`spark.local:5000/vlad/smoke-eager-pp:8ed8980`, code identical to main `c624f13`).
The full per-epoch tables (content frames and all frames) are in [bar.md](bar.md).

## The answer

**No run passes the bar on either base, but the curve is still falling at the full set.**
The best median gap/floor drops with every doubling of data on both bases: PersonaPlex 2.92,
2.87, 2.54, 2.09 and moshiko 2.95, 2.48, 2.45, 2.10 for 15, 30, 59 and 117 dialogues. The
epoch-1 gap does the same: PersonaPlex 0.027, 0.023, 0.020, 0.018 and moshiko 0.029, 0.028,
0.023, 0.021, against 0.046 and 0.059 for the bare bases. Read against the design's three
outcomes:
- **Still falling at the full set:** yes, on both bases.
- **Flat above the floor from a quarter on:** no. The steepest drop is the last rung, 2.54 to
  2.09 on PersonaPlex.
- **Inside the floor before the full set:** no. The closest any run comes is a p90 ratio of
  1.50 (PersonaPlex 1/2, epoch 4), and the median ratio never goes below 2.09.

## The curve (content frames)

The gap is KL(typed || spoken A) and the floor is KL(spoken A || spoken B). A run passes when
median gap <= median floor and p90 gap <= p90 floor.

| base | slice | dialogues | best median gap/floor (epoch) | best p90 gap/floor (epoch) | lowest median gap (epoch) | gap/floor there |
|---|---|---|---|---|---|---|
| personaplex | bare | - | 3.93 (0) | 3.91 (0) | 0.046 (0) | 3.93 |
| personaplex | 1/8 | 15 | 2.92 (6) | 2.11 (8) | 0.025 (2) | 3.27 |
| personaplex | 1/4 | 30 | 2.87 (4) | 2.08 (3) | 0.023 (1) | 3.28 |
| personaplex | 1/2 | 59 | 2.54 (6) | 1.50 (4) | 0.020 (1) | 3.21 |
| personaplex | 1/1 | 117 | 2.09 (6) | 1.58 (2) | 0.018 (1) | 3.04 |
| moshiko | bare | - | 3.08 (0) | 2.64 (0) | 0.059 (0) | 3.08 |
| moshiko | 1/8 | 15 | 2.95 (7) | 2.18 (7) | 0.029 (1) | 3.87 |
| moshiko | 1/4 | 30 | 2.48 (5) | 1.88 (7) | 0.028 (1) | 3.18 |
| moshiko | 1/2 | 59 | 2.45 (8) | 1.65 (3) | 0.023 (1) | 3.42 |
| moshiko | 1/1 | 117 | 2.10 (5) | 1.66 (2) | 0.021 (1) | 2.87 |

How to read it:
- **Best epoch:** `make bar`'s "best epoch" is the lowest median ratio. Late epochs win it,
  but not cleanly. Gap and floor both grow 3 to 7 times from epoch 1 to epoch 8 as every run
  overfits, so the ratio improves partly because the floor inflates. The absolute gap is
  lowest at epoch 1 or 2, where held-out text loss is also lowest.
- **The data trend holds either way:** under both readings the ratio and the gap fall with
  data.
- **Door control:** the door, KL(C || A), lands at or under the floor in every row (bar.md),
  so the harness and the TTS are sane.
- **All frames:** on all frames, trained runs sit about 7 to 10 times above the floor. Those
  frames include the pupil's own turn, where the typed rendering has no audio to match. Only
  content frames are the bar.
- **Base:** PersonaPlex and moshiko track each other within noise at every rung. The base is
  not the lever here.

## Held-out text loss per epoch ("held text" in the train log)

The minimum of each row is in bold. Every run bottoms at epoch 2 or 3 and then overfits.
Train loss keeps falling in every run, to about 1.4 to 1.55 at epoch 8.

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | s/epoch |
|---|---|---|---|---|---|---|---|---|---|
| personaplex-8 | 0.915 | 0.758 | **0.723** | 0.801 | 0.851 | 1.008 | 1.081 | 1.188 | 413 |
| personaplex-4 | 0.747 | **0.648** | 0.673 | 0.759 | 0.891 | 1.014 | 1.023 | 1.102 | 825 |
| personaplex-2 | 0.633 | **0.593** | 0.654 | 0.771 | 0.863 | 0.949 | 0.985 | 1.036 | 1,615 |
| personaplex-1 | 0.551 | **0.540** | 0.611 | 0.707 | 0.809 | 0.917 | 0.973 | 1.038 | 3,220 |
| moshiko-8 | 0.861 | 0.735 | **0.721** | 0.815 | 0.889 | 1.023 | 1.064 | 1.172 | 242 |
| moshiko-4 | 0.715 | **0.638** | 0.676 | 0.748 | 0.886 | 1.007 | 1.042 | 1.116 | 486 |
| moshiko-2 | 0.617 | **0.589** | 0.650 | 0.772 | 0.861 | 0.970 | 1.000 | 1.060 | 955 |
| moshiko-1 | 0.543 | **0.540** | 0.614 | 0.719 | 0.842 | 0.900 | 0.968 | 1.009 | 1,912 |

Peak memory was 49.4 GiB for PersonaPlex and 42.9 GiB for moshiko. Each measure took about
197 min for PersonaPlex and 116 min for moshiko.

## Inputs

- **Render** (render-1, job `job-20261026-212948-vlad-render-1`): 351 train rows; a holdout of
  13 dialogues x 5 pupil turns, each rendered as typed and as spoken voices A, B and C, plus the
  tape base. The longest train row is 2,765 frames. Slices nest at 15, 30, 59 and 117
  dialogues.
- **Zero-training probe** (B1.5, `job-20261026-210539-vlad-probe-bare`): 3 holdout dialogues,
  greedy, a typed second turn after a spoken opening.

  | variant | answer | echo | ignore |
  |---|---|---|---|
  | plain | 2 | 0 | 1 |
  | system | 3 | 0 | 0 |

  On reading, the plain "answers" are moshi continuing its own explanation, and the system ones
  are generic sign-offs. Bare PersonaPlex ignores the typed door and never echoes it.

## Running it

- **Order:** one job at a time, in this order: personaplex 8, 4, 2, 1, then moshiko 8, 4, 2,
  1, each trained then measured.
- **drama3:** stopped for the whole ladder, as Vlad authorised.
- **Hangs and reboots:**
  - The box went dark on 11-02 at 18:54 with drama3 stopped. That killed moshiko-2 during
    epoch 6; it was resumed from `model-epoch-05.pt` with `--resume`.
  - moshiko-2's epoch 5 ran about 1.7 times slower than its earlier epochs just before that
    hang. So drama3's memory is not the only cause of the hangs.
  - Earlier ssh drops (10-27, 11-02) are why `scripts/ladder.sh` now retries ssh in the
    memory gate, the submit and the wait. It looks a failed submit up by name before
    resubmitting.

## What it suggests for the gate

- **Data is the lever the ladder points to.** Fitting the last rungs gives about 18% off the
  median ratio per doubling. On that slope, about 3 or 4 more doublings reach the floor,
  roughly 1,000 to 2,000 dialogues. That is a rough extrapolation from two points per base,
  not a measurement.
- **The epochs could be cut.** Every run overfits after epoch 2 or 3, so a bigger-corpus
  rerun can drop to about 3 epochs. That would bring a full-set PersonaPlex run from 7 h to
  about 2.7 h at this corpus size.
