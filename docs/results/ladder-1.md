# Ladder 1: two bases over four rungs of 117 dialogues

Eight LoRA runs, every epoch measured on the 13-dialogue holdout:
- **Runs:** PersonaPlex and moshiko, each on slices 1/8, 1/4, 1/2 and all of render-1's train
  rows, 8 epochs, same seed.
- **Engine:** compile, the default. No job passed `--eager`, and 413 s/epoch matches B4's
  compiled smoke, not eager's 531 s.
- **Box:** ran 2026-10-27 to 2026-11-03 through sparks.
- **Image:** `spark.local:5000/vlad/smoke-eager-pp:8ed8980`. The tag is the name of the eager
  smoke that built it; the code is identical to main `c624f13`.

The full per-epoch tables (content frames and all frames) are in [bar.md](bar.md). Its
`smoke-train-pp` row is B4's 1-epoch smoke: `make bar` copies every `measure-*` directory on
the box.

## The answer

**No run passes the bar on either base.** Whether more data closes the gap depends on which
epoch you read, and the two readings disagree.

**1. Best epoch of 8 (`make bar`'s pick, the lowest median gap/floor).** The ratio falls with
data on both bases:
- PersonaPlex: 2.92, 2.87, 2.54, 2.09 for 15, 30, 59 and 117 dialogues.
- moshiko: 2.95, 2.48, 2.45, 2.10.
- Slope: about 10.5% per doubling across the whole ladder on both bases. The last rung is
  steeper, 18% on PersonaPlex and 14% on moshiko.

The catch is that these picks are epochs 4 to 8, deep in overfitting:
- Gap and floor both grow about 4 to 8 times from epoch 1.
- Within one run the ratio moves 0.3 to 0.5 between neighbouring epochs, about the size of the
  last-rung drop.
- The p90 ratio does not fall at the last rung: PersonaPlex 1.50 to 1.58, moshiko 1.65 to 1.66.

**2. Least-overfit epoch (lowest absolute gap, epoch 1 or 2).** The gap itself falls with data:
- PersonaPlex: 0.027, 0.023, 0.020, 0.018 (bare 0.046).
- moshiko: 0.029, 0.028, 0.023, 0.021 (bare 0.059).

But the floor falls with it, so the ratio barely moves:
- PersonaPlex: 3.27, 3.28, 3.21, 3.04 (bare 3.93).
- moshiko: 3.87, 3.18, 3.42, 2.87 (bare 3.08).

Read against the design's three outcomes:
- **Still falling at the full set:** yes, but only under reading 1. Under reading 2 the ratio
  is close to flat on PersonaPlex.
- **Flat above the floor:** partly. The ratio is flat at about 3 at the healthy epochs.
  Training cuts the absolute gap by about 60% (PersonaPlex 0.046 to 0.018), but the floor
  shrinks just as fast.
- **Inside the floor:** no. The closest any run comes is a p90 ratio of 1.50 (PersonaPlex
  1/2, epoch 4). The median ratio never goes below 2.09.

These are single runs on 65 holdout turns with no noise estimate. Differences under about 0.3
in the ratio, such as 2.92 against 2.87 or 2.48 against 2.45, are within noise.

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
- **Best epoch:** `make bar`'s "best epoch" is the lowest median ratio, so late epochs win it.
  From epoch 1 to epoch 8 the gap grows about 4 to 6 times and the floor about 5 to 8 times as
  every run overfits. The ratio improves partly because the floor inflates.
- **Lowest gap:** the absolute gap is lowest at epoch 1 or 2. Held-out text loss is lowest at
  epoch 2 or 3.
- **Door control:** the door, KL(C || A), lands within rounding of the floor in every row and
  under it in nearly all. The largest excess is 0.0014 (PersonaPlex 1/2, epoch 5). So the
  harness and the TTS are sane.
- **All frames:** on all frames, trained runs sit about 3 to 19 times above the floor. Those
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

Under reading 1 (the best epoch of 8), data is the lever, and the extrapolations differ by a
factor of 3 to 7:
- **Whole-ladder slope:** about 10.5% per doubling. That reaches the floor after about 6.6 more
  doublings, around 11,000 dialogues, or about 170 h of rendering at the design's 15 h per
  1,000.
- **Last-rung slope:** 14 to 18% per doubling. That takes about 4 to 5 doublings, roughly
  1,600 to 3,300 dialogues, or 25 to 50 h of rendering.
- **Caveat:** both rest on best-of-8 picks from overfit epochs.

Reading 2 (the least-overfit epoch) points the other way:
- **The ratio is flat at about 3.** Data alone does not close it in this recipe.
- **Overfitting comes first.** Every run starts overfitting after epoch 2 or 3, even at the
  full set. That makes the recipe a candidate lever: LR, rank, regularisation, or fewer
  epochs.

Either way:
- **Cut the epochs to 3 or fewer.** The next ladder can run 3 epochs or fewer. A full-set
  PersonaPlex run would drop from 7 h to about 2.7 h at this corpus size.
- **Measure noise.** Repeat one rung with a second seed so that differences of 0.3 in the
  ratio can be told from noise.
