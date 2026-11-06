# Side by side 3: is the lesson opening already broken at epoch 1?

Vlad, 2026-11-06, at B10's GATE ("The pilot listens mid-lesson but goes silent after the spoken opening question. Which fix next?"): **"Check epoch 1 first"**. B10 found the consistency pilot (`cons1-pp-l10`, λ = 10) silent after the pupil's spoken first question at epoch 2 (13/13 after A, 10/13 after B). This batch decodes epoch 1 (R_gap 1.90, held text loss +7.7%) with B10's job unchanged, to see whether that collapse grows with training or comes with the term from the start. No training, no rendering, no code.

## The answer

**At epoch 1 the opening is not silent, but it is already damaged, and the body of the answer is damaged too. The term at λ = 10 hurts from the first epoch; more training moved the damage, it did not create it.**

- **No silence anywhere.** After the spoken opening question epoch 1 speaks in 13/13 (A) and 13/13 (B), where epoch 2 was silent in 13/13 and 10/13. Its typed opening starts on frame 2-5, not epoch 2's 8-9.
- **The opening is already off.** Blind judges, same pool for all three checkpoints:
  - it takes up the pupil's question in 8/13 typed, 11/13 A, 10/13 B (baseline 13/11/12, epoch 2 8/0/1);
  - its move matches the tape tutor's in only **1/13 typed**, 6/13 A, 4/13 B (baseline 12/8/8);
  - judges call it well formed in only **5/13 typed**, 6/13 A, 9/13 B (baseline 11/12/12);
  - 6 of the 13 typed openings stop mid-sentence with a fifth or more of the window left, and 5 are ten words or fewer ("it's a"). The baseline does each once.
- **Mid-lesson the answer stops early.** On turns 2-8 epoch 1 still takes up the pupil (51/49/50 of 52, baseline 52/52/50), but:
  - it falls silent mid-sentence with a fifth or more of the window left in **20/52 typed, 11/52 A and 17/52 B** answers. The baseline does this in 1/2/0 and epoch 2 in 0/1/1;
  - median length is 41/57/47 words against 66/67/66 for the baseline;
  - judges call 36/40/38 of 52 well formed, against 49/49/48 (baseline) and 47/51/51 (epoch 2).
  - This matches epoch 1's held text loss (+7.7%, against +1.0% at epoch 2).
- **Tracking slipped already at epoch 1.** The answer takes up the pupil turn actually played in 95/104 swap-test items (baseline 103/104, epoch 2 94/104; McNemar 9 vs 1 discordant, p ≈ 0.02).
- **Teacher-forced, epoch 1 sits between the two.** The median turn-0 lead-frame KL(typed‖A) is 0.54, against 0.74 for the baseline and 0.095 at epoch 2. So the "agreement by not starting" that B10 saw at epoch 2 is mostly not there yet at epoch 1.

So the lever is the **loss weight**, not training length. An early stop has no healthy checkpoint to stop at: epoch 1 truncates answers mid-lesson and makes a poor opening, and epoch 2 traded the truncation for a silent spoken opening. Leaving the opening out of the term would address epoch 2's symptom only. Epoch 1's truncation sits in turns 2-8, which would stay in the term.

## Counts: baseline, epoch 1, epoch 2

All three checkpoints were judged in one shuffled blind pool (B10's rubrics, 13 opus judges). The baseline is `ladder-personaplex-1` epoch 2 (B7's decode and B10's swap rows). Epoch 2 is B10's decode, re-judged here. These judges agree with B10's labels on the shared items in 238/260 pairs, 382/390 answers and 202/208 swap items, so the B10 columns below are B10's own numbers and the others this batch's.

### Uptake: the opening takes up what the pupil said (single-answer judges)

| after | baseline, all | epoch 1, all | epoch 2, all | baseline, turns 2-8 | epoch 1, turns 2-8 | epoch 2, turns 2-8 | baseline, turn 0 | epoch 1, turn 0 | epoch 2, turn 0 |
|---|---|---|---|---|---|---|---|---|---|
| typed | 65/65 | 59/65 | 58/65 | 52/52 | 51/52 | 50/52 | 13/13 | 8/13 | 8/13 |
| A | 63/65 | 60/65 | 50/65 | 52/52 | 49/52 | 50/52 | 11/13 | 11/13 | **0/13** |
| B | 62/65 | 60/65 | 51/65 | 50/52 | 50/52 | 50/52 | 12/13 | 10/13 | **1/13** |

B10's own labels for comparison: baseline 52/50/50 and epoch 2 50/50/49 on turns 2-8; turn 0 baseline 13/12/11, epoch 2 8/0/2.

### Turn 0 in detail (13 openings each)

| | baseline typed / A / B | epoch 1 typed / A / B | epoch 2 typed / A / B |
|---|---|---|---|
| silent | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 13 / 10 |
| onset frame (tape: 3) | 3 / 2 / 2-3 | 3-5 / 0-2 / 0-1 | 8-9 / - / 3 |
| takes up the question | 13 / 11 / 12 | 8 / 11 / 10 | 8 / 0 / 1 |
| same move as the tape | 12 / 8 / 8 | **1** / 6 / 4 | 5 / 0 / 0 |
| well formed | 11 / 12 / 12 | **5** / 6 / 9 | 8 / 0 / 3 |
| stops mid-sentence with ≥ 20% of the window left | 1 / 1 / 1 | 6 / 4 / 4 | 3 / 0 / 0 |

### Substance (blind pair judges)

| pair | baseline, all | epoch 1, all | epoch 2, all | baseline, turns 2-8 | epoch 1, turns 2-8 | epoch 2, turns 2-8 |
|---|---|---|---|---|---|---|
| typed vs A differs | 20/65 | 18/65 | 25/65 | 15/52 | 10/52 | 14/52 |
| A vs B differs | 22/65 | 19/65 | 20/65 | 20/52 | 12/52 | 17/52 |

B10's labels: typed vs A on turns 2-8, baseline 17/52 and epoch 2 18/52; A vs B 21 and 19. Epoch 1 differs least, but its answers are shorter and the judges read mainly the first two sentences, so part of that is less content to differ on. It is not a clear behavioural gain.

Same move as the tape tutor, turns 2-8, typed / A / B: baseline 34/32/33, epoch 1 33/32/31, epoch 2 33/31/30.

### The swap: which pupil turn does the answer take up?

Turns 2 and 6 of every holdout dialogue, typed and A, with B10's donors (the same map).

| model | condition | typed: played / own / both / neither | A: played / own / both / neither |
|---|---|---|---|
| baseline | true | 26 / - / 0 / 0 | 26 / - / 0 / 0 |
| | swapped | 26 / 0 / 0 / 0 | 25 / 0 / 0 / 1 |
| epoch 1 | true | 24 / - / 1 / 1 | 23 / - / 0 / 3 |
| | swapped | 25 / 0 / 0 / 1 | 23 / 0 / 0 / 3 |
| epoch 2 | true | 24 / - / 1 / 1 | 23 / - / 0 / 3 |
| | swapped | 22 / 1 / 0 / 3 | 25 / 0 / 1 / 0 |

Played turn taken up: baseline 103/104, epoch 1 95/104 (McNemar 9 vs 1, p ≈ 0.02), epoch 2 94/104 (10 vs 1, p ≈ 0.01; B10 had the same 94). Epoch 1 never follows the dialogue's own turn after a swap; its misses are "neither".

### Onset, length and endings (`python -m dual.sbs_report --quality`, plus a span count)

| model | after | n | silent | onset 0 / 1 / 2 / 3 / later | median words | repetitive | unfinished | stops mid-sentence, turns 2-8 | ends on a question, turns 2-8 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | typed | 65 | 0 | 0 / 0 / 0 / 65 / 0 | 66 | 3 | 17 | 1/52 | 34/52 |
| | A | 65 | 0 | 52 / 0 / 13 / 0 / 0 | 67 | 6 | 16 | 2/52 | 37/52 |
| | B | 65 | 0 | 52 / 0 / 12 / 1 / 0 | 66 | 5 | 13 | 0/52 | 38/52 |
| epoch 1 | typed | 65 | 0 | 0 / 0 / 37 / 22 / 6 | **41** | 3 | **32** | **20/52** | 16/52 |
| | A | 65 | 0 | 53 / 11 / 1 / 0 / 0 | 57 | 10 | 23 | **11/52** | 26/52 |
| | B | 65 | 0 | 60 / 5 / 0 / 0 / 0 | **47** | 6 | **27** | **17/52** | 23/52 |
| | typed, swapped | 26 | 0 | 0 / 0 / 16 / 10 / 0 | 40 | 3 | 11 | | |
| | A, swapped | 26 | 0 | 26 / 0 / 0 / 0 / 0 | 40.5 | 4 | 10 | | |
| epoch 2 | typed | 65 | 2 | 0 / 0 / 0 / 49 / 14 | 64 | 7 | 11 | 0/52 | 42/52 |
| | A | 65 | 13 | 52 / 0 / 0 / 0 / 0 | 61 | 7 | 6 | 1/52 | 41/52 |
| | B | 65 | 10 | 52 / 0 / 0 / 3 / 0 | 62 | 5 | 7 | 1/52 | 42/52 |

- **Stops mid-sentence:** the last non-`<pad>` text frame is before 80% of the answer window and the text does not end in `.`, `?` or `!`. The window is the tape answer's length, so a healthy answer runs to near its end (the baseline's median spoken span is 0.84-0.88 of the window on turns 2-8; epoch 1's is 0.50 typed, 0.71 A, 0.48 B).
- **Typed onset moved earlier** at epoch 1: frame 2 in 37/65, where the tape and the baseline open on frame 3. At epoch 2 it moved later (frame 8-9 at turn 0). Spoken onset stays on frame 0-2, the TTS artefact B7 traced.
- **First word, all turns where both spoke** (typed vs A / A vs B / typed vs B): baseline 48/52/43 of 65, epoch 1 59/59/58 of 65, epoch 2 44/46/48 of 52. Epoch 1's openings are the most uniform.

## What the training log and B9's measure say (no GPU)

- **The term is logged as one number per step, never per turn.** The train log prints `cons` every 10th step and a mean per epoch: 0.0081, 0.0050, 0.0052. It fell 3-5× within the first ~50 steps (0.049 at step 10, 0.033 at step 20) and then sat near 0.005-0.01. Nothing gives the opening's share, so the question "how large was the term on opening answers" is answered only through the measure, below.
- **The term never sees the lead frames.** It covers content frames only (target text not `<pad>`). The 3 lead frames where B10's "agreement by not starting" shows are outside the loss at every turn. Turn 0 holds 117 of 585 training pairs and 20.7% of the term's content frames, the same weight as each of turns 2, 4, 6 and 8.
- **Turn 0 is where the term pulled hardest at the start.** The median content-frame gap KL(typed‖A) by turn in B9's measure:

| model | turn 0 | turn 2 | turn 4 | turn 6 | turn 8 |
|---|---|---|---|---|---|
| bare PersonaPlex | **0.093** | 0.047 | 0.039 | 0.042 | 0.031 |
| baseline epoch 2 | 0.030 | 0.025 | 0.024 | 0.025 | 0.023 |
| pilot epoch 1 | 0.0065 | 0.0067 | 0.0053 | 0.0060 | 0.0063 |
| pilot epoch 2 | 0.0061 | 0.0057 | 0.0051 | 0.0073 | 0.0091 |

  By epoch 1 the term had flattened every turn to about 0.006, turn 0 included.
- **Turn-0 lead frames, median KL(typed‖A):** bare 2.51, baseline 0.74, epoch 1 0.54, epoch 2 0.095, epoch 3 0.061. The first content frame: 0.35, 0.10, 0.024, 0.011, 0.016. So the lead-frame closure B10 saw happens between epoch 1 and 2, and it came mostly from the spoken row moving toward the tape at the lead, not the typed row: KL(tape‖A) on turn-0 lead frames is 0.47 (baseline), 0.12 (epoch 1) and 0.037 (epoch 2), while KL(typed‖tape) goes 0.034, 0.77, 0.070. The term reaches the lead frames only through shared weights.
- **B10's untested reading (both lean toward `<pad>`) cannot be tested without a GPU.** The measure stores only per-frame KLs for its five pairs, with no probabilities, entropy or argmax. Testing it needs a measure job that stores P(`<pad>`) on lead and first content frames for typed, A and B, at turn 0 and mid-lesson. The free decodes are consistent with it but do not measure mass: epoch 1 stops mid-answer and epoch 2 does not start, both ways of emitting `<pad>` where the tape speaks.

## Examples (truncated)

**The opening** (dlg-00005-038 turn 0). The pupil asks: "hello, so what does a derivative actually tell you about a curve?"
- baseline, typed: "it tells you how fast the height changes, so let's work one out. I expect this curve to climb to a hump, then fall to a dip..."
- epoch 2, A and B: (silence)
- epoch 1, A: "it tells you how fast a curve moves, so I expect a curve that climbs fast and falls fast, since it climbs fast and falls fast..."
- epoch 1, B: "it tells you how steep a curve is, so I expect a hump to be steepest at its middle..."

**An opening that stops** (dlg-00005-091 turn 0). The pupil asks: "hi, how would I find a curve's slope at one point?"
- baseline, A: "you find a curve's slope at one point by nudging the input a tiny bit and seeing how far the output moves..."
- epoch 1, typed: "it's a natural question, and I'd say it depends on how you define a curve. if you take" (and nothing more)
- epoch 1, B: "it's a" (and nothing more)
- epoch 2, A and B: (silence)

**Mid-lesson, stopping before the hand-back** (dlg-00005-123 turn 2, B; 143 of 339 frames spoken): "...so I hold the curve climbs one for each step along at one, since the input is one." The tape and the baseline go on to the next question for the pupil.

Every answer, all three checkpoints side by side, is in `side-by-side-3/sbs-report.md`.

## Verdict

The opening is not broken at epoch 1 in the way it is at epoch 2: no silence, and the spoken openings take the question up about as often as the baseline's. But epoch 1 is not healthy either. Its typed opening rarely makes the tape's move (1/13) and is often malformed (well formed in 5/13), and mid-lesson a third of its typed and B answers stop mid-sentence with half the window left. The swap shortfall is already there (95/104). Between epoch 1 and 2 the held text loss recovered (+7.7% to +1.0%) and the mid-answer truncation went away, while the spoken opening went silent. That pattern fits one cause, the λ = 10 term, which pulls toward not speaking in different places as training proceeds. It does not fit a cause that only appears with length. So:
- **Training length is not the lever.** There is no epoch to stop at.
- **The loss weight is.** The damage is present from the first epoch, and it is not confined to the opening.
- **The opening's place in the term matters, but is not enough on its own.** Turn 0 started with the largest gap (0.093 on the bare model, twice the others), so the term pulled hardest there, and epoch 2's collapse is there. But epoch 1's truncation is in turns 2-8, which would stay in the term.

## What was run

- **Job (image `sbs2:72868db`, B10's; drama3 stopped, watchdog on):** `job-20261106-121325-vlad-sbs-3`, `--run cons1-pp-l10 --epoch 1 --no-bare --swap-turns 2,6 --mem-cap-gib 64`. 195 true and 52 swap rows, 2,720 s of decoding (2,886 s with load), peak 38.9 GiB, about 48 GPU-minutes. The true rows ran in B7's and B10's exact batch shapes (53/35/29/26/23/21/8), the swap rows in B10's (28/24), with B10's donor map. Adapter norm 26.75 (epoch 2: 27.16).
- **No code changed.** The job already selected the epoch.
- **Judging:** 13 blind opus judges over 390 pairs, 585 answers and 312 swap items (all three checkpoints pooled, seed 13). Prompts in `side-by-side-3/judge-prompts.txt`, unblinded labels in `judge-labels.jsonl`.
- **Side reads (no GPU):** the pilot's train log (`run-20261105-161703-vlad-cons1-pp-l10`) and B9's measure rows for epochs 1-3, with B8's per-frame rerun for the baseline and the bare model. The turn-0 epoch-2 numbers reproduce B10's exactly (0.0953 / 0.7365 / 0.0113 / 0.1033).
- **Files in `docs/results/side-by-side-3/`:** `answers.jsonl` (epoch 1's 247 rows, model `cons1-e1`; the baseline's are in `side-by-side-1/` and `side-by-side-2/`, epoch 2's in `side-by-side-2/`), `meta-sbs-3.json`, `judge-prompts.txt`, `judge-labels.jsonl`, `sbs-report.md`.

## Caveats

- **Greedy, one path per row, bf16.** B7's control showed 6/30 rows change substance under another batch shape. The batch shapes here are identical to B10's, so the epoch comparison is not confounded by shape, but individual answers near a tie can flip.
- **One label per item, model judges.** Agreement with B10's labels on the shared items is 92-98%. Turn-0 cells have 13 items each.
- **"Stops mid-sentence" is a text-stream count,** not judged. The window is cut at the tape answer's length, so "unfinished" alone is partly by design. The 80%-and-no-final-punctuation rule separates a cut by the window from a stop.
- **One seed, λ = 10.** Whether λ = 3 avoids both failures is the next question, not something this batch shows.

## The gate

- **(ii) λ = 3, same recipe and seed** (~3.6 GPU-hours of training, ~1.25 of measure, ~1 of side-by-side at epochs 1 and 2). It tests the lever this batch points at. **Recommended.**
- **(i) λ = 10 without the opening answer in the term** (a small train-side change, ~6 GPU-hours). It targets epoch 2's silent opening, but not epoch 1's mid-lesson truncation, which is in turns that stay in the term.
- **(iii) Fewer epochs or an early stop.** Not supported: epoch 1 is not healthy (truncated answers, a poor typed opening, held text +7.7%).
- **(iv) TTS-voiced teacher rows in voice D** (new render path, ~6.5 GPU-hours, needs a plan). It would likely inherit the same pull toward silence at λ = 10.
