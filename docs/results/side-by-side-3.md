# Side by side 3: is the lesson opening already broken at epoch 1?

Vlad, 2026-11-06, at B10's GATE ("The pilot listens mid-lesson but goes silent after the spoken opening question. Which fix next?"): **"Check epoch 1 first"**. B10 found the consistency pilot (`cons1-pp-l10`, λ = 10) silent after the pupil's spoken first question at epoch 2 (13/13 after A, 10/13 after B). This batch decodes epoch 1 (R_gap 1.90, held text loss +7.7%) with B10's job unchanged, to see whether that collapse grows with training or comes with the term from the start. No training, no rendering, no code.

## The answer

**The silent opening is not there at epoch 1: it grows with training. But epoch 1 is not healthy either. It answers in fragments, often stopping after a word or two. So at λ = 10 no decoded epoch is healthy, and there is no epoch to stop at. Which lever fixes it, λ or the opening's place in the term, this batch cannot say.**

- **No silence anywhere.** After the spoken opening question epoch 1 speaks in 13/13 (A) and 13/13 (B), where epoch 2 was silent in 13/13 and 10/13. Its typed opening starts on frame 3-5, not epoch 2's 8-9.
- **The opening is weaker than the baseline's.** Blind judges, same pool for all three checkpoints:
  - it takes up the pupil's question in 8/13 typed, 11/13 A, 10/13 B (baseline 13/11/12, epoch 2 8/0/1);
  - its move matches the tape tutor's in only **1/13 typed**, 6/13 A, 4/13 B (baseline 12/8/8);
  - judges call it well formed in only **5/13 typed**, 6/13 A, 9/13 B (baseline 11/12/12);
  - 6 of the 13 typed openings stop mid-sentence with a fifth or more of the window left, and 5 are ten words or fewer ("it's a"). The baseline does each once.
- **Mid-lesson it answers in fragments.** On turns 2-8 epoch 1 still takes up the pupil (51/49/50 of 52, baseline 52/52/50), but:
  - it stops mid-sentence with a fifth or more of the window left in **20/52 typed, 11/52 A and 17/52 B** answers (baseline 1/2/0, epoch 2 0/1/1). Most of these stop almost at once: the median stopped answer ends at 0.08-0.11 of the window, and 12/52 typed and 12/52 B answers are ten words or fewer ("yes, zero point");
  - median length on turns 2-8 is 41/57/38 words against 64.5/66/67 for the baseline;
  - judges call 36/40/38 of 52 well formed, against 49/49/48 (baseline) and 47/51/51 (epoch 2).
  - This matches epoch 1's held text loss: 0.593, against the baseline's own epoch 1 at 0.551 (+7.7%). By epoch 2 it is +1.0% and the fragments are gone.
- **The swap shortfall at epoch 1 is mostly the fragments.** The answer takes up the pupil turn actually played in 95/104 swap-test items (baseline 103/104, epoch 2 94/104; McNemar 9 vs 1, p ≈ 0.02). But all 9 of epoch 1's misses are broken answers: 6 fragments of 1-4 words, 1 loop ("one point nine nine nine...") and 2 that stop mid-sentence. Epoch 2's 10 misses are full answers that miss the pupil's turn.
- **Teacher-forced, epoch 1 sits between the two.** The median turn-0 lead-frame KL(typed‖A) is 0.54, against 0.74 for the baseline, 0.095 at epoch 2 and 0.061 at epoch 3. B10's "agreement by not starting" grows over epochs 1-3, as the silence does.

**Two readings fit, and this batch cannot separate them.**
- **One cause.** The λ = 10 term pulls toward not speaking from the start. It shows as fragments at epoch 1 and as a silent spoken opening at epoch 2.
- **Two causes.**
  - The term slowed text learning in epoch 1. An under-trained adapter sits between the bare model, which is silent on all 156 mid-lesson answers in B7, and a trained one, so it answers in fragments. Epoch 2 cures that.
  - Separately, the silent opening grows with training.

The baseline's own epoch 1 was never decoded, so whether its answers also fragment is unknown. It is the cheap test between the readings (~45 GPU-minutes, no code).

## Counts: baseline, epoch 1, epoch 2

All three checkpoints were judged in one shuffled blind pool (B10's rubrics, 13 opus judges). The baseline is `ladder-personaplex-1` epoch 2 (B7's decode and B10's swap rows). Epoch 2 is B10's decode, re-judged here. **Every cell in the tables is this batch's labels**; B10's own labels are in the line after each table. On the shared baseline and epoch-2 items these judges agree with B10's on 238/260 substance labels, 382/390 uptake labels (363/390 on well formed) and 202/208 swap labels.

### Uptake: the opening takes up what the pupil said (single-answer judges)

| after | baseline, all | epoch 1, all | epoch 2, all | baseline, turns 2-8 | epoch 1, turns 2-8 | epoch 2, turns 2-8 | baseline, turn 0 | epoch 1, turn 0 | epoch 2, turn 0 |
|---|---|---|---|---|---|---|---|---|---|
| typed | 65/65 | 59/65 | 58/65 | 52/52 | 51/52 | 50/52 | 13/13 | 8/13 | 8/13 |
| A | 63/65 | 60/65 | 50/65 | 52/52 | 49/52 | 50/52 | 11/13 | 11/13 | **0/13** |
| B | 62/65 | 60/65 | 51/65 | 50/52 | 50/52 | 50/52 | 12/13 | 10/13 | **1/13** |

B10's own labels: on turns 2-8, baseline 52/50/50 and epoch 2 50/50/49; at turn 0, baseline 13/12/11 and epoch 2 8/0/2.

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

B10's own labels on turns 2-8: typed vs A, baseline 17/52 and epoch 2 18/52; A vs B, 21 and 19. Epoch 1 differs least, but many of its answers are fragments, and two fragments that both say "yes, zero" cannot differ much. This is not a behavioural gain.

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

The played turn is taken up in 103/104 for the baseline, 95/104 at epoch 1 (McNemar 9 vs 1, p ≈ 0.02) and 94/104 at epoch 2 (10 vs 1, p ≈ 0.01; B10 also had 94). Epoch 1 never follows the dialogue's own turn after a swap. Its misses are "neither", and every one is a fragment, a loop or a mid-sentence stop.

### Onset, length and endings (`python -m dual.sbs_report --quality`, plus a span count)

| model | after | n | silent | onset 0 / 1 / 2 / 3 / later | median words | repetitive | unfinished | stops mid-sentence, turns 2-8 | ends on a question, turns 2-8 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | typed | 65 | 0 | 0 / 0 / 0 / 65 / 0 | 66 | 3 | 17 | 1/52 | 34/52 |
| | A | 65 | 0 | 52 / 0 / 13 / 0 / 0 | 67 | 6 | 16 | 2/52 | 37/52 |
| | B | 65 | 0 | 52 / 0 / 12 / 1 / 0 | 66 | 5 | 13 | 0/52 | 38/52 |
| | typed, swapped | 26 | 0 | 0 / 0 / 0 / 26 / 0 | 67 | 1 | 6 | | |
| | A, swapped | 26 | 0 | 26 / 0 / 0 / 0 / 0 | 61 | 0 | 1 | | |
| epoch 1 | typed | 65 | 0 | 0 / 0 / 37 / 22 / 6 | **41** | 3 | **32** | **20/52** | 16/52 |
| | A | 65 | 0 | 53 / 11 / 1 / 0 / 0 | 57 | 10 | 23 | **11/52** | 26/52 |
| | B | 65 | 0 | 60 / 5 / 0 / 0 / 0 | **47** | 6 | **27** | **17/52** | 23/52 |
| | typed, swapped | 26 | 0 | 0 / 0 / 16 / 10 / 0 | 40 | 3 | 11 | | |
| | A, swapped | 26 | 0 | 26 / 0 / 0 / 0 / 0 | 40.5 | 4 | 10 | | |
| epoch 2 | typed | 65 | 2 | 0 / 0 / 0 / 49 / 14 | 64 | 7 | 11 | 0/52 | 42/52 |
| | A | 65 | 13 | 52 / 0 / 0 / 0 / 0 | 61 | 7 | 6 | 1/52 | 41/52 |
| | B | 65 | 10 | 52 / 0 / 0 / 3 / 0 | 62 | 5 | 7 | 1/52 | 42/52 |
| | typed, swapped | 26 | 0 | 0 / 0 / 0 / 24 / 2 | 59.5 | 3 | 0 | | |
| | A, swapped | 26 | 0 | 26 / 0 / 0 / 0 / 0 | 59 | 2 | 2 | | |

Median words here are over all turns.

- **Stops mid-sentence:** the last non-`<pad>` text frame comes before 80% of the answer window, and the text does not end in `.`, `?` or `!`. The window is the tape answer's length, so a healthy answer runs close to its end. On turns 2-8 the baseline's median spoken span is 0.84-0.88 of the window. Epoch 1's is 0.50 typed, 0.71 A and 0.48 B.
- **Typed onset moved earlier at epoch 1:** frame 2 in 37/65, all mid-lesson, where the tape and the baseline open on frame 3. At epoch 2 it moved later, to frame 8-9 at turn 0. Spoken onset stays on frame 0-2, the TTS artefact B7 traced.
- **First word**, turns where both answers spoke (typed vs A / A vs B / typed vs B):
  - baseline 48/65, 52/65, 43/65;
  - epoch 1 59/65, 59/65, 58/65;
  - epoch 2 44/52, 46/52, 48/54.

  Epoch 1's openings are the most uniform, mostly "yes".

## What the training log and B9's measure say (no GPU)

- **The term is logged as one number per step, never per turn.** The train log prints `cons` every 10th step and one mean per epoch: 0.0081, 0.0050, 0.0052. The means include steps with no typed rows, which log 0: 10 of the 29 logged steps in epoch 1. In the first ~50 steps the term fell 3-5× (0.049 at step 10, 0.033 at step 20), then sat near 0.005-0.01. Nothing gives the opening's share, so the question "how large was the term on opening answers" is answered only through the measure, below.
- **The term never sees the lead frames.** It covers content frames only, where the target text is not `<pad>`. The 3 lead frames, where B10's "agreement by not starting" shows, are outside it at every turn; text cross-entropy still covers them. The term does see the first content frames, where `<pad>` competes with the first word, so a late start can be rewarded there directly. Turn 0 holds 117 of the 585 training pairs and 20.7% of the term's content frames; turns 2-8 hold 17.8-20.7%. The term is a per-frame mean, so dropping turn 0 would not lower the weight on the other turns.
- **Turn 0 is where the term pulled hardest at the start.** Median content-frame gap by turn in B9's measure:

  | model | turn 0 | turn 2 | turn 4 | turn 6 | turn 8 |
  |---|---|---|---|---|---|
  | bare PersonaPlex | **0.093** | 0.047 | 0.039 | 0.042 | 0.031 |
  | baseline epoch 2 | 0.030 | 0.025 | 0.024 | 0.025 | 0.023 |
  | pilot epoch 1 | 0.0065 | 0.0067 | 0.0053 | 0.0060 | 0.0063 |
  | pilot epoch 2 | 0.0061 | 0.0057 | 0.0051 | 0.0073 | 0.0091 |

  This is KL(typed‖A), the bar's gap. The term itself uses the tape pair, KL(typed‖tape), and on the bare model that gives the same order: 0.096, 0.054, 0.039, 0.035 and 0.030. So turn 0 started at 2-3× the other turns. By epoch 1 the term had flattened every turn to about 0.006, turn 0 included.
- **Turn-0 lead frames, median KL(typed‖A).** Bare 2.51, baseline 0.74, epoch 1 0.54, epoch 2 0.095, epoch 3 0.061. The first content frame goes 0.35, 0.10, 0.024, 0.011, 0.016.
  - Against the baseline, the spoken row moved toward the tape at the lead: KL(tape‖A) is 0.47, 0.12 and 0.037 at baseline, epoch 1 and epoch 2.
  - The typed row detoured at epoch 1: KL(typed‖tape) is 0.034, 0.77 and 0.070. So between epochs 1 and 2 the typed row moved most.
- **B10's untested reading cannot be tested without a GPU.** That reading is that typed and spoken both lean toward `<pad>`. The measure stores only per-frame KLs for its five pairs: no probabilities, entropy or argmax. Testing it needs a measure job that stores P(`<pad>`) on the lead frames and the first content frame, for typed, A and B, at turn 0 and mid-lesson. The free decodes fit the reading but do not measure mass.

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

**Mid-lesson fragments: epoch 1's swap misses.** These are the whole answers: "yes,", "yes, zero point", "yes, minus nine times", "yes, minus one" and "yes, zero". The baseline and epoch 2 answer the same turns in full.

Every answer, all three checkpoints side by side, is in `side-by-side-3/sbs-report.md`.

## Verdict

The opening is not broken at epoch 1 the way it is at epoch 2. There is no silence, and the spoken openings take the question up about as often as the baseline's do. So the silent opening grows with training: the turn-0 lead-frame agreement tightens from 0.54 to 0.095 to 0.061 over epochs 1-3.

Epoch 1 is not healthy either:
- Its typed opening rarely makes the tape's move (1/13) and is often malformed (well formed in 5/13).
- Mid-lesson, a fifth to two fifths of its answers break off within the first tenth of the window.
- Its held text loss is +7.7% over the baseline's own epoch 1.

Between epochs 1 and 2, held text recovered (+1.0%) and the fragments went away, while the spoken opening went silent.

What follows:
- **An early stop is not an option at λ = 10.** No decoded epoch is healthy: epoch 1 fragments and epoch 2 is silent at the spoken opening. Epoch 3 was not decoded, but its lead-frame agreement is tighter still.
- **The loss weight is the candidate left, untested here.** This batch varied the epoch, not λ. Whether epoch 1's fragments come from the term or from ordinary under-training is open until the baseline's own epoch 1 is decoded.
- **The opening's place in the term matters but is unproven as the fix.**
  - Turn 0 started with the largest gap, 2-3× the other turns, so the term pulled hardest there. Epoch 2's collapse is there.
  - But the term never sees the lead frames, so leaving turn 0 out is not sure to restore the start.
  - Epoch 2's mid-lesson swap misses (10 vs 1, full answers that miss the turn) sit in turns that would stay in the term.

## What was run

- **Job (B10's image `sbs2:72868db`; drama3 stopped, watchdog on):** `job-20261106-121325-vlad-sbs-3`, `--run cons1-pp-l10 --epoch 1 --no-bare --swap-turns 2,6 --mem-cap-gib 64`.
  - 195 true and 52 swap rows.
  - 2,720 s of decoding, about 48 GPU-minutes with model load. Peak 38.9 GiB.
  - The true rows ran in B7's and B10's exact batch shapes (53/35/29/26/23/21/8). The swap rows ran in B10's (28/24), with B10's donor map.
  - Adapter norm 26.75 (epoch 2: 27.16).
- **No code changed.** The job already selected the epoch.
- **Judging:** 13 blind opus judges over 390 pairs, 585 answers and 312 swap items. All three checkpoints were pooled (seed 13). Prompts are in `side-by-side-3/judge-prompts.txt`, unblinded labels in `judge-labels.jsonl`.
- **Side reads (no GPU):**
  - the pilot's train log (`run-20261105-161703-vlad-cons1-pp-l10`);
  - B9's measure rows for epochs 1-3;
  - B8's per-frame rerun for the baseline and the bare model.

  The turn-0 epoch-2 numbers reproduce B10's exactly (0.0953 / 0.7365 / 0.0113 / 0.1033).
- **Files in `docs/results/side-by-side-3/`:**
  - `answers.jsonl`: epoch 1's 247 rows, model `cons1-e1`. The baseline's rows are in `side-by-side-1/` and `side-by-side-2/`, and epoch 2's in `side-by-side-2/`.
  - `meta-sbs-3.json`, `judge-prompts.txt`, `judge-labels.jsonl` and `sbs-report.md`.

## Caveats

- **The baseline's epoch 1 was never decoded.** Without it, epoch 1's fragments cannot be attributed to the term rather than to early training.
- **Greedy, one path per row, bf16.** B7's control showed 6/30 rows change substance under another batch shape. The batch shapes here match B10's, so the shape does not confound the epoch comparison, but individual answers near a tie can flip.
- **One label per item, from model judges.** Agreement with B10's labels on the shared items is 91-98% (93% on well formed). The turn-0 cells have 13 items each.
- **"Stops mid-sentence" is a text-stream count, not a judgment.** The window is cut at the tape answer's length, so "unfinished" alone is partly by design. The rule of 80% plus no final punctuation separates a cut by the window from a stop.
- **One seed, λ = 10.**

## The gate

- **(ii) λ = 3, same recipe and seed.** About 6.5 GPU-hours: 3.6 of training, 1.25 of measure and 1.6 of side-by-side at epochs 1 and 2. It changes one thing, B9's held-text trigger already fired for it, and it adds a second λ point. **Recommended**, with the baseline's epoch-1 decode (~45 GPU-minutes, no code) run first or alongside, so that epoch 1's fragments can be read.
- **(i) λ = 10 without the opening answer in the term.** A small train-side change, ~6.5 GPU-hours. It targets epoch 2's silent opening, but that fix is untested: the term never sees the lead frames. Epoch 2's mid-lesson swap misses sit in turns that stay in the term.
- **(iii) Fewer epochs or an early stop.** Not supported. Epoch 1 is not healthy (fragments, a poor typed opening, held text +7.7%).
- **(iv) TTS-voiced teacher rows in voice D.** A new render path, ~6.5 GPU-hours, and it needs a plan. At λ = 10 it would likely inherit the same failures.
