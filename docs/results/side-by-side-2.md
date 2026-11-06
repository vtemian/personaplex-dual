# Side by side 2: does the tutor still listen after the consistency term?

Vlad, 2026-11-06, at GATE-1 of the consistency pilot (`docs/results/consistency-1.md`), on step one of option (b): **"do step by step"**. B9's term cut the content ratio from 3.00 to 1.90 (1.67 at epoch 2), but every KL shrank with it, the floor included (×0.36 per turn). That fits two readings:
- the answer became less sensitive to *how* the pupil turn arrived (healthy);
- it became less sensitive to the pupil turn *itself* (bad).

This batch reads the pilot's greedy answers, as B7 did for the baseline, and adds a direct test of the second reading: swap in another dialogue's pupil turn and see whether the answer follows it. There was no training and no rendering.

## The answer

**Mid-lesson the tutor still listens. The lesson's opening is broken. So epoch 2 is not healthy as a checkpoint.**

- **Not deaf to the pupil.**
  - On pupil turns 2-8, the pilot takes up what the pupil said in **50/52 after typed, 50/52 after A and 49/52 after B**. The baseline, judged by the same judges, does so in 52/50/50.
  - Swap a different pupil turn in, and the pilot's answer follows the swapped turn in **46/52**. It follows the dialogue's own turn only **once**, the one outcome the bad reading predicts.
  - Typed vs A differs in substance in **18/52** turns, against **17/52** for the baseline. A vs B differs in 19/52, against 21/52. So typed and spoken are as alike as before, and the voices no more alike than before.
- **Tracking is slightly weaker.**
  - Across all 104 swap-test answers, the pilot's opening takes up the pupil turn actually played in **94/104**, against **103/104** for the baseline (Fisher p ≈ 0.01).
  - Its 10 misses are mostly "neither": it answers with a number nobody said, such as "about twice the step" to "one point six eight".
- **The opening is broken: pupil turn 0 is the lesson's first question.**
  - After a spoken opening question, the pilot says **nothing for the whole ~30 s answer window in 13/13 turns after A and 10/13 after B**. The baseline always spoke.
  - After the typed opening it is silent in 2/13. It starts 5-6 frames late in the other 11, on frame 8-9 instead of the tape's 3. It takes up the question in only 8/13 (baseline 13/13), and several answers stop after "so let's".
  - This is B9's "position 0 diverging" (first-word gap 0.012 → 0.031 → 0.28 over epochs 1-3), already visible at epoch 2 in free-running decoding.
- **The rest of the answer looks fine.**
  - There are no silent answers after turn 0, and none in the swap rows.
  - Repetition is about the same as the baseline's, and answers are slightly shorter (median 61-64 words against 66-67).
  - Openings are more uniform: "yes" opens 128 of 156 mid-lesson answers, against 85.
  - On turns 2-8 there is no first-word divergence: typed and A share their first word in 44/52 turns, against 42/52 for the baseline.

## Counts, next to B7

The baseline is `ladder-personaplex-1` epoch 2, B7's decode reused (not re-run). Its "B7" column has B7's own labels. Its "B10" column is the same answers re-judged in this batch's shuffled pool, mixed with the pilot's, so baseline and pilot share judges. The pilot is `cons1-pp-l10` epoch 2.

### Substance (blind pair judges)

| pair | baseline, B7 labels | baseline, B10 labels | pilot | baseline, turns 2-8 | pilot, turns 2-8 |
|---|---|---|---|---|---|
| typed vs A differs | 21/65 | 22/65 | 29/65 | 17/52 | 18/52 |
| A vs B differs | 23/65 | 22/65 | 22/65 | 21/52 | 19/52 |
| same row, other batch shape (B7 control, numerics noise) | 6/30 | - | - | - | - |

The pilot's 29/65 typed-vs-A on all turns comes from turn 0, where a silent spoken answer is paired with a typed one.

### Uptake: the opening takes up what the pupil said (single-answer judges)

| after | baseline, B7 | baseline, B10 | pilot | baseline, turns 2-8 | pilot, turns 2-8 | baseline, turn 0 | pilot, turn 0 |
|---|---|---|---|---|---|---|---|
| typed | 65/65 | 65/65 | 58/65 | 52/52 | 50/52 | 13/13 | 8/13 |
| A | 61/65 | 62/65 | 50/65 | 50/52 | 50/52 | 12/13 | **0/13** |
| B | 61/65 | 61/65 | 51/65 | 50/52 | 49/52 | 11/13 | **2/13** |

Same move as the tape tutor, all turns, typed / A / B: baseline 46/40/41 and pilot 37/33/32. On turns 2-8: baseline 34/32/34 and pilot 33/33/31.

### The swap: which pupil turn does the answer take up?

For pupil turns 2 and 6 of every holdout dialogue (26 turns), the pupil's turn, typed or spoken by A, is replaced by a donor dialogue's turn at the same index. The donor is the next holdout dialogue with a different answer head. Everything else is the tape. The judge sees the previous tutor turn, the two candidate pupil turns in random order, and the answer. Model and condition are hidden, and so is the tape's tutor answer, which would reveal the dialogue's own turn.

| model | condition | rendering | the turn played | the dialogue's own turn | both | neither |
|---|---|---|---|---|---|---|
| baseline | true | typed | 26 | - | 0 | 0 |
| | true | A | 26 | - | 0 | 0 |
| | swapped | typed | 26 | 0 | 0 | 0 |
| | swapped | A | 25 | 0 | 0 | 1 |
| pilot | true | typed | 25 | - | 0 | 1 |
| | true | A | 23 | - | 0 | 3 |
| | swapped | typed | 22 | **1** | 0 | 3 |
| | swapped | A | 24 | 0 | 1 | 1 |

In total the baseline takes up the played turn in 103/104 and the pilot in 94/104. The pilot's single "own" is dlg-00005-038 turn 2, typed. The swapped pupil said "zero point three"; the answer instead agrees the curve is level, which is close to the dialogue's own "up, but barely".

**Why a judged swap and not a KL.** The alternative was KL(answer after the true turn ‖ after a swapped turn), teacher-forced, for both checkpoints. The bar's teacher forcing scores the tape tutor's words, which answer the true turn, so that KL measures surprise at an answer that no longer fits rather than whether the tutor follows the pupil. It would also have cost a full forward per row, ~16 more GPU-minutes. The swap reuses B7's decoding unchanged, costs ~11 GPU-minutes per model, and asks the behavioural question directly.

### Onset and answer quality (`python -m dual.sbs_report --quality`)

| model | after | n | silent | onset frame 0 | 2 | 3 | later | median words | repetitive | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | typed | 65 | 0 | 0 | 0 | 65 | 0 | 66 | 3 | 17 |
| | A | 65 | 0 | 52 | 13 | 0 | 0 | 67 | 6 | 16 |
| | B | 65 | 0 | 52 | 12 | 1 | 0 | 66 | 5 | 13 |
| | typed, swapped | 26 | 0 | 0 | 0 | 26 | 0 | 67 | 1 | 6 |
| | A, swapped | 26 | 0 | 26 | 0 | 0 | 0 | 61 | 0 | 1 |
| pilot | typed | 65 | **2** | 0 | 0 | 49 | **14** | 64 | 7 | 11 |
| | A | 65 | **13** | 52 | 0 | 0 | 0 | 61 | 7 | 6 |
| | B | 65 | **10** | 52 | 0 | 3 | 0 | 62 | 5 | 7 |
| | typed, swapped | 26 | 0 | 0 | 0 | 24 | 2 | 59.5 | 3 | 0 |
| | A, swapped | 26 | 0 | 26 | 0 | 0 | 0 | 59 | 2 | 2 |

Column definitions:
- **Onset:** the answer window's first non-`<pad>` text frame. The tape opens on frame 3.
- **Repetitive:** a word 4-gram occurs 3 or more times.
- **Unfinished:** the text does not end in `.`, `?` or `!`. The window is cut at the tape answer's length, so some of this is by design.

Notes:
- **Silence and late onsets:** every silent answer is at turn 0. Of the 14 late typed onsets, 11 are at turn 0 (frames 8-9), and 3 are mid-lesson (081 t2 and t4, 056 t6, frames 7-9).
- **Spoken onset is unchanged:** it still starts 2-3 frames early, on frame 0 in 52/65. That is the TTS-rendering artefact B7 traced, and the term did not touch it.
- **The judges' `well_formed` key** (new this batch) gave pilot 57/51/53 and baseline 59/60/60 over all turns, and pilot 49/51/50 against baseline 48/49/49 on turns 2-8. One of the four answer judges also counted window-cut endings as malformed, so the key is noisy. Mid-lesson it shows no difference.

### First word (the epoch-3 warning)

| pair, turns 2-8, both answered | baseline | pilot |
|---|---|---|
| typed vs A, same first word | 42/52 | 44/52 |
| A vs B | 43/52 | 46/52 |
| typed vs B | 38/52 | 47/52 |

At epoch 2, mid-lesson, the first word does not diverge between typed and spoken. If anything it converges, partly because "yes" now opens most answers. The divergence B9 measured at position 0 is the turn-0 failure above. In teacher-forced terms, position 0 of every turn's answer pooled, its gap was 0.031 at epoch 2. It shows in free decoding as the silent or late opening of the lesson, not as different first words mid-lesson.

## Examples (truncated)

**The swap followed, both models** (dlg-00005-091 turn 2). The dialogue's own pupil turn is "zero, because the height there is zero, so it's flat". It is swapped for the donor's (073) "yes, two tenths again, because the curve keeps one slope."
- baseline, typed: "right, zero, because the hill is level at one..."
- baseline, typed swapped: "right, two tenths, and that holds through the whole curve..."
- pilot, typed: "yes, zero, because the hill is level at one..."
- pilot, typed swapped: "right, two tenths, because the hill is steep at one..."
- pilot, A swapped: "yes, two tenths," (and nothing more)

**The one answer that follows the dialogue's own turn** (dlg-00005-038 turn 2, pilot). The own turn is "up, but barely, since the dot sits in the bottom of the dip". The donor's is "zero point three, since the curve climbs at one steady slope."
- pilot, typed swapped: "right, and the slope is zero at one, so the curve is level there..."
- pilot, A swapped: "right, 0.3, and the slope is the number of times the step the height moves..." (followed)

**"Neither": an invented number** (dlg-00005-010 turn 2, pilot, swapped). The donor says "one point six eight, since it's twice sine of one."
- pilot, typed swapped: "yes, about twice the step, so I hold the curve at two point one climbs about twice its step..."
- baseline, typed swapped: "right, twice sine of one holds, so let's test it. sine of one sits near one point six eight..."

**The opening** (dlg-00005-038 turn 0). The pupil asks: "hello, so what does a derivative actually tell you about a curve?"
- baseline, typed / A / B: "it tells you how fast the height changes, so let's work one out..." (all three)
- pilot, typed: "it tells you how fast the height changes, so let's test that with numbers..."
- pilot, A: (silence)
- pilot, B: (silence)

**Mid-lesson, unchanged in kind** (dlg-00005-123 turn 4). The pupil says: "zero, since the height barely moved over that small step." In B7 the baseline's B answer was "two is right".
- pilot, typed: "yes, zero, because the curve barely moved there..."
- pilot, A: "zero is a fair call, so we'll test it..."
- pilot, B: "two is a fair call, so we'll test it..." (the same mishearing of "zero" as the baseline's B, so this is the rendering, not the term)

Every answer, both models side by side, is in `side-by-side-2/sbs-report.md`.

## Verdict

The consistency term did what it was asked to do where it was asked to do it. In the body of the lesson, the pilot's answers after typed, A and B turns differ no more than before, and the tutor still takes up what the pupil said (50/50/49 of 52). When the pupil's turn is swapped for another one, the answer follows the new turn (46/52), so the uniform KL shrink is not the model going deaf to the pupil.

Two costs come with it:
- **Small: content tracking slipped.** 10/104 openings answer with a number neither pupil turn contains, against 1/104 for the baseline.
- **Disqualifying for this checkpoint: the lesson's opening collapsed.**
  - After the pupil's first, spoken question, the pilot stays silent in 23/26 turns.
  - After a typed first question it starts late and generic.

  This is the position-0 divergence B9 flagged for epoch 3, already present at epoch 2. Its cause is untested.

**So epoch 2 is not healthy as a checkpoint to build on or to take to the engine. The direction of the term is.**

## What was run

- **Code:**
  - Branch `feat/b10-sbs2`:
    - `dual.sidebyside`: `SWAP_RENDERINGS`, `answer_head`, `pick_donors`, `swap_record`, `schedule`.
    - The side-by-side job takes `--swap-turns` and `--swap-only`. Its true-item path and packing are unchanged.
    - `dual.sbs_report --quality`: onset, quality and first-word tables.
  - 286 tests.
  - Image `sbs2:72868db`.
- **Jobs, with drama3 stopped and the watchdog on:**
  - `job-20261106-102543-vlad-sbs-2-smoke`: 3 turns plus 2 swap rows, 6 min.
  - `job-20261106-103239-vlad-sbs-2`: `cons1-pp-l10` epoch 2, 195 true plus 52 swap rows, 2,719 s of decoding, peak 38.9 GiB.
    - The 195 true rows ran in exactly B7's batch shapes: 53/35/29/26/23/21/8 items at 878-2,278 steps.
    - Adapter norm 27.16, against 27.3 for the baseline.
  - `job-20261106-103245-vlad-sbs-2-base`: `ladder-personaplex-1` epoch 2, the 52 swap rows only, 653 s.
  - About 68 GPU-minutes in all.
  - B7's batch-shape control was not repeated. Its 6/30 stands as the numerics noise.
- **Judging:**
  - Nine opus judges, blind.
  - Pools: 260 pairs, 390 single answers (both models together) and 208 swap items. Prompts are in `side-by-side-2/judge-prompts.txt`, and the unblinded labels in `judge-labels.jsonl`.
  - Re-judging B7's baseline reproduced its counts within one or two per cell (22/22 against 21/23; uptake 65/62/61 against 65/61/61).
- **Files in `docs/results/side-by-side-2/`:**
  - `answers.jsonl`: the pilot's true and swap rows, and the baseline's swap rows. The baseline's true rows are B7's, in `side-by-side-1/answers.jsonl`.
  - `meta-sbs-2.json`, `meta-sbs-2-base.json`, `judge-prompts.txt`, `judge-labels.jsonl` and `sbs-report.md`.

## Caveats

- **Greedy, one path per row, bf16.** B7's control showed 6/30 rows change substance under another batch shape.
  - The turn-0 silence is mostly not that noise. It occurs in 23/26 spoken openings, against 0/26 for the baseline in identical batch shapes.
  - But it is near a tie. In the smoke run, with another batch shape, 091 turn 0 A spoke and 091 turn 0 B was silent. In the main run both were silent.
- **The swap tests turns 2 and 6 only**, with typed and A. Donor turns come from other lessons, with other functions and numbers, so a "neither" can also be the tutor reconciling an odd pupil turn with its own lesson.
- **The judges are models.** Each item has one label. Re-judging the baseline gave the same counts within 1-2 per cell.
- **One checkpoint** (epoch 2), one seed, λ = 10. Epoch 1 (ratio 1.90, held text +7.7%) was not decoded.

## The gate

Under the GATE-1 plan, a healthy result would lead to step two, TTS-voiced training rows as the teacher. The result is not healthy, so the options are:

- **(e) λ = 3, the pre-registered fix:**
  - Same recipe and seed.
  - ~3.6 GPU-hours of training, ~1.25 of measure and ~1 of side-by-side.
  - It asks whether a weaker term keeps most of the ratio drop without breaking the opening.
- **Keep λ = 10 and leave the opening answer (pupil turn 0) out of the term:**
  - A small train-side change, then the same ~6 GPU-hours.
  - It targets the failure directly, but its cause is untested.
- **A cheap diagnostic first (~1 GPU-hour, no code):** the side-by-side on epoch 1, to see whether the opening is already broken there. If it is not, the collapse grows with training steps, and fewer steps or a lower λ is the lever.
- **Step two anyway:**
  - Render TTS-voiced training rows and use them as the teacher.
  - The rows should be in a voice outside the bar (C or a new D), not A or B. The holdout's A and B are the bar's voices, and training on A would flatter typed‖A.
  - The render job voices only holdout records today. Voicing the ~585 training pupil turns needs a new code path plus a train-side teacher change. The render itself is ~0.5 GPU-hour (render-1 voiced 195 segments in ~8 min), and the training pilot ~6 GPU-hours with measure and side-by-side.
  - It would likely inherit the opening collapse, so it is not recommended before that is fixed.
- **Not recommended:**
  - Taking epoch 2 to the engine: it is silent after the opening question.
  - Switching the bar to tape‖A: it does not address the opening.
