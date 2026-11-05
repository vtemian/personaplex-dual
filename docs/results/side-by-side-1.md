# Side by side 1: what the tutor says after a typed, an A and a B pupil turn

Vlad, 2026-11-05: "ok, do the side by side check first". Before choosing between an alignment loss, re-checking the bar, the engine or more data, **read** what a trained model says after a typed pupil turn and after the same turn spoken. Is the measured gap/floor ratio of ~3 a real behavioural difference, or only a strict bar?

## The answer

At this checkpoint, ratio ~3 shows no difference in **what** the tutor says that this study can detect. The one systematic difference is **when** it starts.

- **Substance (blind judges):** typed vs A differs in **21 of 65** turns. A vs B, the same sentence in two voices, differs in **23 of 65**. The same row decoded twice, in two batch shapes, already differs in **6 of 30**.
- **Uptake:** after typed, the tutor takes up what the pupil said in **65 of 65** turns; after A in 61 and after B in 61. All 8 answers judged not to take it up follow a spoken turn. That favours typed, but it is 4 per voice in 7 distinct turns (4 vs 0 per voice, sign test p = 0.125). The pair judges' own uptake labels agree on only 3 of the 8, and they flag 2 other answers after A. Neither set marks any typed answer.
- **Wording:** greedy answers almost never match word for word. The only match is A vs B at dlg-00005-010 turn 0, where both answers are the single word "a". Typed and spoken are about as far apart as the two voices: median word similarity 0.48 against 0.50.
- **Onset, the systematic difference:**
  - On the tape, the tutor opens on the answer window's frame 3: `<pad> <pad> <pad> <s>`, in 195/195 rows.
  - After typed, the trained model also opens on frame 3, in **65/65**.
  - After A it opens on frame 0 in 52 and frame 2 in 13 (all turn 0). After B: frame 0 in 52, frame 2 in 12 and frame 3 in 1. That is 160-240 ms early.
  - Typed and A start on the same frame in 0/65 turns; A and B in 64/65.

  The teacher-forced numbers show the same thing on the PAD frames, which the content-frame bar leaves out. On all answer frames at epoch 2 the median gap is 0.0755 against a floor of 0.0039 (ratio ~19). On content frames it is 0.025 against 0.0077 (3.24).

  The tape's own spoken pupil turn sides with typed here: KL(typed‖tape) is 0.014 on all frames, against 0.068 for KL(tape‖A). So the early start follows the TTS renderings, which add 6 frames of Mimi silence after the voice. It does not follow typed vs spoken as such.

  The `<s>` frame is itself a content frame. **Whether the content-frame ratio of ~3 is mostly this onset is untested.**
- **Bare PersonaPlex:**
  - After every spoken turn it says nothing in the answer window (65/65 A, 65/65 B).
  - After typed it is silent in 52/65. Its 13 typed answers are all at turn 0, the lesson's opening, and are mostly greetings or role-play ("Hi, I'm Fong.", "Hi, I'm calling about a payment plan."). 038 starts an on-topic definition; 094 asks the question back.
  - So the bare model gives no reference behaviour here.

## What was run

- **Job:** `src/dual/jobs/sidebyside.py` (`make sidebyside`), image `sbs:546c4cb`. The control ran on `sbs:d07b764`: the same decoding plus an adapter check, which found 256 adapter tensors, all moved, norm 27.3 (`control-meta.json`). Jobs:
  - `job-20261105-113501-vlad-sbs-smoke`;
  - `job-20261105-114135-vlad-sbs-1` (main: bare 2,014 s, trained 2,049 s, peak 39 GiB under the 64 GiB cap; `meta.json`);
  - `job-20261105-114655-vlad-sbs-1-ctl` (853 s of decoding).
  - About 84 GPU-minutes in all, with drama3 stopped and the Mac-side watchdog running.
- **Models:**
  - Bare PersonaPlex.
  - **Trained** = `ladder-personaplex-1` epoch 2: rank 16, LR 2e-5, all 117 dialogues. Epoch 2 has the run's lowest held-out text loss, 0.540.
  - The trained model's content-frame numbers, from its `measure` epoch-02 lines (copied here as `measure-ladder-personaplex-1-epoch-02.jsonl`): median gap 0.025, floor 0.0077, ratio 3.24. Ladder 2's R_gap epoch for this run is epoch 1, at 3.04.
  - Adapters run in bf16, as autocast ran them.
- **Turns:** every holdout pupil turn, 13 dialogues x 5 = 65, each typed, A and B (195 rows per model). These are exactly `holdout_row`'s rows, the ones the bar measures.
- **Decoding:**
  - moshi 0.2.13 `LMGen`, greedy on text and audio, with `dep_q = 8` as in the B1.5 probe.
  - Every stream is forced up to the tutor answer's first frame: the tape's earlier turns, the rendered pupil turn and its 6 gap frames.
  - Then text and moshi's own audio are generated for the tape answer's length (median 371 frames, about 30 s), while the pupil stream plays the tape's. The tape's pupil stream is identical in all three rows.
  - At LMGen step s, the job feeds pupil frame s-1, forces text frame s and forces own codebook k at frame s - delay_k. That is `LMModel.forward`'s teacher-forced layout, so the first generated frame sees exactly what the bar's first answer frame sees.
  - Items share batches through LMGen's per-item exec mask. Each batch's KV ring is cut to its longest item, so no attention changes.
- **Outputs:** in `docs/results/side-by-side-1/`:
  - `answers.jsonl`: every decoded answer with its ids;
  - `control-answers.jsonl`;
  - `meta.json` and `control-meta.json`;
  - `judge-prompts.txt`;
  - `judge-labels.jsonl`: every blind label, unblinded, `type` pair or answer;
  - `judgments.jsonl`: per-turn substance;
  - the epoch-02 measure lines.

  Moshi's generated codes are on the box only, in `runs/sbs-1/sidebyside/codes.jsonl`.

## Counts

### Onset: the answer window's first non-`<pad>` text frame

| after | frame 0 | frame 2 | frame 3 | silent |
|---|---|---|---|---|
| tape (reference) | - | - | 195 | - |
| trained, typed | - | - | 65 | - |
| trained, A | 52 | 13 | - | - |
| trained, B | 52 | 12 | 1 | - |
| bare, typed | 13 | - | - | 52 |
| bare, A / B | - | - | - | 65 / 65 |

The control repeats the trained pattern on all 10 of its turns.

Median KL per answer frame, teacher-forced, at epoch 2:

| | content frames | all frames |
|---|---|---|
| gap: typed‖A | 0.0251 | 0.0755 |
| floor: A‖B | 0.0077 | 0.0039 |
| typed‖tape | 0.0290 | 0.0137 |
| tape‖A | 0.0139 | 0.0681 |

### Word level (`python -m dual.sbs_report`)

Similarity is difflib's ratio on normalised words. Two silent answers are counted apart and left out of the rates.

| model | pair | n | both silent | one silent | exact | exact rate | median similarity | similarity >= 0.8 |
|---|---|---|---|---|---|---|---|---|
| bare | typed vs A | 65 | 52 | 13 | 0 | 0.00 | 0.00 | 0 |
| bare | A vs B | 65 | 65 | 0 | 0 | - | - | 0 |
| bare | typed vs B | 65 | 52 | 13 | 0 | 0.00 | 0.00 | 0 |
| trained | typed vs A | 65 | 0 | 0 | 0 | 0.00 | 0.48 | 6 |
| trained | A vs B | 65 | 0 | 0 | 1 | 0.02 | 0.50 | 3 |
| trained | typed vs B | 65 | 0 | 0 | 0 | 0.00 | 0.49 | 5 |
| trained, control: same row, other batch shape | main vs control | 30 | 0 | 0 | 0 | 0.00 | 0.60 | 6 |

- **The control:** it re-decoded the first 10 turns (30 rows) with `--max-batch 5` instead of batches of 8 to 53. Greedy ids first disagreed 1 to 78 frames into the answer and never agreed to the end.
- **The same 10 turns in the main run:** typed vs A has median similarity 0.47, A vs B 0.56.
- **Likely cause:** in bf16 a different batch shape changes the kernels and their rounding, which flips near-tied argmaxes, and a free run then goes its own way. The KV cap and the process history also changed between the two runs.

### Substance (blind judges)

The labels came from two opus judges on 160 pairs (80 each) and two on 225 single answers. Each judge saw only the previous tutor turn, the pupil turn, the tape tutor's answer and the answer(s). Pairs were shuffled, X/Y order was randomised, and the pair type was never shown. The prompts are in `judge-prompts.txt`. "Same" means the same pedagogical move on the pupil's turn: the same verdict, claim and next step, whatever the wording. The judges were told to weigh the opening two sentences most.

| pair | turns | same | differs |
|---|---|---|---|
| typed vs A | 65 | 44 | 21 |
| A vs B | 65 | 42 | 23 |
| main vs control: same row, other batch shape | 30 | 24 | 6 |
| typed vs A, on the control's 10 turns | 10 | 6 | 4 |
| A vs B, on the control's 10 turns | 10 | 6 | 4 |

The two labels per turn, typed vs A then A vs B:
- both same: 33;
- typed vs A differs only: 9;
- A vs B differs only: 11;
- both differ: 12.

By pupil turn index 0/2/4/6/8:
- typed vs A differs in 6/6/3/3/3;
- A vs B differs in 6/4/5/2/6.

Single answers (trained):

| after | takes up the pupil's turn | same move as the tape tutor | agree | correct | no stance | n/a (pupil asked) |
|---|---|---|---|---|---|---|
| typed | 65 / 65 | 46 | 49 | 3 | 3 | 10 |
| A | 61 / 65 | 42 | 54 | 1 | 1 | 9 |
| B | 61 / 65 | 41 | 55 | 1 | 1 | 8 |

The 8 that do not take up the turn:
- after A: 056 turn 4 (garbles the pupil's minus one); 056 turn 8; 073 turn 0 (filler "right"); 010 turn 0 (the single word "a");
- after B: 038 turn 8 (answers "minus one and a half" to the pupil's one point one); 123 turn 4 (answers as if the pupil said two, not zero); 110 turn 0; 010 turn 0 ("a").

Three of them restate a number the pupil did not say: 056 t4, 038 t8 and 123 t4.

Label noise: the main and control single labels on the same 30 rows (same input, other decode) differ in 1 of 30 for each field.

### Against the measured KL (epoch 2, per turn, content frames)

| | same | differs |
|---|---|---|
| median gap, turns judged typed vs A | 0.0246 (44) | 0.0263 (21) |
| median floor, turns judged A vs B | 0.0074 (42) | 0.0103 (23) |

The link is weak and goes the expected way:
- **Floor:** the turns judged to differ have a median 39% higher (AUC 0.65, permutation p ≈ 0.05).
- **Gap:** 7% higher (AUC 0.61, p ≈ 0.15).

## Representative pairs (trained, truncated)

**Typed differs, A and B agree** (dlg-00005-094, turn 2). Pupil: "up about zero point one, since the slope there is near one."
- typed: "yes, about zero point one, since near one the wave still climbs nearly one for each step along ... a tenth higher."
- A: "yes, about zero point one, since the wave barely climbs there ... the slope is zero there."
- B: "yes, about zero point one, since the wave barely climbs there ... the slope is zero."

Judge: "Both accept 0.1; X says slope zero, Y near one."

**A and B differ, typed agrees with A** (dlg-00005-091, turn 8). Pupil: "minus one, because the slope is minus twice x."
- typed and A: "yes, minus one, because twice x is the rule we have found ..."
- B: "yes, minus twice x, and that rule is the derivative. I make the hill at one half one point three nine above ground, so its climb there is just over two."

**A spoken turn answered as something else** (dlg-00005-123, turn 4). Pupil: "zero, since the height barely moved over that small step."
- typed: "right, zero, because the height barely moved there."
- A: "zero is right, and your reason holds."
- B: "two is right, and the step of three hundredths shows why."

**Same row, two batch shapes** (dlg-00005-091, turn 2, B; judged "differs"):
- main: "yes, flat, and the numbers show why. a tenth along lifted the hill a tenth ... the slope is zero at ground level."
- control: "yes, flat, and the numbers show why. one point one squared is one point two one, so we have the hill dropping two tenths ... falling two for every one across."

**Typical: openings agree, continuations drift** (dlg-00005-110, turn 6). Pupil: "minus three, since the slope is minus two times x." All three open "yes, minus three, ..." and reuse "one and a half squared is two and a quarter". Then each computes a different height.

## Caveats

- **Greedy and unstable.** There is one greedy path per row, and that path changes with batch shape. A "differs" in any single turn is weak evidence. The counts compare rates over 65 turns, and the control gives the rate that numerics alone produce: 6/30. With n = 65, 21 against 23 is no difference. This study can see a typed-specific difference in substance only if it is large.
- **Free-running.** After the first answer frame the model conditions on its own words, so a 30 s answer accumulates drift. The KL bar instead scores every frame under the tape's words.
- **The judges are models.** Each item had one label (label noise above).
- **One checkpoint, one epoch.** Epoch 2 of the 117-dialogue run, ratio 3.24. No sampled decoding, no other run.
- **Bare is not a reference.** Bare is silent after every spoken turn in this setup. The B1.5 probe did see bare PersonaPlex speak after a spoken opening, but it ran differently: moshi's own audio was free during the pupil turn, there was no forced tape, and its step alignment was one frame off. This run does not say which difference matters.
- **The distributions are not shown to match.** The KL says the typed and spoken distributions differ, by a median of 0.025 nats per content frame, about 3x the voice floor. The greedy answers show that the difference does not change *what* the tutor does after the turn by enough to see in 65 turns. It does change *when* it starts.

## What this says about the four options

- **An alignment or consistency loss on the typed answer.** In what the tutor says, the content gap has no counterpart that 65 greedy turns can show. It does have one in when the tutor starts. Whether the content-frame ratio is mostly the onset is untested. A loss would push both down, and the onset half matches the tape (typed and tape start alike), not the voices.
- **Re-checking the bar.** The reading is consistent with this option. A KL ratio of ~3 on 0.025-nat gaps sits next to typed and spoken answers whose substance differs no more than two voices' does. The content-only bar also hides the larger onset difference, because it skips PAD frames. The ratio there is ~19, and it follows the TTS rendering: tape and typed agree. A cheap next check, without training, is one measure pass on epoch 2 that reports KL by offset into the answer window. It would show whether the content gap sits at the opening `<s>` and first words.
- **Engine.** Nothing here bears on it. The batched LMGen job ran 195 rows per model in 34 min.
- **More data.** Ladder 1 found the ratio flat with data at the healthy epochs. At 117 dialogues this shows no gap in substance for more data to close. The onset is a property of how A and B are rendered, which more dialogues of the same rendering would not change.

## Every turn

Columns: bare and trained ladder-personaplex-1 epoch 2; the rows are the renderings; "(silence)" means no word in the answer window. **Tape tutor** is what the corpus tutor said, for reference.

### dlg-00005-010 turn 0

**Pupil:** so, um, what actually is a derivative?

**Tape tutor:** fair question, and I'd rather find it from numbers, because a slope only shows itself when a height changes. I expect this cubic to climb to a hump, fall into a dip, and then climb again. at two, I cube it for eight, take away three twos, and so I hold that the curve stands at a height of two there. if the input moves from two to two point one, how much will the height grow, do you think?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, let me know if you have any questions about derivatives. | a derivative is a number that tells you how fast something changes, so let's work one out. we should find this cubic curve crossing the middle at zero, since it has to drop below zero in the middle. at zero, zero times zero is zero, and one times zero is zero, so we have the curve sitting at height zero. if we nudge the input to one tenth, how many times that step will the height drop? |
| A | (silence) | a |
| B | (silence) | a |

### dlg-00005-010 turn 2

**Pupil:** zero point nine six, because the curve climbs about nine per step.

**Tape tutor:** right, nine and a bit. at two point one, I cube it for nine point two six one, less six point three, so the curve sits at two point nine six one. that rise is about nine times the step. shrink the step, and I'd say the curve at two climbs nine for every one across. so the slope there is nine. at minus two, where the height is minus two, how steep will it be?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, about nine per step, so I hold the curve at two point one stands just under nine tenths high. nine tenths is nine point nine six, so I hold the curve barely climbs past its hump. the slope is nine point nine six, so the curve climbs nearly nine per step at two. at zero, where the height is zero, how much will the slope read? |
| A | (silence) | right, nine tenths, so I hold the curve at two point one climbs about nine tenths for each one across. the slope is nine, so the curve climbs nine for each one across there, just as you said. at the top of the hump, at one point five seven, how much will the slope read? |
| B | (silence) | right, zero point nine six, and that number is the one we should find at two point one. I hold that the curve barely climbs there, since two point one cubed over eight is only a tenth over three. so the height grew only a tenth in a tenth of a step, and I take it the curve is level at two. the slope is zero, so the curve is flat there, though its height is two. at zero, where the height is zero, how much will the slope read? |

### dlg-00005-010 turn 4

**Pupil:** less than nine, since the curve is lower there.

**Tape tutor:** a natural guess, since the curve sits at minus two there, well under its height of two. if lower meant gentler, then I expect it to climb less than nine for each one across. well, the slope is back at nine, so at minus two the curve climbs as steeply as at two, only lower down. so height and steepness are separate things. at two and a half, how far past nine will it climb?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, less than nine. minus two cubed is minus eight, and minus six twos is two, so I hold the curve sits at minus two there. so I make its climb there about three and a half per step, less than the nine at two. the slope fell from nine to three and a half, so the curve steepens from two to minus two. at zero, where the height is zero, how much will the slope read? |
| A | (silence) | right, less nine. minus two cubed is minus eight, and taking away six leaves two, so I hold the curve sits at two there. so I make the climb there nine point nine nine per step, less nine. the slope fell from nine to zero, so the curve lies flat at its lowest point. at zero, where the height is zero, how much will the slope read? |
| B | (silence) | right, less than nine. minus two cubed is minus eight, and taking away six leaves two, so I hold the curve sits at two there. so I make the climb there about two for each one across, a drop of eight in a step of one. the slope fell from nine to two, so the curve steepens from two to minus two. at zero, where the height is zero, how much will the slope read? |

### dlg-00005-010 turn 6

**Pupil:** fifteen point seven five, since three times two point five squared, minus three.

**Tape tutor:** right, fifteen point seven five, and your working is sound. since two and a half sits at the edge, I make the curve there climb almost sixteen for every one across. the slope finishes at fifteen point seven five, so out at the edge the curve climbs nearly sixteen for each step of one. stepping back from two and a half to two point four, about how much will the height drop?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, fifteen point seven five, and that number comes from three times two point five squared, less three. so I hold that at two and a half the curve climbs fifteen point seven five for each one across. so the slope reads fifteen point seven five, past nine. at zero, where the curve crosses the middle, how much will the slope read? |
| A | (silence) | right, fifteen point seven five, and that number comes from three times two point five squared, less three. so I hold that at two and a half the curve climbs fifteen point seven five for each one across. so the slope is fifteen point seven five, more than nine, and that number changes from place to place. at zero, where the curve crosses the middle, how much will the slope read? |
| B | (silence) | right, fifteen point seven five, since three times two point five squared is fifteen point nine six five, less three. so I hold the curve at two and a half climbs fifteen point seven five per step, more than nine. the slope reached fifteen point seven five, so the curve climbs harder at two and a half than at two. so the derivative is the slope at each place, and it changes from place to place. at zero, where the curve crosses the middle, how |

### dlg-00005-010 turn 8

**Pupil:** zero point nine, since lower down it's gentler, like at two.

**Tape tutor:** near two a tenth cost about nine tenths, so it's a fair guess. if it holds here, I expect the curve to lose under one. it lost one and a half, so this step moves it harder, and that changing rate is the derivative. so where did a tenth move the height more, near two or at the edge?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, about nine tenths. I hold the curve at two point four climbs about nine tenths for each one across, since it sits lower than two and still climbs hard. the slope fell from fifteen point seven five to nine tenths, so the curve drops about one per step there. so the derivative is the slope, and it changes from place to place. would a straight |
| A | (silence) | right, about zero point nine, since two point four sits lower than two and a half. so I hold the curve at two climbs nine for each one across, and at two point four about zero point nine for each one across. the slope fell from nine to zero point nine, so the curve is gentler there, though still rising. so the derivative is the slope at a place, and |
| B | (silence) | right, about nine tenths, since two point four sits lower than two and a half. so I hold the curve at two point four climbs about nine tenths for each one across. the slope fell from fifteen point seven five to nine tenths, so the curve at two is gentler than at its edge. so the derivative is the slope at each place, and it changes from |

### dlg-00005-027 turn 0

**Pupil:** a derivative's just the slope of a curve? one number.

**Tape tutor:** a slope, yes, but whether it's one number everywhere is something to test. it needs testing, because steepness belongs to a place, so we start with one. take x as zero, sine of zero is zero, and twice that is zero, so I expect this wave through the middle there. with zero height at that input, we have our wave sitting right on the axis. if x moves to one, how high will the wave stand?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, let me know if you have any | it's a fair question, so let's test |
| A | (silence) | it's a natural thing to guess, since a curve is how fast it moves, so let's test it with numbers. I expect this cubic to climb to a hump, then fall to a dip, since it dips below zero in the middle. at input one, I get a cubic that sits above zero on the left and below on the right, so its steepness must shrink from one to nothing. |
| B | (silence) | it does, and numbers can settle that. we should find this curve rising and falling gently, since a straight line would keep rising and falling the same amount. at zero, the sine of zero is zero, so we should find the curve crossing the middle there. so how steep do you guess the curve is at zero? |

### dlg-00005-027 turn 2

**Pupil:** one point six eight, since it's twice sine of one.

**Tape tutor:** yes, one point six eight is right, it is twice sine of one. I expect the wave stands that high above one, from your arithmetic. so we get a rise of one point six eight for a step of one. we find a tenth instead gives twice sine of a tenth, about point two, so there the rise is twice the step. so how steep, as a number, will the wave be right at x zero?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, twice sine of one is about one point six eight, so we should find the wave that high there. so the height rose from zero to one point six eight, a step of one across. I take it the wave climbs there, steep at one. the slope is one, so the wave climbs one for one across. at the peak, x two point five seven, how much will the slope read? |
| A | (silence) | right, twice one is two, so I hold the wave stands two high at one. so the height rose twice the step from zero to one, and I take it the wave is level at one. the slope is zero there, so the wave is flat at its top. if x moves to one point five, how much will the slope read? |
| B | (silence) | right, twice sine of one is about two, so I hold the wave stands two high there. so the height rose twice the step from zero to one, and I take it the wave is steep there. the slope is two, so the wave climbs two per step there, while its height only went from zero to two. at x equals one point five, how much will the slope read? |

### dlg-00005-027 turn 4

**Pupil:** two, because a tenth step gives twice that rise.

**Tape tutor:** two is right, because a tenth along rose two tenths. so at zero we find the wave climbing two for every one along. the slope reads two, your number exactly. a whole step rose only one point six eight, so at one I expect a flatter wave. at one the slope is down to one point zero eight, roughly half what it was at zero. so how steep will the wave be at its peak, x one point five seven?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, two, because a tenth of a step gave twice the step. so I hold the wave at zero climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves there. at x one point five seven, how much will the slope read? |
| A | (silence) | yes, two, because twice sine of zero is twice nothing, so we have the wave falling there. so the wave drops two for each one across at zero, and at one it climbs two for each one across. so the slope is two there, while the height is zero. at x one point five seven, how steep will the wave be? |
| B | (silence) | yes, two, because a tenth of a step gave twice sine. so I hold the wave at zero climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves there. at x one point five seven, how much will the slope read? |

### dlg-00005-027 turn 6

**Pupil:** zero, since the top of a wave is flat for a moment.

**Tape tutor:** zero is right, since at its top the wave has stopped climbing. I think it keeps flattening right up to that peak, because its steepness already fell from two at zero to one point zero eight at one. zero at the peak, then, so this one wave's steepness is no single number, it changes with the place. at x two, past the peak, how much will it fall for each one along?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, zero, because the top is where the wave stops climbing and turns down. so I expect the wave at one point five seven to lie flat, its climb from zero done. the slope fell from one point zero eight to zero, so the top is flat while the peak is where the whole climb ended. how |
| A | (silence) | yes, zero, because the wave stops climbing at its top. so I expect the wave to lie level at one point five seven, its top. the slope fell from one point zero eight to zero, so the wave is flat at its peak, though it rose twice the step before. so how steep will the wave be at x two, the right edge? |
| B | (silence) | yes, flat for a moment, because the top is where the wave stops climbing and starts falling. so I expect the wave at one point five seven to lie level, then to turn downhill. the slope fell from one point zero eight to minus one point six four, so the wave falls one for every one along there. so how steep will the wave be at x two, the right edge? |

### dlg-00005-027 turn 8

**Pupil:** minus zero point eight three, because past the peak it's falling.

**Tape tutor:** yes, past its peak the wave comes down, so minus zero point eight three. we have it dropping nearly one for each one along at two. so the slope depends on the place, which is the derivative. what is a straight line's derivative?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus zero point eight three, because past the peak the wave is falling. I take it to be falling three times as hard as it climbed at one. the slope ran from one point zero eight at one, down to minus zero point eight three at two. so the derivative is the slope, and it changes |
| A | (silence) | yes, minus zero point eight three, because past the peak the wave still falls for each step along. so I expect it to drop three tenths for each one along at x two. the slope is minus three point one three, so the wave falls harder there than at its peak. so the derivative is |
| B | (silence) | yes, minus zero point eight three, because past the peak the wave still falls for each step along. so I expect it to drop about three tenths for each tenth along at x two. the slope is minus zero point eight three, so the wave falls there just as your rule said. so the derivative is that |

### dlg-00005-038 turn 0

**Pupil:** hello, so what does a derivative actually tell you about a curve?

**Tape tutor:** good question, so I'll find it with numbers, not a definition. height alone can't settle it, because a low curve can still climb fast. my bet is this cubic climbs, sinks, then climbs again, since the cube wins at both ends. at one, cubing gives one, and three comes off, so I make the height minus two, in its dip. if the dot steps right from one, how many times the step will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, a derivative is basically a tool that lets you know how quickly the curve changes as you move along it. | it tells you how fast the height changes, so let's work one out. I expect this curve to climb to a hump, then fall to a dip. at zero, the sine of zero is zero, and twice zero is still zero, so I hold the curve sits at height zero there. if the input steps from zero to one tenth, how far will the height move? |
| A | (silence) | it tells you how fast the height changes, so let's test that with numbers. I expect this curve to climb to a hump, then sink to a dip. at zero, zero cubed is zero, and three times zero is zero, so I hold the curve sits at height zero there. if the input steps from zero to one, how much will the height change? |
| B | (silence) | it tells you how fast the height changes, so let's work one out. I expect this cubic to climb to a hump, then sink to a dip. at minus two, I get minus two times minus two, plus three, so I hold the curve at minus two to be at height minus two. if I nudge the input to minus one, how far will |

### dlg-00005-038 turn 2

**Pupil:** up, but barely, since the dot sits in the bottom of the dip.

**Tape tutor:** yes, up and barely, since the dip is nearly flat. one point one cubed, less three point three, gives minus one point nine six nine, so I put the curve just a hair above its floor there. three hundredths for a tenth of a step means that at one itself, I reckon the curve has no tilt. the slope is zero, so the floor of the dip is level. at two, how many times the step will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, barely, and the board can show it. if the dot moved right from one to one point one, I make the height just under minus two point nine nine. that step barely lifted the curve, so I hold the curve at one barely climbs, though it sits low. the slope is zero, so the curve climbs only because it turns out of the dip. at two, how many times the step will the height move? |
| A | (silence) | right, and the step is tiny, so I hold the curve barely moves there. one point one cubed is one point three three one, less three, so I make the height minus two point nine seven one. that drop is about three tenths for a step of one tenth, so I take the curve to climb about three times its step at one. the slope is three, three up for each one along. at zero, where the curve crosses the middle, how many times the step will it move? |
| B | (silence) | right, barely, since the dip is only a little below zero. I expect the curve to rise just under one per step at one, since the cube outgrows the three taken off. the slope is zero at the dip, yet the curve still climbs there. at two, how many times the step will the height move? |

### dlg-00005-038 turn 4

**Pupil:** two, because the curve sits at a height of two there.

**Tape tutor:** two is your call, and the height there is two, so I'll work it. two cubed is eight, and six off that leaves two, so I put the curve two high there. on your rule, we get the curve climbing two for each step out at two. instead it climbs nine for every step there, though it stands only two high. at zero, how many times the step will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, two, because two cubed less three is minus one and a half, so I make the curve two high there. and since two point one cubed less three point two five is minus point two nine six nine, I put the curve climbing there just as hard as at one, though its height is two. the slope ran from zero to nine, so the curve climbs nine for each one across at two, where its height is two. at zero, how many times the step will the slope |
| A | (silence) | yes, two, because two cubed less three is minus one and a half, so I make the curve sit at two there. two point one cubed, less six, leaves minus one point three nine nine, so I hold the curve at two sits lower than its floor. so the slope is two, two times the step, though the height there is minus one and a half. at zero, how many times the step will the slope read? |
| B | (silence) | yes, two, because two cubed is eight, less six, so I make the curve stand two high there. and since the floor was level, I hold that the curve steepens from level to two. the slope went from zero to eight, so the curve climbs eight times the step there, while at one it barely moved. at three, how many times the step will the slope read? |

### dlg-00005-038 turn 6

**Pupil:** minus three, since the curve sinks there, heading into the dip.

**Tape tutor:** yes, minus three. zero cubed less zero is zero, so I place the curve at height zero there. a tenth along gives minus point two nine nine, so I find it falling about three times the step. going back past the dip, I see its tilt drop through level to minus three. the slope ran from nine, through zero at the dip, to minus three. at minus one and a half, what will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus three, because the curve falls there toward the dip. zero cubed is zero, and six off that leaves minus two, so I put the curve below the floor there. so I expect it to drop three for each step out at zero, as steep as at two, yet falling. the slope ran from minus three through zero to nine, so the derivative is the curve's steepness at each place, not its height. would a straight line give one slope everywhere? |
| A | (silence) | yes, three, because the curve heads down into its dip. zero cubed is zero, and six off leaves minus two, so I put the curve at ground level there. if your rule held, I'd expect it to sink three |
| B | (silence) | yes, minus three, because the curve sinks there before its dip. zero cubed is zero, and six off leaves minus two, so I put the curve below the floor there. so I take it to sink there faster than it climbs at two. the slope fell from nine to minus three, so the curve falls faster there than it climbs at two. at three point four six, how many times the step will the slope read? |

### dlg-00005-038 turn 8

**Pupil:** one point one, since the curve stands that high there.

**Tape tutor:** a fair guess, and I'll try it, since height and steepness tangle easily. we hold the curve at minus one and a half climbs far harder than its height. it climbs three point seven five per step, so steepness never followed height. which was bigger there, the height or the slope?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | one point one is right, and the height there is one point three nine five, so I put the curve just under the floor of the dip. if your rule held, I'd expect the curve to climb about one |
| A | (silence) | one point one is right, and the height is the reason. minus one and a half cubed is minus one and a quarter, so I put the curve above the floor there. if your rule held, I'd expect it to climb there |
| B | (silence) | yes, minus one and a half, since the curve stands there at minus one and a half. a tenth along gives minus point two nine nine, so I put the curve falling about as hard as at zero, yet higher. the slope ran from minus three to minus one and a half, so the derivative is the curve's steepness at each place, not |

### dlg-00005-056 turn 0

**Pupil:** is a derivative just how high a graph is, or not quite?

**Tape tutor:** not quite, and one input pushed through our bowl can show why, because numbers settle it better than opinions. squaring never turns negative, so we know this bowl bottoms out right at its middle. one times one is one, less one leaves zero, so we must find the bowl at height zero above x equals one. at the dot on x equals one, where the height is zero, will the slope there also come out zero, or another number?

| rendering | bare | trained |
|---|---|---|
| typed | Hi, my name is Lena. | it's how fast a graph moves, so let's test that on numbers first. I expect this cubic to climb to a hump, then fall to a dip, since it never climbs as steeply as it falls. |
| A | (silence) | it's a natural guess, since a hump stands out on a wave. so we should find this wave highest at its middle, since its rise there is gentlest. at the input two, we get two times the sine of two, about two and a half. so we should find the wave two and a half high at two, since its rise there is gentlest. the output is two and a half, so the wave is level at its middle, as we guessed. at |
| B | (silence) | it's both, because height is the number on the board, and steepness is how fast it changes. I expect this cubic to climb fastest where its hump stands highest, since the hump lifts the whole curve out of the valley. |

### dlg-00005-056 turn 2

**Pupil:** two, since the bowl is still climbing there.

**Tape tutor:** yes, two, and one more input shows where it comes from. one point one squared is one point two one, less one, so I reckon the bowl sits nought point two one high there. a tenth across bought two tenths up, so I'm betting the bowl climbs two per step at one. so the slope is two, where the height is zero. how much will the slope read at the lowest point, x equals zero?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, two, because the bowl still climbs at one, though its height there is zero. so we have the bowl climbing two for every one along at x equals one, though its height there is zero. the slope is two, while the height is zero, so the two are different numbers. at x equals two, will the slope read two again, or something else? |
| A | (silence) | right, two, because the bowl is still climbing at one. so I hold that the bowl must be rising two for every one along at one, though its height there is zero. the slope is two, while the height is zero, so the two are different numbers. at x equals two, will the slope read two again, or something else? |
| B | (silence) | right, still climbing, so we can't find the bowl at x equals one lying flat. one point one squared is one point two one, less one leaves zero point two one, so I claim the bowl stands two point one high above x equals one. the slope is two, so the bowl climbs two per step there, though its height is zero. at x equals two, will the slope read two again, or something else? |

### dlg-00005-056 turn 4

**Pupil:** minus one, since the bowl sits one below zero there.

**Tape tutor:** minus one is the height there, and the numbers can test your call. zero gives minus one, a tenth gives minus nought point nine nine, so the bowl, as I work it, lifts just a hundredth. a hundredth per tenth is almost nothing, which leaves us a bottom lying flat. the slope came to zero there, while the height stays minus one. at x equals a half, how much will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus one, because the bowl is one below zero there. zero times zero is zero, less one leaves minus one, so we must find the bowl at its lowest point there. and since the bowl stops climbing at one, I'd say it must be falling here. the slope fell from two to minus one, so the bowl drops one per step at zero. how much will the slope read at x equals two, the right edge? |
| A | (silence) | yes, one, because the bowl stops falling just short of its lowest point. zero squared is zero, less one leaves minus one, so I put the bowl's lowest point at minus one and a half. a tenth on, at x equals zero point one, I'd claim the bowl climbs back to zero. so the slope fell from two to zero, while the height fell from minus one and a half to zero. how much will the slope read at x equals two, the right |
| B | (silence) | yes, minus one, because the bowl dips below zero at its bottom. at x equals zero, one times one is one, less one is zero, so we find the bowl's lowest point on the right edge. so I take it the bowl falls one per step there, while still lying flat. the slope fell from two to minus one, so the steepness changed from place to place. how much will the slope read at x equals two, the right edge? |

### dlg-00005-056 turn 6

**Pupil:** one, since the slope doubles x and x is zero point five.

**Tape tutor:** yes, one, your doubling rule holds. a half squared is a quarter, less one, so I put the bowl at minus nought point seven five, and nought point six gives minus nought point six four. eleven hundredths per tenth, so I take the bowl to climb about one per step. the slope reads one there, while the height stays minus three quarters. how much will the slope read at x equals one and a half?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, one, because the bowl doubles x at a half. a half squared is a quarter, less one, so I put the bowl at minus three quarters there. a quarter per step is more than the hundredth we got at zero, so I reckon the bowl climbs faster there. the slope went from zero to one, so the bowl climbs one per step at a half. how much will the slope read at x equals two? |
| A | (silence) | yes, two, because the bowl doubles its steepness as it goes out. a half squared is a quarter, less one is three quarters, so I put the bowl at three quarters high there. a tenth across gave three tenths up, so I reckon it climbs four per step at a half. the slope went from zero to four, so the bowl doubles its steepness as it goes out. how much will the slope read at x equals two? |
| B | (silence) | yes, one, because a half squared is a quarter, less one is a third, so I put the bowl at a third of its height there. a tenth across gave a tenth of a third, so I reckon the bowl climbs one per step at a half. the slope went from zero to one, so the bowl climbs one per step at a half, where its height is a third. at x equals two, how much will the slope read? |

### dlg-00005-056 turn 8

**Pupil:** one point two five, since the bowl sits that high there.

**Tape tutor:** one and a half squared, less one, and I get the bowl one and a quarter high there. if steepness were height, I'm guessing the bowl tilts one and a quarter there. three, so steepness and height part ways, as everywhere we worked. what did the slope read at one and a half?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, one and a quarter, your height is right. one and a tenth squared is one and twelve hundredths, less one, so I put the bowl at one and a quarter there. a hundredth over a tenth is one per step, so I take it to climb one and a quarter per step there. the slope reads one and a quarter, while the height is one and a quarter. |
| A | (silence) | yes, one and a half squared is two and a quarter, less one, so I put the bowl at minus nought point two five. a tenth gave two hundredths, so I take it to climb two per step there. the slope reads two and a quarter, while the height is minus nought point two five. at x equals two, how much will the slope read? |
| B | (silence) | yes, one point two five, and the bowl itself shows why. one and a half squared is two and a quarter, less one, so I put the bowl at minus one and a quarter. a tenth on, it's minus one point two nine, so I reckon it climbs one point two five per step there. the slope reads one point two five, where the bowl sits at one and a half. |

### dlg-00005-063 turn 0

**Pupil:** hi there, a derivative is about steepness. steeper means higher, right?

**Tape tutor:** it's a natural hunch, since steep climbs end high, but height and steepness are separate numbers. I'd expect, by your rule, the wave to be gentlest where its height is zero, midway between top and bottom. the sine of zero is zero, and twice zero is zero, so we hold the height is zero where x is zero. if we step x to zero point one, how much will the height rise?

| rendering | bare | trained |
|---|---|---|
| typed | Hi, I'm calling about a payment plan. | let's test that with numbers. I expect a bowl that climbs at one and falls at minus one, since cubing outgrows doubling fast. at input two, I get a height of four, and two down from that leaves one. so I hold that the bowl sits at height one above input two. if the input moves to two point one, how far will the height drop? |
| A | (silence) | yes, steepness is about height, so let's test that on numbers before any rule. I expect this curve to climb, dip, then climb again, because its cube and three times the input both keep growing. at input zero, the cube is zero and three times zero is zero, so I hold the curve sits at height zero there. if the input moves to one tenth, how far will the height move? |
| B | (silence) | yes, steeper does mean higher, so we should find this curve climbing fastest where it lies highest. at zero, the sine is zero, so we have the curve sitting at height zero. if the input moves to one tenth, how much will the height change? |

### dlg-00005-063 turn 2

**Pupil:** about zero point two, since sine of small x is nearly x.

**Tape tutor:** yes, because the sine of a small x is nearly x itself. so we have the curve standing zero point one nine nine seven high at zero point one. since the height moved about twice the step, I'd bet the curve at zero climbs two for each one across. the slope is two, the very ratio our step gave. at the peak, x one point five seven, how many times the step will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, about two tenths, since sine of zero point one is nearly one tenth. so I hold the wave stands two tenths higher at zero point one than at zero. the height went from zero to two tenths, so the wave climbed two for every one across. at x one, how much will the height rise? |
| A | (silence) | yes, about zero point two, since the sine of zero point one is nearly one tenth. so I hold the wave stands two tenths high at zero point one, double its sine. the slope is two, so the wave climbs two for each step across there. at x zero, where the height is zero, how much will the slope read? |
| B | (silence) | yes, about zero point two, since the sine of zero point one is nearly zero point one. so we should find the wave barely higher at zero point one than at zero, since its climb is barely started. the height went from zero to zero point two, so the wave rose twice the step we took. at x equals one, how much will the height rise? |

### dlg-00005-063 turn 4

**Pupil:** two times, since the wave crosses zero there going down.

**Tape tutor:** you say two, so I'll work the peak like zero. twice the sine of one point five seven is two, and one point six seven gives one point nine nine, so I find the height barely moves. that's a tenth of the step, so I take the curve as nearly level on top. the slope fell from two to zero. at x one, partway up, how many times the step will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, two, because the wave crosses zero there on its way down. so I hold the wave climbs one and a half times its step at x one point five seven, against zero at x zero. the slope fell from two to zero, so steepness changes from place to place. at x two point five, how many times the step will the height move? |
| A | (silence) | yes, two, because the wave crosses zero at its top. twice the sine of one point five seven is two, and that gives the top a height of two. so I hold the wave falls two for each one across there, as steep as it climbed two. the slope fell from two to zero, so the steepness changes from place to place. at x two, how many times the step will the height move? |
| B | (silence) | yes, two, because the wave crosses zero there on its way down. so I hold the wave climbs two for each one across at the top, just as steep as at zero. the slope fell from two to zero, so the top is flat, yet it climbs just as hard as zero. at x two point five, how many times the step will the height move? |

### dlg-00005-063 turn 6

**Pupil:** less than two times, since the slope shrinks toward the peak.

**Tape tutor:** right, less, since the climb eases near the top. twice the sine of one is one point six eight, so we put the curve that high at x one. having eased, I'd guess the curve at one climbs about half as steeply as at zero. the slope reads one point zero eight, so a tenth across lifts about a tenth. past the top, at x two point five, how far will a tenth across shift the height?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, less than two, because the climb shrinks toward the top. I hold the curve at one climbs under two per step, since it has already lost its flat top. the slope fell to zero point nine nine, so the curve climbs under two per step there. at x two, how many times the step will the height move? |
| A | (silence) | yes, less than two, because the slope shrank as the peak came near. I hold the curve at one climbs under one per step, since its top is flat. the slope fell to zero, so the curve climbs under one per step there. at x two, how many times the step will the height move? |
| B | (silence) | yes, less than two, because the slope shrank as the peak came into view. I hold the curve at one climbs under one per step, since it has already lost its flat top. the slope fell to zero, so the curve at one climbs nothing per step. at x two, how many times the step will the height move? |

### dlg-00005-063 turn 8

**Pupil:** down, since past the peak the curve falls.

**Tape tutor:** right, it falls, so the height goes down. I'll stake it that the curve at two point five loses more than one for each one across. the slope is minus one point six, though the height still stands at one point two. so each x gave its own number, two, zero, minus one point six, and that number for each x is the derivative. could a curve's derivative stay one number everywhere?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, down, since past the peak the curve turns back toward zero. twice the sine of two point five is one point six eight, so I hold the curve sits below zero there. a tenth across dropped the height two tenths, so I take the curve at two point five to fall about twice as hard as it climbed at zero. the slope ran from zero to minus two, so steepness changes from place to place, and that changing number is the derivative. |
| A | (silence) | yes, past the top the curve falls, and a tenth across at two point five is a fair test. twice the sine of two point five is one point six eight, so I make the height there minus one point nine nine. so I hold the curve drops about twice the step there. the slope fell to minus two, so the derivative is steepness, and that number changes from place to place. would a straight line keep one slope everywhere? |
| B | (silence) | yes, down, because past the top the wave has turned away from its peak. twice the sine of two point five is one point six nine, so we put the wave below zero there. so I'd say it drops about twice the step there, as at zero. the slope fell to minus two, so the derivative is the slope, and it changes from place to place. would a straight line keep changing like that? |

### dlg-00005-073 turn 0

**Pupil:** a derivative is just the slope? a curve has one slope everywhere?

**Tape tutor:** a slope, yes, but whether it's one slope, numbers can test. one height shows no steepness, so we compare two close ones. doubling just stretches sine, so we know this wave swings evenly. at zero, sine is zero, and doubled stays zero, so our wave crosses the middle there. a tenth along, sine is nearly a tenth, so I put the wave two tenths up. at one point five seven, will a tenth step lift the height two tenths again?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, let me know if you have any | let's test that with numbers on one wave. I expect this wave to climb and fall twice for every step along, because its sine doubles twice. at input zero, sine of zero is zero, and doubled it's still zero, so I hold the wave crosses the middle there. if the input moves from zero to a tenth, how far will the output move? |
| A | (silence) | right, so I expect this curve to climb and fall in equal steps. I hold that at zero, the sine of zero is zero, and twice zero is still zero. so I say the curve crosses the middle at zero, at height zero. if the input moves to one tenth, how much will the output move? |
| B | (silence) | it does, and that slope is what tells us how fast the height changes. I expect this cubic to climb to a hump, then fall to a dip. at minus two, I get minus two times minus two, plus three, so I hold the curve sits at minus two there. if the input moves to minus one, how far will the height move? |

### dlg-00005-073 turn 2

**Pupil:** yes, two tenths again, because the curve keeps one slope.

**Tape tutor:** two tenths held near zero, so your guess earns its trial up high. at one point five seven, we have the wave at its peak, height two. on one slope, we need the wave a tenth further on to sit two tenths above that top. the second height reads one point nine nine, so up here a tenth step dropped the wave a hundredth. how steep would you guess the wave is at one point five seven?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, two tenths, because the curve keeps one slope. so I hold that at one point five seven the wave climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves. at the peak, one point five six, how much will the slope read? |
| A | (silence) | yes, two tenths, because the wave keeps one slope. so I hold that at one point five seven the wave stands two tenths up, just as the tenth before did. the slope is two, so the wave climbs two for each step along. at the peak, x one point five seven, how much will the slope read? |
| B | (silence) | right, two tenths, because the curve keeps one slope. so I hold that at one point five seven the wave lifts two tenths for each tenth along. the slope is two, so the wave climbs two for each one along there. at the peak, one point five six, will the slope read two again? |

### dlg-00005-073 turn 4

**Pupil:** zero, because at the peak the wave is flat.

**Tape tutor:** zero holds, since a tenth past the top the wave dropped only a hundredth. so at one point five seven, we have a wave whose steepness comes to nothing. yet near zero the wave rose two tenths for every tenth, so there we have a wave twice as steep as it runs. the slope climbed from nothing to two, so one wave carries many slopes. what will the slope read at three point one four?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | a flat top is a natural guess, so let's test it. if the wave were flat at its peak, I'd expect a tenth step barely to move it. the second height is one point nine nine, so the wave dropped a hundredth in a tenth. I take it the wave at one point five seven falls one for every one across. the slope is minus one, so the wave falls one for every one across at its top. at the peak, how much will the slope read? |
| A | (silence) | flat is a fair guess, since the top is level. a tenth further on, at one point six seven, I get the wave near minus one point nine nine. so I take it the wave falls here as steeply as it rose at the peak. the slope is minus one point three nine, so the wave falls here as steeply as it rose. at the peak, height two, I get a slope of zero. at the crossing, height zero, how much will the slope read? |
| B | (silence) | flat is right, so we can test that with numbers. a tenth past the peak, sine is near one, so I put the wave near its top there. a tenth further, it has dropped to about zero, so I take the wave to be falling one for one past its top. the slope reads minus one, so the wave falls one for every one along there. at the peak, height two, did the slope read zero, or minus one? |

### dlg-00005-073 turn 6

**Pupil:** minus two, since the wave falls steeply through the middle there.

**Tape tutor:** minus two holds, and arithmetic earns it. doubled sine a tenth past three point one four is minus two tenths, so I place the wave just under the middle there. the second height now reads minus point two, two tenths down. so I'm betting it falls there as steeply as it rose at zero. the slope lands on minus two, two down for each one. at one, will the slope read nearer two or nearer one?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | minus two holds, and your reason fits the wave we have. at three point one four, sine is zero, and doubled it stays zero, so the wave crosses the middle there. a tenth along, it's near minus two tenths, so I put the wave falling two for each one across. the slope went from zero to minus two, so the wave falls twice as fast as it runs. how steep will the slope read at minus one point five? |
| A | (silence) | two holds, since the wave falls steeply through its middle. at three point one four, sine is near one point nine seven, so doubled it is two. so I put the wave below the axis there, falling two for every one across. the slope reads minus two, so the wave drops two for each one across there. how much will the slope read at zero? |
| B | (silence) | minus two holds, since the wave falls twice as hard as it rises at zero. at three point one four, I make the wave sit below the axis, two tenths down for every tenth along. so I take it to be falling there at two per step. the slope came to minus two, so the wave drops two for each step along. how much will the slope read at minus one point five? |

### dlg-00005-073 turn 8

**Pupil:** nearer one, since one sits closer to the flat peak than zero.

**Tape tutor:** nearer one is right, since one sits close to the flat peak. so I'll claim the wave at one climbs about one for each step along. there the wave rises one point zero eight for each step, as you guessed. so the derivative is a rule giving each place its own slope. would three sine x keep these same slopes?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | nearer one holds, and the numbers show why. at one, doubled sine is about one point six eight, so I put the wave just under its top. so I claim it falls there about as gently as it climbs at zero. the slope eased from minus two to about zero, so the wave falls gently at one and climbs steeply at zero. how much will the slope read at minus one point |
| A | (silence) | nearer one holds, and the numbers show why. at one, doubled sine of one is zero, so I put the wave on the middle line there. a tenth on, it's twice sine of one tenth, so I claim the wave climbs there. the slope went from minus two to about one point nine nine, so near one the wave climbs near one per step. so the slope changes from place to place, |
| B | (silence) | one holds, and the numbers show why. at one, doubled sine is about one point six eight, so I put the wave just under its top. near one, I make the wave climb about one for each one along. the slope reads one point zero eight, so near one the wave climbs about one per step. so the slope changes from place to place, and that changing number is the derivative. |

### dlg-00005-074 turn 0

**Pupil:** a curve has just one slope all the way along?

**Tape tutor:** worth testing rather than trusting, so let's work numbers on this cubic first. if one steepness held everywhere, I'd expect our curve to be straight. at one, one cubed is one, three times one is three, and one take away three is minus two, so I'm betting our curve sits at minus two above one. if the input nudges a bit past one, how many times that nudge will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, let me know if you have any | it's a natural guess, since a straight line has one slope everywhere, so let's test it on a curve. I expect this curve to climb at first, then turn over and fall, since it doubles a sine. at zero, sine of zero is zero, and twice zero is still zero, so I hold the curve sits at height zero there. if the input moves to one tenth, how far will the output move? |
| A | (silence) | it does, and that slope is what tells us how fast the height changes. if the height changes slowly, we should find the curve flat where it crosses the middle. at two, two times two is four, and one taken from that leaves three. so I hold that the curve stands three above the axis at two. if I nudge the input to two point one, how far will the height move? |
| B | (silence) | it does, because a wave is only steep where its height changes fast, and this wave climbs only once. at minus two, I expect the wave to sit below the axis, since two cubed less three twos is minus two. if I nudge the input to minus one, how far will the output move? |

### dlg-00005-074 turn 2

**Pupil:** up only slightly, since the curve bottoms out near there.

**Tape tutor:** yes, only slightly, and you're right that the dip is there. we should find it barely above minus two at one point one, since one point one cubed is one point three three one, less three point three. a tenth of a step lifted it three hundredths, so I'd say the curve is flat right at one. zero slope, so at the bottom it neither climbs nor falls. at two, what will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | a fair guess, so let's try it. if your guess holds, I'd expect the curve to rise only a tiny bit past one, since it falls away to zero there. |
| A | (silence) | right, only slightly, so let's test your call. if your rule holds, I'd expect the curve to rise just under one for one point one, since one point one cubed is one point three three one, less three point three. the height went from minus two to minus one point nine seven, so the curve climbed about once the step. at zero, where the curve crosses the axis, how many times the step will the height move? |
| B | (silence) | right, and that little more than one cubed is what we need to turn a drop of one into a climb of many times that nudge. one point one cubed is one point three three one, so we have the curve climbing about three for each one across at one. the slope is three, so the climb is three times the step. at zero, where the curve crosses the middle, how many times the step will the climb be? |

### dlg-00005-074 turn 4

**Pupil:** zero, since a curve keeps one slope all the way along.

**Tape tutor:** a natural thing to think, since at one the slope does read zero. by your reckoning, we ought to find the curve as level at two as at one. the slope climbed to nine, so near two the height moves nine times any small step. so at zero, between the hump and the dip, how many times the step will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, zero, because the bottom has no climb left. two cubed is eight, less six, so I'm betting our curve sits at minus two there. a tenth past two, the |
| A | (silence) | yes, zero, because the curve keeps one slope all along. two cubed is eight, less three, so I'm betting our curve sits at minus two there. if the input |
| B | (silence) | yes, zero, because the curve lies flat at its bottom and climbs out of it. two cubed is eight, less three, so I'm betting our curve stands two above the axis at two. |

### dlg-00005-074 turn 6

**Pupil:** down, because the curve falls between the hump and the dip.

**Tape tutor:** right, it falls, and at zero the height is zero. one tenth cubed is a thousandth, less three tenths, so I make our curve nearly three tenths under zero at one tenth. a drop of three tenths over one tenth, so I'd put the fall at zero near three times the step. minus three, so here the height drops three times the step. at one half, how many times the step will the height fall?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, down, because past the dip the curve heads toward the bottom. I'd put the curve at zero below minus two, since zero cubed is zero, and taking away three makes minus three. so I expect it to climb nine times the step at zero, as it did near two. the slope ran from minus three to nine, so the curve climbs nine for each one across there. at minus one and a half, how many times the step |
| A | (silence) | right, it falls, and between the hump and the dip it falls just as hard. zero cubed is zero, and three times zero is zero, so I'm betting our curve crosses the axis at zero. if you're right, it must drop from its hump to zero with no change in steepness. the slope ran from nine down to minus two, so the curve falls twice as hard between the hump and the dip as it climbs between one and two. at minus one and a half, |
| B | (silence) | yes, two, because the curve falls between the hump and the dip. at zero, the cube of zero is zero, and three times zero is zero, so I'm betting our curve sits at height zero there. if you're right, I'd expect it to drop two for every one across at zero. the slope ran from nine down to minus two, so the curve drops two for every one across there. so at minus one and a half, how many times the step will the height move? |

### dlg-00005-074 turn 8

**Pupil:** three, since the slope at zero was minus three.

**Tape tutor:** a fair guess, since zero gave three. going by your three, I'd make our curve drop as steeply at one half as at zero. instead it falls two and a quarter times the step there. nine at two, zero at one, minus three at zero, so the steepness changes from place to place, and that changing number is the derivative. so did the curve fall more gently at one half than at zero?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, three, since the slope at zero was minus three. one half cubed is a quarter, less three quarters, so I make our curve below zero at one half. three tenths down for one tenth of step, so I'd say it falls three times the step there. the slope ran from minus three to minus three and a half, so the fall changes from place to place. at minus two and a half, how many times the step will the height fall? |
| A | (silence) | yes, three, and your three at zero is the slope we should find at one half too. one half cubed is a quarter, less three quarters, so I make the curve below zero there. three tenths down over one tenth, so I'd say it falls three times the step there. the slope ran from minus three to minus three and a half, so the fall changes from place to place. at minus two and a half, how many times the step will the height fall? |
| B | (silence) | right, minus three, and the slope at one half is minus two and a half, so the curve falls harder there. so I'd say the curve at one half drops two and a half times the step, just as the slope said. the slope ran from minus three to minus two and a half, so the steepness changes from place to place. at minus one and a half, how many times the step will the height fall? |

### dlg-00005-081 turn 0

**Pupil:** so, how would i work out a derivative? i'm not sure.

**Tape tutor:** you work one out from numbers, one input at a time, so I'll start with a wave. we know this wave never climbs past two, because a sine tops out at one and we double it. at one, the sine is about point eight four, so we have the wave sitting about one point six eight high there. if I nudge the input to one point one, how much do you think the height climbs?

| rendering | bare | trained |
|---|---|---|
| typed | Hi, I'm Fong. | let's work numbers before naming them, because a derivative is about how fast something changes. this wave doubles its sine, so we should find it climbing and falling. at input one, the sine of one is about point eight four, so we should find the wave there sitting about two point five six high. if the input moves to one point one, how far will the height move? |
| A | (silence) | it's a natural thing to wonder, because derivatives sound like numbers you find on a curve. so let's work one out on numbers before trusting a rule. we expect this curve to climb to a hump, then fall to a dip. at input zero, the cube is zero, and three times zero is still zero, so we have the curve crossing the axis there. if the input moves from zero to a tenth, how far do you guess the height will move? |
| B | (silence) | you can work out a derivative with numbers, so let's try it. I expect this curve to climb and fall, since a cube outgrows a cube. at input one, one cubed is one, and one minus three is minus two, so I say the curve sits at height minus two there. if the input moves to one point one, how far will the height move? |

### dlg-00005-081 turn 2

**Pupil:** it climbs to about one point seven eight, since sine still rises.

**Tape tutor:** right, it climbs to about one point seven eight. we should find it about point one above the old height, because the sine still rises there. so a step of point one lifting it point one tells me the wave at one climbs about one per one along. the slope reads just over one, so the climb matches the step. at zero, where the wave has no height, how much will it climb per step?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, about one point seven eight, since sine still climbs. so I hold the wave must be near its top at one point one, since it barely climbs there. the slope is zero point two eight, so the wave climbs two point eight times the step. at the input zero, where the height is zero, how much do you think the slope reads? |
| A | (silence) | yes, about one point seven eight, and the wave itself shows it. we should find the wave at one point one about one point six nine high, since the sine there is about point one nine four. so the second height rose about as much as the step, and I take it the wave climbs about one per one here. the slope is one, so the wave climbs one for one at one. at zero, |
| B | (silence) | yes, about one point seven eight, and the wave itself shows that. we get about point eight four from the sine of one point one, so we have the wave climbing about one per step there. so the slope is one, and it matches our one point six eight high. at the input zero, where the height is zero, how much do you think the slope reads? |

### dlg-00005-081 turn 4

**Pupil:** less than at one, since the wave sits lower there.

**Tape tutor:** let's try zero. two sine of zero is zero, and two sine of point one is about point two. so I'd say the wave at zero climbs twice its step, steeper than at one, though it sits lower. the slope rose to two by zero, so the wave climbs fastest where its height is nothing. at the peak, one point five seven, how much will it climb per step?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, less than one, because the wave is lower there. at zero the sine is zero, so we have the wave sitting at ground level. if you nudge the input to zero point one, how much will the slope read? |
| A | (silence) | right, less than one. at zero the wave sits at height zero, so we have it climbing nothing per step there. so the slope fell from one to nothing, and the wave's height there is zero, so the two are separate numbers. at the peak, two point five seven, how much will the slope read? |
| B | (silence) | right, less than one, because the wave lies flat there. we should find it barely climbs at zero, since the sine there is near zero. so the slope fell from one to about zero, and the flat wave at zero is the same height as the steep one at one. at two, how much will |

### dlg-00005-081 turn 6

**Pupil:** neither up nor down, since the top of the wave is flat.

**Tape tutor:** right, the top is flat, so it neither climbs nor falls there. two sine of one point five seven is two, so I doubt any point on this wave stands higher. and zero climbed steepest with no height, so I think the climb is all gone up here at the most height. the slope fell from two to zero, so height and steepness differ. at two, just past the top, how much will it fall per step?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, flat. one point five seven sine is about one point five seven, so I put the wave's top at height zero. if you nudge the input to one point six, how much will the slope read? |
| A | (silence) | right, flat. one point five seven sine is about one point five seven, so I take it the wave stops climbing there. so I claim the wave climbs nothing per step at its top, though its height there is one. the slope fell from two to zero, so the steepest climb came at the lowest place. at four point five, how much will the slope read? |
| B | (silence) | right, flat. one point five seven sine is about one point five seven, so I take the wave to sit there at height one. if you nudge the input to one point six, how much will the slope read? |

### dlg-00005-081 turn 8

**Pupil:** up, since the wave still stands high there.

**Tape tutor:** let's try two, which is high. I get two sine of two as one point eight two, and two sine of two point one as one point seven three, so we drop. that's point oh nine down per step, so I'd put the fall at two just under one per one along. the slope reads minus point eight three, so a high wave still falls. at two, is the wave climbing or falling?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, it still stands high there, so I expect it to fall. two point five sine is about one point nine nine, so I make the wave below the top there. and since it still stands high, I hold it falls there more gently than at the peak. the slope fell to minus one point nine nine, so the wave falls gently there while still high. at three point five, how much will it fall per step? |
| A | (silence) | right, it still stands high there, so the fall must be gentle. two point five seven sine is about one point six eight, so I put the wave below the top by that much. and since it falls gently, I take the fall to be about one per one along. the slope fell to about one, so the fall is gentle there while the wave stands high. at three, how much will it fall per step? |
| B | (silence) | right, it still stands high there, so the fall must come from its turn away from the top. two point five seven is about two point one five, so I put the wave below the top there. and I'd bet it falls there faster than at zero. the slope fell to minus one point three one, so the wave drops one and a third per step there. at three point four six, how much will it fall per step? |

### dlg-00005-087 turn 0

**Pupil:** sorry, how would I work out a derivative of a curve myself?

**Tape tutor:** no need for sorry, you work it out with arithmetic. the height here is the input cubed less three times it, so any height can be worked by hand. I doubt such a curve stays level anywhere. at two, two cubed is eight and three twos are six, so I expect eight less six. that leaves two, so I hold the curve stands two high there. at two point one, how high do you guess?

| rendering | bare | trained |
|---|---|---|
| typed | Hi, I'm Sarah. | you work it out from a second height, because a derivative is how fast the height changes. I expect this curve to climb, since the cube outgrows the square. |
| A | (silence) | you work out a derivative yourself, so let's find one on numbers instead of rules. I expect this curve to climb and fall, since a cube and a half cube both grow faster than one would guess. at input one, I get a cube of one and a half cube of one as one and a half, so I hold the curve stands one and a half high there. if the input moves to one point one, how far will the height drop? |
| B | (silence) | you work out a curve by feeding numbers to an equation and seeing how far the output moves for a small step. so we should find this cubic climbing to a hump and then falling to a dip. at zero, the cube is zero, and three times zero is zero, so we have the curve crossing the axis there. if the input moves from zero to zero point one, how far will the output move? |

### dlg-00005-087 turn 2

**Pupil:** two point nine six one, because the cube grows fast.

**Tape tutor:** yes, that is right, because the cube climbs quickly once we are past two. we should find the curve nearly a whole unit higher, just a tenth of a step past two. the height now reads two point nine six, so a tenth of a step lifted the curve almost ten times as far. at nought, where the curve sits at height nought, how many times the step do you guess the height moves?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, two point nine six one, because the cube grows fast. so I hold the curve climbs past two to two point nine six one, since two point one cubed is three point four six one and three point three two cubed is three point nine six one. that tenth of a step lifted it nearly a whole unit, so I take the curve at two to climb about nine per step. the slope is nine, so the curve climbs nine for every one across. at zero, |
| A | (silence) | right, it climbs fast there, so I hold the curve stands just under two point nine six one at two point one. the slope is nine point six one, nine point six one up for one along. at two, the top, how much do you guess the slope reads? |
| B | (silence) | right, it climbs fast there, so I take your point to be the top sits just past two. the height went from two to two point nine six one, so the curve climbs nine point six one times the step. that steepness is the derivative, and I hold the curve climbs harder at two than at two. the slope is nine point six one, while the height is two. at three, how steep do you guess the curve climbs? |

### dlg-00005-087 turn 4

**Pupil:** zero point one, since a low curve should be fairly flat.

**Tape tutor:** a fair guess, so let us try it a tenth past nought. I am betting the curve drops there, because a tenth cubed is only a thousandth while three tenths come off. so we must land nearly three tenths below where the curve started. the second height sits three tenths below nought, three times the step, so low is not flat. at one, in the dip, how many times the step will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, about one tenth, since the curve at nought barely climbs. so I hold the curve at nought moves about one tenth for each tenth we step along. the slope is one point zero eight, so the curve climbs nearly one for each step there. at one, how many times the step do you guess the height moves? |
| A | (silence) | yes, zero point one is right, because a flat curve barely moves with its place. at nought, the cube is zero, and three times zero is still zero, so I hold the curve sits at nought there. a tenth on, at nought point one, I expect it to rise just under |
| B | (silence) | yes, zero point one is right, because a low curve barely moves. we should find the curve at nought barely above nought, since a tenth of a step lifted it only a hundredth. the height went from two point nine six to nought point one six, so about one tenth of a step lifted one tenth of the height. at one, how many times the step do you guess the height moves? |

### dlg-00005-087 turn 6

**Pupil:** less than one time the step, since the dip is nearly level.

**Tape tutor:** yes, well under once, because the dip is nearly level. I make one point one cubed one point three three one, so taking three point three off leaves the curve just above minus two. so a tenth of a step barely lifts it, and right at one I hold the curve dead level. the slope is zero, flat at the bottom of the dip. at two, how much will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, less than one time the step, because the dip is nearly level. one cubed is one, and three times one is three, so I hold the curve sits at nought there. a tenth on, it is one point one cubed less three point one, so I claim the curve climbs there. the slope is one point one against nought, so low is not |
| A | (silence) | yes, one and a half times the step is right, because the dip is nearly level. we should find the curve barely climb out of its dip at one, since its climb from nought has barely started. the slope is one and a half, so the curve moves that many times the step there, while at nought it moved three times the step. at minus one and a half, how many times the step will the height move?; |
| B | (silence) | yes, less than one is right, because the dip is nearly level. one cubed is one, and three times one is three, so I hold the curve sits at nought there. if the step were a tenth bigger, how many times the step would the height move? |

### dlg-00005-087 turn 8

**Pupil:** two, since the curve stands two high there.

**Tape tutor:** fair guess, two is the height there, so let us try it. if height sets steepness, I expect the curve at two to climb only twice its step. the slope says nine, so the curve climbs nine times its step there, though it stands only two high. that number differs from place to place, and that changing number is the derivative. so what gave us the nine at two, the height or the climb?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, two, because two cubed less three twos is eight less six, so the curve stands two high there. so I hold it climbs two for every one across at two, its steepness. the slope reads two, so the curve climbs two for every one across at two, while at one it was zero. at nought, how much will the slope read? |
| A | (silence) | yes, two, because two cubed is eight, less six, so we find the curve two high there. and since two cubed over eight is two, I hold the curve climbs twice as fast as it goes out. the slope ran from zero to two, so the curve climbs two for every one across at two. at the peak, one point five seven, how much will the slope read? |
| B | (silence) | yes, two, because two cubed less three twos is minus one point three three, so I expect the curve to climb out of its dip there. the slope went from zero to nine, so the curve climbs nine times the step at two, while at one it barely moved. at minus one and a half, how much will the slope read? |

### dlg-00005-091 turn 0

**Pupil:** hi, how would I find a curve's slope at one point?

**Tape tutor:** good question, so let's find out with numbers. since one minus a square never climbs past one, we should get a hill. at one, one squared is one, and one minus one is zero, so we have the hill at ground level there. a tenth further, one point one squared is one point two one, so we must already be below ground. so at one, where the height is zero, how steep is the hill?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, let me know if you need help with that math stuff | good question, so let's work one out on numbers. I expect this cubic to climb fastest where its hump is highest, since a hump is what a climb follows. at minus two, cubing gives minus two, and taking away three times minus two leaves two, so I hold the curve sits at height two there. if I nudge the input to minus one, how far will the height drop? |
| A | (silence) | you find a curve's slope at one point by nudging the input a tiny bit and seeing how far the output moves, so let's test that on this wave. I expect this wave to climb to a hump, then fall back to zero. at zero, the sine of zero is zero, and twice zero is still zero, so I hold the wave sits at height zero there. if the input moves from zero to zero point one, how far will the output move? |
| B | (silence) | you find it from an input number, because a curve is how a number gets turned into a height. we expect this curve to climb to a hump, then fall back to zero. at zero, the cube is zero, and three times zero is still zero, so we have the curve sitting at height zero. if the input moves to one, how far will the height move? |

### dlg-00005-091 turn 2

**Pupil:** zero, because the height there is zero, so it's flat.

**Tape tutor:** you call it flat since the height is zero, so let's try it. the height fell zero point two one over a step of one tenth, two point one times the step, so I suspect the hill at one is anything but level. the slope is minus two, so the hill drops two for each step across, even at zero height. at x minus one, also at zero height, how steep is the hill?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, zero, because the hill is level at one. so I hold that the hill must lie flat at one, with no climb or fall there. the slope is zero, so the hill is level at one, though its height is zero. at two, how much will the slope read? |
| A | (silence) | right, it is flat there, because the hill is level at one. so I hold that the hill must be falling just as hard as it climbed there, since one point one squared is one point two one. the slope is minus one, so the hill drops one for each step along at one. at two, where the height is zero again, how steep do you guess the hill is? |
| B | (silence) | yes, flat, and the numbers show why. a tenth along lifted the hill a tenth, so we expect it to climb about one for one at one. the slope is zero at ground level, so the hill is level there though its height is zero. at two, how much will the slope read? |

### dlg-00005-091 turn 4

**Pupil:** two, going up, since the hill climbs toward its top there.

**Tape tutor:** yes, two, and climbing, because that side still rises toward the top. we have a hill mirrored across x zero, so its climb at minus one must match its drop at one, sign turned over. the slope went from minus two, through zero on the peak, to two at minus one, so steepness changes from place to place. out at x minus two, how steep is the hill?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, two, because the hill climbs toward its top at minus one. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. a tenth on, minus one point one squared is one point two one, so we must be near its top. so I take it the hill climbs two for each step across there, its steepest place yet. the slope went |
| A | (silence) | yes, two, because the hill climbs toward its top. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. two point one squared is four point two one, so we must be near its top by minus one. so I take it the hill climbs two for each step across there, just as you said. |
| B | (silence) | yes, up, because the hill climbs toward its top at zero. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. a tenth on, minus one point one squared is one point two one, so we must be near its top. so I take it the hill climbs three for each step across there, its steepest place yet. the slope reached three, |

### dlg-00005-091 turn 6

**Pupil:** four, going up, since the slope doubles as x doubles.

**Tape tutor:** yes, four, since the steepness runs at twice how far out we stand. I make the height minus three at minus two, and minus two point six one a tenth inward. that rise of zero point three nine is three point nine times the step, so to my mind the hill climbs four right at minus two. it climbs four there, twice its pace at minus one. at x one half, how steep is the hill?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, four, and up, because the slope doubled as x doubled. two squared |
| A | (silence) | yes, four, and doubling the step is the right rule. two times the step gave two, so we expect the hill at minus two to climb four for each step across. the slope went from two at minus one, to zero on the top, to four at minus two, so steepness changes from place to place. at x zero, how steep is the hill? |
| B | (silence) | yes, four, and doubling holds because the hill grew steeper as we went from one to minus one. so I expect the hill at minus two to climb four for each step across. the slope went from two back to zero, so the hill climbs four at minus two, where its height is minus two. at x zero, the very top, how steep is the hill? |

### dlg-00005-091 turn 8

**Pupil:** minus one, because the slope is minus twice x.

**Tape tutor:** yes, minus one, as twice a half is one. a half squared is a quarter, so I put the hill three quarters high at one half. it stands above ground, yet I claim it falls one per step there. so the derivative changes place to place, whatever the height. would lifting the hill change that slope?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus one, because twice x is the rule we have found. so I hold the hill at one half drops one for each step across, and falls harder than at minus two. the slope fell from four to minus one, so the hill steepens as it goes out. at x two, how steep is the hill? |
| A | (silence) | yes, minus one, because twice x is the rule we have found. so I hold the hill at one half drops one for each step across, and falls one for each step at x one. the slope fell from four to minus one, so the hill drops one for each step there, its own pace. at x two, how steep is the hill? |
| B | (silence) | yes, minus twice x, and that rule is the derivative. I make the hill at one half one point three nine above ground, so its climb there is just over two. the slope fell from four to just over two, so the hill steepens as it goes out. at x two, how steep is the hill? |

### dlg-00005-094 turn 0

**Pupil:** so, if I had a curve, how could I find its steepness myself?

**Tape tutor:** you nudge an input and watch how far the output moves, so we'll try it on two times sine x. sine never passes one, so I expect this wave to stay within two either way. at one, sine is about point eight four, so doubled I claim the wave stands one point six eight high there. stepping from one to one point one, how far will the height climb?

| rendering | bare | trained |
|---|---|---|
| typed | Hey, if you had a curve, how would you measure its steepness yourself? | you can find it yourself with numbers, because a curve's steepness at a point is how fast its height changes as you move a small step along. I expect this curve to climb, since the cube outgrows the square past one. at one, one times one is one, and three times one is three, so I hold the curve sits at height three there. if I step the input from one to one point one, |
| A | (silence) | you can find a curve's steepness yourself with numbers, so let's work one out. I expect this curve to climb fastest where its hump is highest, since a hump is where the climb is steepest. at input two, I get the curve standing two high, since two cubed less three twos is minus one, plus two. so I hold that the curve stands two high at input two, and climbs toward |
| B | (silence) | you can find a curve's steepness at a place with numbers, because a steep curve climbs fast where it rises fast. I expect this cubic to climb fastest where its hump stands highest, since a hump is the steepest place. at minus two, cubing gives minus two, and taking two away leaves two, so I hold the curve sits at height two there. if the input moves to minus one, how much |

### dlg-00005-094 turn 2

**Pupil:** up about zero point one, since the slope there is near one.

**Tape tutor:** right, a tenth, just as you said. I reckon two sine of one point one comes to one point seven eight, a tenth higher than at one. so a tenth of climb for a tenth of step is one rise per step. my guess is the curve right at one climbs just about that steeply too. and the slope agrees, just over one. so at zero, where the height is zero, how much will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, about zero point one, since near one the wave still climbs nearly one for each step along. so I hold that the wave must stand about one point zero eight at one point one, a tenth higher. the slope is zero point one zero, so the wave climbs nearly one for each step along there. at the peak, x zero, how much will the slope read? |
| A | (silence) | yes, about zero point one, since the wave barely climbs there. so I hold that the wave barely rises from one to one point one, since it barely moves there. the slope is zero there, while the height is one point six eight. at zero, where the wave crosses the middle, how much will the slope read? |
| B | (silence) | yes, about zero point one, since the wave barely climbs there. so I hold that the wave barely climbs at one, since its steepness is nearly zero. the slope is zero, so the wave climbs nothing there, though its height is one point six eight. at zero, where the height is zero, how much will the slope read? |

### dlg-00005-094 turn 4

**Pupil:** two, since the wave climbs steepest as it crosses zero.

**Tape tutor:** yes, two, the wave crosses the middle at full tilt. and I get two sine of point one at point two, so the curve rises twice its step. I count the wave nearly twice as steep at zero as at one. and the slope reaches two at the crossing, with no height at all. so at the peak, one point five seven, how much will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, two, because the wave crosses zero upwards at zero. I take it to be a flat wave at zero, since its height there is zero. the slope went from one to two, so the wave climbs twice as hard at zero as at one. so the slope changes from place to place, and that changing number is the derivative. at three point five seven, how much will the slope read? |
| A | (silence) | right, two, because the wave doubles sine there, and sine of zero is still zero. so I claim the wave crosses the middle at zero with no height at all, yet climbing two for every one along. the slope went from one to two, so the wave climbs twice as hard at zero as at one. at three point five seven, how much will the slope read? |
| B | (silence) | right, zero height and steepest climb go together. sine of zero is zero, so I claim the wave crosses the middle there at its steepest. so I make the wave climb about one per step there, just as your rule says. the slope came out one there, where the height is zero. at three point one four, how much will the slope read? |

### dlg-00005-094 turn 6

**Pupil:** zero, since the wave stops climbing right at the top.

**Tape tutor:** yes, zero, the top is where climbing turns to falling. two sine of one point six seven is one point nine nine, so a tenth past the peak, I'm betting the wave has sunk just a hundredth. so to my mind, steepness drains from two at the crossing to nothing up top. the slope ran from two down to zero, flat on the peak. so at two, past the top, how much will it read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, zero, because the climb stops at the top. I make the wave one point six eight high at one point five seven, so its climb is gone. so the slope fell from two to zero, while the height rose from zero to one point six eight. so the steepness followed the height, not the place on the board. at three point one four, how much will the slope read? |
| A | (silence) | right, zero, because the climb stops dead at the top. I make the wave one point six eight at one point five seven, just under its peak. so I claim it climbs nothing at the top, though it stands highest there. the slope fell from two to zero, so the top is flat while it stands highest. so at three point five seven, how much will the slope read? |
| B | (silence) | right, zero, because the climb stops at the top. so I take it the wave can't be rising there, since its climb ran out at the crossing. the slope fell from two to zero, so the top is flat. so at three point one four, how much will the slope read? |

### dlg-00005-094 turn 8

**Pupil:** minus zero point eight three, since past the top it falls.

**Tape tutor:** yes, minus zero point eight three, since the wave is falling there. I suspect that at two the wave drops about eight tenths for every step, because its top lies behind it. and the slope settles at minus zero point eight three. so every place gave its own number, two where the height was zero, and that changing number is the derivative. if you found that number at every x, what curve would it trace?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus zero point eight three, because past the top the wave turns back down. so I take it the wave at two falls just as steeply as it climbed at one, only down. the slope reads minus zero point eight three, so the wave falls there just as hard as it climbed at one. so the derivative is the slope, the steepness at a place. would doubling the input give the steepest climb? |
| A | (silence) | yes, minus zero point eight three, because past the top the wave falls. I put two sine of two at point one eight four, so I claim the wave drops about eight hundredths in a tenth. so I make the wave fall about one per step past the top. the slope fell from zero to minus one, so the wave drops one per step there. so at three point one four, how much will the slope read? |
| B | (silence) | yes, minus zero point eight three, because past the top the wave stops climbing and starts to fall. so I take it the wave falls here just as gently as it climbed at one. the slope fell from zero to minus zero point eight three, so the wave falls gently at two, just as its climb at one showed. so at three point one four, how much will the slope read? |

### dlg-00005-110 turn 0

**Pupil:** derivatives lose me completely, honestly. can you build one up from scratch?

**Tape tutor:** yes, because a derivative only counts how fast an output moves when its input moves, so we start with one output in hand. take one half, square it to get one quarter, and take that from one, which leaves three quarters. so, by that sum, we know our hill, one minus x squared, stands three quarters high at one half. if one half moves a tenth right, to six tenths, how far will the height drop?

| rendering | bare | trained |
|---|---|---|
| typed | Hello, welcome to the Voice of the People radio network. How can I help you today? | sure, so let's work one out on numbers before naming it. we know a wave doubles whatever input it gets, so we should find this wave crossing zero in the middle. at input zero, the sine of zero is zero, and twice zero is still zero, so we have the wave crossing at height zero there. if the input moves from zero to a tenth, how far will the height move?; |
| A | (silence) | let's try it with numbers first. we expect a sine wave to look like a wave, because it keeps doubling whatever its input is, and doubling a number never cuts it in half. at zero, the sine of nothing is nothing, and doubling nothing still leaves nothing, so we have the wave crossing the middle there. if the input moves from zero to one tenth, how far will the output move? |
| B | (silence) | let's test that with numbers. we know a sine never passes one, so we should find this wave never climbs above one. at input zero, the sine of zero is zero, and twice zero is still zero, so we should find the wave crossing the middle there. if the input moves from zero to one tenth, how far will the height move? |

### dlg-00005-110 turn 2

**Pupil:** up, I think, since steeper ground should mean higher ground.

**Tape tutor:** a natural hunch, so let's test it just past one half. if steeper meant higher, I should find the hill standing above three quarters there. six tenths squared is thirty-six hundredths, so one minus that leaves sixty-four hundredths, lower. I put the fall at eleven hundredths, from a step of one tenth. so the height dropped one point one times the step. how many times the step will the steepness read right at one half?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, up, and the board can show it. a tenth right only lifts the height by about a tenth, so we should find the hill barely above three quarters there. the slope is zero at one half, so the hill lies level there, though its height is three quarters. at x equals zero, where the height is zero, how much will the slope read? |
| A | (silence) | right, steeper ground means higher ground, so I hold the hill stands lower at one and a tenth than at one half. the height fell from three quarters to three quarters minus two tenths, so the hill dropped two tenths in a tenth. so I reckon the hill at one and a tenth falls two for every one across. the slope is two, the same drop per step we found. at x equals zero, where the height is zero, how much will the slope read? |
| B | (silence) | right, and the board can test that. if steeper meant higher, we should find the hill at one point five seven higher at one point three nine than at one half. |

### dlg-00005-110 turn 4

**Pupil:** minus one, because the height falls as x moves right.

**Tape tutor:** yes, minus one, since the height falls as x moves right. a tenth gave one point one, so we should find the hill's own lean at one half a touch gentler, one down per step. the slope reads minus one, so here the hill drops one for every one along. at one and a half the hill stands much lower, so how many times the step will the slope read there?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus one, because the hill drops one for each step right of one half. so I hold that the hill falls one for each one across at one half, though its height there is three quarters. the slope is minus one, so the hill drops one for each one |
| A | (silence) | yes, one, because the fall is one for every one across. so I hold that the hill drops one for every one across at one half, just as the numbers show. the slope is one, so the hill drops one for every one across, at one half. how many times the step will the slope read at two? |
| B | (silence) | minus one is right, so let's test it at one half. if your rule held, I'd find the hill falling one for every step right there. instead it climbs one for every step, so the hill climbs one and a half times the step there. so the slope is one and a half, while the height is three quarters. at x equals zero, where the height is zero, how many times the step will the slope read? |

### dlg-00005-110 turn 6

**Pupil:** minus three, since the slope is minus two times x.

**Tape tutor:** yes, minus three, because your two times x doubles one and a half. one and a half squared is two and a quarter, so we put the hill below the ground, at minus one and a quarter. since you double one and a half, we should find the hill falling three for every one along. minus one grew to minus three, so the lower ground was steeper. at x one, how much will the slope read?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus three, since the slope fell from minus one to minus two. one and a half squared is two and a quarter, so I claim the hill sits below three quarters there. so I expect it to drop three times the step there, as at one half. the slope ran from minus one to minus three, so the hill drops three for every one along there. at two, how many times the step will the slope read? |
| A | (silence) | yes, minus three, because the hill drops two for every one along there. one and a half squared is two and a quarter, so we find the hill below zero there. and since the fall eases from one half, I'd say its steepness there is minus three, not minus one. the slope reads minus three, so the hill drops three for every one along there, not one. at two, how many times the step will the slope read? |
| B | (silence) | yes, minus three, since the slope is minus two there. one and a half squared is two and a quarter, less three and a quarter, so we find the hill two and a quarter below ground there. so I take it the hill drops three per step there, as you said. the slope ran from minus one to minus three, so the steepness changes from place to place. at x equals zero, where the hill is flat, how many times the step will the slope |

### dlg-00005-110 turn 8

**Pupil:** minus two, since two times one is two.

**Tape tutor:** yes, minus two, because doubling one gives two. the ground at one sits higher, at zero, so we expect a gentler lean there. the slope eased to minus two, gentler on higher ground, so the derivative is a number that changes from place to place. is a hill's highest point ever its steepest?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus two, because two times one is two. one squared is one, so we get the hill two below ground there. so I take it the hill falls two for every one along at one. the slope ran from minus one through minus three, so the hill fell two per step at one, just as you said. at x zero, the very top, how |
| A | (silence) | yes, minus two, because two times one is two. one squared is one, so we have the hill two below ground at one. so I take it the hill falls two for every one along there. the slope ran from minus three to minus two, so the steepness changes from place to place. at x zero, where the hill is flat, how much |
| B | (silence) | yes, minus two, because two times one is two. one squared is one, so we have the hill two below ground at one. so I take it the hill falls two for every one along at one. the slope ran from minus one through minus three, so the hill fell two per step at one, just as your doubling said. at x minus one and a half, |

### dlg-00005-123 turn 0

**Pupil:** hi there, derivatives lose me completely. can you build them up?

**Tape tutor:** yes, we'll build it from one machine, because the idea only shows once we watch an output change. this machine cubes an input and takes three of it away, so we should find a rise, a dip, and then a climb. feed it one, and since one cubed is one and three ones make three, our curve sits at minus two there. if the input steps to one point one, how far will the height move?

| rendering | bare | trained |
|---|---|---|
| typed | Hi, I'm Fong. | sure, so let's work one out on numbers before trusting a rule. we know a wave doubles whatever its input is, so we should find this wave climbing and falling in equal steps. at input zero, the sine of zero is zero, and doubled it stays zero, so we have the wave crossing the axis there. if the input moves to one tenth, how far will the height move? |
| A | (silence) | good question, so let's work one out on numbers. I expect this wave to stay below zero, since its sine never passes one. at input one, sine of one is about point eight four, so I hold the wave sits about that high there. if the input moves to one point one, how far will the height move? |
| B | (silence) | they grow from numbers, so let's work one out. if height is how far up the curve sits, we should find this curve highest above the input two. two cubed is eight, and eight minus three times two is three, so we have the curve sitting three above the input two. if the input moves to two point one, how far will the height move? |

### dlg-00005-123 turn 2

**Pupil:** zero point three, since the curve climbs at one steady slope.

**Tape tutor:** zero point three is a fair guess, so we'll try it. if your guess holds, we should sit well clear of minus two just past one. by my sums, one point one cubed is one point three three one, and less three point three leaves us just above minus two. odd, it crept up just three hundredths. how steep, then, do you expect the curve right at one?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, and the slope is the steepness, so we should find the curve climbing about one per step there. one point one cubed is one point three three one, so we have the curve above minus two at one. so the height moved three tenths, about one per step. at the input zero, how many times the step will the height move? |
| A | (silence) | right, three tenths is right, and the climb is gentle because the slope is gentle. so we should find the curve barely higher at one point one than at one, since the climb is still gentle. the height went from minus two to minus one point nine five, a tenth of a step for a tenth of a rise. at zero, where the curve crosses the middle, how much will the slope read? |
| B | (silence) | right, and the slope is the steepness, so we should find the curve climbing just under its top at one. one point one cubed is one point three three one, less three, so we get a height of minus one point six six one. that rise of about a tenth over a tenth is about one per tenth, so we expect the curve at one to climb about one for every one along. the slope is one, so the curve climbs one for every |

### dlg-00005-123 turn 4

**Pupil:** zero, since the height barely moved over that small step.

**Tape tutor:** yes, zero, because three hundredths over a tenth of a step is barely a climb. so our curve must lie level right at one. the slope reads zero, so the curve climbs nothing per step there. past one, we should find it steepening, because cubing outgrows taking three away. by two the slope reads nine, so steepness changes from place to place. how steep will the curve be at zero, between the hump and the dip?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | right, zero, because the height barely moved there. so I hold the curve lies level at one, since its height barely moved. the slope is zero, so the curve lies level at one, though its height barely moved. at two, how far will the slope read? |
| A | (silence) | zero is right, and your reason holds. so I hold the curve at one barely lifts from its floor, even though it stands just above minus two. the slope is zero, so the curve lies flat there, though its height is minus two. at two, how steep do you guess the curve climbs? |
| B | (silence) | two is right, and the step of three hundredths shows why. so I hold the curve at one climbs two for every one across, though its height barely budged. the slope is two, while the height is minus two. at the input zero, where the curve crosses the middle, how much will the slope read? |

### dlg-00005-123 turn 6

**Pupil:** minus three, since it is heading down from the hump.

**Tape tutor:** yes, minus three, and at zero the height is zero. at zero point one I get a thousandth less three tenths, minus zero point two nine nine, so the curve drops about three times its step. that means we must have the curve falling three for every one along, right at zero. the slope came down from nine to minus three, as the sums said. how steep will it be at one half?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus three is right, because the curve falls from the hump to the dip. zero cubed is zero, and three take away zero leaves minus three, so we put the curve at minus three at zero. the slope reads minus three, so the curve drops three per step there. so the steepness changes from place to place, and that changing number is the derivative. what would the slope read at two? |
| A | (silence) | yes, three, because the hump is level and the dip is flat. so we should find the curve falling three per step at zero, as steep as it climbs at one. the slope fell from nine to minus three, so steepness changes from place to place. how steep will the curve be at two and a |
| B | (silence) | yes, minus three, because the hump is the highest point and the curve only falls from there. so I expect the curve to drop three per step at zero, as steep as at two, yet falling. the slope fell from nine to minus three, so steepness changes from place to place. how steep will the curve be at three point five? |

### dlg-00005-123 turn 8

**Pupil:** minus two point two five, since it is flattening toward the dip.

**Tape tutor:** yes, minus two and a quarter, since it eases toward the dip. I make it fall that much per step at one half. nine at two, zero at one, minus three at zero, so the derivative is the slope as it changes with place. could two places on one curve share a slope?

| rendering | bare | trained |
|---|---|---|
| typed | (silence) | yes, minus two point two five, because the curve is level at one and still falling there. so I expect it to drop two and a quarter for each one along at one and a half. |
| A | (silence) | yes, minus two and a quarter, because the hump is level and the dip is flat. so I expect the curve to level off at one half, just as flatly as the dip was. |
| B | (silence) | yes, minus two and a quarter, because the hump is level at zero and still falling here. so I take it the curve falls two and a quarter per step at one half. the slope fell from minus three to minus two and a quarter, so steepness |
