# Consistency 5: voice D as the consistency teacher (B15)

Plan: `docs/plans/2026-11-08-voice-d-teacher.md` (pre-registered at b6e4ca8, before any voice-D number).
Run `voiced1-pp-l3`: λ = 3, seed 0, from scratch, 3 epochs, the teacher row is the typed row's typed turns
voiced by TTS voice D (`vctk/p294_023.wav`), every other pupil turn the tape. Everything else is
`cons1-pp-l3` (B12): same rows, init, batch order.

## The answer

**D9 verdict: NO BETTER. The part that failed is R; behaviour passes.**

- **PASS fails.** `content.passes` is false at every epoch (R 2.28 / 1.87 / 1.53; p90 ratio 1.47 / 1.17 / 1.36).
- **LOWERS THE CEILING fails on both legs.** It needs R at epoch 3 ≤ 1.37 (1.87 − T) and the paired e3 CI
  against `cons1-pp-l3` excluding 0. R e3 is **1.53**; the CI is **−0.34 [−0.66, +0.03]**.
- **Behaviour at epoch 3 passes every D9 health check** against `ladder-personaplex-1` e3, one of them exactly
  at the limit: speaks after the spoken opening 13/13 A and 13/13 B; typed-opening onset frame 3 in 13/13;
  mid-lesson stops 2 / 1 / 1 of 52; swap-follow **91** ≥ 96 − 5 = 91.
- **The rule's epoch is 1, and epoch 1 is broken** the way λ3-tape epoch 1 was: silent after the spoken
  opening in 13/13 (A) and 10/13 (B), typed opening late or silent. The noise rule reads it as "down"
  (R_gap 2.28 at e1, CIs vs both λ0 seeds exclude 0), but that epoch is not usable.

**Why it did not work (read from the ceiling numbers, post hoc):**

- **The ceiling moved; the run did not reach it.** D‖A is 0.75 × floor at epoch 3 (tape‖A is 1.55-1.90 × floor),
  so a typed answer that matched voice D could pass the bar. But typed‖D on the holdout is **0.0148** at e3,
  as large as the bar's gap itself (typed‖A 0.0139). The term did not pull typed close to D on unseen turns.
- **R fell mostly because the floor rose, not because the gap shrank.** At epoch 3, against `cons1-pp-l3`:
  gap 0.01387 vs 0.01458 (**−5%**), floor 0.00907 vs 0.00781 (**+16%**). The same at positions 26+:
  gap 0.0104 vs 0.0102, floor 0.0056 vs 0.0047, ratio 1.88 vs 2.18. A run that makes A and B differ more
  is not a run that makes typed sound more like speech.

## The voice screen (GATE-0, D3, decided by rule)

`render-d-screen` (render-1 + D1/D2/D3), measured on `cons1-pp-l3` epoch 3. All ratios over the floor of the same pass.

| | r_L = L‖A | r_L_b = L‖B | s_L = typed‖L / typed‖A | admissible |
|---|---|---|---|---|
| D1 `vctk/p226_023.wav` | **0.730** | 1.030 | 1.157 | yes |
| D2 `vctk/p280_023.wav` | 0.782 | 0.906 | 1.138 | yes |
| D3 `vctk/p294_023.wav` | 0.755 | **0.937** | 0.954 | yes |
| reference: tape‖A | 1.669 | | | |
| reference: C‖A (door) | 0.926 | | | |

- **Reproduction:** gap 0.014583, floor 0.0078059, R 1.868: equal to B12's epoch-3 summary (0.00%). The
  copied A/B/typed rows are byte copies.
- **Pick:** smallest r_L is D1; D3 is within the 0.05 tie band (+0.025), D2 is not (+0.052); the smaller
  r_L_b of D1/D3 is D3. **D = D3 = `vctk/p294_023.wav`.** Lever check 0.755 ≤ 1.669 − 0.3: passes.

## render-2 and the smokes (GATE-1, decided by rule)

- **render-2** (`--from render-1 --add-voice D=vctk/p294_023.wav --teacher-voice D`, 1,208 s):
  teacher rows 234; answer pairs 585 voiced = 585 tape; typed-turn pupil frames voiced 34,210 vs tape 32,766
  (+4.4%); longest teacher row 2,769 frames (dlg-00005-109, ≤ 3,000); holdout rows 390; TTS deterministic.
  `check_same_tape` passed on all 117 train stems and 13 holdout bases.
- **voiced-smoke-1** (`--only-stem dlg-00005-109 --max-steps 3`): peak 39.2 GiB, cons 0.042 / 0 / 0.036, teacher 32% of step.
- **voiced-smoke-wide** (`--widest-first --max-steps 4`): peak 50.1 GiB (51.4 reserved), cons 0.04-0.08,
  teacher 8.8-9.0 of 26.8 s (33%). Startup line = `cons1-pp-l3`'s: 287 steps/epoch, typed rows 227,
  answer pairs 585, content frames 94,464, plus `teacher voiced D` (D4 check 2).

## Bar R per epoch

| | e1 | e2 | e3 |
|---|---|---|---|
| **voiced1-pp-l3** R | **2.28** | **1.87** | **1.53** |
| gap / floor | .00879 / .00385 | .01065 / .00569 | .01387 / .00907 |
| p90 ratio | 1.47 | 1.17 | 1.36 |
| cons1-pp-l3 (λ3 tape, s0) R | 2.51 | 1.95 | 1.87 |
| ladder-personaplex-1 (λ0 s0) R | 3.04 | 3.24 | 2.87 |
| vs cons1-pp-l3 same epoch | −0.23 [−0.45, +0.33] | −0.07 [−0.60, +0.10] | −0.34 [−0.66, +0.03] |
| vs λ0 s0 same epoch | −0.76 [−1.17, −0.39] | −1.37 [−1.89, −0.71] | −1.35 [−1.91, −0.62] |
| vs λ0 s1 same epoch | −0.68 [−1.44, −0.21] | −1.20 [−1.82, −0.57] | −1.61 [−1.90, −1.08] |

- **Noise rule** (`make noise`, d 0.08, R0 3.00, T 0.50): R_gap **2.28 at e1**, R_gap − R0 −0.72, CIs
  −0.76 [−1.17, −0.39] and −0.68 [−1.44, −0.21]: **"down"**. B12's seed-0 λ3-tape was "not moved" (2.51 at e1).
- `pool/epoch_ci.py` (consistency-4's plus the new rows) reproduces consistency-4's six λ3 rows exactly.

## The ceiling, per epoch (from `summary.json`)

| | e1 | e2 | e3 |
|---|---|---|---|
| typed‖D (`gap_D_median`, what the term minimises on train) | .00752 | .00986 | .01479 |
| typed‖A (the bar's gap) | .00879 | .01065 | .01387 |
| D‖A / floor (the new ceiling) | 0.876 | 0.751 | 0.745 |
| D‖B / floor | 0.940 | 0.877 | 0.823 |
| tape‖A / floor (the old ceiling) | 1.861 | 1.903 | 1.550 |
| C‖A / floor (door) | 0.959 | 0.954 | 0.907 |
| typed‖tape (`tape_gap_median`); cons1-pp-l3 | .01028; .00997 | .01507; .01156 | .01991; .01563 |

- typed‖tape *rose* against λ3-tape (e3 .0199 vs .0156): with the tape no longer the teacher, typed drifts from it.
- **Turn-0 lead-frame KL(typed‖A)** (B14's `t0`): 0.102 / 0.232 / 0.527 (λ3-tape s0 0.062 / 0.252 / 0.701,
  reproduced). Lead KL(typed‖D): 0.070 / 0.155 / 0.391.

## Behaviour next to the bar

Epoch 3 (and the rule's epoch 1) against the two controls, all in one fresh blind pool (below). Text counts
from `pool/qual4.py`; judged counts from the 13-checkpoint pool.

| | λ0 s0 e3 (`ladder-personaplex-1`) | λ3-tape s0 e3 (`cons1-pp-l3`) | **voice-D e3** | **voice-D e1** |
|---|---|---|---|---|
| bar R | 2.87 | 1.87 | **1.53** | 2.28 |
| held text loss | 0.6105 | 0.5775 | 0.5867 | 0.5582 |
| speaks after spoken opening, A / B (of 13) | 13 / 13 | 13 / 13 | **13 / 13** | **0 / 3** |
| typed-opening onset frame (tape 3) | 3 (13/13) | 3 (13/13) | **3 (13/13)** | 3 (1), 7-12 (9), silent 3 |
| mid-lesson stops, typed / A / B (of 52) | 0 / 0 / 0 | 0 / 1 / 0 | **2 / 1 / 1** | 13 / 11 / 8 |
| mid-lesson median words, typed / A / B | 63 / 65 / 61 | 63.5 / 60.5 / 64 | 64 / 62 / 61.5 | 54 / 58.5 / 59 |
| opening uptake, typed / A / B (of 13) | 13 / 12 / 13 | 12 / 11 / 13 | 12 / 12 / 12 | 5 / 0 / 3 |
| opening makes the tape's move, typed / A / B | 12 / 11 / 9 | 10 / 11 / 8 | 11 / 10 / 9 | 2 / 0 / 0 |
| mid-lesson uptake, typed / A / B (of 52) | 51 / 50 / 48 | 50 / 49 / 49 | 49 / **46** / 49 | 51 / 50 / 50 |
| mid-lesson substance differs, typed vs A / A vs B | 18 / 14 | 16 / 18 | 16 / 16 | 6 / 8 |
| swap: played turn taken up (of 104) | 96 | 92 | **91** | 102 |

- **D9 health at e3** (against λ0 s0 e3 in this pool): speaks ≥ 12/13 A and B: 13/13 ✓; onset by frame 4
  in ≥ 11/13: 13/13 ✓; stops ≤ 3 of 52 per rendering: 2/1/1 ✓; swap ≥ 96 − 5 = 91: **91 ✓ (at the limit)**.
- **Swap, paired (post hoc):** λ0 e3 vs voice-D e3 10 vs 5 discordant (p 0.30); λ3-tape e3 vs voice-D e3
  7 vs 6 (p 1.0). Voice-D e3 sits where λ3-tape e3 sits; neither is shown below λ0.
- **One soft spot (post hoc):** mid-lesson uptake after A is 46/52 (λ0 50, λ3-tape 49), and the A rendering
  has 3 "neither" true-condition swap items. Small, not a pre-registered criterion.
- **Epoch 1 is not usable:** silent after the spoken opening (A 13/13 silent), like λ3-tape e1 (B13: 12/13 silent).
  Its 102 swap and low "differs" counts come from short, generic answers (stops 13/11/8).

## The blind pool

- **Input:** `pool/combine13.py` (consistency-4's `combine8.py`, unchanged logic) reproduces B14's
  `combined11.jsonl` byte for byte with B14's specs; two more specs add `v1-e1` and `v1-e3`
  (`consistency-5/answers.jsonl`), 3,211 rows.
- **Blinding:** `pool/blind5.py` (blind4, seed 13); two builds byte-identical. 1,690 pairs, 2,535 answers,
  1,352 swap items in 18 / 26 / 13 chunks.
- **Judges:** 57 opus judges (xhigh), one per chunk, each reading only that chunk's `items.jsonl` and
  `rubric.txt` (B10's rubrics verbatim). All 5,577 items labelled.
- **Agreement with B14 on the 11 shared checkpoints:** swap followed-vs-not 1,122/1,144; substance
  1,312/1,430; uptake 2,117/2,145; tape's move 2,058/2,145; well formed 1,981/2,145. Old swap counts move by
  ≤ 3 (λ3 s1 e2 90 → 93, base e3 94 → 96). This pool's numbers supersede B14's for the shared checkpoints.
- Unblinded labels: `consistency-5/judge-labels.jsonl` (`score4.py`); swap: `pool/swap7.py`.

## Training

| | e1 | e2 | e3 |
|---|---|---|---|
| s/epoch | 4,317 | 4,311 | 4,311 |
| peak (GiB, 64 cap) | 56.0 | 56.0 | 56.0 |
| mean cons | 0.00891 | 0.00561 | 0.00650 |
| (cons1-pp-l3) | 0.00989 | 0.00709 | 0.00772 |
| max grad | 4.45 | 10.15 | 16.0 |
| held text | 0.5582 | 0.5344 | 0.5867 |
| (cons1-pp-l3 / λ0 s0) | 0.5574 / 0.5511 | 0.5366 / 0.5397 | 0.5775 / 0.6105 |

## GPU-hours (D10 estimate 8.1-8.9 h)

| job | s |
|---|---|
| render-d-screen | 486 |
| measure-d-screen (8 renderings, 1 epoch) | 2,450 |
| render-2-smoke / render-2 | 255 / 1,208 |
| voiced-smoke-1 / voiced-smoke-wide | 184 / 223 |
| train voiced1-pp-l3 | 13,227 |
| measure-voiced1-pp-l3 (6 renderings, 3 epochs) | 5,366 |
| sbs-v1-e3 / sbs-v1-e1 | 2,827 / 2,830 |
| **total** | **29,056 s = 8.07 h** |

## What was run

- Code: B15 V1-V4 merged to main at 80d7766 (render `--from/--add-voice/--teacher-voice`, train `--teacher voiced`,
  measure pairs for added voices, `LADDER=6`), image `voiced1:80d7766`.
- Jobs (sparks, one at a time): render-d-screen, measure-d-screen, render-2-smoke, render-2, voiced-smoke-1,
  voiced-smoke-wide, `LADDER=6` (train + measure + `make bar`), sbs-v1-e3 and sbs-v1-e1 on B14's image
  `sbs2:72868db` (`--no-bare --swap-turns 2,6 --mem-cap-gib 64`, render-1). sbs2 read the new checkpoint
  (the `teacher` key is in the run meta, not the adapter meta); batch shapes 53/35/29/26/23/21/8 and 28/24, peak 38.9 GiB.
- Reproduce the pool:
  - `python3 pool/combine13.py <repo> combined13.jsonl base-e3=docs/results/consistency-3/answers.jsonl:base-e3 l3-e1=docs/results/consistency-3/answers.jsonl:l3-e1 base-s1-e3=docs/results/consistency-4/answers.jsonl:base-s1-e3 l3s1-e2=docs/results/consistency-4/answers.jsonl:l3s1-e2 l3s1-e3=docs/results/consistency-4/answers.jsonl:l3s1-e3 v1-e1=docs/results/consistency-5/answers.jsonl:v1-e1 v1-e3=docs/results/consistency-5/answers.jsonl:v1-e3`
  - `python3 pool/blind5.py <out> $(for m in base-e2 l10-e1 l10-e2 base-e1 l3-e2 l3-e3 base-e3 l3-e1 base-s1-e3 l3s1-e2 l3s1-e3 v1-e1 v1-e3; do echo $m=combined13.jsonl:$m; done)`
  - text counts: `uv run python pool/qual4.py <repo> v1-e3=docs/results/consistency-5/answers.jsonl:v1-e3 ...`
  - bar: `PYTHONPATH=src uv run python docs/results/consistency-5/pool/epoch_ci.py runs/`

## Caveats

- One seed. The R drop vs λ3-tape is not shown (CI includes 0 at every epoch).
- The screen chose D on the λ3-tape e3 checkpoint; D‖A under the new run stayed low (0.75), so the choice held.
- The side-by-side decodes render-1 (A/B/typed), not render-2's D: no decode shows the model after a D turn.
- The swap criterion passes exactly at its limit (91 = 96 − 5).

## The gate (GATE-2, Vlad)

Pre-registered NO BETTER options: **(a)** stop the consistency line and send λ3-tape e3 to the engine;
**(b)** revisit the bar. The numbers add one observation for (b): the bar's R moves with the floor (A‖B),
which a teacher can change without bringing typed closer to speech.
