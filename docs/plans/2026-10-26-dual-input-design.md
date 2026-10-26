# personaplex-dual: one speech model, every pupil turn typed or spoken

The question this project answers: can one Moshi-class model take a pupil's turn as
typed text or as speech, at any point in a conversation, and teach the same either way?
Deterministically measured, no judge.

What is in scope is the door and its measurement. What is out: the drawing (chalkboard's
scribe and DSL), which comes back on top once the door passes, and whether the lesson is
good, which chalkboard's gates measure.

## Fixed points

- One model, Moshi alone. Knowledge in the weights: no RAG, no recogniser, no writer.
- Voice in and text in, both. Voice and text out, always both.
- A typed turn and a spoken turn with the same words must produce the same teaching.

## Section 1: what runs

**The model** is PersonaPlex's weights (`nvidia/personaplex-7b-v1`, NVIDIA Open Model
License, gated download), unchanged in shape: moshiko's 32-layer, 4096-wide temporal
transformer, its text head, its 16-step depformer (moshi's 8 codebooks, then the pupil's
8), Mimi. Plus one trunk LoRA trained here. No other module.

**Every 80 ms frame** the trunk takes three inputs summed: a text token, moshi's 8 Mimi
codes, the pupil's 8 Mimi codes. It emits one text token and 8 codes of moshi's voice.
This loop never changes.

**A spoken pupil turn**: the microphone's codes on the pupil stream; the text and
own-audio inputs are what moshi just produced. Plain Moshi.

**A typed pupil turn, at any frame**: the engine writes the typed words on the text input
one token a frame, PersonaPlex's silence codes (948, 243, 1178, 546, 1736, 1030, 1978,
2008) on moshi's own audio input, PersonaPlex's 440 Hz sine frame (430, 1268, 381, 1611,
1095, 1495, 56, 472) on the pupil input, then 0.5 s of the same with PAD on the text.
Frames are stepped as fast as the box computes them, about 89 ms each on the GB10, so a
12-word question costs about a second. Then the loop returns to normal and moshi answers
in voice.

This is PersonaPlex's own prompt mechanism (`step_system_prompts` in its fork's `lm.py`)
moved from "once, before frame 0" to "any frame". Its loop only ever forces text before
the dialogue; the typed-turn path is that call in a loop at a later moment, plus a text
box in the client.

**The marker.** While the pupil stream carries the sine, text on moshi's stream is the
pupil's words; while it carries real codes, the text is moshi's own transcript. Disjoint
by construction, no new token. A fourth input stream for pupil text was considered and
not taken: the trunk has never seen a fourth term, it needs the same data, and the
convention already separates the cases.

**Why PersonaPlex and not moshiko.** Mechanically the released moshiko takes the same
three inputs and nothing refuses text on its stream; what its weights do with it is read
it as their own words and collapse (chalkboard, `docs/results/text-in-speech-out.md`).
PersonaPlex trained a second reading, "text with silence under it and a sine beside it is
someone else's words: read, do not voice, respond", on about 2,250 h of rows. The LoRA
here moves that reading from frame 0 to any frame. Whether it could install the reading
from moshiko alone on a tenth of the data is unknown, so the first experiment trains both
bases on the same rows and the bar says.

Costs of the head start, each a thing to budget: the 16-step depformer (16.7 GB against
moshiko's 15.4; the engine's per-frame graph assumes 8 steps), its silence codes instead
of chalkboard's, a voice and persona shaped by customer-service data, English only.

## Section 2: how it is trained

**Source dialogues**: the chalkboard derivatives round-2 set. Measured at the time of
writing: 5,501 dialogues, 3.64 M frames, 80.8 h rendered, median dialogue 674 frames
(54 s), about five pupil turns each, about 27,000 pupil turns in all. A tenth is held out
whole, before rendering, and never rendered into a training row.

**Rendering**: each training dialogue becomes three rows on the three streams. The
teacher's turns are identical in all three: transcript on the text stream, voice on its
own audio stream, from the two-voice tape. The pupil's turns differ by a mask, one bit per
pupil turn, drawn once per dialogue from a fixed seed:

- row 1, the mask: turns with bit 1 typed, the rest spoken
- row 2, the complement: the same turns the other way round
- row 3, all spoken: the tape as it exists

A spoken turn: the pupil's TTS voice on the pupil stream, PAD on the text stream, silence
generated on moshi's stream. A typed turn: the triple above, then the 0.5 s gap. The
acoustic delays apply after, by the existing code. About 16,500 rows and 240 h of frames,
with about 13,000 pupil turns typed and the same 13,000 spoken, each followed by the
identical teacher answer.

**Loss**: on the teacher's text and audio. Zero on every frame of a typed turn, so the
model is never asked to predict the pupil's words, only to act on them (PersonaPlex did
the same for its prompt). The pupil's spoken frames carry no loss either.

**What trains**: the trunk LoRA only, rank 16, scaling 2.0, on the attention in/out
projections and the gated-FFN in/out of all 32 layers. Depformer, embeddings, heads
frozen. fp32 adapters under bf16 autocast, B at zero so step 0 is the base to the bit.

A full fine-tune is not forbidden, it is budgeted out: 8B parameters under AdamW is
128 GB before activations, the box has 128 GB unified, and a bf16 weight update at
PersonaPlex's rate of 2e-6 rounds to zero below one ulp. 8-bit Adam plus an fp32-safe
update fits, at days per run. It is the recipe if the LoRA runs flatten above the floor.

**Where it runs**: the sparks queue, one `make` verb, an image built on the PersonaPlex
fork, the rendered rows shipped as the job's data mount. Nothing starts by hand on the
box.

## Section 3: how it is measured

**The holdout**: the dialogues held out whole. For every pupil turn in them, three
renderings of the same words: typed, spoken in voice A, spoken in voice B. The teacher's
answer after each is the one fixed answer from the dialogue.

**The number**: teacher-force the dialogue up to and through that answer under each
rendering, read the text head's logits at every answer token, take the KL divergence per
token between two renderings, averaged over the answer. Per turn:

- the gap: KL(typed, spoken A)
- the floor: KL(spoken A, spoken B)

**The bar**: over the holdout, the gap's distribution sits inside the floor's: median gap
at or below median floor, and the 90th percentile likewise. Two voices saying the same
words are the case the product already treats as identical, so the floor is what "the
same teaching" means in this model.

**Determinism**: fixed seeds, teacher-forced, no sampling in the number; a rerun on the
same checkpoint reproduces it. One forward per rendering per turn, cheap enough for
every checkpoint.

**Controls**, same holdout:

- bare PersonaPlex and bare moshiko before training: the starting gap
- the conversion door, typed to TTS to the pupil stream, a third voice: must land on the
  floor by construction, or the harness or the TTS is the fault, not the door

**Not in the bar**: whether the lesson is good. That returns with the scribe.

## The first experiment: a data ladder, two bases

Eight runs, not one. The same LoRA on nested slices of the rows, an eighth, a quarter, a
half, all, from each base, same seed, read on the same holdout. The four points draw a
curve per base:

- still falling at the full set: more data buys more, and the generator that wrote these
  dialogues writes more, at about 15 h of rendering per 1,000 dialogues
- flat above the floor from a quarter on: data is not the lever; then rank, base, or the
  rendering itself
- inside the floor before the full set: done, with the surplus as margin for broader
  subjects

The rendering for every rung is done once. This is the ladder that settled chalkboard's
narrator data-scaling question, where five adapters from 336 to 5,384 rows tracked data
all the way.

**Before the ladder**, one probe on bare PersonaPlex, zero training: a spoken exchange,
then a typed turn forced mid-dialogue with the triple. Answer, echo, or ignore. It sets
the starting gap and says whether the frame-0 habit transfers at all.

## Precedents and the record

- PersonaPlex, arXiv 2602.06053, NVIDIA/personaplex: the triple, the 0.5 s slots as
  delimiter, no loss on the prompt, text-first against voice-first made no difference.
- "Can Speech LLMs Think while Listening?", arXiv 2510.07497: a second Moshi fine-tune
  that puts unspoken text on the monologue channel with silent audio; needed new
  delimiter tokens and a fine-tune.
- Kyutai never put non-speech text on Moshi's stream; MoshiRAG adds an embedding instead,
  and the STT prompt is labelled "very sensitive".
- Typed beats spoken on every paired benchmark (3 points for well-mixed models, 20 to 50
  for the rest), so the typed door is the easier one once the model has seen the shape.
- chalkboard `reports/Text input into speech models.md` holds the full survey.

## Open points

- The engine's per-frame graph on the 16-step depformer: run all 16 or stop at 8; cost
  per frame against the 89 ms today.
- PersonaPlex's voice prompt against the tape's voice: whether the fine-tune pulls the
  voice back, or a voice prompt is chosen and kept.
- A typed turn while moshi is mid-sentence: the triple forces its audio silent, so it
  cuts moshi off. Whether that is the right product behaviour is a later decision.
