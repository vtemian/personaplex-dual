# Step 0: bare PersonaPlex live, with its own session prompts

Date: 2026-11-10. Branch `feat/step0-prompts` off main 1f2de1e.

## Why, and where this sits

After the live fail of the consistency checkpoint (B16 GATE-3: silent after the
opening, babble in Romanian), Vlad: "hai sa facem o chestie si mai simpla. doar un
mic fine tune ... pur si simplu sa mergem din aproape in aproape", and to the
4-step ladder ("da, hai sa facem asa"):

0. bare PersonaPlex live, with its official prompts (this plan);
1. a do-no-harm LoRA on spoken-only dialogues;
2. typed text at session start only;
3. typed text mid-dialogue.

Every step is judged by free-running behaviour (a replay with no forced context and a
short live session), never by the KL bar.

Step 0's question: does PersonaPlex, served by our engine the way NVIDIA serves it
(voice prompt + text system prompt), talk to a pupil? The B16 bare control replay
had no prompts (GATE-1 bare row: typed 7/1/5, spoken 9/0/4 answer/echo/ignore).

## The prompt mechanism (NVIDIA/personaplex 16419a2, moshi/models/lm.py, server.py)

`LMGen.step_system_prompts`, at the start of every session, in order:

1. the voice prompt: each Mimi frame of the voice, forced as moshi's own audio; text
   PAD (3); pupil stream SINE;
2. `audio_silence_frame_cnt` = int(0.5 * 12.5) = 6 frames: own SILENCE, pupil SINE,
   text PAD;
3. the text prompt, one token a frame on text: own SILENCE, pupil SINE. The server
   wraps it as `<system> {text} <system>` (unless already wrapped) and encodes it with
   the SPM (`text_tokenizer.encode`);
4. 6 more silence frames, as 2.

SILENCE / SINE are `dual.constants.SILENCE` / `SINE` (identical to lm.py's
SILENCE_TOKENS / SINE_TOKENS). The server then resets Mimi's streaming state; the
prompt frames are never played to the user.

Voices. `voices.tgz` in `nvidia/personaplex-7b-v1` holds only `.pt` files
(`{"embeddings": [T-1, 1, 1, 4096] bf16, "cache": [1, 17, 4]}`): the transformer
input embeddings of each voice-prompt frame plus LMGen's token ring at the end. No
wavs. Our engine steps codes, so the codes are recovered offline: each embedding is
`text_emb[PAD] + sum_k emb[k][token_k]` over the 16 audio codebooks with the pupil
(SINE) and text known; the 8 own tokens are solved by nearest-row coordinate descent
against `emb.0..7` (fetched from the safetensors by HTTP range). NATF2: residual at
bf16 rounding (mean 0.9 %, max 2.9 % of the norm; one wrong code would be ~30 %), and
the frame the embeddings and the cache both hold agrees exactly. 52 frames (4.2 s);
frame 0 is invisible to the model (initial tokens), as in PersonaPlex.
On the box: `/srv/spark/models/personaplex-7b-v1/voices/{voices.tgz, NATF2.pt,
NATF2.codes.json}` (rsync, not in the image). The tool is the session scratchpad's
`tools/invert_voice.py` (not committed: a one-off; the codes file records its source).

Server defaults differ from ours in two ways, kept as known differences:
- NVIDIA's server now prefills and pins the system prompt in the KV cache. We step it
  frame by frame (lm.py: same cache up to floating point, only RNG use differs) and do
  not pin: the prompt slides out of the 2048-frame context after ~2.7 min of session.
- Sampling: LMGen defaults temp 0.8, temp_text 0.7, top_k 250, top_k_text 25; ours
  are greedy unless `--temp/--temp-text` (moshi's top_k defaults are the same).

## The prompts used

- voice: `NATF2.codes.json` (NATF2: the README's assistant example voice);
- text: the README's assistant role, verbatim: "You are a wise and friendly teacher.
  Answer questions or provide advice in a clear and engaging way."

## Code (batch S0.1, one sonnet agent, TDD)

- `dual.engine` (pure): `wrap_system(text)` (PersonaPlex's `wrap_with_system_tags`),
  `system_prompt_frames(voice_codes, text_tokens, silence=6) -> list[Frame]` in the
  order above; `"prompt"` joins STEP_KINDS.
- `dual.jobs.engine`: `--voice-prompt FILE` (a `.codes.json`, `{"codes": [[8 ints]...]}`,
  or a `.wav` encoded by Mimi after peak normalisation, noted as not LUFS) resolved
  under `models/personaplex-7b-v1/voices/`; `--text-prompt TEXT`. Both checked before
  the model loads. `Engine.system_prompt(frames)`: `step_codes` per frame with forced
  text and forced own codes, nothing decoded, then `mimi.reset_streaming()`. Applied
  right after `open()` in every replay session and every live session (before the
  READY message and before an `--opening`); mic frames queued meanwhile are dropped
  and not counted as lag. Recorded: live `meta.json["prompt"]`, steps kind "prompt",
  replay rows `frames.prompt`, `summary.json` variant_flags `voice_prompt`/`text_prompt`.
- Replay, `--context none` only: turn 1's reply is classified too. Row `first_class`
  = `classify(turn-1 words, the 150 frames after spoken turn 1, decode)` (turn-1 words
  from `renderings[turn 1]["typed"]["text"][:-GAP_FRAMES]`, as turn 2's), and
  `summary.json["first_classes"]` counts it per variant. `--after-end` keeps its
  wait; `first_class` then reads the frames up to its stop.
- Targeted: `uv run pytest tests/test_engine_job.py tests/test_engine.py -q -p no:xdist`.

## Runs

1. Image engine1:<merge sha> (b16-engine-run's method), queue empty, MemAvailable
   >= 60 GiB, drama3 stopped.
2. Greedy: `make replay NAME=bare-prompt ARGS='--bare --voice-prompt NATF2.codes.json
   --text-prompt "You are a wise ..."'`.
3. Sampled at the server's defaults: same + `--temp 0.8 --temp-text 0.7 --seed 0`,
   NAME=bare-prompt-t08.
4. Live, whatever the replay says, with the better setting:
   `make engine NAME=bare-prompt ARGS='--bare --voice-prompt ... --text-prompt ...
   <temps> --opening none'` (`--opening none` is required: the serve default is the
   first holdout stem's forced opening). GATE-2 health from the box host, then
   the tunnel from the Mac; Vlad's live session is the step's real test.

## Pre-registered reading (written before any run)

Replay = the existing free-running replay (`--context none`): empty session after the
prompts, pupil turn 1 spoken in voice A from render-1, 150 free frames, then pupil
turn 2 typed (the burst) or spoken, 150 free frames, `classify`.

Bare PersonaPlex with prompts **talks live** if, in a replay run, its reply to the
**spoken pupil turn 1** (`first_class` of the `spoken` rows) is `answer` >= 10/13
**and** that reply (`first_answer`), read row by row, is English and on the topic of
the pupil's turn. Recorded for information, not as a bar: the turn-2 classes (`class`)
for spoken and typed (the burst; typed input is step 2/3's business). Below 10/13, or off-topic/non-English
text, the replay says "does not talk"; the live session still runs (Vlad's decision),
and the report says which.

Compared against GATE-1's bare row without prompts (typed 7/1/5, spoken 9/0/4).
