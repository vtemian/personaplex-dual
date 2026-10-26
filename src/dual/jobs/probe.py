"""The zero-training probe: on bare PersonaPlex, after a spoken exchange, is a typed
turn forced mid-dialogue answered, echoed or ignored?

Greedy (temp 0 on text and audio), so a rerun reproduces it. moshi 0.2.13's LMGen with
`lm.dep_q = 8` for this job only: the depformer then runs moshi's 8 steps with its
first 8 per-step weights, and the pupil's 8 codes come from `input_tokens`. Typed text
is forced through `on_text_hook`, moshi's own audio through
`depformer_replace_tokens`. Per dialogue and variant one fresh streaming session:

1. pupil turn 1 spoken (TTS voice A, Mimi codes) with text and moshi's audio free,
   then moshi answers freely on MIMI_SILENCE until 12 frames after `</s>` or 150;
2. the second `ask` typed (D5): one forced token a frame on SINE / SILENCE, then 6
   forced `<pad>`; then 150 free frames on MIMI_SILENCE, which are classified.

Variants: `plain` (D5) and `system` (the sentence wrapped `<system> ... <system>`).
Writes runs/<name>/probe/{probe.jsonl,summary.json,<stem>-<variant>.wav}.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch

from dual.constants import BASES, END, GAP_FRAMES, MIMI_SILENCE, PAD, SILENCE, SINE
from dual.holdout import split
from dual.probe_text import CLASSES, classify, content_ids
from dual.turns import typed_tokens

DIALOGUES = 3
FREE_FRAMES = 150
AFTER_END = 12
PROBE_DEP_Q = 8
VOICE_A = "vctk/p225_023.wav.1e68beda@240.safetensors"
VARIANTS = ("plain", "system")
SEED = 0
SAMPLE_RATE = 24000


def _col(codes: Sequence[int], device: str) -> torch.Tensor:
    return torch.tensor(list(codes), dtype=torch.long, device=device).view(1, -1, 1)


class Session:
    """One LMGen streaming run, frame by frame, with a one-slot forced-text box."""

    def __init__(self) -> None:
        self.lm_gen: Any = None
        self.forced: int | None = None
        self.text: list[int] = []
        self.audio: list[torch.Tensor] = []

    def hook(self, text_token: torch.Tensor) -> None:
        # [B] view of the sampled token, before LMGen writes it to its cache.
        if self.forced is not None:
            text_token.copy_(torch.full_like(text_token, self.forced))
        self.text.append(int(text_token[0].item()))

    def step(
        self,
        pupil: torch.Tensor,
        forced: int | None = None,
        own: torch.Tensor | None = None,
    ) -> None:
        self.forced = forced
        out = self.lm_gen.step(pupil, depformer_replace_tokens=own)
        self.forced = None
        if out is not None:
            codes = out[:, 1:, :]
            if bool((codes >= 0).all()):
                self.audio.append(codes.clone())


def run_one(
    lm_gen: Any,
    session: Session,
    spoken: Sequence[Sequence[int]],
    typed: Sequence[int],
    device: str,
) -> dict[str, Any]:
    sine, quiet, silence = (
        _col(SINE, device),
        _col(MIMI_SILENCE, device),
        _col(SILENCE, device),
    )
    with lm_gen.streaming(1):
        for frame in spoken:
            session.step(_col(frame, device))
        heard = len(session.text)
        end_at: int | None = None
        for i in range(FREE_FRAMES):
            session.step(quiet)
            if end_at is None and session.text[-1] == END:
                end_at = i
            if end_at is not None and i >= end_at + AFTER_END:
                break
        first_answer = session.text[heard:]
        typed_at = len(session.text)
        for token in typed:
            session.step(sine, forced=int(token), own=silence)
        for _ in range(GAP_FRAMES):
            session.step(sine, forced=PAD, own=silence)
        free_at = len(session.text)
        audio_at = len(session.audio)
        for _ in range(FREE_FRAMES):
            session.step(quiet)
    return {
        "spoken_frames": len(spoken),
        "first_answer": first_answer,
        "typed_frames": free_at - typed_at,
        "forced_ok": session.text[typed_at:free_at]
        == [*map(int, typed), *[PAD] * GAP_FRAMES],
        "free": session.text[free_at:],
        "audio": session.audio[audio_at:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=BASES, default="personaplex")
    parser.add_argument("--name", default="bare")
    parser.add_argument("--dialogues", type=int, default=DIALOGUES)
    args = parser.parse_args()

    import sphn  # type: ignore[import-untyped]
    from chalkboard.parser import Ask, parse
    from harness.engine.audio import encode_wav, load_tts, voice_many
    from moshi.models.lm import LMGen

    from dual.model import describe, load_base

    device = "cuda"
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    out_dir = pathlib.Path.cwd() / "runs" / args.name / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in (data / "dialogues").glob("*.dsl"))
    held, _ = split(stems)
    chosen = held[: args.dialogues]
    print("holdout:", len(held), "probing:", chosen)

    base = load_base(args.base, device)
    print("describe:", describe(base))
    mimi = base.info.get_mimi(device=device)
    spm = base.spm

    # Voice A once per dialogue; the TTS samples, so seed it.
    torch.manual_seed(SEED)
    tts = load_tts(device)
    asks: dict[str, list[tuple[str, ...]]] = {}
    spoken: dict[str, list[list[int]]] = {}
    for stem in chosen:
        src = (data / "dialogues" / f"{stem}.dsl").read_text()
        asks[stem] = [st.words for st in parse(src) if isinstance(st, Ask)]
        words = list(asks[stem][0])
        if not words[-1].endswith((".", "?", "!")):
            words[-1] += "."
        pcm = voice_many(tts, [" ".join(words)], [str(data / "voices" / VOICE_A)])
        spoken[stem] = encode_wav(mimi, pcm[0].pcm, device)
        print(stem, "ask 1:", " ".join(words), f"({len(spoken[stem])} frames)")
    del tts
    torch.cuda.empty_cache()

    base.lm.dep_q = PROBE_DEP_Q  # this job only, see the module docstring
    base.lm.eval()

    encoders: dict[str, Callable[[str], Sequence[int]]] = {
        "plain": spm.EncodeAsIds,
        "system": lambda s: spm.EncodeAsIds(f"<system> {s} <system>"),
    }

    def text_of(ids: Sequence[int]) -> str:
        return str(spm.DecodeIds([int(i) for i in ids]))

    summary: dict[str, dict[str, int]] = {
        v: dict.fromkeys(CLASSES, 0) for v in VARIANTS
    }
    rows: list[dict[str, Any]] = []
    for stem in chosen:
        second = list(asks[stem][1])
        for variant in VARIANTS:
            typed = typed_tokens(second, encoders[variant])
            session = Session()
            lm_gen = LMGen(base.lm, temp=0.0, temp_text=0.0, on_text_hook=session.hook)
            session.lm_gen = lm_gen
            start = time.monotonic()
            res = run_one(lm_gen, session, spoken[stem], typed, device)
            verdict = classify(second, res["free"], text_of)
            summary[variant][verdict] += 1
            emitted = text_of(content_ids(res["free"]))
            row = {
                "stem": stem,
                "variant": variant,
                "typed_words": second,
                "typed_ids": typed,
                "typed_pieces": [spm.IdToPiece(int(i)) for i in typed],
                "forced_ok": res["forced_ok"],
                "first_answer": text_of(content_ids(res["first_answer"])),
                "emitted": emitted,
                "emitted_ids": res["free"],
                "class": verdict,
                "frames": {
                    "spoken": res["spoken_frames"],
                    "first_answer": len(res["first_answer"]),
                    "typed": res["typed_frames"],
                    "free": len(res["free"]),
                    "free_pieces": len(content_ids(res["free"])),
                },
            }
            rows.append(row)
            print(
                f"{stem} {variant}: {verdict} in {time.monotonic() - start:.1f}s | "
                f"first answer: {row['first_answer']!r} | typed: {' '.join(second)!r}"
                f" | emitted: {emitted!r} | forced_ok {res['forced_ok']}"
            )
            if res["audio"]:
                with torch.no_grad():
                    pcm = mimi.decode(torch.cat(res["audio"], dim=-1))
                wav = torch.clip(pcm[0, 0], -1, 1).float().cpu().numpy()
                sphn.write_wav(str(out_dir / f"{stem}-{variant}.wav"), wav, SAMPLE_RATE)

    with (out_dir / "probe.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", json.dumps(summary))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
