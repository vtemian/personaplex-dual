"""The live engine: a trained checkpoint streaming on the box, a pupil on a web page
typing or speaking, moshi answering in voice and text.

Two modes, one Engine:

- `--replay N` (no network): for the first N holdout dialogues, pupil turn 1 is heard
  in voice A (render-1's codes, decoded to PCM and fed as the mic would be), moshi
  answers; pupil turn 2 is then typed (the burst) or spoken in voice A, and what moshi
  says next is classified answer / echo / ignore as the B1.5 probe did. Reports the
  cost of a frame. Writes runs/<name>/engine/{replay.jsonl,summary.json}.
- default: an aiohttp server. GET / is the page (src/dual/client.html), GET /health,
  GET /ws the websocket (the wire is dual.engine). One session at a time, each a fresh
  streaming state, each recorded under runs/<name>/engine/<session>/.

The model loads as the side-by-side job loads it: adapters from the checkpoint, cast
to bf16, `lm.dep_q = 8`, greedy by default. Prompts are off unless `--voice-prompt` /
`--text-prompt`: PersonaPlex's session prompts, stepped right after a session opens
(docs/plans/2026-11-10-step0-bare-live.md); without them, the measured condition of
B10-B15.

A sparks job cannot publish a port. The server binds 0.0.0.0 in the container and
prints the container's bridge address first; the box host reaches it directly, the
laptop through `ssh -L` (make tunnel). The job ends at --max-minutes or on SIGTERM
(sparks abort), saving the open session first.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import pathlib
import queue
import signal
import socket
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
from aiohttp import web

from dual.constants import CODEBOOKS, END, GAP_FRAMES, MIMI_SILENCE, PAD
from dual.engine import (
    AUDIO,
    FRAME_BYTES,
    FRAME_SAMPLES,
    READY,
    SAMPLE_RATE,
    STATUS_EVERY,
    TEXT,
    TYPED,
    Counters,
    Frame,
    PcmBuffer,
    Timing,
    burst,
    check_voice,
    first_class_counts,
    is_word,
    pack,
    piece_text,
    prompt_steps,
    replay_summary,
    session_name,
    step_row,
    system_prompt_frames,
    trim_backlog,
    typed_words,
    unpack,
    wrap_system,
)
from dual.model import MEM_CAP_GIB
from dual.probe_text import classify, content_ids, words_of
from dual.sidebyside import StepInput, step_input
from dual.turns import typed_tokens

DEP_Q = 8  # the probe's and the side-by-side's: moshi's 8 depformer steps
FREE_FRAMES = 150  # replay: frames moshi may answer in
AFTER_END = 12  # replay: frames kept after </s>
AFTER_END_LIMIT = 600  # replay --after-end: frames moshi may take to reach </s>
MAX_TYPED_WORDS = 60  # a typed turn longer than this is refused (D4's cap)
WARMUP_FRAMES = 4  # pays one-time compile/allocation costs, checks the path runs
FULL_SCALE = 32767.0
VOICE_DIR = ("models", "personaplex-7b-v1", "voices")
VOICE_PEAK = 0.5  # a voice .wav is peak-normalised to this fraction of full scale
ZERO_FRAME = b"\x00" * FRAME_BYTES  # digital silence into the mic path
CLIENT = pathlib.Path(__file__).resolve().parents[1] / "client.html"
STATE: web.AppKey[dict[str, Any]] = web.AppKey("state", dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="the live engine: serve or --replay")
    p.add_argument("--name", required=True)
    p.add_argument("--base", default="personaplex")
    p.add_argument("--run", default="cons1-pp-l3", help="a train run under runs/")
    p.add_argument("--epoch", type=int, default=3)
    p.add_argument("--bare", action="store_true", help="no adapters: the base alone")
    p.add_argument(
        "--replay", type=int, default=0, help="N holdout dialogues offline; 0 serves"
    )
    p.add_argument("--render", default="render-1", help="the holdout the replay reads")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8998)
    p.add_argument("--max-minutes", type=float, default=180.0)
    p.add_argument("--temp", type=float, default=0.0, help="audio temperature")
    p.add_argument("--temp-text", type=float, default=0.0)
    p.add_argument(
        "--eager",
        action="store_true",
        help="NO_TORCH_COMPILE=1 (moshi's torch.compile off)",
    )
    p.add_argument("--seed", type=int, default=0, help="torch seed per replay session")
    p.add_argument(
        "--context",
        choices=("none", "tutor", "opening"),
        default="none",
        help="tutor: teacher-force the tape history instead of turn 1 + a free run; "
        "opening: force the --opening stem's opening in every stem",
    )
    p.add_argument(
        "--opening",
        default=None,
        metavar="STEM|none",
        help="the holdout stem whose opening is forced at the start of a session "
        "(default: the first record; none: no opening)",
    )
    p.add_argument(
        "--after-end",
        action="store_true",
        help="replay: wait up to 600 frames for </s> after turn 1 (context none only)",
    )
    p.add_argument(
        "--voice-prompt",
        default=None,
        metavar="FILE",
        help="PersonaPlex's voice prompt: a .codes.json or .wav under "
        "models/personaplex-7b-v1/voices/",
    )
    p.add_argument(
        "--text-prompt",
        default=None,
        metavar="TEXT",
        help="PersonaPlex's text system prompt, wrapped in <system> tags",
    )
    p.add_argument("--mem-cap-gib", type=float, default=MEM_CAP_GIB)
    args = p.parse_args(argv)
    if args.replay < 0:
        p.error("--replay must be >= 0")
    if args.max_minutes <= 0:
        p.error("--max-minutes must be positive")
    if not 0 < args.port < 65536:
        p.error("--port must be in 1..65535")
    if args.temp < 0 or args.temp_text < 0:
        p.error("temperatures must be >= 0")
    if args.seed < 0:
        p.error("--seed must be >= 0")
    if args.context != "none" and args.after_end:
        p.error(
            "--after-end is for the empty-session flow, not --context " + args.context
        )
    if args.context == "opening" and args.opening == "none":
        p.error("--context opening needs an opening, not --opening none")
    if args.context == "opening" and not args.replay:
        p.error("--context opening is a replay context: pass --replay N")
    if args.replay and args.context != "opening" and args.opening is not None:
        p.error("--opening with --replay needs --context opening")
    if args.voice_prompt is not None:
        try:
            check_voice_name(args.voice_prompt)
        except ValueError as e:
            p.error(str(e))
    if args.text_prompt is not None and not args.text_prompt.strip():
        p.error("--text-prompt is blank")
    return args


def check_voice_name(name: str) -> None:
    """A voice file is a name under the voices directory: no absolute path, no `..`."""
    rel = pathlib.PurePath(name)
    if not name or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"--voice-prompt {name!r}: a file name under voices/ only")


def voice_path(name: str) -> pathlib.Path:
    check_voice_name(name)
    return pathlib.Path.cwd().joinpath(*VOICE_DIR, name)


@dataclass
class Prompt:
    """PersonaPlex's session prompt. `codes` is None for a .wav until Mimi exists;
    `tokens` and `frames` are filled by `complete_prompt`."""

    voice: str | None
    path: pathlib.Path | None
    codes: list[list[int]] | None
    text: str | None
    tokens: list[int] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)


def check_prompt(args: argparse.Namespace) -> Prompt | None:
    """The prompt flags checked before the model loads: the voice file exists and, for
    json, parses to valid codes; the text is not blank. None when neither is given."""
    if args.voice_prompt is None and args.text_prompt is None:
        return None
    path: pathlib.Path | None = None
    codes: list[list[int]] | None = None
    if args.voice_prompt is not None:
        path = voice_path(args.voice_prompt)
        if path.suffix not in (".json", ".wav"):
            raise ValueError(f"{path}: a voice prompt is a .json or a .wav")
        if not path.is_file():
            raise ValueError(f"{path}: voice prompt not found")
        if path.suffix == ".json":
            try:
                codes = [
                    [int(c) for c in f] for f in json.loads(path.read_text())["codes"]
                ]
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}: not a voice codes file ({e!r})") from e
            if not codes:
                raise ValueError(f"{path}: no voice frames")
            check_voice(codes)
    if args.text_prompt is not None:
        wrap_system(args.text_prompt)
    return Prompt(args.voice_prompt, path, codes, args.text_prompt)


def encode_voice_wav(path: pathlib.Path, mimi: Any, device: str) -> list[list[int]]:
    """A voice .wav -> its Mimi codes, one row of CODEBOOKS ints a frame. Read at
    24 kHz (resampled if needed), first channel, peak-normalised to VOICE_PEAK of full
    scale (PersonaPlex normalises to -24 LUFS; there is no pyloudnorm here), zero-padded
    to whole frames, encoded in one non-streaming pass outside any session.
    PersonaPlex encodes frame by frame while streaming, so edge codes may differ
    slightly; the `.codes.json` path is the one used."""
    import sphn  # type: ignore[import-untyped]

    data, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    pcm = np.asarray(data[0], dtype=np.float32)
    peak = float(np.abs(pcm).max()) if pcm.size else 0.0
    if peak == 0.0:
        raise ValueError(f"{path}: silent voice prompt")
    pcm = pcm * (VOICE_PEAK / peak)
    pcm = np.pad(pcm, (0, -len(pcm) % FRAME_SAMPLES))
    with torch.no_grad():
        t = torch.from_numpy(pcm).view(1, 1, -1).to(device)
        codes = mimi.encode(t)  # [1, K, T]
    return [[int(c) for c in row] for row in codes[0, :CODEBOOKS].t().tolist()]


def complete_prompt(prompt: Prompt, spm: Any, mimi: Any, device: str) -> Prompt:
    """Once the model is loaded: the wav's codes, the text's tokens, the frames."""
    if prompt.codes is None and prompt.path is not None:
        prompt.codes = encode_voice_wav(prompt.path, mimi, device)
    prompt.tokens = (
        [int(t) for t in spm.EncodeAsIds(wrap_system(prompt.text))]
        if prompt.text is not None
        else []
    )
    prompt.frames = system_prompt_frames(prompt.codes or [], prompt.tokens)
    return prompt


def resolve_prompt(
    args: argparse.Namespace, spm: Any, mimi: Any = None, device: str = "cuda"
) -> Prompt | None:
    """`check_prompt` then `complete_prompt`, for callers past the load."""
    prompt = check_prompt(args)
    return None if prompt is None else complete_prompt(prompt, spm, mimi, device)


def wait_for_end(ids: Sequence[int]) -> int | None:
    """The index of the first END (</s>) in `ids`, or None."""
    for i, t in enumerate(ids):
        if t == END:
            return i
    return None


def context_steps(base: dict[str, Any], prefix: int) -> list[StepInput]:
    """The teacher-forced steps of the first `prefix` frames of an undelayed tape row.
    Inside the prefix every text and own entry is forced (an int)."""
    if prefix < 1:
        raise ValueError(f"prefix {prefix} < 1")
    steps = [step_input(base, prefix, s) for s in range(prefix)]
    for s, x in enumerate(steps):
        if x.text is None or any(c is None for c in x.own):
            raise ValueError(f"step {s} of the prefix {prefix} is not fully forced")
    return steps


def opening_prefix(rec: dict[str, Any]) -> int:
    """The frames of a stem's opening: the tape up to the start of the pupil's second
    turn (`spans[pupil_turns[1]]`)."""
    turns = rec["pupil_turns"]
    if len(turns) < 2:
        raise ValueError(f"{rec['id']}: fewer than two pupil turns")
    prefix = int(rec["base"]["spans"][int(turns[1])]["start"])
    if prefix < 1:
        raise ValueError(f"{rec['id']}: opening prefix {prefix} < 1")
    return prefix


def pick_opening(
    records: Sequence[dict[str, Any]], stem: str | None
) -> dict[str, Any] | None:
    """The record `--opening` names: None for "none", the first for None."""
    if stem == "none":
        return None
    if not records:
        raise ValueError("no holdout records; --opening none serves without an opening")
    if stem is None:
        return records[0]
    for rec in records:
        if rec["id"] == stem:
            return rec
    raise ValueError(f"--opening {stem!r}: no such holdout stem")


def holdout_path(render: str) -> pathlib.Path:
    return pathlib.Path.cwd() / "runs" / render / "render" / "holdout.jsonl"


def read_holdout(render: str) -> list[dict[str, Any]]:
    path = holdout_path(render)
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def resolve_opening(args: argparse.Namespace) -> tuple[str, list[StepInput]] | None:
    """`--opening` as (stem, its forced steps), built once; None for `none`."""
    if args.opening == "none":
        return None
    path = holdout_path(args.render)
    try:
        records = read_holdout(args.render)
    except FileNotFoundError:
        records = []
    if not records:
        raise ValueError(
            f"{path}: holdout missing or empty; --opening none serves without it"
        )
    rec = pick_opening(records, args.opening)
    if rec is None:
        return None
    return str(rec["id"]), context_steps(rec["base"], opening_prefix(rec))


def _col(codes: Sequence[int], device: str) -> torch.Tensor:
    return torch.tensor(list(codes), dtype=torch.long, device=device).view(1, -1, 1)


def pcm_to_tensor(pcm: bytes, device: str) -> torch.Tensor:
    """int16 LE bytes of one frame -> [1, 1, 1920] float32 in [-1, 1] on `device`."""
    if len(pcm) != FRAME_BYTES:
        raise ValueError(f"{len(pcm)} bytes is not one frame of {FRAME_BYTES}")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / FULL_SCALE
    return torch.from_numpy(samples).clamp(-1.0, 1.0).view(1, 1, -1).to(device)


def tensor_to_pcm(pcm: torch.Tensor) -> bytes:
    """[1, 1, N] float -> int16 LE bytes."""
    out = (pcm[0, 0].float().clamp(-1.0, 1.0) * FULL_SCALE).round().to(torch.int16)
    return out.cpu().numpy().astype("<i2").tobytes()


def wav_of(frames: Sequence[bytes]) -> Any:
    """Frames of int16 LE bytes -> one float32 mono array (sphn.write_wav's input)."""
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32) / FULL_SCALE


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    """The base with the run's epoch adapters (unless --bare), dep_q 8, eval; and
    Mimi. Exactly the side-by-side job's loading (jobs/sidebyside.py:419-434)."""
    from dual.model import (
        adapter_setup,
        attach_lora,
        describe,
        load_base,
        load_blob_adapters,
        read_blob,
    )

    t0 = time.monotonic()
    base = load_base(args.base, "cuda")
    print(f"loaded {args.base} in {time.monotonic() - t0:.0f}s:", describe(base))
    if not args.bare:
        from training.moshi.trunk_lora import unmoved, wrappers

        ckpt = (
            pathlib.Path.cwd()
            / "runs"
            / args.run
            / "train"
            / f"model-epoch-{args.epoch:02d}.pt"
        )
        blob = read_blob(ckpt)
        attach_lora(base, *adapter_setup(blob["meta"]))
        load_blob_adapters(base, blob)
        # moshi's KV cache takes the first in_proj's dtype: fp32 adapters would make
        # it fp32. autocast ran the adapters' matmuls in bf16 in train and measure.
        lora = [p for n, p in base.lm.named_parameters() if ".lora_" in n]
        for p in lora:
            p.data = p.data.to(torch.bfloat16)
        if len(wrappers(base.lm)) != 128 or unmoved(base.lm):
            raise RuntimeError(f"{ckpt}: adapters missing or zero after loading")
        norm = float(sum(float(p.float().norm()) ** 2 for p in lora) ** 0.5)
        print(f"adapters: {ckpt} norm {norm:.3f} run {blob.get('run', {})}")
    base.lm.dep_q = DEP_Q  # this job only, as the probe and side-by-side
    base.lm.eval()
    mimi = base.info.get_mimi(device="cuda")
    return base, mimi


class Engine:
    """One streaming model with Mimi in and out. One thread drives it at a time; the
    thread that calls open() must be the one that steps (torch.no_grad is per thread).
    """

    def __init__(
        self, base: Any, mimi: Any, device: str, temp: float, temp_text: float
    ) -> None:
        from moshi.models.lm import LMGen

        self.base, self.mimi, self.device = base, mimi, device
        self.forced: int | None = None
        self.last_text: int = PAD
        self.lm_gen = LMGen(
            base.lm,
            use_sampling=temp > 0 or temp_text > 0,
            temp=temp,
            temp_text=temp_text,
            on_text_hook=self._hook,
        )
        self.quiet = _col(MIMI_SILENCE, device)
        self._stack: contextlib.ExitStack | None = None
        self.counters = Counters()

    def _hook(self, text_token: torch.Tensor) -> None:
        # [B] view of the sampled token, before LMGen writes it to its cache.
        if self.forced is not None:
            text_token.copy_(torch.full_like(text_token, self.forced))
        self.last_text = int(text_token[0].item())

    @property
    def is_open(self) -> bool:
        return self._stack is not None

    def open(self) -> None:
        """A fresh session: moshi's and Mimi's streaming states, no gradients."""
        if self._stack is not None:
            raise RuntimeError("a session is already open")
        stack = contextlib.ExitStack()
        stack.enter_context(torch.no_grad())
        stack.enter_context(self.mimi.streaming(1))
        stack.enter_context(self.lm_gen.streaming(1))
        self._stack = stack
        self.counters = Counters()
        self.last_text = PAD

    def close(self) -> None:
        stack, self._stack = self._stack, None  # cleared even if close() raises
        if stack is not None:
            stack.close()

    def step_codes(
        self,
        pupil: torch.Tensor,
        forced: int | None = None,
        own: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        """One frame: the pupil's 8 codes [1, 8, 1] in; moshi's text id out and, once
        the delays are filled, its 8 own codes [1, 8, 1] (None before that)."""
        if self._stack is None:
            raise RuntimeError("open() first")
        self.forced = forced
        out = self.lm_gen.step(pupil, depformer_replace_tokens=own)
        self.forced = None
        self.counters.frames += 1
        if out is None:
            return self.last_text, None
        codes = out[:, 1:, :]
        if not bool((codes >= 0).all()):
            return self.last_text, None
        return self.last_text, codes

    def decode(self, own: torch.Tensor) -> bytes:
        return tensor_to_pcm(self.mimi.decode(own))

    def hear(self, pcm: bytes) -> tuple[int, bytes | None]:
        """A mic frame in; moshi's text id and its voice frame (or None) out."""
        t0 = time.monotonic()
        codes = self.mimi.encode(pcm_to_tensor(pcm, self.device))
        text, own = self.step_codes(codes)
        voice = None if own is None else self.decode(own)
        self.counters.hear.add(time.monotonic() - t0)
        self.counters.heard += 1
        return text, voice

    def type(self, tokens: Sequence[int]) -> list[int]:
        """The typed burst, as fast as the GPU steps: one token a frame on SINE with
        moshi's own audio forced to SILENCE, then GAP_FRAMES of PAD. Returns the text
        ids as forced. The forced frames are decoded and dropped so Mimi's decoder
        stays in step; the page hears nothing until the first free frame."""
        t0 = time.monotonic()
        ids: list[int] = []
        for text, own, pupil in burst(tokens):
            got, codes = self.step_codes(
                _col(pupil, self.device), forced=int(text), own=_col(own, self.device)
            )
            if codes is not None:
                self.mimi.decode(codes)
            ids.append(got)
        self.counters.burst.add(time.monotonic() - t0)
        self.counters.typed += 1
        return ids

    def system_prompt(self, frames: Sequence[Frame]) -> list[int]:
        """PersonaPlex's session prompt: each frame's text and own codes forced, the
        pupil stream as given; nothing decoded or played. The frames are undelayed;
        `prompt_steps` lays them out as moshi 0.2.13 expects. Mimi's streaming state is
        reset after, as PersonaPlex's server does. Returns the text ids as forced."""
        ids: list[int] = []
        for text, own, pupil in prompt_steps(frames):
            got, _ = self.step_codes(
                _col(pupil, self.device), forced=int(text), own=_col(own, self.device)
            )
            ids.append(got)
        self.mimi.reset_streaming()
        return ids

    def force_prefix(
        self,
        steps: Sequence[StepInput],
        emit: Callable[[int, bytes | None], None] | None = None,
    ) -> list[int]:
        """Teacher-forces tape frames: each step's pupil codes in, its text and all own
        codebooks forced. A frame's own codes, once out, are decoded and dropped so
        Mimi's decoder stays in step; with `emit`, each step calls
        `emit(text_id, pcm)`, pcm None before the delays are filled. Returns the text
        ids as forced."""
        ids: list[int] = []
        for x in steps:
            if x.text is None or any(c is None for c in x.own):
                raise ValueError("force_prefix needs every text and own entry forced")
            got, codes = self.step_codes(
                _col(x.pupil, self.device),
                forced=int(x.text),
                own=_col([int(c) for c in x.own if c is not None], self.device),
            )
            if emit is None:
                if codes is not None:
                    self.mimi.decode(codes)
            else:
                emit(got, None if codes is None else self.decode(codes))
            ids.append(got)
        return ids

    def warmup(self) -> None:
        """A throwaway session that pays the one-time compile and allocation costs
        (moshi, Mimi) and checks the path runs."""
        self.open()
        try:
            for _ in range(WARMUP_FRAMES):
                _, own = self.step_codes(self.quiet)
                if own is not None:
                    self.decode(own)
            self.hear(ZERO_FRAME)
        finally:
            self.close()


def pcm_frames_of(
    mimi: Any, codes: Sequence[Sequence[int]], device: str
) -> list[bytes]:
    """A voiced pupil turn's Mimi codes -> its PCM, one frame of int16 bytes per code
    frame, decoded in one non-streaming pass (outside any session)."""
    with torch.no_grad():
        t = torch.tensor(codes, dtype=torch.long, device=device).t()[None]  # [1, 8, T]
        pcm = mimi.decode(t)  # [1, 1, T * 1920]
    total = pcm.shape[-1] // FRAME_SAMPLES
    return [
        tensor_to_pcm(pcm[:, :, i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
        for i in range(total)
    ]


def replay(
    args: argparse.Namespace,
    engine: Engine,
    spm: Any,
    out: pathlib.Path,
    opening: tuple[str, list[StepInput]] | None = None,
    prompt: Prompt | None = None,
) -> None:
    """The B1.5 probe's scenario on this checkpoint through the live path (D6).
    `opening` is `resolve_opening(args)`, given with `--context opening`; `prompt`
    is stepped right after every session opens."""
    path = holdout_path(args.render)
    records = read_holdout(args.render)
    chosen = records[: args.replay]
    print(f"replay: {len(chosen)} of {len(records)} holdout dialogues from {path}")

    def decode(ids: list[int]) -> str:
        return str(spm.DecodeIds([int(i) for i in ids]))

    rows: list[dict[str, Any]] = []
    hear_all, burst_all = Timing(), Timing()
    for rec in chosen:
        turns = [int(k) for k in rec["pupil_turns"]]
        if len(turns) < 2:
            print(f"{rec['id']}: fewer than two pupil turns, skipped")
            continue
        first = pcm_frames_of(
            engine.mimi, rec["renderings"][str(turns[0])]["A"]["pupil"], engine.device
        )
        second = rec["renderings"][str(turns[1])]
        tokens = [int(t) for t in second["typed"]["text"][:-GAP_FRAMES]]
        words = words_of(decode(content_ids(tokens)))
        words_1: list[str] | None = None
        if args.context == "none":
            tokens_1 = [
                int(t)
                for t in rec["renderings"][str(turns[0])]["typed"]["text"][:-GAP_FRAMES]
            ]
            words_1 = words_of(decode(content_ids(tokens_1)))
        second_pcm = pcm_frames_of(engine.mimi, second["A"]["pupil"], engine.device)
        for variant in ("typed", "spoken"):
            torch.manual_seed(args.seed)
            engine.open()
            try:
                if prompt is not None:
                    engine.system_prompt(prompt.frames)
                t0 = time.monotonic()
                answer: list[int] = []
                ended: bool | None = None
                end_at: int | None = None
                if args.context != "none":
                    if opening is not None:
                        forced_steps = opening[1]
                    else:
                        base = rec["base"]
                        prefix = int(base["spans"][turns[1]]["start"])
                        forced_steps = context_steps(base, prefix)
                    answer = engine.force_prefix(forced_steps)
                    spoken_1 = len(forced_steps)
                else:
                    spoken_1 = len(first)
                    for pcm in first:
                        engine.hear(pcm)
                    limit = AFTER_END_LIMIT if args.after_end else FREE_FRAMES
                    for _ in range(limit):
                        text, _ = engine.hear(ZERO_FRAME)
                        answer.append(text)
                        if end_at is None:
                            end_at = wait_for_end(answer)
                        if end_at is not None and len(answer) > end_at + AFTER_END:
                            break
                    ended = end_at is not None
                typed_at = engine.counters.frames
                if variant == "typed":
                    engine.type(tokens)
                else:
                    for pcm in second_pcm:
                        engine.hear(pcm)
                free_at = engine.counters.frames
                free: list[int] = []
                for _ in range(FREE_FRAMES):
                    text, _ = engine.hear(ZERO_FRAME)
                    free.append(text)
                verdict = classify(words, free, decode)
                first_class = (
                    None if words_1 is None else classify(words_1, answer, decode)
                )
                c = engine.counters
                hear_all.extend(c.hear)
                burst_all.extend(c.burst)
                row = {
                    "stem": rec["id"],
                    "variant": variant,
                    "class": verdict,
                    "first_class": first_class,
                    "first_words": words_1,
                    "typed_words": words,
                    "first_answer": decode(content_ids(answer)),
                    "forced": args.context != "none",
                    "ended": ended,
                    "end_frame": end_at,
                    "emitted": decode(content_ids(free)),
                    "emitted_ids": free,
                    "frames": {
                        "prompt": 0 if prompt is None else len(prompt.frames),
                        "spoken_1": spoken_1,
                        "answer_1": len(answer),
                        "turn_2": free_at - typed_at,
                        "free": len(free),
                        "free_pieces": len(content_ids(free)),
                    },
                    "hear_ms": c.hear.summary(),
                    "burst_ms": c.burst.summary(),
                    "seconds": time.monotonic() - t0,
                }
                if args.context != "none":
                    row["frames"]["prefix"] = spoken_1
                if opening is not None:
                    row["opening"] = opening[0]
                rows.append(row)
                print(
                    f"{rec['id']} {variant}: {verdict} (first {first_class}) in "
                    f"{row['seconds']:.0f}s | "
                    f"hear {c.hear.summary().get('mean', 0):.0f} ms | typed: "
                    f"{' '.join(words)!r} | emitted: {row['emitted']!r}"
                )
            finally:
                engine.close()
    with (out / "replay.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        **replay_summary(rows),
        "first_classes": first_class_counts(rows),
        "hear_ms": hear_all.summary(),
        "burst_ms": burst_all.summary(),
        "compile": not args.eager,
        "bare": args.bare,
        "run": None if args.bare else args.run,
        "epoch": None if args.bare else args.epoch,
        "temp": [args.temp, args.temp_text],
        "variant_flags": {
            "seed": args.seed,
            "context": args.context,
            "after_end": args.after_end,
            "opening": None if opening is None else opening[0],
            "temps": [args.temp, args.temp_text],
            "voice_prompt": args.voice_prompt,
            "text_prompt": args.text_prompt,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", json.dumps(summary))


class _OpeningStopped(Exception):
    """Raised from the opening's emit when stop() came in: ends the forcing early."""


class Session:
    """One websocket's model session. The worker thread steps the model; the asyncio
    side feeds it mic frames and typed turns and sends what it posts to `outbox`."""

    def __init__(
        self,
        engine: Any,
        spm: Any,
        out: pathlib.Path,
        loop: asyncio.AbstractEventLoop,
        opening: tuple[str, list[StepInput]] | None = None,
        prompt: Prompt | None = None,
    ) -> None:
        self.engine, self.spm, self.loop = engine, spm, loop
        self.opening = opening
        self.prompt = prompt
        self.prompt_meta: dict[str, Any] | None = None
        self.opening_meta: dict[str, Any] | None = None
        self.stopping = threading.Event()  # set by stop(); the opening polls it
        self.name = session_name()
        self.dir = out / self.name
        self.inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.outbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.buffer = PcmBuffer()
        self.steps: list[dict[str, Any]] = []
        self.heard: list[bytes] = []
        self.voiced: list[bytes] = []
        self.typed_log: list[dict[str, Any]] = []
        self.error: str | None = None

    # the asyncio side
    def feed(self, data: bytes) -> None:
        kind, payload = unpack(data)
        if kind == AUDIO:
            for frame in self.buffer.push(payload):
                self.inbox.put(("hear", frame))
        elif kind == TYPED:
            text = payload.decode("utf-8")
            n = len(text.split())
            if n == 0 or n > MAX_TYPED_WORDS:
                why = "blank" if n == 0 else f"{n} words, over {MAX_TYPED_WORDS}"
                print(f"typed text refused ({why})", flush=True)
                return
            self.inbox.put(("type", text))
        else:
            raise ValueError(f"the page sent a server-only kind {kind}")

    def stop(self) -> None:
        self.stopping.set()
        self.inbox.put(("stop", None))

    # the worker side
    def post(self, message: bytes | None) -> None:
        self.loop.call_soon_threadsafe(self.outbox.put_nowait, message)

    def run(self) -> None:
        try:
            self.engine.open()
            if self.prompt is not None:
                self.play_prompt(self.prompt)
                if self.stopping.is_set():  # the prompt runs to its end, then the stop
                    return
            self.post(pack(READY, json.dumps({"session": self.name}).encode()))
            if self.opening is not None:
                self.play_opening(*self.opening)
                if self.stopping.is_set():  # a stop mid-opening beats queued turns
                    return
            while True:
                cmd, arg = self.inbox.get()
                if cmd == "stop":
                    break
                if cmd == "type":
                    self.handle_typed(str(arg))
                else:
                    self.handle_hear(bytes(arg))
        except Exception as e:  # broad on purpose: the record must say why it died
            self.error = repr(e)
            print("session error:", self.error, flush=True)
        finally:
            try:
                self.engine.close()
            except Exception as e:  # the outbox must still end, or the socket hangs
                self.error = self.error or repr(e)
                print("session close error:", repr(e), flush=True)
            finally:
                self.post(None)

    def play_prompt(self, prompt: Prompt) -> None:
        """Steps PersonaPlex's session prompt before READY. Nothing is played or heard;
        the mic frames queued meanwhile are dropped, not counted as lag."""
        at = self.engine.counters.frames
        t0 = time.monotonic()
        ids = self.engine.system_prompt(prompt.frames)
        secs = time.monotonic() - t0
        each = 1000 * secs / len(ids)
        for i, t in enumerate(ids):
            self.steps.append(step_row(at + 1 + i, "prompt", t, each))
        self.prompt_meta = {
            "voice": prompt.voice,
            "voice_frames": len(prompt.codes or []),
            "text": prompt.text,
            "tokens": len(prompt.tokens),
            "frames": len(ids),
            "seconds": secs,
            "dropped": self.drop_hears(self.inbox.qsize(), live=False),
            "wav_lead": "the prompt is not in either wav",
        }
        print(
            f"prompt: {len(ids)} frames in {secs:.1f}s, dropped "
            f"{self.prompt_meta['dropped']} mic frames",
            flush=True,
        )

    def play_opening(self, stem: str, steps: list[StepInput]) -> None:
        """Forces the opening's frames, posting its voice and words as they come. A stop
        ends the forcing at the next frame. The mic frames queued meanwhile are
        dropped: the pupil was listening. A typed turn queued meanwhile is kept: the
        hears ahead of it go here, it runs in the inbox loop, and handle_typed drops
        the hears behind it."""
        t0 = time.monotonic()
        last = t0
        played: list[int] = []
        stopped = False

        def emit(text: int, voice: bytes | None) -> None:
            nonlocal last
            now = time.monotonic()
            c = self.engine.counters
            self.steps.append(step_row(c.frames, "opening", text, 1000 * (now - last)))
            played.append(text)
            last = now
            if voice is not None:
                self.voiced.append(voice)
                self.post(pack(AUDIO, voice))
            if is_word(text):
                self.post(pack(TEXT, piece_text(self.spm.IdToPiece(text)).encode()))
            if c.frames % STATUS_EVERY == 0:
                self.post(c.status())
            if self.stopping.is_set():
                raise _OpeningStopped

        try:
            self.engine.force_prefix(steps, emit=emit)
        except _OpeningStopped:
            stopped = True
        finally:  # an error midway still leaves what was played in the record
            self.opening_meta = {
                "stem": stem,
                "frames": len(played),
                "stopped": stopped,
                "text": str(self.spm.DecodeIds(content_ids(played))),
                "seconds": time.monotonic() - t0,
                # not counters.dropped: that is the live lag, the opening is no lag
                "dropped": self.drop_hears(self.inbox.qsize(), live=False),
                "pupil_wav_lead": "zeros; the model heard the tape pupil's opener",
            }
            print(
                f"opening {stem}: {len(played)} frames in "
                f"{self.opening_meta['seconds']:.1f}s"
                f"{' (stopped)' if stopped else ''}, dropped "
                f"{self.opening_meta['dropped']} mic frames",
                flush=True,
            )

    def queued_hears(self) -> int:
        """Mic frames waiting, not counting a typed turn or the stop."""
        with self.inbox.mutex:  # the asyncio side appends under the same lock
            return sum(1 for cmd, _ in self.inbox.queue if cmd == "hear")

    def drop_hears(self, n: int, live: bool = True) -> int:
        """Drops up to `n` queued mic frames from the front, never a typed turn. They
        count in `counters.dropped` (the live lag) unless `live` is False."""
        dropped = 0
        q = self.inbox.queue  # the worker is the only consumer; a peek is atomic
        while dropped < n and q and q[0][0] == "hear":
            self.inbox.get_nowait()
            dropped += 1
        if live:
            self.engine.counters.dropped += dropped
        return dropped

    def handle_hear(self, frame: bytes) -> None:
        self.drop_hears(trim_backlog(self.queued_hears()))
        t0 = time.monotonic()
        text, voice = self.engine.hear(frame)
        ms = 1000 * (time.monotonic() - t0)
        self.heard.append(frame)
        c = self.engine.counters
        self.steps.append(step_row(c.frames, "hear", text, ms))
        if voice is not None:
            self.voiced.append(voice)
            self.post(pack(AUDIO, voice))
        if is_word(text):
            self.post(pack(TEXT, piece_text(self.spm.IdToPiece(text)).encode()))
        if c.frames % STATUS_EVERY == 0:
            self.post(c.status())

    def handle_typed(self, text: str) -> None:
        words = typed_words(text)
        tokens = typed_tokens(words, self.spm.EncodeAsIds)
        at = self.engine.counters.frames
        t0 = time.monotonic()
        ids = self.engine.type(tokens)
        secs = time.monotonic() - t0
        dropped = self.drop_hears(self.inbox.qsize())  # the mic while the pupil typed
        each = 1000 * secs / len(ids)
        for i, t in enumerate(ids):
            kind = "typed" if i < len(tokens) else "gap"
            self.steps.append(step_row(at + 1 + i, kind, t, each))
        self.typed_log.append(
            {
                "at": at,
                "words": words,
                "tokens": tokens,
                "seconds": secs,
                "dropped": dropped,
            }
        )
        self.post(self.engine.counters.status())
        print(
            f"typed at {at}: {' '.join(words)!r}, {len(tokens)} tokens in {secs:.2f}s, "
            f"dropped {dropped} mic frames",
            flush=True,
        )

    def save(self) -> None:
        import sphn

        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "steps.jsonl").open("w") as f:
            for row in self.steps:
                f.write(json.dumps(row) + "\n")
        # the opening played frames with no mic: pupil.wav leads with as many zero
        # frames, so both wavs share frame 0
        lead = 0 if self.opening_meta is None else int(self.opening_meta["frames"])
        if self.opening_meta is not None:
            self.opening_meta["pupil_wav_lead_frames"] = lead
        pupil = [ZERO_FRAME] * lead + self.heard
        sphn.write_wav(str(self.dir / "pupil.wav"), wav_of(pupil), SAMPLE_RATE)
        sphn.write_wav(str(self.dir / "moshi.wav"), wav_of(self.voiced), SAMPLE_RATE)
        c = self.engine.counters
        said = [int(r["text"]) for r in self.steps if r["kind"] == "hear"]
        meta = {
            "session": self.name,
            "frames": c.frames,
            "heard": c.heard,
            "dropped": c.dropped,  # live lag; the opening's is in opening
            "typed": self.typed_log,
            "opening": self.opening_meta,
            "prompt": self.prompt_meta,
            "hear_ms": c.hear.summary(),
            "burst_ms": c.burst.summary(),
            "transcript": str(self.spm.DecodeIds(content_ids(said))),
            "error": self.error,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print("saved", self.dir, flush=True)


async def pump(session: Session, ws: web.WebSocketResponse) -> None:
    """Sends what the worker posts; a None closes the socket (the session ended). A
    client that dropped mid-send stops the sending, not the draining."""
    dropped = False
    while True:
        message = await session.outbox.get()
        if message is None:
            if not ws.closed:
                with contextlib.suppress(ConnectionError):
                    await ws.close()
            return
        if not ws.closed and not dropped:
            try:
                await ws.send_bytes(message)
            except ConnectionError:
                dropped = True  # keep draining to the sentinel, send no more


async def page(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(CLIENT)


async def health(request: web.Request) -> web.Response:
    s = request.app[STATE]
    return web.json_response(
        {
            "ok": True,
            "name": s["name"],
            "ready": s["engine"] is not None,
            "busy": s["lock"].locked(),
            "sessions": s["sessions"],
        }
    )


async def finish(
    s: dict[str, Any],
    session: Session,
    worker: asyncio.Future[None],
    sender: asyncio.Task[None],
) -> None:
    """Ends a session: stops the worker, saves the record, frees the engine. A task of
    its own because aiohttp cancels the handler when the page goes away, and the record
    must still be written then. Releases the session lock last."""
    try:
        session.stop()
        # No timeout: the worker always reaches the stop command, and the engine is
        # not ours to release while it may still hold it open.
        await asyncio.gather(worker, return_exceptions=True)
        await asyncio.gather(sender, return_exceptions=True)
        try:
            await asyncio.to_thread(session.save)
        except Exception as e:
            print("session save failed:", repr(e), flush=True)
        s["sessions"] += 1
    finally:
        s["open"].discard(session)
        s["lock"].release()


def same_origin(request: web.Request) -> bool:
    """No Origin (not a browser) passes; else its host must be localhost, 127.0.0.1 or
    the host the request was addressed to."""
    origin = request.headers.get("Origin")
    if origin is None:
        return True
    host = (urlsplit(origin).hostname or "").lower()
    own = (urlsplit("//" + request.host).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", own) and host != ""


async def websocket(request: web.Request) -> web.StreamResponse:
    s = request.app[STATE]
    if not same_origin(request):
        raise web.HTTPForbidden(text="cross-origin websocket refused")
    if s["engine"] is None:
        raise web.HTTPServiceUnavailable(text="the model is still loading")
    if s["closing"]:
        raise web.HTTPServiceUnavailable(text="the engine is shutting down")
    if s["lock"].locked():
        raise web.HTTPConflict(text="a session is open; one at a time")
    await s["lock"].acquire()  # no await since the check: nobody else got in
    done: asyncio.Future[None] | None = None
    try:
        ws = web.WebSocketResponse(max_msg_size=4 * 2**20)
        await ws.prepare(request)
        session = Session(
            s["engine"],
            s["spm"],
            s["out"],
            asyncio.get_running_loop(),
            opening=s["opening"],
            prompt=s["prompt"],
        )
        s["open"].add(session)
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(s["executor"], session.run)
        sender = asyncio.create_task(pump(session, ws))
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    session.feed(msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    break
        finally:
            done = asyncio.ensure_future(finish(s, session, worker, sender))
            s["finishing"].add(done)
            done.add_done_callback(s["finishing"].discard)
            await asyncio.shield(done)
        return ws
    except BaseException:
        if done is None:
            s["lock"].release()  # failed before the worker started
        raise


def make_app(state: dict[str, Any]) -> web.Application:
    app = web.Application()
    app[STATE] = state
    app.router.add_get("/", page)
    app.router.add_get("/health", health)
    app.router.add_get("/ws", websocket)
    return app


def new_state(name: str, out: pathlib.Path) -> dict[str, Any]:
    return {
        "name": name,
        "out": out,
        "engine": None,
        "spm": None,
        "lock": asyncio.Lock(),
        "sessions": 0,
        "finishing": set(),
        "opening": None,  # (stem, steps) every session starts with, or None
        "prompt": None,  # the Prompt every session starts with, or None
        "closing": False,  # set by shutdown(): /ws refuses new sessions
        "open": set(),  # sessions whose worker has started and not yet finished
        # ONE long-lived thread: moshi captures CUDA graphs per process on first use,
        # and capture and replay must share a thread.
        "executor": concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="engine"
        ),
    }


SHUTDOWN_BUDGET_S = 90.0


async def shutdown(
    state: dict[str, Any], runner: Any, budget: float = SHUTDOWN_BUDGET_S
) -> None:
    """Stops and saves every open session, then tears the server down. Order matters:
    runner.cleanup() does not close websockets, it waits for their handlers, so the
    sessions are stopped first (the worker posts the sentinel, pump closes the socket,
    the handler ends, finish saves) and only then is the runner cleaned up. New
    sessions are refused from the start (503), every open session is stopped again on
    each pass (stop only queues a command), and the wait is bounded by `budget`."""
    state["closing"] = True
    deadline = time.monotonic() + budget
    while state["open"] or state["finishing"]:
        for session in list(state["open"]):
            session.stop()
        left = deadline - time.monotonic()
        if left <= 0:
            print(
                f"shutdown budget spent, {len(state['open'])} session(s) still open",
                flush=True,
            )
            break
        if state["finishing"]:
            await asyncio.wait(list(state["finishing"]), timeout=min(0.05, left))
        else:
            await asyncio.sleep(
                min(0.05, left)
            )  # a handler is winding down to finish()
    await runner.cleanup()


async def serve(args: argparse.Namespace, out: pathlib.Path) -> None:
    opening = resolve_opening(args)  # before the load: an unknown stem fails at once
    state = new_state(args.name, out)
    state["opening"] = opening
    prompt = check_prompt(args)  # before the load: a missing voice file fails at once
    state["prompt"] = prompt
    if prompt is not None:
        print(f"engine: prompt voice {prompt.voice} text {prompt.text!r}", flush=True)
    if opening is not None:
        print(f"engine: opening {opening[0]}, {len(opening[1])} frames", flush=True)
    runner = web.AppRunner(
        make_app(state), shutdown_timeout=5.0
    )  # sessions are finished before cleanup
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "unknown"
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):  # before the load: an abort is honoured
        loop.add_signal_handler(sig, stop.set)
    print(
        f"engine: listening on {args.host}:{args.port}; container address {ip}; "
        f"page http://{ip}:{args.port}/ health http://{ip}:{args.port}/health",
        flush=True,
    )
    executor = state["executor"]
    started = time.monotonic()
    base, mimi = await loop.run_in_executor(executor, load_model, args)
    engine = Engine(base, mimi, "cuda", args.temp, args.temp_text)
    if prompt is not None:
        await loop.run_in_executor(
            executor, complete_prompt, prompt, base.spm, mimi, "cuda"
        )
        print(f"engine: prompt {len(prompt.frames)} frames", flush=True)
    await loop.run_in_executor(executor, engine.warmup)
    if not stop.is_set():
        state["engine"], state["spm"] = engine, base.spm
        print(
            f"engine: ready; stops after {args.max_minutes:g} min or SIGTERM",
            flush=True,
        )
        left = max(0.0, 60 * args.max_minutes - (time.monotonic() - started))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=left)
    print(
        "engine: stopping" + (" (signal)" if stop.is_set() else " (max minutes)"),
        flush=True,
    )
    await shutdown(state, runner)
    executor.shutdown(wait=True)
    print(f"engine: done, {state['sessions']} session(s) recorded", flush=True)


def main() -> None:
    args = parse_args()
    from dual.model import cap_gpu_memory, set_compile_mode

    set_compile_mode(args.eager)  # before anything imports moshi
    cap_gpu_memory(args.mem_cap_gib)
    out = pathlib.Path.cwd() / "runs" / args.name / "engine"
    out.mkdir(parents=True, exist_ok=True)
    if args.replay:
        # before the load: an unknown stem fails at once
        opening = resolve_opening(args) if args.context == "opening" else None
        prompt = check_prompt(args)
        base, mimi = load_model(args)
        engine = Engine(base, mimi, "cuda", args.temp, args.temp_text)
        engine.warmup()
        if prompt is not None:
            complete_prompt(prompt, base.spm, mimi, "cuda")
        replay(args, engine, base.spm, out, opening, prompt)
    else:
        asyncio.run(serve(args, out))
    print("engine job ok")


if __name__ == "__main__":
    main()
