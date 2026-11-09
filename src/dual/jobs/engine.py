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
to bf16, `lm.dep_q = 8`, greedy by default. No voice prompt, no frame-0 text prompt:
the measured condition of B10-B15.

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
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import torch
from aiohttp import web

from dual.constants import END, GAP_FRAMES, MIMI_SILENCE, PAD
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
    PcmBuffer,
    Timing,
    burst,
    is_word,
    pack,
    piece_text,
    replay_summary,
    session_name,
    step_row,
    trim_backlog,
    typed_words,
    unpack,
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
        choices=("none", "tutor"),
        default="none",
        help="tutor: teacher-force the tape history instead of turn 1 + a free run",
    )
    p.add_argument(
        "--after-end",
        action="store_true",
        help="replay: wait up to 600 frames for </s> after turn 1 (not with tutor)",
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
    if args.context == "tutor" and args.after_end:
        p.error("--after-end is for the empty-session flow, not --context tutor")
    return args


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

    def force_prefix(self, steps: Sequence[StepInput]) -> list[int]:
        """Teacher-forces tape frames: each step's pupil codes in, its text and all own
        codebooks forced. A frame's own codes, once out, are decoded and dropped so
        Mimi's decoder stays in step. Returns the text ids as forced."""
        ids: list[int] = []
        for x in steps:
            if x.text is None or any(c is None for c in x.own):
                raise ValueError("force_prefix needs every text and own entry forced")
            got, codes = self.step_codes(
                _col(x.pupil, self.device),
                forced=int(x.text),
                own=_col([int(c) for c in x.own if c is not None], self.device),
            )
            if codes is not None:
                self.mimi.decode(codes)
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
    args: argparse.Namespace, engine: Engine, spm: Any, out: pathlib.Path
) -> None:
    """The B1.5 probe's scenario on this checkpoint through the live path (D6)."""
    path = pathlib.Path.cwd() / "runs" / args.render / "render" / "holdout.jsonl"
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
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
        second_pcm = pcm_frames_of(engine.mimi, second["A"]["pupil"], engine.device)
        for variant in ("typed", "spoken"):
            torch.manual_seed(args.seed)
            engine.open()
            try:
                t0 = time.monotonic()
                answer: list[int] = []
                ended: bool | None = None
                end_at: int | None = None
                if args.context == "tutor":
                    base = rec["base"]
                    prefix = int(base["spans"][turns[1]]["start"])
                    answer = engine.force_prefix(context_steps(base, prefix))
                    spoken_1 = prefix
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
                c = engine.counters
                hear_all.extend(c.hear)
                burst_all.extend(c.burst)
                row = {
                    "stem": rec["id"],
                    "variant": variant,
                    "class": verdict,
                    "typed_words": words,
                    "first_answer": decode(content_ids(answer)),
                    "forced": args.context == "tutor",
                    "ended": ended,
                    "end_frame": end_at,
                    "emitted": decode(content_ids(free)),
                    "emitted_ids": free,
                    "frames": {
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
                if args.context == "tutor":
                    row["frames"]["prefix"] = spoken_1
                rows.append(row)
                print(
                    f"{rec['id']} {variant}: {verdict} in {row['seconds']:.0f}s | "
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
            "temps": [args.temp, args.temp_text],
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("summary:", json.dumps(summary))


class Session:
    """One websocket's model session. The worker thread steps the model; the asyncio
    side feeds it mic frames and typed turns and sends what it posts to `outbox`."""

    def __init__(
        self, engine: Any, spm: Any, out: pathlib.Path, loop: asyncio.AbstractEventLoop
    ) -> None:
        self.engine, self.spm, self.loop = engine, spm, loop
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
        self.inbox.put(("stop", None))

    # the worker side
    def post(self, message: bytes | None) -> None:
        self.loop.call_soon_threadsafe(self.outbox.put_nowait, message)

    def run(self) -> None:
        try:
            self.engine.open()
            self.post(pack(READY, json.dumps({"session": self.name}).encode()))
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

    def queued_hears(self) -> int:
        """Mic frames waiting, not counting a typed turn or the stop."""
        with self.inbox.mutex:  # the asyncio side appends under the same lock
            return sum(1 for cmd, _ in self.inbox.queue if cmd == "hear")

    def drop_hears(self, n: int) -> int:
        """Drops up to `n` queued mic frames from the front, never a typed turn."""
        dropped = 0
        q = self.inbox.queue  # the worker is the only consumer; a peek is atomic
        while dropped < n and q and q[0][0] == "hear":
            self.inbox.get_nowait()
            dropped += 1
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
        import sphn  # type: ignore[import-untyped]

        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "steps.jsonl").open("w") as f:
            for row in self.steps:
                f.write(json.dumps(row) + "\n")
        sphn.write_wav(str(self.dir / "pupil.wav"), wav_of(self.heard), SAMPLE_RATE)
        sphn.write_wav(str(self.dir / "moshi.wav"), wav_of(self.voiced), SAMPLE_RATE)
        c = self.engine.counters
        said = [int(r["text"]) for r in self.steps if r["kind"] == "hear"]
        meta = {
            "session": self.name,
            "frames": c.frames,
            "heard": c.heard,
            "dropped": c.dropped,
            "typed": self.typed_log,
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
        session = Session(s["engine"], s["spm"], s["out"], asyncio.get_running_loop())
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
    state = new_state(args.name, out)
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
        base, mimi = load_model(args)
        engine = Engine(base, mimi, "cuda", args.temp, args.temp_text)
        engine.warmup()
        replay(args, engine, base.spm, out)
    else:
        asyncio.run(serve(args, out))
    print("engine job ok")


if __name__ == "__main__":
    main()
