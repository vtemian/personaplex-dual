"""The live engine's pure parts: the wire, the mic frame buffer, the backlog rule,
the typed burst, the timings and the session record. No torch, no network.

The wire. Every websocket message is binary: one kind byte, then the payload.

    client -> server   AUDIO   int16 little-endian mono PCM at 24 kHz, any length
                       TYPED   the UTF-8 words of a typed pupil turn
    server -> client   READY   JSON once the model streams: {"session": name}
                       AUDIO   one 1920-sample frame of moshi's voice, int16 LE
                       TEXT    a UTF-8 word piece moshi emitted
                       STATUS  JSON counters, see Counters.status

The typed turn is PersonaPlex's prompt mechanism at any frame (the design doc,
section 1): one token a frame on the text stream, SILENCE on moshi's own audio, SINE
on the pupil stream, then GAP_FRAMES of PAD on the same codes. `typed_segment` in
frames.py is that triple; `burst` reads it out frame by frame for the GPU loop.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dual.constants import AUDIO_CARD, CODEBOOKS, FRAME_RATE, PAD, SILENCE, SINE
from dual.frames import typed_segment
from dual.probe_text import CLASSES

SAMPLE_RATE = 24000
FRAME_SAMPLES = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples, 80 ms
FRAME_BYTES = 2 * FRAME_SAMPLES  # int16

READY = 0
AUDIO = 1
TEXT = 2
TYPED = 3
STATUS = 4
KINDS = (READY, AUDIO, TEXT, TYPED, STATUS)

# Mic frames queued for the model. Past MAX_BACKLOG the oldest are dropped down to
# BACKLOG_KEEP: the model is behind real time (a typed burst, a slow step) and the
# pupil would hear every answer later and later. 25 frames is 2 s.
MAX_BACKLOG = 25
BACKLOG_KEEP = 2

SPECIALS = 4  # <unk> <s> </s> <pad>: no words
STATUS_EVERY = 25  # frames between STATUS messages

# PersonaPlex's audio_silence_frame_cnt: int(0.5 * 12.5) frames either side of the
# text prompt.
PROMPT_SILENCE_FRAMES = int(0.5 * FRAME_RATE)

STEP_KINDS = ("hear", "typed", "gap", "opening", "prompt")
REPLAY_VARIANTS = ("typed", "spoken")

Frame = tuple[int, tuple[int, ...], tuple[int, ...]]  # text, own 8, pupil 8


def pack(kind: int, payload: bytes = b"") -> bytes:
    if kind not in KINDS:
        raise ValueError(f"unknown message kind {kind}")
    return bytes([kind]) + payload


def unpack(message: bytes) -> tuple[int, bytes]:
    if not message:
        raise ValueError("empty message")
    kind = message[0]
    if kind not in KINDS:
        raise ValueError(f"unknown message kind {kind}")
    return kind, bytes(message[1:])


class PcmBuffer:
    """Mic bytes in any lengths, whole 1920-sample frames out."""

    def __init__(self) -> None:
        self._held = bytearray()

    @property
    def pending(self) -> int:
        """Bytes held back, short of a frame."""
        return len(self._held)

    def push(self, data: bytes) -> list[bytes]:
        if len(data) % 2:
            raise ValueError(f"{len(data)} bytes is not whole int16 samples")
        self._held.extend(data)
        out: list[bytes] = []
        while len(self._held) >= FRAME_BYTES:
            out.append(bytes(self._held[:FRAME_BYTES]))
            del self._held[:FRAME_BYTES]
        return out


def trim_backlog(
    queued: int, limit: int = MAX_BACKLOG, keep: int = BACKLOG_KEEP
) -> int:
    """How many of `queued` mic frames to drop: none within `limit`, else all but
    the newest `keep`."""
    if keep < 0 or limit < keep:
        raise ValueError(f"bad backlog rule: limit {limit}, keep {keep}")
    if queued <= limit:
        return 0
    return queued - keep


def typed_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        raise ValueError("a typed turn with no words")
    return words


def burst(tokens: Sequence[int]) -> list[Frame]:
    """The typed turn frame by frame: (text, own, pupil) a step, tokens then the gap."""
    seg = typed_segment(tokens)
    return list(zip(seg.text, seg.own, seg.pupil, strict=True))


def wrap_system(text: str) -> str:
    """PersonaPlex's `wrap_with_system_tags`: `<system> text <system>`, unless the
    text already starts and ends with `<system>`."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("a blank text prompt")
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def check_voice(voice: Sequence[Sequence[int]]) -> None:
    """Every voice frame is CODEBOOKS ints in 0..AUDIO_CARD-1, or ValueError."""
    for i, codes in enumerate(voice):
        if len(codes) != CODEBOOKS or not all(0 <= int(c) < AUDIO_CARD for c in codes):
            raise ValueError(
                f"voice frame {i}: need {CODEBOOKS} codes in 0..{AUDIO_CARD - 1}"
            )


def system_prompt_frames(
    voice: Sequence[Sequence[int]],
    text: Sequence[int],
    silence: int = PROMPT_SILENCE_FRAMES,
) -> list[Frame]:
    """PersonaPlex's session prompt (LMGen.step_system_prompts) frame by frame: the
    voice's codes as moshi's own audio on text PAD, `silence` frames, one text token a
    frame, `silence` frames; the pupil stream is SINE throughout."""
    if not voice and not text:
        raise ValueError("an empty prompt: no voice frames and no text tokens")
    check_voice(voice)
    frames: list[Frame] = [(PAD, tuple(int(c) for c in codes), SINE) for codes in voice]
    frames.extend([(PAD, SILENCE, SINE)] * silence)
    frames.extend((int(t), SILENCE, SINE) for t in text)
    frames.extend([(PAD, SILENCE, SINE)] * silence)
    return frames


def is_word(token: int) -> bool:
    return token >= SPECIALS


def piece_text(piece: str) -> str:
    """A SentencePiece piece as the page shows it: the word boundary is a space."""
    return piece.replace("\u2581", " ")


@dataclass
class Timing:
    """Wall-clock milliseconds of one kind of step."""

    ms: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.ms.append(1000.0 * seconds)

    def extend(self, other: Timing) -> None:
        self.ms.extend(other.ms)

    def summary(self) -> dict[str, float | int]:
        if not self.ms:
            return {"n": 0}
        s = sorted(self.ms)
        return {
            "n": len(s),
            "mean": sum(s) / len(s),
            "p50": s[len(s) // 2],
            "p90": s[min(len(s) - 1, (9 * len(s)) // 10)],
            "max": s[-1],
        }


@dataclass
class Counters:
    """What one session has done so far; `status` is the STATUS message of it."""

    frames: int = 0  # model steps
    heard: int = 0  # mic frames stepped
    typed: int = 0  # typed turns
    dropped: int = 0  # mic frames dropped: the backlog rule, or during a burst
    hear: Timing = field(default_factory=Timing)
    burst: Timing = field(default_factory=Timing)

    def status(self) -> bytes:
        body = {
            "frames": self.frames,
            "heard": self.heard,
            "typed": self.typed,
            "dropped": self.dropped,
            "hear_ms": self.hear.summary(),
            "burst_ms": self.burst.summary(),
        }
        return pack(STATUS, json.dumps(body).encode())


def step_row(index: int, kind: str, text: int, ms: float) -> dict[str, Any]:
    """One line of steps.jsonl: what the model said at step `index` and what the
    step cost. `kind`: "hear" (a mic frame), "typed" (a forced token), "gap" (a
    forced PAD), "opening" (a forced frame of the session's opening), "prompt" (a forced
    frame of the voice/text system prompt). The audio of
    both sides is in the session's wavs."""
    if kind not in STEP_KINDS:
        raise ValueError(f"unknown step kind {kind!r}")
    return {"i": index, "kind": kind, "text": int(text), "ms": round(ms, 2)}


def session_name(now: float | None = None) -> str:
    stamp = time.time() if now is None else now
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(stamp))


def replay_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per variant, the probe's classes counted over the replayed dialogues."""
    classes: dict[str, dict[str, int]] = {
        v: dict.fromkeys(CLASSES, 0) for v in REPLAY_VARIANTS
    }
    for r in rows:
        if r["variant"] not in classes:
            raise ValueError(f"unknown replay variant {r['variant']!r}")
        classes[r["variant"]][r["class"]] += 1
    return {"dialogues": len({r["stem"] for r in rows}), "classes": classes}


def first_class_counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per variant, the classes of the rows' `first_class` (turn 1's reply), skipping
    rows where it is None (a forced context)."""
    counts: dict[str, dict[str, int]] = {
        v: dict.fromkeys(CLASSES, 0) for v in REPLAY_VARIANTS
    }
    for r in rows:
        if r["variant"] not in counts:
            raise ValueError(f"unknown replay variant {r['variant']!r}")
        if r.get("first_class") is not None:
            counts[r["variant"]][r["first_class"]] += 1
    return counts
