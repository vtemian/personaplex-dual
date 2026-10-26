"""Renders every training row and the holdout records once. GPU for the TTS and Mimi."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from collections.abc import Sequence
from typing import Any

from dual.holdout import split
from dual.render import holdout_record, holdout_row, tape_segments, training_rows
from dual.turns import spans, text_row, typed_tokens

VOICES = {"A": "vctk/p225_023.wav", "B": "vctk/p248_023.wav", "C": "vctk/p302_023.wav"}
VOICE_SUFFIX = ".1e68beda@240.safetensors"


def stable_seed(words: Sequence[str], voice: str) -> int:
    """A 32-bit seed that is the same in every process (str hash() is not)."""
    digest = hashlib.sha256(f"{' '.join(words)}|{voice}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def validate_holdout(
    records: Sequence[dict[str, Any]], voices: Sequence[str]
) -> tuple[int, int, int]:
    """Reads every holdout row the way the measure job will (holdout_row raises on a
    row over the context or without a teacher answer). Returns (rows, max frames,
    min answer window)."""
    rows, max_frames, min_window = 0, 0, 10**9
    for record in records:
        for k in record["pupil_turns"]:
            for rendering in ("typed", *voices, "tape"):
                row, (start, end) = holdout_row(record, k, rendering)
                rows += 1
                max_frames = max(max_frames, row["frames"])
                min_window = min(min_window, end - start)
    return rows, max_frames, min_window


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=0, help="dialogues, for a smoke render")
    return p.parse_args()


def main() -> None:
    import torch
    from harness.engine.audio import encode_wav, load_tts, voice_many

    from dual.model import load_base

    args = parse_args()
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    out = pathlib.Path.cwd() / "runs" / args.name / "render"
    out.mkdir(parents=True, exist_ok=True)
    records = {
        r["id"]: r
        for r in map(
            json.loads, (data / "tape" / "speech.jsonl").read_text().splitlines()
        )
    }
    sources = {
        p.stem: p.read_text() for p in sorted((data / "dialogues").glob("*.dsl"))
    }
    stems = sorted(set(records) & set(sources))
    if len(stems) != len(sources):
        raise SystemExit(f"{len(sources) - len(stems)} dialogues have no tape record")
    held, train = split(stems)
    if args.limit:
        held, train = held[: max(1, args.limit // 10)], train[: args.limit]

    base = load_base("moshiko")  # the SPM and Mimi; the weights are not used here
    spm = base.spm
    mimi = base.info.get_mimi(device="cuda")
    tts = load_tts("cuda")
    voice_cache: dict[tuple[str, str], list[list[int]]] = {}

    def pieces(word: str) -> Sequence[str]:
        return list(spm.EncodeAsPieces(word))

    def typed(words: Sequence[str]) -> Sequence[int]:
        return typed_tokens(words, spm.EncodeAsIds)

    def voiced(words: Sequence[str], voice: str) -> list[list[int]]:
        key = (" ".join(words), voice)
        if key not in voice_cache:
            seed = stable_seed(words, voice)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            text = " ".join(words)
            if not text.endswith((".", "?", "!")):
                text += "."
            path = str(data / "voices" / f"{voice}{VOICE_SUFFIX}")
            (v,) = voice_many(tts, [text], [path])
            voice_cache[key] = encode_wav(mimi, v.pcm, "cuda")
        return voice_cache[key]

    (out / "stems.json").write_text(
        json.dumps(
            {
                "held": held,
                "train": train,
                "voices": VOICES,
                "chalkboard_sha": pathlib.Path("/app/vendor/SHA").read_text().strip(),
            },
            indent=1,
        )
    )

    n_rows = 0
    train_max = 0
    held_records: list[dict[str, Any]] = []
    with (
        (out / "rows-train.jsonl").open("w") as rows_f,
        (out / "holdout.jsonl").open("w") as hold_f,
    ):
        for stem in train + held:
            rec, src = records[stem], sources[stem]
            sp = spans(src, rec)
            text = text_row(src, rec, pieces, spm.PieceToId)
            segs = tape_segments(sp, text, rec)
            if stem in held:
                h = holdout_record(stem, sp, segs, typed, voiced, VOICES)
                hold_f.write(json.dumps(h) + "\n")
                held_records.append(h)
                print("held", stem, "pupil turns", sum(s.party == "pupil" for s in sp))
            else:
                stem_rows = 0
                for row in training_rows(stem, sp, segs, typed):
                    rows_f.write(json.dumps(row) + "\n")
                    n_rows += 1
                    stem_rows += 1
                    train_max = max(train_max, row["frames"])
                print("train", stem, "rows", stem_rows)
    # Determinism: the same words in the same voice encode to the same codes twice.
    words = ("the", "same", "words")
    first = voiced(words, VOICES["A"])
    voice_cache.clear()
    again = voiced(words, VOICES["A"])
    print("tts deterministic:", first == again)
    n_hold, hold_max, min_window = validate_holdout(held_records, tuple(VOICES))
    print(f"holdout rows {n_hold} max frames {hold_max} min answer window {min_window}")
    print("train max row frames:", train_max)
    print(f"rendered train rows {n_rows} holdout {len(held)}")


if __name__ == "__main__":
    main()
