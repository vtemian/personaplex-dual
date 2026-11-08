"""Renders every training row and the holdout records once, or (--from) extends an
earlier render with new voices (B15). GPU for the TTS and Mimi."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import types
from collections.abc import Mapping, Sequence
from typing import Any

from dual.consistency import answer_pairs
from dual.frames import to_record
from dual.holdout import split
from dual.render import (
    add_rendering,
    assemble,
    check_same_tape,
    holdout_record,
    holdout_row,
    tape_segments,
    training_rows,
    voiced_rows,
)
from dual.turns import spans, text_row, typed_tokens

VOICES = {"A": "vctk/p225_023.wav", "B": "vctk/p248_023.wav", "C": "vctk/p302_023.wav"}
VOICE_SUFFIX = ".1e68beda@240.safetensors"
SHA_FILE = pathlib.Path("/app/vendor/SHA")  # the chalkboard commit the image vendors


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


LABEL = re.compile(r"^[A-Z][A-Za-z0-9]*$")  # a measure pair key suffix: gap_<label>


def parse_voices(specs: Sequence[str], taken: Mapping[str, str]) -> dict[str, str]:
    """--add-voice LABEL=VOICE words -> {label: voice}. Refuses a label or a voice the
    render already has (A, B, C: the bar and the door are never a teacher), a
    lower-case or reserved label, and a label or voice given twice."""
    out: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--add-voice {spec!r}: want LABEL=VOICE")
        label, voice = spec.split("=", 1)
        if not LABEL.match(label):
            raise ValueError(f"--add-voice {spec!r}: label must match {LABEL.pattern}")
        if label in taken:
            raise ValueError(f"--add-voice {spec!r}: label {label} already exists")
        if voice in taken.values():
            raise ValueError(f"--add-voice {spec!r}: {voice} already voices a label")
        if label in out or voice in out.values():
            raise ValueError(f"--add-voice {spec!r}: label or voice given twice")
        out[label] = voice
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int, default=0, help="dialogues, for a smoke render")
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="an earlier render: copy its rows, stems and holdout; voice only the "
        "added labels (B15)",
    )
    p.add_argument(
        "--add-voice",
        action="append",
        default=[],
        metavar="LABEL=VOICE",
        help="a holdout rendering to add, e.g. D=vctk/p280_023.wav (with --from)",
    )
    p.add_argument(
        "--teacher-voice",
        default=None,
        help="an added label: also write rows-teacher.jsonl in that voice",
    )
    args = p.parse_args()
    if (args.add_voice or args.teacher_voice) and not args.source:
        p.error("--add-voice and --teacher-voice need --from")
    labels = [s.split("=", 1)[0] for s in args.add_voice]
    if args.teacher_voice and args.teacher_voice not in labels:
        p.error(f"--teacher-voice {args.teacher_voice}: not an --add-voice label")
    return args


def load_dialogues(
    data: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """The tape records by id, the dialogue sources by stem, and the stems with both."""
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
    return records, sources, stems


def load_engines(data: pathlib.Path) -> types.SimpleNamespace:
    """The SPM, Mimi and the TTS on cuda; pieces, typed and the cached, seeded voiced
    exactly as render-1 ran them. GPU; imports torch and harness inside."""
    import torch
    from harness.engine.audio import encode_wav, load_tts, voice_many

    from dual.model import load_base

    base = load_base("moshiko")  # the SPM and Mimi; the weights are not used here
    spm = base.spm
    mimi = base.info.get_mimi(device="cuda")
    tts = load_tts("cuda")
    cache: dict[tuple[str, str], list[list[int]]] = {}

    def pieces(word: str) -> Sequence[str]:
        return list(spm.EncodeAsPieces(word))

    def typed(words: Sequence[str]) -> Sequence[int]:
        return typed_tokens(words, spm.EncodeAsIds)

    def voiced(words: Sequence[str], voice: str) -> list[list[int]]:
        key = (" ".join(words), voice)
        if key not in cache:
            seed = stable_seed(words, voice)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            text = " ".join(words)
            if not text.endswith((".", "?", "!")):
                text += "."
            path = str(data / "voices" / f"{voice}{VOICE_SUFFIX}")
            (v,) = voice_many(tts, [text], [path])
            cache[key] = encode_wav(mimi, v.pcm, "cuda")
        return cache[key]

    return types.SimpleNamespace(
        spm=spm, pieces=pieces, typed=typed, voiced=voiced, cache=cache
    )


def main() -> None:
    args = parse_args()
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    out = pathlib.Path.cwd() / "runs" / args.name / "render"
    out.mkdir(parents=True, exist_ok=True)
    if args.source:
        return extend_render(args, data)
    records, sources, stems = load_dialogues(data)
    held, train = split(stems)
    if args.limit:
        held, train = held[: max(1, args.limit // 10)], train[: args.limit]

    eng = load_engines(data)

    (out / "stems.json").write_text(
        json.dumps(
            {
                "held": held,
                "train": train,
                "voices": VOICES,
                "chalkboard_sha": SHA_FILE.read_text().strip(),
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
            text = text_row(src, rec, eng.pieces, eng.spm.PieceToId)
            segs = tape_segments(sp, text, rec)
            if stem in held:
                h = holdout_record(stem, sp, segs, eng.typed, eng.voiced, VOICES)
                hold_f.write(json.dumps(h) + "\n")
                held_records.append(h)
                print("held", stem, "pupil turns", sum(s.party == "pupil" for s in sp))
            else:
                stem_rows = 0
                for row in training_rows(stem, sp, segs, eng.typed):
                    rows_f.write(json.dumps(row) + "\n")
                    n_rows += 1
                    stem_rows += 1
                    train_max = max(train_max, row["frames"])
                print("train", stem, "rows", stem_rows)
    # Determinism: the same words in the same voice encode to the same codes twice.
    words = ("the", "same", "words")
    first = eng.voiced(words, VOICES["A"])
    eng.cache.clear()
    again = eng.voiced(words, VOICES["A"])
    print("tts deterministic:", first == again)
    n_hold, hold_max, min_window = validate_holdout(held_records, tuple(VOICES))
    print(f"holdout rows {n_hold} max frames {hold_max} min answer window {min_window}")
    print("train max row frames:", train_max)
    print(f"rendered train rows {n_rows} holdout {len(held)}")


def extend_render(args: argparse.Namespace, data: pathlib.Path) -> None:
    """--from (B15): copy an earlier render's rows-train.jsonl byte for byte and its
    stems and holdout; voice only the added labels. Every holdout record gains each
    added rendering; with --teacher-voice, rows-teacher.jsonl gets each train stem's
    two voiced rows (mask, complement). The tape is rebuilt from the data and must equal
    the stored rows, so the voiced rows sit on exactly the rows the train job reads."""
    runs = pathlib.Path.cwd() / "runs"
    src = runs / args.source / "render"
    out = runs / args.name / "render"
    if out.resolve() == src.resolve():
        raise SystemExit("--from names this render itself")
    stems = json.loads((src / "stems.json").read_text())
    sha = SHA_FILE.read_text().strip()
    if stems["chalkboard_sha"] != sha:
        raise SystemExit(
            f"{args.source} was rendered at chalkboard {stems['chalkboard_sha']}, "
            f"this image has {sha}"
        )
    added = parse_voices(args.add_voice, stems["voices"])
    for voice in added.values():
        if not (data / "voices" / f"{voice}{VOICE_SUFFIX}").is_file():
            raise SystemExit(f"no voice file for {voice} under {data / 'voices'}")
    rows_text = (src / "rows-train.jsonl").read_text()
    by_stem: dict[str, dict[str, dict[str, Any]]] = {}
    for r in map(json.loads, rows_text.splitlines()):
        by_stem.setdefault(r["stem"], {})[r["kind"]] = r
    held_in = [
        json.loads(line) for line in (src / "holdout.jsonl").read_text().splitlines()
    ]
    records, sources, _ = load_dialogues(data)
    train, held = list(stems["train"]), list(stems["held"])
    if args.limit:
        held, train = held[: max(1, args.limit // 10)], train[: args.limit]
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows-train.jsonl").write_text(rows_text)  # the same rows, the same batches
    (out / "stems.json").unlink(missing_ok=True)  # written last, once all checks pass
    eng = load_engines(data)

    def tape_of(stem: str) -> tuple[Any, Any]:
        rec, source = records[stem], sources[stem]
        sp = spans(source, rec)
        text = text_row(source, rec, eng.pieces, eng.spm.PieceToId)
        return sp, tape_segments(sp, text, rec)

    if args.teacher_voice:
        label = args.teacher_voice
        n_rows = pairs_tape = pairs_voiced = pupil_tape = pupil_voiced = 0
        longest = ("", 0)
        with (out / "rows-teacher.jsonl").open("w") as f:
            for stem in train:
                sp, segs = tape_of(stem)
                kinds = by_stem[stem]
                spoken = kinds["spoken"]
                check_same_tape(
                    to_record(assemble(f"{stem}#spoken", sp, segs, {})), spoken
                )
                for row in voiced_rows(stem, sp, segs, eng.voiced, label, added[label]):
                    typed_row = kinds[row["pairs_with"]]
                    if sorted(row["voiced_turns"]) != sorted(typed_row["typed_turns"]):
                        raise SystemExit(f"{row['id']}: voices other turns than typed")
                    pairs_tape += len(answer_pairs(typed_row, spoken))
                    pairs_voiced += len(answer_pairs(typed_row, row))
                    for k in row["voiced_turns"]:
                        v, t = row["spans"][k], spoken["spans"][k]
                        pupil_voiced += v["end"] - v["start"]
                        pupil_tape += t["end"] - t["start"]
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
                    if row["frames"] > longest[1]:
                        longest = (stem, row["frames"])
                print("teacher", stem, "rows 2")
        if pairs_voiced != pairs_tape:
            raise SystemExit(f"answer pairs: voiced {pairs_voiced}, tape {pairs_tape}")
        print(
            f"teacher rows {n_rows} voice {label}={added[label]}; answer pairs "
            f"{pairs_voiced} (tape {pairs_tape}); typed-turn pupil frames voiced "
            f"{pupil_voiced} vs tape {pupil_tape}; longest teacher row "
            f"{longest[1]} frames, stem {longest[0]}"
        )

    held_out: list[dict[str, Any]] = []
    keep = set(held)
    with (out / "holdout.jsonl").open("w") as f:
        for h in held_in:
            if h["id"] not in keep:
                continue
            sp, segs = tape_of(h["id"])
            check_same_tape(to_record(assemble(h["id"], sp, segs, {})), h["base"])
            h2 = add_rendering(h, sp, eng.voiced, added)
            f.write(json.dumps(h2) + "\n")
            held_out.append(h2)
            print("held", h["id"], "added", ",".join(added))
    words = ("the", "same", "words")
    for lab, voice in added.items():
        first = eng.voiced(words, voice)
        eng.cache.clear()
        print(f"tts deterministic {lab}:", first == eng.voiced(words, voice))
    labels = (*stems["voices"], *added)
    n_hold, hold_max, min_window = validate_holdout(held_out, tuple(labels))
    print(f"holdout rows {n_hold} max frames {hold_max} min answer window {min_window}")
    (out / "stems.json").write_text(
        json.dumps(
            {
                **stems,
                "held": held,  # the stems this render covers (--limit truncates them)
                "train": train,
                "voices": {**stems["voices"], **added},
                "from": args.source,
                "teacher_voice": args.teacher_voice,
            },
            indent=1,
        )
    )
    print(
        f"extended {args.source} -> {args.name}: {len(train)} train, {len(held)} held"
    )


if __name__ == "__main__":
    main()
