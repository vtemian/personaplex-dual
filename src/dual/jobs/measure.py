"""The bar on one base: bare (epoch 0) or every checkpoint of a train run.

Per holdout record and pupil turn k, five rows (typed, A, B, C, tape) through one
teacher-forced forward each; the text log-probs over each row's answer window give
gap = KL(typed || A), floor = KL(A || B), door = KL(C || A),
tape_gap = KL(typed || tape), tape_floor = KL(tape || A).
Writes runs/<name>/measure/<base>[-<run>]/{epoch-NN.jsonl,summary.json}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import statistics
import time
from collections.abc import Sequence
from typing import Any

import torch

from dual.kl import AnswerKL, answer_kl, bar, log_probs

RENDERINGS = ("typed", "A", "B", "C", "tape")
# (key, p, q): KL(p || q)
PAIRS = (
    ("gap", "typed", "A"),
    ("floor", "A", "B"),
    ("door", "C", "A"),
    ("tape_gap", "typed", "tape"),
    ("tape_floor", "tape", "A"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--render", default="render-1")
    p.add_argument("--run", default=None, help="a train run's name; none = bare")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", action="store_true", help="no torch.compile")
    mode.add_argument("--compile", action="store_true", help="torch.compile")
    return p.parse_args()


def _bar_dict(gaps: Sequence[float], floors: Sequence[float]) -> dict[str, Any]:
    b = bar(gaps, floors)
    return {**dataclasses.asdict(b), "passes": b.passes}


def summarise(
    lines: Sequence[dict[str, Any]],
    epoch: int,
    base: str,
    run: str | None,
    slice_: int | None,
) -> dict[str, Any]:
    def col(key: str, field: str) -> list[float]:
        return [float(x[key][field]) for x in lines]

    out: dict[str, Any] = {
        "run": run,
        "base": base,
        "slice": slice_,
        "epoch": epoch,
        "content": _bar_dict(col("gap", "content"), col("floor", "content")),
        "all": _bar_dict(col("gap", "all_frames"), col("floor", "all_frames")),
    }
    for key in ("door", "tape_gap", "tape_floor"):
        out[f"{key}_median"] = statistics.median(col(key, "content"))
        out[f"{key}_all_median"] = statistics.median(col(key, "all_frames"))
    return out


def turn_lines(lm: Any, records: Sequence[dict[str, Any]], epoch: int) -> list[Any]:
    from dual.model import codes_from_row, forward
    from dual.render import holdout_row

    lines: list[dict[str, Any]] = []
    for record in records:
        for k in record["pupil_turns"]:
            lps: dict[str, torch.Tensor] = {}
            targets: list[int] | None = None
            for rendering in RENDERINGS:
                row, (start, end) = holdout_row(record, k, rendering)
                window = list(row["text"][start:end])
                if targets is None:
                    targets = window
                elif window != targets:
                    raise ValueError(f"{row['id']}: answer text differs by rendering")
                out = forward(lm, codes_from_row(row))
                if not bool(out.text_mask[0, start:end].all()):
                    raise ValueError(f"{row['id']}: masked frames in the answer")
                lps[rendering] = log_probs(out.text_logits[0, start:end])
            assert targets is not None
            line: dict[str, Any] = {"stem": record["id"], "turn": k, "epoch": epoch}
            kls: dict[str, AnswerKL] = {}
            for key, p, q in PAIRS:
                kls[key] = answer_kl(lps[p], lps[q], targets)
                line[key] = dataclasses.asdict(kls[key])
            line["answer_frames"] = len(targets)
            line["content_frames"] = kls["gap"].content_frames
            lines.append(line)
    return lines


def main() -> None:
    from dual.model import (
        EAGER_DEFAULT,
        attach_lora,
        load_base,
        load_blob_adapters,
        read_blob,
        set_compile_mode,
    )

    args = parse_args()
    set_compile_mode(args.eager or (EAGER_DEFAULT and not args.compile))
    root = pathlib.Path.cwd() / "runs"
    render = root / args.render / "render"
    records = [
        json.loads(line) for line in (render / "holdout.jsonl").read_text().splitlines()
    ]
    label = args.base + (f"-{args.run}" if args.run else "")
    out = root / args.name / "measure" / label
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(args.base)
    if args.run:
        ckpts = sorted((root / args.run / "train").glob("model-epoch-*.pt"))
        if not ckpts:
            raise SystemExit(f"no checkpoints in runs/{args.run}/train")
        attach_lora(base)
    else:
        ckpts = []
    print(
        f"measure {label}: {len(records)} records, "
        f"{sum(len(r['pupil_turns']) for r in records)} turns, "
        f"{len(ckpts) or 'bare'} checkpoints"
    )

    summaries: list[dict[str, Any]] = []
    for ckpt in ckpts or [None]:
        t0 = time.monotonic()
        if ckpt is None:
            epoch, slice_ = 0, None
        else:
            blob = read_blob(ckpt)  # once; the "resume" keys are ignored
            meta = blob.get("run", {})
            load_blob_adapters(base, blob)
            epoch, slice_ = int(meta["epoch"]), meta.get("slice")
        with torch.no_grad():
            lines = turn_lines(base.lm, records, epoch)
        with (out / f"epoch-{epoch:02d}.jsonl").open("w") as f:
            f.writelines(json.dumps(x) + "\n" for x in lines)
        s = summarise(lines, epoch, args.base, args.run, slice_)
        summaries.append(s)
        (out / "summary.json").write_text(json.dumps(summaries, indent=1))
        c, a = s["content"], s["all"]
        print(
            f"epoch {epoch} ({time.monotonic() - t0:.0f} s): content gap "
            f"{c['median_gap']:.4f}/{c['p90_gap']:.4f} floor "
            f"{c['median_floor']:.4f}/{c['p90_floor']:.4f} pass {c['passes']} | "
            f"all gap {a['median_gap']:.4f}/{a['p90_gap']:.4f} floor "
            f"{a['median_floor']:.4f}/{a['p90_floor']:.4f} | door "
            f"{s['door_median']:.4f} tape gap {s['tape_gap_median']:.4f} "
            f"tape floor {s['tape_floor_median']:.4f}"
        )
    from dual.report import table

    print(table(summaries, "content"))
    print(table(summaries, "all"))
    print("measured", out)


if __name__ == "__main__":
    main()
