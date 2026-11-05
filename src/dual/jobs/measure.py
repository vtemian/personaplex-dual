"""The bar on one base: bare (epoch 0) or every checkpoint of a train run.

Per holdout record and pupil turn k, five rows (typed, A, B, C, tape) through one
teacher-forced forward each; the text log-probs over each row's answer window give
gap = KL(typed || A), floor = KL(A || B), door = KL(C || A),
tape_gap = KL(typed || tape), tape_floor = KL(tape || A).
Writes runs/<name>/measure/<base>[-<run>][-<tag>]/{epoch-NN.jsonl,summary.json}.
Each jsonl line also has content_mask (0/1 per answer frame, 1 where the target text
is not PAD) and per_frame ({pair: [KL per frame, 6 significant digits]}).
--epochs N [N ...] measures only those checkpoints (with --run); --tag T labels a
partial pass apart from a full run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import re
import statistics
import time
from collections.abc import Sequence
from typing import Any

import torch

from dual.constants import PAD
from dual.kl import answer_kl, bar, kl_per_frame, log_probs
from dual.model import MEM_CAP_GIB

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
    p.add_argument(
        "--mem-cap-gib",
        type=float,
        default=MEM_CAP_GIB,
        help="cap this process's GPU memory; 0 disables the cap",
    )
    p.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="only these checkpoint epochs (with --run); default all",
    )
    p.add_argument("--tag", default=None, help="appended to the output label")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", action="store_true", help="no torch.compile")
    mode.add_argument("--compile", action="store_true", help="torch.compile")
    return p.parse_args()


def label(base: str, run: str | None, tag: str | None) -> str:
    return base + (f"-{run}" if run else "") + (f"-{tag}" if tag else "")


_CKPT = re.compile(r"model-epoch-(\d+)\.pt$")


def select_ckpts(
    paths: Sequence[pathlib.Path], epochs: Sequence[int] | None
) -> list[pathlib.Path]:
    """Checkpoints sorted by name; only the given epochs (by filename) if any."""
    by_epoch: dict[int, pathlib.Path] = {}
    for path in sorted(paths):
        m = _CKPT.search(path.name)
        if m is None:
            raise SystemExit(f"not a checkpoint name: {path.name}")
        by_epoch[int(m.group(1))] = path
    if epochs is None:
        return [by_epoch[e] for e in sorted(by_epoch)]
    missing = sorted(set(epochs) - by_epoch.keys())
    if missing:
        raise SystemExit(
            f"no checkpoint for epoch {missing[0]} (have {sorted(by_epoch)})"
        )
    return [by_epoch[e] for e in sorted(set(epochs))]


def pair_fields(lps: dict[str, torch.Tensor], targets: list[int]) -> dict[str, Any]:
    """One turn's per-pair AnswerKL dicts, content_mask and per_frame KL."""
    out: dict[str, Any] = {}
    per_frame: dict[str, list[float]] = {}
    for key, p, q in PAIRS:
        out[key] = dataclasses.asdict(answer_kl(lps[p], lps[q], targets))
        per_frame[key] = [
            float(f"{x:.6g}") for x in kl_per_frame(lps[p], lps[q]).tolist()
        ]
    out["content_mask"] = [int(t != PAD) for t in targets]
    out["per_frame"] = per_frame
    return out


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
            fields = pair_fields(lps, targets)
            line: dict[str, Any] = {"stem": record["id"], "turn": k, "epoch": epoch}
            for key, _, _ in PAIRS:
                line[key] = fields[key]
            line["answer_frames"] = len(targets)
            line["content_frames"] = fields["gap"]["content_frames"]
            line["content_mask"] = fields["content_mask"]
            line["per_frame"] = fields["per_frame"]
            lines.append(line)
    return lines


def main() -> None:
    from dual.model import (
        EAGER_DEFAULT,
        adapter_setup,
        attach_lora,
        cap_gpu_memory,
        load_base,
        load_blob_adapters,
        read_blob,
        set_compile_mode,
    )

    args = parse_args()
    set_compile_mode(args.eager or (EAGER_DEFAULT and not args.compile))
    cap_gpu_memory(args.mem_cap_gib)
    root = pathlib.Path.cwd() / "runs"
    render = root / args.render / "render"
    records = [
        json.loads(line) for line in (render / "holdout.jsonl").read_text().splitlines()
    ]
    name = label(args.base, args.run, args.tag)
    out = root / args.name / "measure" / name
    if args.epochs and not args.run:
        print("note: --epochs is ignored without --run (bare is epoch 0)")
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(args.base)
    if args.run:
        found = list((root / args.run / "train").glob("model-epoch-*.pt"))
        if not found:
            raise SystemExit(f"no checkpoints in runs/{args.run}/train")
        ckpts = select_ckpts(found, args.epochs)
        # the first checkpoint's own setup; load_blob_adapters checks every one
        attach_lora(base, *adapter_setup(read_blob(ckpts[0])["meta"]))
    else:
        ckpts = []
    print(
        f"measure {name}: {len(records)} records, "
        f"{sum(len(r['pupil_turns']) for r in records)} turns, "
        f"{len(ckpts) or 'bare'} checkpoints"
    )

    summaries: list[dict[str, Any]] = []
    todo: list[pathlib.Path | None] = [*ckpts] if ckpts else [None]
    for ckpt in todo:
        t0 = time.monotonic()
        if ckpt is None:
            epoch, slice_ = 0, None
        else:
            blob = read_blob(ckpt)  # once; the "resume" keys are ignored
            meta = blob.get("run", {})
            load_blob_adapters(base, blob)
            epoch, slice_ = int(meta["epoch"]), meta.get("slice")
            want = int(_CKPT.search(ckpt.name).group(1))  # type: ignore[union-attr]
            assert epoch == want, f"{ckpt.name}: meta epoch {epoch}"
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
