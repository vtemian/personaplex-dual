"""The trunk LoRA on one base over one rung of the rendered rows."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from typing import Any

import torch
from sparks.emit import track

from dual.ladder import rung
from dual.model import EAGER_DEFAULT, set_compile_mode

LR = 2e-5
BETAS = (0.9, 0.95)
CLIP = 1.0
TEXT_WEIGHT = 2.0
AUDIO_WEIGHT = 0.5
FRAMES_PER_BATCH = 4000  # rows are packed by length into batches under this many frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--slice", type=int, default=1)
    p.add_argument("--render", default="render-1")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames-per-batch", type=int, default=FRAMES_PER_BATCH)
    p.add_argument(
        "--resume", action="store_true", help="continue from the latest epoch"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", action="store_true", help="no torch.compile")
    mode.add_argument("--compile", action="store_true", help="torch.compile")
    return p.parse_args()


def losses(lm: Any, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Text and audio CE over scored frames only. Rows are padded to the longest;
    padding is unscored. NaN logits at moshi's masked frames are excluded by index,
    never multiplied."""
    from dual.model import codes_from_row, forward

    width = max(r["frames"] for r in batch)
    codes = torch.cat(
        [
            torch.nn.functional.pad(
                codes_from_row(r), (0, width - r["frames"]), value=3
            )
            for r in batch
        ]
    )
    scored = torch.zeros(len(batch), width, dtype=torch.bool, device=codes.device)
    for i, r in enumerate(batch):
        scored[i, : r["frames"]] = torch.tensor(r["scored"], dtype=torch.bool)
    out = forward(lm, codes)
    text_target = codes[:, 0]
    text_keep = scored & out.text_mask
    text = torch.nn.functional.cross_entropy(
        out.text_logits[text_keep], text_target[text_keep]
    )
    audio_target = codes[:, 1:9]
    audio_keep = scored[:, None, :] & out.audio_mask
    audio = torch.nn.functional.cross_entropy(
        out.audio_logits[audio_keep], audio_target[audio_keep]
    )
    return text, audio


def batches(
    rows: list[dict[str, Any]], rng: random.Random, limit: int = FRAMES_PER_BATCH
) -> list[list[dict[str, Any]]]:
    """Rows sorted by length, packed while rows x longest stays under `limit` (a row
    over it alone is its own batch), then the batches shuffled by `rng`."""
    by_len = sorted(rows, key=lambda r: r["frames"])
    out: list[list[dict[str, Any]]] = [[]]
    for r in by_len:
        if out[-1] and (len(out[-1]) + 1) * r["frames"] > limit:
            out.append([])
        out[-1].append(r)
    rng.shuffle(out)
    return out


def latest_checkpoint(out: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    """(epoch, path) of the highest model-epoch-NN.pt in `out`, none if empty."""
    found = [(int(p.stem.rsplit("-", 1)[1]), p) for p in out.glob("model-epoch-*.pt")]
    return max(found, default=None)


def refuse_stale(out: pathlib.Path, resume: bool) -> None:
    if not resume and latest_checkpoint(out) is not None:
        raise SystemExit(
            f"{out} already holds checkpoints: pass --resume to continue, "
            "or use a new --name"
        )


def capture_rng(rng: random.Random) -> dict[str, Any]:
    """Every RNG the loop draws from, as weights_only-loadable values."""
    state: dict[str, Any] = {
        "batches": rng.getstate(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(rng: random.Random, state: dict[str, Any]) -> None:
    rng.setstate(state["batches"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main() -> None:
    args = parse_args()
    set_compile_mode(args.eager or (EAGER_DEFAULT and not args.compile))
    from training.moshi.trunk_lora import unmoved

    from dual.model import (
        attach_lora,
        load_base,
        load_blob_adapters,
        read_blob,
        save_adapters,
    )

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    render = pathlib.Path.cwd() / "runs" / args.render / "render"
    stems = json.loads((render / "stems.json").read_text())
    keep = set(rung(stems["train"], args.slice))
    rows = [
        r
        for r in map(json.loads, (render / "rows-train.jsonl").read_text().splitlines())
        if r["stem"] in keep
    ]
    held = [
        json.loads(line)["base"]
        for line in (render / "holdout.jsonl").read_text().splitlines()
    ]
    out = pathlib.Path.cwd() / "runs" / args.name / "train"
    out.mkdir(parents=True, exist_ok=True)
    refuse_stale(out, args.resume)
    limit = args.frames_per_batch
    steps_per_epoch = len(batches(rows, random.Random(0), limit))
    print(
        f"base {args.base} slice 1/{args.slice}: {len(keep)} dialogues, "
        f"{len(rows)} rows, {len(held)} held, {steps_per_epoch} steps/epoch "
        f"(<= {limit} frames a batch)"
    )

    base = load_base(args.base)
    attach_lora(base)
    lm = base.lm
    params = [p for p in lm.parameters() if p.requires_grad]
    # weight_decay is chalkboard's: its dv3a trunk group
    # (vendor/training/jobs/train_moshi.py:514) runs AdamW's default 0.01.
    opt = torch.optim.AdamW(params, lr=LR, betas=BETAS, weight_decay=0.01)
    first = 1
    if args.resume:
        latest = latest_checkpoint(out)
        if latest is None:
            raise SystemExit(f"--resume: no checkpoints in {out}")
        blob = read_blob(latest[1])
        load_blob_adapters(base, blob)
        opt.load_state_dict(blob["resume"]["opt"])
        restore_rng(rng, blob["resume"]["rng"])
        first = int(blob["run"]["epoch"]) + 1
        print(f"resumed from {latest[1].name}: continuing at epoch {first}")

    with track(
        total=args.epochs * steps_per_epoch, arm=f"{args.base}-{args.slice}"
    ) as run:
        for epoch in range(first, args.epochs + 1):
            t0 = time.monotonic()
            norms: list[float] = []
            totals: list[float] = []
            for i, batch in enumerate(batches(rows, rng, limit)):
                text, audio = losses(lm, batch)
                loss = TEXT_WEIGHT * text + AUDIO_WEIGHT * audio
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                norm = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
                opt.step()
                vals = (
                    float(loss.detach()),
                    float(text.detach()),
                    float(audio.detach()),
                    norm,
                )
                if not all(map(math.isfinite, vals)):
                    raise RuntimeError(f"epoch {epoch} step {i}: non-finite {vals}")
                norms.append(norm)
                totals.append(vals[0])
                run.step(
                    loss={"total": vals[0], "text": vals[1], "audio": vals[2]},
                    grad_norm=norm,
                    learning_rate=LR,
                    epoch=epoch - 1 + (i + 1) / steps_per_epoch,
                )
                if i % 10 == 0:
                    print(
                        f"epoch {epoch} step {i}: loss {vals[0]:.4f} text "
                        f"{vals[1]:.4f} audio {vals[2]:.4f} grad {norm:.3f} "
                        f"rows {len(batch)}x{max(r['frames'] for r in batch)}"
                    )
            seconds = time.monotonic() - t0
            if epoch == first:
                print("unmoved:", unmoved(lm))
            with torch.no_grad():
                held_text = [float(losses(lm, [r])[0]) for r in held]
            eval_loss = sum(held_text) / len(held_text)
            run.log(eval_loss=eval_loss, epoch=float(epoch))
            final = out / f"model-epoch-{epoch:02d}.pt"
            tmp = out / f"model-epoch-{epoch:02d}.pt.tmp"
            save_adapters(
                base,
                tmp,
                {
                    "epoch": epoch,
                    "slice": args.slice,
                    "render": args.render,
                    "seed": args.seed,
                    "name": args.name,
                },
                resume={"opt": opt.state_dict(), "rng": capture_rng(rng)},
            )
            tmp.rename(final)
            print(
                f"epoch {epoch}: mean loss {sum(totals) / len(totals):.4f} "
                f"max grad {max(norms):.3f} held text {eval_loss:.4f} "
                f"{seconds:.0f} s, peak mem "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                f"{final.name} {final.stat().st_size / 2**20:.0f} MiB"
            )
    print("trained", out)


if __name__ == "__main__":
    main()
