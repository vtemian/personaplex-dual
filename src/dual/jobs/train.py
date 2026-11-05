"""The trunk LoRA on one base over one rung of the rendered rows."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from typing import Any, cast

import torch
from sparks.emit import track

from dual.kl import DIRECTIONS, consistency_loss, log_probs
from dual.ladder import rung
from dual.model import (
    EAGER_DEFAULT,
    MEM_CAP_GIB,
    RANK,
    SCALING,
    cap_gpu_memory,
    set_compile_mode,
)

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
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--scaling", type=float, default=SCALING)
    p.add_argument("--frames-per-batch", type=int, default=FRAMES_PER_BATCH)
    p.add_argument(
        "--mem-cap-gib",
        type=float,
        default=MEM_CAP_GIB,
        help="cap this process's GPU memory; 0 disables the cap",
    )
    p.add_argument(
        "--resume", action="store_true", help="continue from the latest epoch"
    )
    p.add_argument(
        "--consistency",
        type=float,
        default=0.0,
        help="weight of KL(typed answer || spoken answer); 0 = the baseline recipe",
    )
    p.add_argument("--consistency-dir", choices=DIRECTIONS, default="bar")
    p.add_argument(
        "--only-stem",
        default=None,
        help="train on one stem's rows only ('longest' = the stem of the longest row)",
    )
    p.add_argument(
        "--max-steps", type=int, default=0, help="stop after N steps, no checkpoint"
    )
    p.add_argument(
        "--widest-first",
        action="store_true",
        help="run the batches with the most padded frames first (smoke)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eager", action="store_true", help="no torch.compile")
    mode.add_argument("--compile", action="store_true", help="torch.compile")
    args = p.parse_args()
    if args.consistency < 0:
        p.error(f"--consistency {args.consistency}: must be >= 0")
    return args


Teacher = tuple[
    torch.Tensor, torch.Tensor
]  # typed-row frame index [N], log-probs [N, V]


def losses(
    lm: Any,
    batch: list[dict[str, Any]],
    teachers: dict[int, Teacher] | None = None,
    direction: str = "bar",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Text and audio CE over scored frames only, and the consistency KL of each typed
    answer toward its teacher (zero without teachers). Rows are padded to the longest;
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
    students: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for i, (idx, lp) in (teachers or {}).items():
        keep = out.text_mask[i, idx]
        students.append(out.text_logits[i, idx[keep]])
        targets.append(lp[keep])
    if students:
        cons = consistency_loss(torch.cat(students), torch.cat(targets), direction)
    else:
        cons = out.text_logits.new_zeros(())
    return text, audio, cons


def build_teachers(
    lm: Any, batch: list[dict[str, Any]], spoken: dict[str, dict[str, Any]]
) -> dict[int, Teacher]:
    """For each typed row of the batch (by position), the frame indices of its answer
    content frames and the model's own text log-probs at the same frames after the
    spoken (all-tape) rendering of its stem: a no_grad forward with the current adapter,
    one stem at a time, so a single spoken forward is alive at once."""
    from dual.consistency import answer_pairs, flat_indices
    from dual.model import codes_from_row, forward

    out: dict[int, Teacher] = {}
    have: tuple[str, Any] | None = None
    with torch.no_grad():
        for i in sorted(range(len(batch)), key=lambda j: str(batch[j]["stem"])):
            row = batch[i]
            if not row.get("typed_turns"):
                continue
            stem = row["stem"]
            student, teacher = flat_indices(answer_pairs(row, spoken[stem]))
            if not student:
                continue
            if have is None or have[0] != stem:
                t = have = None  # drop the previous stem's logits first
                have = (stem, forward(lm, codes_from_row(spoken[stem])))
            t = have[1]
            s_idx = torch.tensor(student, device=t.text_logits.device)
            t_idx = torch.tensor(teacher, device=t.text_logits.device)
            keep = t.text_mask[0, t_idx]
            out[i] = (s_idx[keep], log_probs(t.text_logits[0, t_idx[keep]]))
    return out


def dynamo_stats() -> dict[str, int]:
    """torch.compile's own counters (graphs compiled, recompiles), for the smoke."""
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return {}
    return {k: int(v) for k, v in counters["stats"].items()}


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


def check_resume_lr(run_meta: dict[str, Any], lr: float) -> None:
    """--resume continues the optimiser state, so the rate must be the run's own.
    A checkpoint without "lr" predates the flag and ran at LR."""
    had = float(run_meta.get("lr", LR))
    if had != lr:
        raise ValueError(f"--resume: checkpoint ran at lr {had}, --lr is {lr}")


def check_resume_consistency(
    run_meta: dict[str, Any], consistency: float, direction: str
) -> None:
    """--resume keeps the recipe: the same weight and direction. A checkpoint without
    the keys predates B9 and ran without the term."""
    had = float(run_meta.get("consistency", 0.0))
    had_dir = str(run_meta.get("consistency_dir", "bar"))
    if (had, had_dir) != (consistency, direction):
        raise ValueError(
            f"--resume: checkpoint ran at consistency {had} ({had_dir}), "
            f"flags say {consistency} ({direction})"
        )


def select_rows(rows: list[dict[str, Any]], only: str | None) -> list[dict[str, Any]]:
    """All rows, or one stem's ('longest': the stem of the longest row)."""
    from dual.consistency import longest_stem

    if only is None:
        return rows
    stem = longest_stem(rows) if only == "longest" else only
    kept = [r for r in rows if r["stem"] == stem]
    if not kept:
        raise ValueError(f"--only-stem {only}: no rows")
    return kept


def widest_first(bs: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Batches by padded frames (rows x longest), largest first; stable on ties."""
    return sorted(bs, key=lambda b: -len(b) * max(r["frames"] for r in b))


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
    # torch is already imported here, but the caching allocator reads
    # PYTORCH_CUDA_ALLOC_CONF lazily, at its first CUDA allocation, so this is in time.
    cap_gpu_memory(args.mem_cap_gib)
    from training.moshi.trunk_lora import unmoved

    from dual.consistency import answer_pairs, flat_indices, spoken_by_stem
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
    # the rows are json dicts; spoken_by_stem types them as read-only Mappings
    spoken = cast(
        dict[str, dict[str, Any]],
        spoken_by_stem(rows) if args.consistency > 0 else {},
    )
    rows = select_rows(rows, args.only_stem)
    typed_rows = [r for r in rows if r.get("typed_turns")]
    consistency = f"consistency {args.consistency} ({args.consistency_dir}); "
    consistency += f"typed rows {len(typed_rows)}"
    if args.consistency > 0:
        # every typed row pairs with its spoken row now, not mid-epoch
        pairs = [answer_pairs(r, spoken[r["stem"]]) for r in typed_rows]
        content = sum(len(flat_indices(p)[0]) for p in pairs)
        consistency += (
            f", answer pairs {sum(map(len, pairs))}, content frames {content}"
        )
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
        f"(<= {limit} frames a batch); rank {args.rank} scaling {args.scaling} "
        f"lr {args.lr} mem cap {args.mem_cap_gib} GiB; {consistency}"
    )

    base = load_base(args.base)
    attach_lora(base, args.rank, args.scaling)
    lm = base.lm
    params = [p for p in lm.parameters() if p.requires_grad]
    # weight_decay is chalkboard's: its dv3a trunk group
    # (vendor/training/jobs/train_moshi.py:514) runs AdamW's default 0.01.
    opt = torch.optim.AdamW(params, lr=args.lr, betas=BETAS, weight_decay=0.01)
    first = 1
    if args.resume:
        latest = latest_checkpoint(out)
        if latest is None:
            raise SystemExit(f"--resume: no checkpoints in {out}")
        blob = read_blob(latest[1])
        try:
            check_resume_lr(blob.get("run", {}), args.lr)
            check_resume_consistency(
                blob.get("run", {}), args.consistency, args.consistency_dir
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        load_blob_adapters(base, blob)
        opt.load_state_dict(blob["resume"]["opt"])
        restore_rng(rng, blob["resume"]["rng"])
        first = int(blob["run"]["epoch"]) + 1
        print(f"resumed from {latest[1].name}: continuing at epoch {first}")

    done = 0
    with track(
        total=args.epochs * steps_per_epoch, arm=f"{args.base}-{args.slice}"
    ) as run:
        for epoch in range(first, args.epochs + 1):
            t0 = time.monotonic()
            norms: list[float] = []
            totals: list[float] = []
            conses: list[float] = []
            plan = batches(rows, rng, limit)
            if args.widest_first:
                plan = widest_first(plan)
            for i, batch in enumerate(plan):
                t_step = time.monotonic()
                teachers = (
                    build_teachers(lm, batch, spoken) if args.consistency > 0 else None
                )
                t_teacher = time.monotonic() - t_step
                text, audio, cons = losses(lm, batch, teachers, args.consistency_dir)
                loss = TEXT_WEIGHT * text + AUDIO_WEIGHT * audio
                if args.consistency > 0:
                    loss = loss + args.consistency * cons
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                norm = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
                opt.step()
                vals = (
                    float(loss.detach()),
                    float(text.detach()),
                    float(audio.detach()),
                    norm,
                    float(cons.detach()),
                )
                if not all(map(math.isfinite, vals)):
                    raise RuntimeError(f"epoch {epoch} step {i}: non-finite {vals}")
                norms.append(norm)
                totals.append(vals[0])
                conses.append(vals[4])
                frames_t = sum(len(idx) for idx, _ in (teachers or {}).values())
                run.step(
                    loss={
                        "total": vals[0],
                        "text": vals[1],
                        "audio": vals[2],
                        "consistency": vals[4],
                    },
                    grad_norm=norm,
                    learning_rate=args.lr,
                    epoch=epoch - 1 + (i + 1) / steps_per_epoch,
                )
                done += 1
                if i % 10 == 0 or args.max_steps:
                    print(
                        f"epoch {epoch} step {i}: loss {vals[0]:.4f} text "
                        f"{vals[1]:.4f} audio {vals[2]:.4f} cons {vals[4]:.5f} "
                        f"({frames_t} frames) grad {norm:.3f} "
                        f"rows {len(batch)}x{max(r['frames'] for r in batch)} "
                        f"teacher {t_teacher:.1f} s step "
                        f"{time.monotonic() - t_step:.1f} s"
                    )
                if args.max_steps and done >= args.max_steps:
                    break
            if args.max_steps:  # a smoke: no held-out eval, never a checkpoint
                if done >= args.max_steps or epoch == args.epochs:
                    print(
                        f"smoke: {done} steps, peak mem "
                        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                        f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} "
                        "GiB); no checkpoint written"
                    )
                    print("dynamo:", dynamo_stats())
                    return
                continue
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
                    "rank": args.rank,
                    "scaling": args.scaling,
                    "lr": args.lr,
                    "mem_cap_gib": args.mem_cap_gib,
                    "consistency": args.consistency,
                    "consistency_dir": args.consistency_dir,
                    "only_stem": args.only_stem,
                },
                resume={"opt": opt.state_dict(), "rng": capture_rng(rng)},
            )
            tmp.rename(final)
            print(
                f"epoch {epoch}: mean loss {sum(totals) / len(totals):.4f} "
                f"mean cons {sum(conses) / len(conses):.5f} "
                f"max grad {max(norms):.3f} held text {eval_loss:.4f} "
                f"{seconds:.0f} s, peak mem "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
                f"{final.name} {final.stat().st_size / 2**20:.0f} MiB"
            )
    print("trained", out)


if __name__ == "__main__":
    main()
