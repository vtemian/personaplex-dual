"""Side by side: what the tutor says after a pupil turn typed, spoken by A, spoken by B.

For every holdout record and pupil turn k, the three rows of `holdout_row` (typed, A, B)
are fed to moshi 0.2.13's LMGen step by step: every stream is forced up to the start of
the tutor's answer, then text and moshi's own audio are decoded greedily over the
answer window while the pupil stream keeps the tape's (the same in all three rows).
Step alignment is `dual.sidebyside.step_input`'s; `lm.dep_q = 8` as in the probe, so the
pupil's codes come from the input and the depformer makes only moshi's 8.

Items of different lengths share a batch through LMGen's per-item exec mask; each
batch shrinks the trunk's KV ring to its longest item, which no item outgrows, so the
attention is the same as with moshi's 3000-frame context.

Models: bare PersonaPlex, then (with --run/--epoch) the same base with that checkpoint's
adapter, cast to bf16 as autocast ran it. Same greedy settings for both.
Writes runs/<name>/sidebyside/{answers.jsonl,codes.jsonl,meta.json}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections.abc import Sequence
from typing import Any

import torch

from dual.constants import CODEBOOKS
from dual.model import MEM_CAP_GIB
from dual.sidebyside import (
    RENDERINGS,
    Item,
    answer_words,
    batch_cap,
    item_of,
    pack,
    step_input,
)

PROBE_DEP_Q = 8
# KV slots (items x frames) a batch may hold: 0.5 MiB a slot (32 layers x k,v x 4096
# bf16), so 48k slots is 24 GiB beside the ~16 GiB of weights.
BUDGET = 48_000
MAX_BATCH = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--base", default="personaplex")
    p.add_argument("--render", default="render-1")
    p.add_argument("--run", default=None, help="a train run; none = bare only")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--no-bare", action="store_true", help="skip the bare model")
    p.add_argument("--limit", type=int, default=0, help="first N (stem, turn) pairs")
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--max-batch", type=int, default=MAX_BATCH)
    p.add_argument("--mem-cap-gib", type=float, default=MEM_CAP_GIB)
    p.add_argument("--compile", action="store_true", help="default eager")
    args = p.parse_args()
    if (args.run is None) != (args.epoch is None):
        p.error("--run and --epoch go together")
    if args.no_bare and args.run is None:
        p.error("--no-bare needs --run")
    if args.compile and args.run is not None:
        # compiled, bare takes moshi's fused gating kernel and an adapted model the
        # generic path (LoRALinear is no nn.Linear): the two would differ by kernel
        p.error("--compile with --run: bare and adapted gating would take two kernels")
    return args


def build_items(
    records: Sequence[dict[str, Any]], limit: int
) -> tuple[list[Item], dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    from dual.render import holdout_row

    items: list[Item] = []
    rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    texts: dict[str, Any] = {}
    pairs = [(r, k) for r in records for k in r["pupil_turns"]]
    if limit:
        pairs = pairs[:limit]
    for record, k in pairs:
        reference: list[int] | None = None
        for rendering in RENDERINGS:
            row, answer = holdout_row(record, k, rendering)
            item = item_of(record["id"], k, rendering, answer)
            items.append(item)
            rows[(item.stem, k, rendering)] = row
            window = list(row["text"][answer[0] : answer[1]])
            if reference is None:
                reference = window
            elif window != reference:
                raise ValueError(f"{row['id']}: answer text differs by rendering")
        texts[f"{record['id']}#{k}"] = {
            "typed": record["renderings"][str(k)]["typed"]["text"],
            "reference": reference,
        }
    return items, rows, texts


class Batch:
    """Per-step forcing for one packed batch, held on the GPU."""

    def __init__(
        self, items: Sequence[Item], rows: Sequence[dict[str, Any]], device: str
    ) -> None:
        self.items = list(items)
        self.steps = max(it.total for it in items)
        n, s = len(items), self.steps
        pupil = torch.zeros(s, n, CODEBOOKS, dtype=torch.long)
        text = torch.zeros(s, n, dtype=torch.long)
        text_on = torch.zeros(s, n, dtype=torch.bool)
        own = torch.zeros(s, n, CODEBOOKS, dtype=torch.long)
        own_on = torch.zeros(s, n, CODEBOOKS, dtype=torch.bool)
        for i, (it, row) in enumerate(zip(items, rows, strict=True)):
            for t in range(it.total):
                x = step_input(row, it.prefix, t)
                pupil[t, i] = torch.tensor(x.pupil)
                if x.text is not None:
                    text[t, i], text_on[t, i] = x.text, True
                for k, c in enumerate(x.own):
                    if c is not None:
                        own[t, i, k], own_on[t, i, k] = c, True
        self.pupil = pupil.to(device)[..., None]  # [S, B, 8, 1]
        self.text, self.text_on = text.to(device), text_on.to(device)
        self.own, self.own_on = own.to(device), own_on.to(device)
        totals = torch.tensor([it.total for it in items])
        self.active = (torch.arange(s)[:, None] < totals[None]).to(device)  # [S, B]
        self.t = 0
        self.text_out: list[torch.Tensor] = []
        self.own_out: list[torch.Tensor] = []

    def text_hook(self, token: torch.Tensor) -> None:
        t = self.t
        token.copy_(torch.where(self.text_on[t], self.text[t], token))
        self.text_out.append(token.clone())

    def audio_hook(self, codes: torch.Tensor) -> None:
        t = self.t
        codes.copy_(torch.where(self.own_on[t], self.own[t], codes))
        self.own_out.append(codes.clone())


def run_batch(lm: Any, batch: Batch) -> tuple[list[list[int]], list[list[list[int]]]]:
    """Every item's generated text ids over its window, and moshi's own codes as
    LMGen emitted them over the same steps (delayed layout, see the module doc)."""
    from moshi.models.lm import LMGen

    layers = lm.transformer.layers
    kept = [layer.self_attn.context for layer in layers]
    cap = batch_cap(batch.items)
    for layer in layers:
        layer.self_attn.context = cap
    try:
        gen = LMGen(
            lm,
            use_sampling=False,
            temp=0.0,
            temp_text=0.0,
            on_text_hook=batch.text_hook,
            on_audio_hook=batch.audio_hook,
            support_out_of_sync=True,
        )
        with gen.streaming(len(batch.items)):
            mask: torch.Tensor | None = None
            for t in range(batch.steps):
                batch.t = t
                now = batch.active[t]
                if mask is None or not bool(torch.equal(now, mask)):
                    gen.set_exec_mask(now)
                    mask = now
                gen.step(batch.pupil[t])
    finally:
        for layer, c in zip(layers, kept, strict=True):
            layer.self_attn.context = c
    text = torch.stack(batch.text_out).cpu()  # [S, B]
    own = torch.stack(batch.own_out).cpu()  # [S, B, 8]
    ids, codes = [], []
    for i, it in enumerate(batch.items):
        ids.append([int(x) for x in text[it.prefix : it.total, i]])
        codes.append(own[it.prefix : it.total, i].tolist())
    return ids, codes


def decode_all(
    lm: Any,
    spm: Any,
    label: str,
    items: Sequence[Item],
    rows: dict[tuple[str, int, str], dict[str, Any]],
    texts: dict[str, Any],
    out: pathlib.Path,
    budget: int,
    max_batch: int,
) -> list[dict[str, Any]]:
    def decode(ids: list[int]) -> str:
        return str(spm.DecodeIds([int(i) for i in ids]))

    timings: list[dict[str, Any]] = []
    batches = pack(items, budget, max_batch)
    print(f"{label}: {len(items)} items in {len(batches)} batches")
    for b, group in enumerate(batches):
        t0 = time.monotonic()
        batch = Batch(
            group, [rows[(it.stem, it.turn, it.rendering)] for it in group], "cuda"
        )
        built = time.monotonic() - t0
        ids, codes = run_batch(lm, batch)
        secs = time.monotonic() - t0 - built
        timings.append(
            {
                "batch": b,
                "items": len(group),
                "steps": batch.steps,
                "cap": batch_cap(group),
                "seconds": secs,
                "build_seconds": built,
            }
        )
        print(
            f"{label} batch {b}: {len(group)} items, {batch.steps} steps, cap "
            f"{batch_cap(group)}, {secs:.0f} s "
            f"({1000 * secs / batch.steps:.0f} ms/step)"
            f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        )
        with (
            (out / "answers.jsonl").open("a") as fa,
            (out / "codes.jsonl").open("a") as fc,
        ):
            for it, gen_ids, own in zip(group, ids, codes, strict=True):
                t = texts[f"{it.stem}#{it.turn}"]
                key = {
                    "model": label,
                    "stem": it.stem,
                    "turn": it.turn,
                    "rendering": it.rendering,
                }
                fa.write(
                    json.dumps(
                        {
                            **key,
                            "pupil": answer_words(t["typed"], decode),
                            "reference": answer_words(t["reference"], decode),
                            "answer": answer_words(gen_ids, decode),
                            "ids": gen_ids,
                            "prefix": it.prefix,
                            "window": it.window,
                        }
                    )
                    + "\n"
                )
                fc.write(json.dumps({**key, "own_steps": own}) + "\n")
    return timings


def main() -> None:
    from dual.model import (
        adapter_setup,
        attach_lora,
        cap_gpu_memory,
        describe,
        load_base,
        load_blob_adapters,
        read_blob,
        set_compile_mode,
    )

    args = parse_args()
    set_compile_mode(not args.compile)
    cap_gpu_memory(args.mem_cap_gib)
    root = pathlib.Path.cwd() / "runs"
    records = [
        json.loads(line)
        for line in (root / args.render / "render" / "holdout.jsonl")
        .read_text()
        .splitlines()
    ]
    items, rows, texts = build_items(records, args.limit)
    out = root / args.name / "sidebyside"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("answers.jsonl", "codes.jsonl"):
        (out / f).unlink(missing_ok=True)
    print(f"side by side: {len(texts)} turns, {len(items)} items")

    base = load_base(args.base)
    print("describe:", describe(base))
    base.lm.dep_q = PROBE_DEP_Q  # this job only, as in the probe
    meta: dict[str, Any] = {"args": vars(args), "models": {}}

    def go(label: str, info: dict[str, Any]) -> None:
        base.lm.eval()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        with torch.no_grad():
            timings = decode_all(
                base.lm,
                base.spm,
                label,
                items,
                rows,
                texts,
                out,
                args.budget,
                args.max_batch,
            )
        meta["models"][label] = {
            **info,
            "seconds": time.monotonic() - t0,
            "batches": timings,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=1))

    if not args.no_bare:
        go("bare", {"base": args.base})
    if args.run is not None:
        ckpt = root / args.run / "train" / f"model-epoch-{args.epoch:02d}.pt"
        blob = read_blob(ckpt)
        attach_lora(base, *adapter_setup(blob["meta"]))
        load_blob_adapters(base, blob)
        # moshi's KV cache takes the first in_proj's dtype: fp32 adapters would make it
        # fp32. autocast ran the adapters' matmuls in bf16 in train and measure anyway.
        lora = [p for n, p in base.lm.named_parameters() if ".lora_" in n]
        for p in lora:
            p.data = p.data.to(torch.bfloat16)
        from training.moshi.trunk_lora import unmoved, wrappers

        if len(wrappers(base.lm)) != 128 or unmoved(base.lm):
            raise RuntimeError(f"{ckpt}: adapters missing or zero after loading")
        norm = float(sum(float(p.float().norm()) ** 2 for p in lora) ** 0.5)
        print(f"adapters: {len(lora)} tensors, norm {norm:.3f}")
        go(
            "trained",
            {
                "run": args.run,
                "epoch": args.epoch,
                "ckpt": str(ckpt),
                "run_meta": blob.get("run", {}),
                "adapter_norm": norm,
                "adapter_meta": blob["meta"],
            },
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
