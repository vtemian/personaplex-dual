"""Proves the path: sparks import, /data, the shared dir, moshiko and Mimi, the GPU.

With `--base NAME`: loads that base, prints describe(), runs one teacher-forced
forward on a 40-frame row of PAD / SILENCE / SINE and checks the logits, then attaches
the trunk LoRA and round-trips its state. No training."""

import argparse
import gc
import os
import pathlib
import time

import torch
from sparks.emit import track

from dual.constants import BASES, CODEBOOKS, PAD, SILENCE, SINE, TEXT_CARD

SMOKE_FRAMES = 40


def path_smoke() -> None:
    data = pathlib.Path(os.environ["SPARKS_DATA"])
    print("data:", sorted(p.name for p in data.iterdir()))
    models = pathlib.Path.cwd() / "models"
    print(
        "cwd:", pathlib.Path.cwd(), "models:", sorted(p.name for p in models.iterdir())
    )
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    from moshi.models.loaders import CheckpointInfo

    info = CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    spm = info.get_text_tokenizer()
    print(
        "spm pieces:", spm.GetPieceSize(), "hello ->", spm.EncodeAsIds("hello there.")
    )
    mimi = info.get_mimi(device="cuda")
    with torch.no_grad(), mimi.streaming(1):
        codes = mimi.encode(torch.zeros(1, 1, 1920, device="cuda"))
    print("mimi silence frame:", [int(c) for c in codes[0, :, 0]])
    with track(total=10) as run:
        for step in range(10):
            run.step(loss=1.0 / (step + 1))
            time.sleep(0.2)


def base_smoke(name: str) -> None:
    from dual.model import (
        attach_lora,
        codes_from_row,
        describe,
        forward,
        load_base,
        read_adapters,
        save_adapters,
    )

    start = time.monotonic()
    base = load_base(name)
    print(f"loaded {name} in {time.monotonic() - start:.1f}s")
    print("describe:", name, describe(base))
    print("spm pieces:", base.spm.GetPieceSize())
    row = {
        "text": [PAD] * SMOKE_FRAMES,
        "own": [list(SILENCE)] * SMOKE_FRAMES,
        "pupil": [list(SINE)] * SMOKE_FRAMES,
    }
    codes = codes_from_row(row)
    with torch.no_grad():
        out = forward(base.lm, codes)
    text, mask = out.text_logits, out.text_mask
    print("text logits:", list(text.shape), "mask frames:", int(mask.sum()))
    assert tuple(text.shape) == (1, SMOKE_FRAMES, TEXT_CARD), text.shape
    nan_in = bool(torch.isnan(text[mask]).any())
    nan_out = bool(torch.isnan(text[~mask]).all()) if bool((~mask).any()) else True
    print("text NaN inside mask:", nan_in, "| masked frames all NaN:", nan_out)
    audio_nan = bool(torch.isnan(out.audio_logits[out.audio_mask]).any())
    print("audio logits:", list(out.audio_logits.shape), "NaN inside mask:", audio_nan)
    if nan_in or audio_nan:
        raise RuntimeError("NaN outside the masked frames")
    top = text[mask].argmax(-1).tolist()
    print("greedy text on frames:", top[:SMOKE_FRAMES])
    assert out.audio_logits.shape[1] == CODEBOOKS
    modules = attach_lora(base)
    # lora_B starts at zero, so a round-trip of it proves nothing: make it real.
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in base.lm.named_parameters():
            if ".lora_B" in n:
                noise = torch.randn(p.shape, generator=gen) * 0.02
                p.copy_(noise.to(device=p.device, dtype=p.dtype))
    with torch.no_grad():
        first = forward(base.lm, codes).text_logits
    bare = text
    moved = float((first[mask] - bare[mask]).abs().max())
    print("random lora_B moved text logits by", moved)
    if moved == 0.0:
        raise RuntimeError("random adapters changed nothing: the round-trip is vacuous")
    path = "/tmp/adapters-smoke.pt"
    save_adapters(base, path)
    n_params = sum(p.numel() for n, p in base.lm.named_parameters() if ".lora_" in n)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    again = load_base(name)
    attach_lora(again)
    read_adapters(again, path)
    with torch.no_grad():
        second = forward(again.lm, codes).text_logits
    delta = float((second[mask] - first[mask]).abs().max())
    equal = bool(torch.equal(second[mask], first[mask]))
    print(
        f"lora: {modules} modules, {n_params} params, round-trip equal {equal}, "
        f"max |delta logits| {delta}"
    )
    if delta >= 1e-3:
        raise RuntimeError(f"round-trip delta {delta} >= 1e-3")
    if not equal:
        print("note: not bit-equal but under 1e-3 (bf16 nondeterminism)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=BASES)
    args = parser.parse_args()
    if args.base is None:
        path_smoke()
    else:
        base_smoke(args.base)
    print("smoke ok")


if __name__ == "__main__":
    main()
