"""Proves the path: sparks import, /data, the shared dir, the baked moshiko and
Mimi, the GPU. No PersonaPlex, no training."""

import os
import pathlib
import time

import torch
from sparks.emit import track


def main() -> None:
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
    print("smoke ok")


if __name__ == "__main__":
    main()
