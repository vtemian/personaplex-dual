"""GPU-only: the two bases, the trunk LoRA and the teacher-forced forward.

`moshi` and chalkboard's `training` are imported inside functions, so the laptop's
checks never load them.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import torch

from dual.constants import (
    AUDIO_CARD,
    BASES,
    CODEBOOKS,
    CONTEXT,
    MOSHIKO,
    MOSHIKO_REPO,
    PERSONAPLEX_FILES,
    STREAMS,
    TEXT_CARD,
)

RANK = 16
SCALING = 2.0
# chalkboard training/moshi/trunk_lora.py: 66,560 per unit of rank per layer, x16 x32.
ADAPTER_PARAMS = 34_078_720
# PersonaPlex's depformer predicts 16 streams: moshi's 8, then the pupil's 8.
PERSONAPLEX_DEP_Q = 16


@dataclass
class Base:
    name: str
    lm: Any  # moshi LMModel
    spm: Any  # sentencepiece processor
    info: Any  # CheckpointInfo, for get_mimi


def personaplex_dir() -> pathlib.Path:
    return pathlib.Path.cwd() / "models" / "personaplex-7b-v1"


def load_base(name: str, device: str = "cuda") -> Base:
    from moshi.models.loaders import CheckpointInfo

    if name not in BASES:
        raise ValueError(f"base {name!r}: one of {BASES}")
    if name == MOSHIKO:
        info = CheckpointInfo.from_hf_repo(MOSHIKO_REPO)
        lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    else:
        root = personaplex_dir()
        paths = {k: root / v for k, v in PERSONAPLEX_FILES.items()}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"PersonaPlex files missing: {missing}")
        # lm_config None: moshi 7B's defaults (n_q 16, 32 layers, the 17 delays), and
        # an 8-codebook Mimi from get_mimi; only dep_q differs. The file is 0.1-era
        # (fused in_proj_weight), which 0.2.13's load hook splits, as for moshiko.
        info = CheckpointInfo(
            moshi_weights=paths["moshi"],
            mimi_weights=paths["mimi"],
            tokenizer=paths["spm"],
            lm_config=None,
        )
        lm = info.get_moshi(
            device=device,
            dtype=torch.bfloat16,
            lm_kwargs_overrides={"dep_q": PERSONAPLEX_DEP_Q},
        )
    lm.eval()
    spm = info.get_text_tokenizer()
    return Base(name, lm, spm, info)


def describe(base: Base) -> dict[str, int]:
    return {
        "params": sum(p.numel() for p in base.lm.parameters()),
        "dep_q": int(base.lm.dep_q),
        "n_q": int(base.lm.num_audio_codebooks),
        "layers": len(base.lm.transformer.layers),
    }


def attach_lora(base: Base) -> int:
    """Wraps the trunk's 128 linears; leaves the lm in train mode (attach_measured)."""
    from training.moshi.trunk_lora import attach_measured

    modules, drift = attach_measured(base.lm, RANK, SCALING)
    if drift != 0.0:
        raise RuntimeError(f"a fresh adapter moved the trunk by {drift}")
    for name, param in base.lm.named_parameters():
        param.requires_grad_(".lora_" in name)
    trainable = sum(p.numel() for p in base.lm.parameters() if p.requires_grad)
    if trainable != ADAPTER_PARAMS:
        raise RuntimeError(f"{trainable} trainable parameters, not {ADAPTER_PARAMS}")
    return int(modules)


def adapter_state(base: Base) -> dict[str, torch.Tensor]:
    return {
        n: p.detach().cpu().clone()
        for n, p in base.lm.named_parameters()
        if ".lora_" in n
    }


def load_adapters(base: Base, state: dict[str, torch.Tensor]) -> None:
    # strict=False only because `state` holds the adapters alone: every unexpected key
    # and every missing adapter key is still refused.
    missing, unexpected = base.lm.load_state_dict(state, strict=False)
    unexpected_l = list(unexpected)
    missing_lora = [m for m in missing if ".lora_" in m]
    if unexpected_l or missing_lora:
        raise RuntimeError(
            f"adapter keys off: unexpected {unexpected_l[:3]}, "
            f"missing lora {missing_lora[:3]}"
        )


ADAPTER_META_KEYS = ("base", "rank", "scaling", "params")


def check_adapter_meta(got: dict[str, Any], want: dict[str, Any]) -> None:
    for key in ADAPTER_META_KEYS:
        if key not in got:
            raise ValueError(f"adapter file has no {key!r} in its metadata")
        if got[key] != want[key]:
            raise ValueError(
                f"adapter {key} mismatch: file {got[key]!r}, base {want[key]!r}"
            )


def adapter_meta(base: Base) -> dict[str, Any]:
    return {
        "base": base.name,
        "rank": RANK,
        "scaling": SCALING,
        "params": ADAPTER_PARAMS,
    }


def save_adapters(
    base: Base, path: str | pathlib.Path, extra: dict[str, Any] | None = None
) -> None:
    """The adapter tensors plus the base, rank, scaling and params of their setup;
    `extra` (epoch, slice, ...) rides along under "run" and is never validated."""
    blob: dict[str, Any] = {"meta": adapter_meta(base), "state": adapter_state(base)}
    if extra:
        blob["run"] = dict(extra)
    torch.save(blob, path)


def adapter_run(path: str | pathlib.Path) -> dict[str, Any]:
    """The "run" metadata of a save_adapters file ({} when none was given)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return dict(blob.get("run", {}))


def read_adapters(base: Base, path: str | pathlib.Path) -> None:
    """Loads a save_adapters file into `base`, refusing one made for another setup."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    check_adapter_meta(blob.get("meta", {}), adapter_meta(base))
    load_adapters(base, blob["state"])


def check_row(record: dict[str, Any]) -> None:
    """Refuses a row moshi would index out of range or run past its context."""
    t = len(record["text"])
    if t > CONTEXT:
        raise ValueError(f"row has {t} frames, over moshi's CONTEXT of {CONTEXT}")
    if (
        min(record["text"], default=0) < 0
        or max(record["text"], default=0) >= TEXT_CARD
    ):
        raise ValueError(f"text ids outside [0, {TEXT_CARD})")
    for stream in ("own", "pupil"):
        rows = record[stream]
        if len(rows) != t:
            raise ValueError(f"{stream} has {len(rows)} frames, text has {t}")
        flat = [c for frame in rows for c in frame]
        if any(len(frame) != CODEBOOKS for frame in rows):
            raise ValueError(f"{stream} frames are not {CODEBOOKS} codes wide")
        if flat and (min(flat) < 0 or max(flat) >= AUDIO_CARD):
            raise ValueError(f"{stream} ids outside [0, {AUDIO_CARD})")


def codes_from_row(record: dict[str, Any], device: str = "cuda") -> torch.Tensor:
    """[1, 17, T] long: text, own 8, pupil 8. Undelayed; LMModel.forward delays."""
    check_row(record)
    text = torch.tensor(record["text"], dtype=torch.long)
    own = torch.tensor(record["own"], dtype=torch.long).t()
    pupil = torch.tensor(record["pupil"], dtype=torch.long).t()
    codes = torch.cat([text[None], own, pupil], dim=0)
    if codes.shape != (STREAMS, len(record["text"])):
        raise ValueError(f"codes {tuple(codes.shape)}, not [{STREAMS}, T]")
    return codes[None].to(device)


@dataclass
class Forward:
    text_logits: torch.Tensor  # [B, T, 32000] fp32, NaN where moshi masks
    text_mask: torch.Tensor  # [B, T] bool
    audio_logits: torch.Tensor  # [B, 8, T, 2048] (moshi's own streams only)
    audio_mask: torch.Tensor  # [B, 8, T] bool


def forward(lm: Any, codes: torch.Tensor) -> Forward:
    # The one place the mode is fixed. Train mode, always: in moshi 0.2.13 the only
    # thing `training` changes in the lm is ActivationGating, which in eval takes a
    # compiled kernel and in train runs eagerly (no_compile). attach_measured leaves
    # adapted models in train mode, so bare models must run it too, or bare-vs-adapted
    # (KL) would differ by compilation. No dropout exists in the lm (SDPA dropout_p=0.0,
    # no nn.Dropout) and activation checkpointing is a constructor flag, off by default.
    lm.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lm(codes)
    return Forward(
        text_logits=out.text_logits[:, 0].float(),
        text_mask=out.text_mask[:, 0],
        audio_logits=out.logits[:, :CODEBOOKS].float(),
        audio_mask=out.mask[:, :CODEBOOKS],
    )
