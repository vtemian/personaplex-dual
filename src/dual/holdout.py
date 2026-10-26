"""Which dialogues are held out, the ladder's order, and which turns a row types.

All by sha256 of the stem with a salt, never by file order or run order.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

HOLDOUT_FRACTION = 0.1


def _digest(stem: str, salt: str) -> int:
    return int(hashlib.sha256(f"{stem}:{salt}".encode()).hexdigest(), 16)


def ordered(stems: Sequence[str], salt: str) -> list[str]:
    if len(set(stems)) != len(stems):
        raise ValueError("duplicate stems")
    return sorted(stems, key=lambda stem: (_digest(stem, salt), stem))


def split(stems: Sequence[str]) -> tuple[list[str], list[str]]:
    by_hash = ordered(stems, "holdout")
    held = math.ceil(HOLDOUT_FRACTION * len(by_hash))
    return by_hash[:held], sorted(by_hash[held:])


def mask_bits(stem: str, turns: int) -> tuple[int, ...]:
    digest = _digest(stem, "mask")
    return tuple((digest >> k) & 1 for k in range(turns))
