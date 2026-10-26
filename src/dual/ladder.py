"""Nested slices of the training stems for the data ladder."""

from __future__ import annotations

import math
from collections.abc import Sequence

from dual.holdout import ordered

RUNGS = (8, 4, 2, 1)  # denominators: an eighth, a quarter, a half, all


def rung(train_stems: Sequence[str], denominator: int) -> list[str]:
    if denominator not in RUNGS:
        raise ValueError(f"denominator {denominator}: one of {RUNGS}")
    by_hash = ordered(train_stems, "ladder")
    return sorted(by_hash[: math.ceil(len(by_hash) / denominator)])
