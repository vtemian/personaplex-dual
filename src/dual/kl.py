"""The number: per-frame KL between two renderings' text distributions, and the bar."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dual.constants import PAD


def log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def kl_per_frame(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per frame; both [T, V] log-probs over the same frames."""
    if lp_p.shape != lp_q.shape:
        raise ValueError(f"shapes {tuple(lp_p.shape)} vs {tuple(lp_q.shape)}")
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


DIRECTIONS = ("bar", "forward")


def consistency_loss(
    student_logits: torch.Tensor, teacher_lp: torch.Tensor, direction: str = "bar"
) -> torch.Tensor:
    """Mean per-frame KL between the student's text distribution and a fixed teacher's.

    student_logits [N, V] (gradient flows); teacher_lp [N, V] log-probs, detached here.
    "bar": KL(student || teacher), the bar's own direction (gap = KL(typed || A)).
    "forward": KL(teacher || student). No frames: a zero that still backpropagates."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}: one of {DIRECTIONS}")
    if student_logits.shape != teacher_lp.shape:
        raise ValueError(
            f"shapes {tuple(student_logits.shape)} vs {tuple(teacher_lp.shape)}"
        )
    if student_logits.shape[0] == 0:
        return student_logits.float().sum() * 0.0
    lp_s = log_probs(student_logits)
    lp_t = teacher_lp.detach().float()
    if direction == "bar":
        return kl_per_frame(lp_s, lp_t).mean()
    return kl_per_frame(lp_t, lp_s).mean()


@dataclass(frozen=True)
class AnswerKL:
    content: float  # mean over frames whose target text is not PAD
    all_frames: float
    content_frames: int
    frames: int


def answer_kl(
    lp_p: torch.Tensor, lp_q: torch.Tensor, targets: Sequence[int]
) -> AnswerKL:
    """lp_* are the answer window's log-probs [W, V]; targets the window's text ids."""
    per = kl_per_frame(lp_p, lp_q)
    if per.shape[0] != len(targets):
        raise ValueError(f"{per.shape[0]} frames of KL, {len(targets)} targets")
    content = torch.tensor([t != PAD for t in targets])
    if not bool(content.any()):
        raise ValueError("an answer with no content frame")
    return AnswerKL(
        content=float(per[content].mean()),
        all_frames=float(per.mean()),
        content_frames=int(content.sum()),
        frames=int(per.shape[0]),
    )


@dataclass(frozen=True)
class Bar:
    median_gap: float
    median_floor: float
    p90_gap: float
    p90_floor: float
    turns: int

    @property
    def passes(self) -> bool:
        return self.median_gap <= self.median_floor and self.p90_gap <= self.p90_floor


def bar(gaps: Sequence[float], floors: Sequence[float]) -> Bar:
    if len(gaps) != len(floors) or not gaps:
        raise ValueError("gaps and floors are one number per holdout turn")
    g: Any = np.asarray(gaps, dtype=float)
    f: Any = np.asarray(floors, dtype=float)
    return Bar(
        median_gap=float(np.median(g)),
        median_floor=float(np.median(f)),
        p90_gap=float(np.percentile(g, 90)),
        p90_floor=float(np.percentile(f, 90)),
        turns=len(gaps),
    )
