import math

import pytest
import torch

from dual.constants import PAD
from dual.kl import (
    DIRECTIONS,
    answer_kl,
    bar,
    consistency_loss,
    kl_per_frame,
    log_probs,
)


def lp(*rows: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor(rows, dtype=torch.float32))


def test_kl_of_identical_distributions_is_zero():
    p = lp([0.9, 0.1], [0.3, 0.7])
    assert torch.allclose(kl_per_frame(p, p), torch.zeros(2))


def test_kl_matches_the_hand_computation():
    got = kl_per_frame(lp([0.9, 0.1]), lp([0.5, 0.5]))
    want = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
    assert float(got[0]) == pytest.approx(want, abs=1e-6)


def test_kl_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes"):
        kl_per_frame(lp([0.5, 0.5]), lp([0.5, 0.5], [0.5, 0.5]))


def test_log_probs_is_a_fp32_log_softmax():
    out = log_probs(torch.zeros(3, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.float32
    assert torch.allclose(out.exp().sum(-1), torch.ones(3))


def test_answer_kl_averages_content_frames_only():
    p = lp([0.9, 0.1], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8])
    q = lp([0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.6, 0.4])
    per = kl_per_frame(p, q)
    got = answer_kl(p, q, [PAD, 7, PAD, 8])
    assert got.content == pytest.approx(float((per[1] + per[3]) / 2), abs=1e-7)
    assert got.all_frames == pytest.approx(float(per.mean()), abs=1e-7)
    assert (got.content_frames, got.frames) == (2, 4)
    assert got.content != pytest.approx(got.all_frames)


def test_answer_kl_refuses_an_all_pad_window():
    p = lp([0.9, 0.1], [0.5, 0.5])
    with pytest.raises(ValueError, match="no content frame"):
        answer_kl(p, p, [PAD, PAD])


def test_answer_kl_refuses_a_target_length_mismatch():
    p = lp([0.9, 0.1], [0.5, 0.5])
    with pytest.raises(ValueError, match="targets"):
        answer_kl(p, p, [7])


def test_bar_passes_when_gaps_are_below_floors():
    got = bar([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    assert got.passes
    assert got.turns == 3
    assert got.median_gap == pytest.approx(0.2)
    assert got.median_floor == pytest.approx(0.3)


def test_bar_fails_when_only_the_p90_is_above():
    gaps = [0.1, 0.1, 0.1, 0.1, 5.0]
    floors = [0.2, 0.2, 0.2, 0.2, 0.3]
    got = bar(gaps, floors)
    assert got.median_gap <= got.median_floor
    assert got.p90_gap > got.p90_floor
    assert not got.passes


def test_bar_refuses_mismatched_or_empty_input():
    with pytest.raises(ValueError, match="one number per"):
        bar([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="one number per"):
        bar([], [])


def test_bar_percentile_is_numpys_linear_interpolation():
    got = bar([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert got.median_gap == 3
    assert got.p90_gap == pytest.approx(4.6)


def _logits(seed: int, n: int = 5, v: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, v, generator=g)


def test_consistency_is_zero_when_student_equals_teacher():
    x = _logits(0)
    assert float(consistency_loss(x, log_probs(x))) == 0.0


def test_consistency_bar_direction_is_the_bars_kl():
    s, t = _logits(1), _logits(2)
    want = kl_per_frame(log_probs(s), log_probs(t)).mean()
    assert torch.allclose(consistency_loss(s, log_probs(t), "bar"), want)


def test_consistency_forward_direction_swaps_the_arguments():
    s, t = _logits(1), _logits(2)
    want = kl_per_frame(log_probs(t), log_probs(s)).mean()
    assert torch.allclose(consistency_loss(s, log_probs(t), "forward"), want)


def test_consistency_gradient_reaches_the_student_only():
    s = _logits(3).requires_grad_(True)
    t_logits = _logits(4).requires_grad_(True)
    consistency_loss(s, log_probs(t_logits)).backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
    assert t_logits.grad is None


def test_consistency_on_no_frames_is_zero_and_backpropagates():
    s = torch.zeros(0, 7, requires_grad=True)
    out = consistency_loss(s, torch.zeros(0, 7))
    out.backward()
    assert float(out.detach()) == 0.0
    assert s.grad is not None and s.grad.shape == (0, 7)


def test_consistency_refuses_bad_shapes_and_directions():
    with pytest.raises(ValueError, match="shapes"):
        consistency_loss(_logits(0, n=3), log_probs(_logits(1, n=4)))
    with pytest.raises(ValueError, match="direction"):
        consistency_loss(_logits(0), log_probs(_logits(1)), "reverse")
    assert DIRECTIONS == ("bar", "forward")
