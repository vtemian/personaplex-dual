import math

import pytest
import torch

from dual.constants import PAD
from dual.kl import answer_kl, bar, kl_per_frame, log_probs


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
