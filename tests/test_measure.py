import pathlib

import pytest
import torch

from dual.constants import PAD
from dual.jobs.measure import PAIRS, label, pair_fields, select_ckpts, summarise
from dual.kl import log_probs


def kl(content, all_frames):
    return {
        "content": content,
        "all_frames": all_frames,
        "content_frames": 3,
        "frames": 5,
    }


def line(g, f, d, t):
    return {
        "gap": kl(g, g / 2),
        "floor": kl(f, f / 2),
        "door": kl(d, d / 2),
        "tape_gap": kl(t, t / 2),
        "tape_floor": kl(0.1, 0.05),
    }


def test_summarise_reads_the_bar_on_content_and_on_all_frames():
    lines = [
        line(1.0, 0.5, 0.4, 2.0),
        line(3.0, 0.7, 0.6, 4.0),
        line(2.0, 0.6, 0.5, 3.0),
    ]
    s = summarise(lines, epoch=2, base="moshiko", run="r", slice_=4)
    assert (s["run"], s["base"], s["slice"], s["epoch"]) == ("r", "moshiko", 4, 2)
    assert s["content"]["median_gap"] == pytest.approx(2.0)
    assert s["content"]["median_floor"] == pytest.approx(0.6)
    assert s["all"]["median_gap"] == pytest.approx(1.0)
    assert s["content"]["turns"] == 3
    assert s["content"]["passes"] is False
    assert s["door_median"] == pytest.approx(0.5)
    assert s["door_all_median"] == pytest.approx(0.25)
    assert s["tape_gap_median"] == pytest.approx(3.0)
    assert s["tape_floor_median"] == pytest.approx(0.1)


def test_pair_fields_per_frame_agrees_with_the_means():
    g = torch.Generator().manual_seed(0)
    lps = {
        r: log_probs(torch.randn(6, 11, generator=g))
        for r in ("typed", "A", "B", "C", "tape")
    }
    targets = [PAD, PAD, 5, PAD, 7, 8]
    out = pair_fields(lps, targets)
    assert out["content_mask"] == [0, 0, 1, 0, 1, 1]
    mask = torch.tensor(out["content_mask"], dtype=torch.bool)
    for key, _, _ in PAIRS:
        pf = torch.tensor(out["per_frame"][key])
        assert len(pf) == len(targets)
        assert float(pf[mask].mean()) == pytest.approx(out[key]["content"], rel=1e-5)
        assert float(pf.mean()) == pytest.approx(out[key]["all_frames"], rel=1e-5)
        assert out[key]["frames"] == 6 and out[key]["content_frames"] == 3
        assert all(float(f"{x:.6g}") == x for x in out["per_frame"][key])
    assert set(out["per_frame"]) == {k for k, _, _ in PAIRS}


def test_select_ckpts_by_filename_epoch():
    paths = [pathlib.Path(f"d/model-epoch-{n:02d}.pt") for n in (3, 1, 2)]
    assert [p.name for p in select_ckpts(paths, None)] == [
        "model-epoch-01.pt",
        "model-epoch-02.pt",
        "model-epoch-03.pt",
    ]
    assert [p.name for p in select_ckpts(paths, [3, 2])] == [
        "model-epoch-02.pt",
        "model-epoch-03.pt",
    ]
    with pytest.raises(SystemExit, match="epoch 7"):
        select_ckpts(paths, [2, 7])


def test_label_appends_run_then_tag():
    assert label("b", None, None) == "b"
    assert label("b", "r", None) == "b-r"
    assert label("b", None, "pos") == "b-pos"
    assert label("b", "r", "pos") == "b-r-pos"
