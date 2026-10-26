import pytest

from dual.jobs.measure import summarise


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
