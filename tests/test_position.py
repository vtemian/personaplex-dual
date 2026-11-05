import json

import numpy as np
import pytest

from dual.position import (
    BUCKETS,
    CUTS,
    MODES,
    bar_from,
    bucket,
    bucket_table,
    main,
    positions,
    render,
    turn_means,
)


def make_line(stem, mask, gap, floor, door):
    """A measure line whose scalars are the means of per_frame, as measure computes."""

    def scalars(x):
        m = np.asarray(mask, dtype=bool)
        v = np.asarray(x, dtype=float)
        return {"content": float(v[m].mean()), "all_frames": float(v.mean())}

    return {
        "stem": stem,
        "gap": scalars(gap),
        "floor": scalars(floor),
        "door": scalars(door),
        "content_mask": mask,
        "per_frame": {
            "gap": gap,
            "floor": floor,
            "door": door,
            "tape_gap": gap,
            "tape_floor": floor,
        },
    }


def synthetic(n=7, frames=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f0 = int(rng.integers(0, 6))
        mask = [0] * f0 + [1 if rng.random() < 0.7 else 0 for _ in range(frames - f0)]
        mask[f0] = 1
        g, f, d = (rng.random(frames).round(4).tolist() for _ in range(3))
        out.append(make_line(f"s{i}", mask, g, f, d))
    return out


def test_bucket_boundaries():
    cases = {-3: "lead", 0: "0", 1: "1", 2: "2", 3: "3-5", 5: "3-5", 6: "6-10"}
    cases |= {10: "6-10", 11: "11-25", 25: "11-25", 26: "26+", 99: "26+"}
    for pos, want in cases.items():
        assert bucket(pos) == want
    assert set(cases.values()) == set(BUCKETS)


def test_positions_count_from_the_first_content_frame():
    assert positions([0, 0, 1, 0, 1]) == [-2, -1, 0, 1, 2]
    with pytest.raises(ValueError):
        positions([0, 0, 0])


def test_turn_means_by_hand():
    mask = [0, 0] + [1] * 28  # f0 = 2, positions -2..27
    kl = [10.0, 20.0] + [float(p) for p in range(28)]  # kl at position p is p
    all_ = turn_means(kl, mask, content_only=False)
    assert all_["lead"] == 15.0
    assert all_["2"] == 2.0
    assert all_["3-5"] == 4.0
    assert all_["6-10"] == 8.0
    assert all_["11-25"] == 18.0
    assert all_["26+"] == 26.5
    cont = turn_means(kl, mask, content_only=True)
    assert cont["lead"] is None
    assert cont["0"] == 0.0
    # a PAD frame inside the answer is dropped in content mode only
    mask2 = [1, 1, 0, 1]
    assert turn_means([1.0, 1.0, 9.0, 3.0], mask2, content_only=True)["2"] is None
    assert turn_means([1.0, 1.0, 9.0, 3.0], mask2, content_only=True)["3-5"] == 3.0
    assert turn_means([1.0, 1.0, 9.0, 3.0], mask2, content_only=False)["2"] == 9.0
    with pytest.raises(ValueError):
        turn_means([1.0], [1, 1], content_only=True)


def test_bar_from_reproduces_the_scalar_bars():
    lines = synthetic()
    c = bar_from(lines, 0, "content")
    a = bar_from(lines, None, "all")
    for got, mode, key in [(c, "content", "content"), (a, "all", "all_frames")]:
        for pair in ("gap", "floor"):
            scal = [x[pair][key] for x in lines]
            assert got[f"median_{pair}"] == pytest.approx(np.median(scal))
            assert got[f"p90_{pair}"] == pytest.approx(np.percentile(scal, 90))
        assert got["ratio"] == pytest.approx(got["median_gap"] / got["median_floor"])
        assert got["turns"] == len(lines), mode
    assert c["frames"] == sum(sum(x["content_mask"]) for x in lines)
    assert bar_from(lines, None, "content")["median_gap"] == c["median_gap"]


def test_lead_has_no_content_frames_and_shares_sum_to_one():
    lines = synthetic()
    for mode in MODES:
        rows = {r["bucket"]: r for r in bucket_table(lines, mode)}
        assert list(rows) == list(BUCKETS)
        for pair in ("gap", "floor"):
            assert sum(r[f"{pair}_share"] for r in rows.values()) == pytest.approx(1)
    content = {r["bucket"]: r for r in bucket_table(lines, "content")}
    assert content["lead"]["frames"] == 0
    assert content["lead"]["turns"] == 0
    assert content["lead"]["median_gap"] is None
    assert content["lead"]["ratio"] is None
    assert content["lead"]["gap_share"] == 0
    assert sum(r["frames"] for r in content.values()) == sum(
        sum(x["content_mask"]) for x in lines
    )


def test_bucket_table_medians_over_turns():
    mask = [1] * 4
    lines = [
        make_line("a", mask, [1, 1, 1, 1], [2, 2, 2, 2], [0, 0, 0, 0]),
        make_line("b", mask, [3, 3, 3, 3], [2, 2, 2, 2], [0, 0, 0, 0]),
        make_line("c", mask, [5, 5, 5, 5], [4, 4, 4, 4], [0, 0, 0, 0]),
    ]
    row = {r["bucket"]: r for r in bucket_table(lines, "content")}["3-5"]
    assert (row["turns"], row["frames"]) == (3, 3)
    assert row["median_gap"] == 3 and row["median_floor"] == 2
    assert row["ratio"] == 1.5 and row["gap_share"] == pytest.approx(9 / 36)


def test_cuts_skip_turns_without_frames():
    lines = [
        make_line("a", [1] * 4, [1.0] * 4, [1.0] * 4, [0.0] * 4),
        make_line("b", [1] * 20, [2.0] * 20, [1.0] * 20, [0.0] * 20),
    ]
    assert CUTS == (None, 0, 1, 2, 3, 6, 11)
    b = bar_from(lines, 11, "content")
    assert (b["turns"], b["frames"], b["median_gap"]) == (1, 9, 2.0)


def test_render_and_cli(tmp_path, capsys):
    lines = synthetic()
    md = render(lines, "run/epoch-02.jsonl")
    assert md.startswith("### run/epoch-02.jsonl")
    assert md.count("| bucket |") == 2 and "| from position |" in md
    assert "-" in md
    d = tmp_path / "run"
    d.mkdir()
    (d / "epoch-02.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    main([str(d / "epoch-02.jsonl")])
    assert capsys.readouterr().out.startswith("### run/epoch-02.jsonl")
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
