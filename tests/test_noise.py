import json
import pathlib

import numpy as np
import pytest

from dual.noise import (
    FLOOR_INFLATED,
    floor_guard,
    lowest_gap,
    main,
    p90_ratio,
    paired_ci,
    pick,
    ratio,
    read_epochs,
    report,
    verdict,
)

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "noise_pp1_epoch01.jsonl"


def line(stem, turn, gap, floor, epoch=1):
    return {
        "stem": stem,
        "turn": turn,
        "epoch": epoch,
        "gap": {"content": gap},
        "floor": {"content": floor},
    }


def run(gaps, floors, epoch=1, per_stem=2):
    return [
        line(f"s{i // per_stem}", i % per_stem, g, f, epoch)
        for i, (g, f) in enumerate(zip(gaps, floors, strict=True))
    ]


GAPS = [0.5, 1.0, 2.0, 4.0, 8.0, 3.0, 1.5, 6.0]
FLOORS = [0.5, 0.4, 0.6, 1.0, 0.8, 1.2, 0.5, 2.0]


def test_ratio_and_p90_match_numpy():
    lines = run(GAPS, FLOORS)
    assert ratio(lines) == pytest.approx(np.median(GAPS) / np.median(FLOORS))
    assert p90_ratio(lines) == pytest.approx(
        np.percentile(GAPS, 90) / np.percentile(FLOORS, 90)
    )


def test_pick_prefers_a_passing_epoch_then_the_smallest_ratio():
    fail_small = run([1.0] * 4, [0.5] * 4)  # ratio 2, fails
    passing = run([0.4] * 4, [0.5] * 4)  # ratio 0.8, passes
    worse_pass = run([0.5] * 4, [0.5] * 4)  # ratio 1.0, passes
    assert pick({1: fail_small, 2: passing, 3: worse_pass}) == (2, pytest.approx(0.8))
    assert pick({1: fail_small, 2: run([3.0] * 4, [0.5] * 4)})[0] == 1


def test_pick_and_lowest_gap_ignore_epoch_4_and_later():
    e1 = run([2.0] * 4, [0.5] * 4)
    e4 = run([0.1] * 4, [0.5] * 4, epoch=4)
    assert pick({1: e1, 4: e4}) == (1, pytest.approx(4.0))
    assert lowest_gap({1: e1, 4: e4}) == (1, pytest.approx(4.0))
    assert pick({1: e1, 4: e4}, max_epoch=4)[0] == 4


def test_lowest_gap_is_the_epoch_with_the_smallest_median_gap():
    e1 = run([2.0] * 4, [0.5] * 4)  # gap 2, ratio 4
    e2 = run([1.0] * 4, [0.1] * 4)  # gap 1, ratio 10
    e3 = run([1.5] * 4, [0.5] * 4)
    assert lowest_gap({1: e1, 2: e2, 3: e3}) == (2, pytest.approx(10.0))


def test_pick_with_no_epoch_in_range_raises():
    with pytest.raises(ValueError):
        pick({4: run([1.0] * 4, [1.0] * 4)})


def test_paired_ci_of_a_run_against_itself_is_zero():
    lines = run(GAPS, FLOORS)
    assert paired_ci(lines, lines) == (0.0, 0.0, 0.0)


def test_paired_ci_of_a_shifted_run_excludes_zero():
    a = run([g + 3.0 for g in GAPS] * 3, FLOORS * 3, per_stem=4)
    b = run(GAPS * 3, FLOORS * 3, per_stem=4)
    delta, lo, hi = paired_ci(a, b)
    assert delta == pytest.approx(ratio(a) - ratio(b))
    assert lo > 0 and hi >= lo


def test_paired_ci_is_seeded():
    a = run(GAPS, FLOORS)
    b = run([g * 1.1 for g in GAPS], FLOORS)
    assert paired_ci(a, b, seed=3) == paired_ci(a, b, seed=3)


def test_paired_ci_raises_on_a_stem_missing_from_either_side():
    a = run(GAPS, FLOORS)
    with pytest.raises(ValueError, match="s3"):
        paired_ci(a, a[:6])
    with pytest.raises(ValueError, match="s3"):
        paired_ci(a[:6], a)


def test_verdict_covers_all_three_outcomes():
    excl_neg = (-1.0, -1.5, -0.5)
    excl_pos = (1.0, 0.5, 1.5)
    straddle = (-1.0, -2.0, 0.3)
    # seeds 3.0 and 3.2: R0 = 3.1
    assert verdict(2.0, 3.0, 3.2, excl_neg, excl_neg, 0.5) == "down"
    assert verdict(4.2, 3.0, 3.2, excl_pos, excl_pos, 0.5) == "up"
    # big enough, but one interval includes 0
    assert verdict(2.0, 3.0, 3.2, excl_neg, straddle, 0.5) == "not moved"
    # intervals clear, but under the threshold
    assert verdict(2.9, 3.0, 3.2, excl_neg, excl_neg, 0.5) == "not moved"
    # opposite signs
    assert verdict(2.0, 3.0, 3.2, excl_neg, excl_pos, 0.5) == "not moved"


def test_read_epochs_and_the_real_ladder_1_lines(tmp_path):
    d = tmp_path / "measure" / "lab"
    d.mkdir(parents=True)
    (d / "epoch-01.jsonl").write_text(FIXTURE.read_text())
    (d / "epoch-02.jsonl").write_text("")
    (d / "notes.txt").write_text("x")
    by_epoch = read_epochs(tmp_path, "lab")
    assert sorted(by_epoch) == [1]
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines()]
    g = [r["gap"]["content"] for r in rows]
    f = [r["floor"]["content"] for r in rows]
    assert ratio(by_epoch[1]) == pytest.approx(np.median(g) / np.median(f))
    with pytest.raises(FileNotFoundError):
        read_epochs(tmp_path, "nope")


def test_main_prints_the_noise_line_and_a_table(tmp_path, capsys):
    for label, scale in (("s0", 1.0), ("s1", 1.1), ("v", 0.2)):
        d = tmp_path / "measure" / label
        d.mkdir(parents=True)
        lines = run([g * scale for g in GAPS] * 3, FLOORS * 3, per_stem=4)
        (d / "epoch-01.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    main([str(tmp_path), "--seeds", "s0", "s1", "--variants", "v"])
    out = capsys.readouterr().out
    assert "d = " in out and "R0 = " in out and "T = " in out
    rows = [x for x in out.splitlines() if x.startswith("| ")]
    assert len(rows) == 1 + 3  # header, two seeds, one variant
    assert rows[-1].startswith("| v ")
    assert rows[-1].rstrip().endswith("down |")


def test_ratio_and_paired_ci_refuse_a_non_finite_ratio():
    zero_floor = run(GAPS, [0.0] * 8)
    with pytest.raises(ValueError, match="finite"):
        ratio(zero_floor)
    with pytest.raises(ValueError, match="finite"):
        paired_ci(zero_floor, run(GAPS, FLOORS))


def test_paired_ci_refuses_a_resample_whose_median_floor_is_zero():
    # the point ratio is finite, but a resample of the stems can pick only zero floors
    floors = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    a, b = run(GAPS, floors), run(GAPS, FLOORS)
    assert np.isfinite(ratio(a))
    with pytest.raises(ValueError, match="finite"):
        paired_ci(a, b, draws=500)


def test_report_gives_seed_rows_no_verdict_and_no_self_ci():
    runs = {}
    for label, scale in (("s0", 1.0), ("s1", 1.1), ("v", 0.2)):
        runs[label] = {1: run([g * scale for g in GAPS] * 3, FLOORS * 3, per_stem=4)}
    rows = report(runs, ["s0", "s1"], ["v"]).splitlines()[4:]
    cells = [[c.strip() for c in x.strip("|").split("|")] for x in rows]
    assert [c[0] for c in cells] == ["s0", "s1", "v"]
    s0, s1, v = cells
    assert s0[-1] == "-" and s1[-1] == "-"
    assert s0[7] == "-" and s1[8] == "-"  # own self-comparison
    assert s0[8] != "-" and s1[7] != "-"  # the cross-seed intervals stay
    assert "+0.00 [+0.00, +0.00]" not in " ".join(s0 + s1)
    assert v[-1] == "down" and v[7] != "-" and v[8] != "-"


def _epochs(gaps_floors):
    """{epoch: run} from [(gap, floor), ...], one constant-valued run per epoch."""
    return {
        e: run([g] * 12, [f] * 12, epoch=e, per_stem=4)
        for e, (g, f) in enumerate(gaps_floors, start=1)
    }


def _cells(report_text, label):
    for x in report_text.splitlines():
        c = [y.strip() for y in x.strip("|").split("|")]
        if x.startswith("| ") and c[0] == label:
            return c
    raise AssertionError(label)


def test_report_decides_on_r_gap_not_on_the_best_of_three_r():
    seeds = {
        "s0": _epochs([(1.0, 0.5), (2.0, 0.4), (3.0, 0.3)]),  # R_gap 2.0 (e1)
        "s1": _epochs([(1.1, 0.5), (2.0, 0.4), (3.0, 0.3)]),  # R_gap 2.2 (e1)
    }
    # the lowest-gap epoch (1) has the seeds' ratio; epoch 3's floor inflates and R
    # (the best of three) is far lower, 0.3, but R_gap is unchanged: not moved
    v = _epochs([(1.05, 0.5), (2.0, 0.5), (0.9, 3.0)])
    runs = {**seeds, "v": {1: v[1], 2: v[2], 3: v[3]}}
    # make epoch 3 the best-R pick yet not the lowest-gap one
    runs["v"][3] = run([1.2] * 12, [4.0] * 12, epoch=3, per_stem=4)  # R 0.3
    assert pick(runs["v"])[0] == 3 and lowest_gap(runs["v"])[0] == 1
    c = _cells(report(runs, ["s0", "s1"], ["v"]), "v")
    assert c[-1] == "not moved"
    assert c[1].endswith("(3)") and c[2].endswith("(1)")


def test_floor_guard_labels_a_down_whose_floor_is_inflated():
    assert floor_guard("down", 1.6, 1.0, 1.0) == FLOOR_INFLATED
    assert floor_guard("down", 1.5, 1.0, 1.0) == "down"  # not above 1.5x
    assert floor_guard("down", 1.0, 0.8, 1.2) == "down"
    assert floor_guard("up", 5.0, 1.0, 1.0) == "up"
    assert floor_guard("not moved", 5.0, 1.0, 1.0) == "not moved"
    assert FLOOR_INFLATED == "down, floor inflated"


def test_report_flags_floor_inflation_and_has_the_new_columns():
    seeds = {
        k: {1: run([g] * 12, [0.5] * 12, per_stem=4)}
        for k, g in (("s0", 1.0), ("s1", 1.1))
    }
    runs = {
        **seeds,
        "v": {1: run([0.4] * 12, [1.0] * 12, per_stem=4)},
    }  # R 0.4, floor 2x
    text = report(runs, ["s0", "s1"], ["v"])
    head = [c.strip() for c in text.splitlines()[2].strip("|").split("|")]
    assert head[:6] == [
        "run",
        "R (epoch)",
        "R_gap (epoch)",
        "median gap",
        "median floor",
        "p90 ratio",
    ]
    v = _cells(text, "v")
    assert v[3] == "0.400" and v[4] == "1.000"
    assert v[-1] == "down, floor inflated"
    assert _cells(text, "s0")[4] == "0.500"


def test_p90_ratio_is_read_at_the_decision_epoch():
    runs = {
        "s0": {1: run([1.0] * 12, [0.5] * 12, per_stem=4)},
        "s1": {1: run([1.0] * 12, [0.5] * 12, per_stem=4)},
        # epoch 1 has the lowest gap; epoch 2 is the best-R pick (p90 3.0 vs 4.0 differ)
        "v": {
            1: run([1.0] * 12, [0.5] * 12, per_stem=4),
            2: run([1.5] * 12, [4.0] * 12, per_stem=4),  # passes: the best-R pick
        },
    }
    c = _cells(report(runs, ["s0", "s1"], ["v"]), "v")
    assert c[5] == f"{p90_ratio(runs['v'][1]):.2f}"
