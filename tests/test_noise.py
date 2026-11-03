import json
import pathlib

import numpy as np
import pytest

from dual.noise import (
    lowest_gap,
    main,
    p90_ratio,
    paired_ci,
    pick,
    ratio,
    read_epochs,
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
