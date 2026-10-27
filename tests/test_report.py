import json

from dual.report import best, main, table


def bar(mg, mf, pg, pf, turns=65):
    return {
        "median_gap": mg,
        "median_floor": mf,
        "p90_gap": pg,
        "p90_floor": pf,
        "turns": turns,
    }


def summary(run, epoch, mg, mf, pg, pf, slice_=8):
    return {
        "run": run,
        "base": "personaplex",
        "slice": slice_,
        "epoch": epoch,
        "content": bar(mg, mf, pg, pf),
        "all": bar(mg / 2, mf / 2, pg / 2, pf / 2),
        "door_median": 0.25,
        "tape_gap_median": 1.5,
    }


BARE = summary(None, 0, 2.0, 0.5, 4.0, 1.0, slice_=None)
E1 = summary("s", 1, 1.0, 0.5, 2.0, 1.0)
E2 = summary("s", 2, 0.4, 0.5, 0.9, 1.0)


def test_table_has_the_columns_and_one_line_per_summary():
    out = table([BARE, E1]).splitlines()
    assert out[0] == (
        "| run | base | slice | epoch | median gap | median floor | p90 gap |"
        " p90 floor | pass | door | tape gap |"
    )
    assert out[1].startswith("|---")
    assert len(out) == 4
    assert out[2].startswith("| bare | personaplex | - | 0 | 2.000 | 0.500 |")
    assert out[3].startswith("| s | personaplex | 1/8 | 1 | 1.000 |")


def test_table_marks_a_pass_only_when_median_and_p90_are_both_under():
    rows = table([E1, E2]).splitlines()[2:]
    assert "| no |" in rows[0]
    assert "| yes |" in rows[1]
    half = summary("h", 1, 0.4, 0.5, 2.0, 1.0)
    assert "| no |" in table([half]).splitlines()[2]


def test_table_reads_the_all_frames_numbers_when_asked():
    row = table([BARE], key="all").splitlines()[2]
    assert "| 1.000 | 0.250 | 2.000 | 0.500 |" in row


def test_door_and_tape_gap_are_the_last_columns():
    assert table([BARE]).splitlines()[2].endswith("| 0.250 | 1.500 |")


def test_best_picks_per_run_the_smallest_gap_to_floor_ratio():
    picked = best([E1, E2, BARE])
    assert [(s["run"], s["epoch"]) for s in picked] == [(None, 0), ("s", 2)]


def test_main_reads_every_summary_under_runs(tmp_path, capsys):
    for name, entries in {"personaplex": [BARE], "personaplex-s": [E1, E2]}.items():
        d = tmp_path / "measure" / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(entries))
    main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "content frames" in out and "all frames" in out and "best epoch" in out
    assert out.count("| s | personaplex | 1/8 | 2 |") == 3


def test_read_refuses_two_summaries_for_one_run_base_epoch(tmp_path):
    import pytest

    from dual.report import read

    for name in ("personaplex-s", "personaplex-s2"):
        d = tmp_path / "measure" / name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps([E1]))
    with pytest.raises(ValueError, match=r"personaplex-s.*personaplex-s2"):
        read(tmp_path)


def test_best_prefers_a_passing_epoch_over_a_smaller_ratio():
    low_ratio_fails_p90 = summary("s", 1, 0.1, 0.5, 2.0, 1.0)
    passes = summary("s", 2, 0.4, 0.5, 0.9, 1.0)
    picked = best([low_ratio_fails_p90, passes])
    assert [s["epoch"] for s in picked] == [2]


def test_best_falls_back_to_smallest_ratio_when_none_pass():
    a = summary("s", 1, 1.0, 0.5, 2.0, 1.0)
    b = summary("s", 2, 0.8, 0.5, 2.0, 1.0)
    assert best([a, b])[0]["epoch"] == 2


def test_best_survives_a_zero_floor():
    zero = summary("s", 1, 0.0, 0.0, 0.0, 0.0)
    other = summary("s", 2, 1.0, 0.5, 2.0, 1.0)
    assert best([zero, other])[0]["epoch"] == 1  # 0 <= 0 passes; other does not
    bad = summary("s", 1, 0.3, 0.0, 2.0, 1.0)
    assert best([bad, other])[0]["epoch"] == 2  # inf vs 2.0
