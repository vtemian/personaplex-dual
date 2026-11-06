import json
import pathlib

import pytest

from dual.sbs_report import (
    first_word_table,
    judgments_table,
    load,
    main,
    onset,
    pair_stats,
    quality_table,
    similarity,
    stats_table,
    turn_blocks,
)


def row(model, stem, turn, rendering, answer, pupil="what is it", reference="ref"):
    return {
        "model": model,
        "stem": stem,
        "turn": turn,
        "rendering": rendering,
        "pupil": pupil,
        "reference": reference,
        "answer": answer,
        "ids": [],
        "prefix": 1,
        "window": 2,
    }


ROWS = [
    row("bare", "s1", 1, "typed", "A half."),
    row("bare", "s1", 1, "A", "a half"),
    row("bare", "s1", 1, "B", "something else entirely"),
    row("bare", "s1", 2, "typed", ""),
    row("bare", "s1", 2, "A", ""),
    row("bare", "s1", 2, "B", "x | y\nz"),
    row("trained", "s1", 1, "typed", "one two"),
    row("trained", "s1", 1, "A", "one two"),
]


def test_load(tmp_path: pathlib.Path):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n")
    assert load(p) == ROWS


def test_similarity():
    assert similarity("", "...") == 1.0
    assert similarity("a b", "A, b") == 1.0
    assert similarity("a b", "") == 0.0
    assert 0 < similarity("a b c d", "a b c e") < 1


def test_pair_stats():
    # two silent answers are not an agreement: counted apart, left out of the rates
    s = pair_stats(ROWS, "bare", "typed", "A")
    assert s["n"] == 2
    assert s["both_empty"] == 1 and s["one_empty"] == 0 and s["said"] == 1
    assert s["exact"] == 1
    assert s["exact_rate"] == 1.0
    assert s["empty_p"] == 1 and s["empty_q"] == 1
    assert s["sim_ge_0_8"] == 1
    t = pair_stats(ROWS, "bare", "typed", "B")
    assert t["both_empty"] == 0 and t["one_empty"] == 1 and t["said"] == 2
    assert t["exact"] == 0 and t["median_sim"] == pytest.approx(
        (similarity("A half.", "something else entirely") + 0.0) / 2
    )
    assert t["empty_p"] == 1 and t["empty_q"] == 0


def test_pair_stats_all_silent():
    rows = [row("m", "s", 0, "typed", ""), row("m", "s", 0, "A", "")]
    s = pair_stats(rows, "m", "typed", "A")
    assert s["n"] == 1 and s["both_empty"] == 1 and s["said"] == 0
    assert s["exact"] == 0 and s["exact_rate"] is None and s["median_sim"] is None
    out = stats_table(rows, ["m"]).splitlines()
    assert "| m | typed vs A | 1 | 1 | 0 | 0 | - | - | 0 |" in out[2]


def test_pair_stats_none():
    s = pair_stats(ROWS, "trained", "A", "B")
    assert s["n"] == 0 and s["exact_rate"] is None and s["median_sim"] is None


def test_duplicates_raise():
    with pytest.raises(ValueError):
        pair_stats([*ROWS, ROWS[0]], "bare", "typed", "A")
    with pytest.raises(ValueError):
        turn_blocks([*ROWS, ROWS[0]], ["bare"])


def test_stats_table():
    out = stats_table(ROWS, ["bare", "trained"]).splitlines()
    assert out[0].startswith(
        "| model | pair | n | both silent | one silent | exact | exact rate"
    )
    assert len(out) == 2 + 6
    assert "| bare | typed vs A | 2 | 1 | 0 | 1 | 1.00 |" in out[2]


def test_turn_blocks():
    out = turn_blocks(ROWS, ["bare", "trained"])
    assert "### s1 turn 1" in out and "### s1 turn 2" in out
    assert "**Pupil:** what is it" in out
    assert "**Tape tutor:** ref" in out
    assert "| rendering | bare | trained |" in out
    assert "| typed | (silence) | (silence) |" not in out
    assert "x \\| y z" in out
    assert "| B | something else entirely |" in out
    assert out.index("turn 1") < out.index("turn 2")


def test_judgments_table():
    j = [
        {
            "model": "bare",
            "stem": "s1",
            "turn": 1,
            "typed_vs_A": "same",
            "A_vs_B": "differs",
        },
        {
            "model": "bare",
            "stem": "s1",
            "turn": 2,
            "typed_vs_A": "differs",
            "A_vs_B": "differs",
        },
    ]
    out = judgments_table(j, ["bare", "trained"]).splitlines()
    assert out[0] == "| model | turns | typed vs A differs | A vs B differs |"
    assert out[2] == "| bare | 2 | 1 | 2 |"
    assert out[3] == "| trained | 0 | 0 | 0 |"


def test_main(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS))
    jp = tmp_path / "j.jsonl"
    jp.write_text(
        json.dumps(
            {
                "model": "bare",
                "stem": "s1",
                "turn": 1,
                "typed_vs_A": "same",
                "A_vs_B": "same",
            }
        )
    )
    main([str(p), "--judgments", str(jp), "--models", "bare"])
    out = capsys.readouterr().out
    assert out.index("typed vs A") < out.index("A vs B differs") < out.index("### s1")
    assert "trained" not in out
    main([str(p)])
    assert "| rendering | bare | trained |" in capsys.readouterr().out


def qrow(model, rendering, answer, ids, turn=1, stem="s1"):
    r = row(model, stem, turn, rendering, answer)
    r["ids"] = ids
    return r


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def test_onset():
    assert onset([3, 3, 1, 5]) == 2
    assert onset([7, 3]) == 0
    assert onset([3, 3, 3]) is None
    assert onset([]) is None


def test_quality_table():
    rep = "a b c d a b c d a b c d."
    rows = [
        qrow("m", "typed", "Hello there.", [3, 3, 3, 9], 1),
        qrow("m", "typed", "Why not?", [1, 9], 2),
        qrow("m", "typed", "", [3, 3], 3),
        qrow("m", "typed", "and then", [3, 3, 3, 3, 4], 4),
        qrow("m", "A", rep, [3, 4], 1),
        qrow("m", "typed-swap", "one two three!", [3, 3, 5], 1),
        qrow("zzz", "typed", "x.", [3, 3, 3, 3, 3, 8], 1),
    ]
    t = quality_table(rows, ["m"])
    assert "zzz" not in t
    head = _cells(t.splitlines()[0])
    assert head == [
        "model", "rendering", "n", "silent", "onset 0", "onset 1", "onset 2",
        "onset 3", "onset other", "median words", "repetitive", "unfinished",
    ]  # fmt: skip
    lines = [ln for ln in t.splitlines() if ln.startswith("| m ")]
    assert [_cells(ln)[1] for ln in lines] == ["typed", "A", "typed-swap"]
    # typed: 4 rows, 1 silent; onsets 0:1 (the <s> row) 3:1 other:1; words 2,2,2
    assert _cells(lines[0])[2:] == ["4", "1", "1", "0", "0", "1", "1", "2", "0", "1"]
    # A: one answer with the 4-gram "a b c d" three times, ends with "."
    assert _cells(lines[1])[2:] == ["1", "0", "0", "1", "0", "0", "0", "12", "1", "0"]
    assert _cells(lines[2])[2:] == ["1", "0", "0", "0", "1", "0", "0", "3", "0", "0"]


def test_first_word_table():
    rows = [
        qrow("m", "typed", "Well, yes.", [], 1),
        qrow("m", "A", "well no", [], 1),
        qrow("m", "B", "", [], 1),
        qrow("m", "typed", "so", [], 2),
        qrow("m", "A", "Then", [], 2),
        qrow("m", "B", "then", [], 2),
    ]
    t = first_word_table(rows, ["m"])
    got = [_cells(ln) for ln in t.splitlines() if ln.startswith("| m ")]
    assert got == [
        ["m", "typed vs A", "2", "1"],
        ["m", "A vs B", "1", "1"],
        ["m", "typed vs B", "1", "0"],
    ]


def test_unknown_renderings_ignored():
    rows = [*ROWS, row("bare", "s1", 1, "typed-swap", "swapped")]
    assert stats_table(rows, ["bare"]) == stats_table(ROWS, ["bare"])
    assert turn_blocks(rows, ["bare"]) == turn_blocks(ROWS, ["bare"])


def test_main_quality(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]):
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS))
    main([str(p), "--quality"])
    out = capsys.readouterr().out
    assert "repetitive" in out and "typed vs A" in out and "### s1" in out
    main([str(p)])
    assert "repetitive" not in capsys.readouterr().out
