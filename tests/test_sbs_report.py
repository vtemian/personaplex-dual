import json
import pathlib

import pytest

from dual.sbs_report import (
    judgments_table,
    load,
    main,
    pair_stats,
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
