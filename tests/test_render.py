import pytest

from dual.constants import GAP_FRAMES, PAD
from dual.frames import tape_segment, typed_segment
from dual.holdout import mask_bits
from dual.render import (
    add_rendering,
    assemble,
    check_same_tape,
    holdout_record,
    holdout_row,
    pupil_indices,
    tape_segments,
    training_rows,
    voiced_rows,
)
from dual.turns import TurnSpan

FRAMES = 60
OWN = (7,) * 8
PUPIL = (9,) * 8


def two_ask_spans() -> list[TurnSpan]:
    return [
        TurnSpan(0, "pupil", ("a", "b"), 3, 0, 10),
        TurnSpan(1, "teacher", ("c", "d"), 14, 10, 28),
        TurnSpan(2, "pupil", ("e",), 32, 28, 37),
        TurnSpan(3, "teacher", ("f", "g"), 41, 37, 60),
    ]


def fake_record() -> dict:
    return {"audio": [OWN] * FRAMES, "user_audio": [PUPIL] * FRAMES}


def fake_text() -> list[int]:
    # distinct ids on the teacher spans so a shifted copy is recognisable
    text = [PAD] * FRAMES
    for f in range(10, 28):
        text[f] = 500 + f
    for f in range(37, 60):
        text[f] = 500 + f
    return text


def typed(words):
    return [100 + len(w) for w in words]


def voiced(words, voice):
    return [[len(voice)] * 8] * (2 * len(words))


VOICES = {"A": "va", "B": "vbb", "C": "vccc"}


def base():
    return tape_segments(two_ask_spans(), fake_text(), fake_record())


def test_tape_segments_score_only_the_teacher():
    segs = base()
    assert [s.scored for s in segs] == [False, True, False, True]
    assert [s.frames for s in segs] == [10, 18, 9, 23]


def test_training_rows_kinds_and_partition():
    rows = training_rows("stem", two_ask_spans(), base(), typed)
    assert [r["kind"] for r in rows] == ["mask", "complement", "spoken"]
    mask, comp, spoken = rows
    assert sorted(mask["typed_turns"] + comp["typed_turns"]) == [0, 2]
    assert not set(mask["typed_turns"]) & set(comp["typed_turns"])
    assert spoken["typed_turns"] == []
    assert [r["id"] for r in rows] == ["stem#mask", "stem#complement", "stem#spoken"]
    assert all(r["stem"] == "stem" for r in rows)


def test_training_rows_follow_the_hash_mask():
    rows = training_rows("stem", two_ask_spans(), base(), typed)
    bits = mask_bits("stem", 2)
    assert rows[0]["typed_turns"] == [k for k, b in zip([0, 2], bits, strict=True) if b]


def test_spoken_row_is_the_tape():
    spoken = training_rows("stem", two_ask_spans(), base(), typed)[2]
    assert spoken["frames"] == FRAMES
    assert spoken["text"] == fake_text()
    assert spoken["own"] == [list(OWN)] * FRAMES
    assert spoken["pupil"] == [list(PUPIL)] * FRAMES


def test_typed_rows_have_the_typed_length():
    spans = two_ask_spans()
    for row in training_rows("stem", spans, base(), typed)[:2]:
        expected = FRAMES
        for k in row["typed_turns"]:
            expected += (
                len(spans[k].words) + GAP_FRAMES - (spans[k].end - spans[k].start)
            )
        assert row["frames"] == expected


def test_scored_is_on_teacher_spans_only():
    for row in training_rows("stem", two_ask_spans(), base(), typed):
        for span in row["spans"]:
            flags = row["scored"][span["start"] : span["end"]]
            assert set(flags) == {1 if span["party"] == "teacher" else 0}


def test_holdout_record_renders_every_pupil_turn_four_ways():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    assert rec["id"] == "stem"
    assert rec["pupil_turns"] == [0, 2]
    assert set(rec["renderings"]) == {"0", "2"}
    for per_turn in rec["renderings"].values():
        assert set(per_turn) == {"typed", "A", "B", "C"}
    assert rec["base"]["frames"] == FRAMES


def test_holdout_row_swaps_one_turn_and_shifts_the_answer():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    base_row = rec["base"]
    row, (start, end) = holdout_row(rec, 2, "A")
    assert row["spans"][2]["rendering"] == "A"
    assert row["id"] == "stem#2#A"
    answer = row["spans"][3]
    assert (start, end) == (answer["start"], answer["end"])
    b3 = base_row["spans"][3]
    assert row["text"][start:end] == base_row["text"][b3["start"] : b3["end"]]
    # A voiced 'e' is 2 frames + the gap
    assert row["frames"] == FRAMES - 9 + 2 + GAP_FRAMES
    assert row["scored"][start:end] == [1] * (end - start)
    assert start == row["spans"][2]["end"]


def test_holdout_row_tape_reproduces_the_base_row():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    for turn in (0, 2):
        row, window = holdout_row(rec, turn, "tape")
        expected = {k: v for k, v in rec["base"].items() if k != "id"}
        assert {k: v for k, v in row.items() if k != "id"} == expected
        assert row["id"] == f"stem#{turn}#tape"
        answer = rec["base"]["spans"][turn + 1]
        assert window == (answer["start"], answer["end"])


def test_holdout_row_refuses_a_turn_with_no_answer():
    spans = two_ask_spans()[:3]  # ends on a pupil turn
    segs = tape_segments(spans, fake_text()[:37], fake_record_of(37))
    rec = holdout_record("stem", spans, segs, typed, voiced, VOICES)
    with pytest.raises(ValueError, match="no teacher answer"):
        holdout_row(rec, 2, "typed")


def fake_record_of(n):
    return {"audio": [OWN] * n, "user_audio": [PUPIL] * n}


def test_pupil_indices():
    assert pupil_indices(two_ask_spans()) == [0, 2]


def test_assemble_refuses_past_the_context():
    spans = [
        TurnSpan(0, "pupil", ("a",), 0, 0, 5),
        TurnSpan(1, "teacher", ("b",), 5, 5, 2999),
    ]
    segs = [
        tape_segment([PAD] * 5, [OWN] * 5, [PUPIL] * 5, scored=False),
        tape_segment([PAD] * 2994, [OWN] * 2994, [PUPIL] * 2994, scored=True),
    ]
    assert assemble("s", spans, segs, {}).frames == 2999
    long = typed_segment([100, 101, 102, 103])  # 4 + gap = 10 frames
    with pytest.raises(ValueError, match="over moshi's"):
        assemble("s", spans, segs, {0: ("typed", long)})


def test_tape_segments_refuse_text_on_a_pupil_frame():
    text = fake_text()
    text[30] = 777  # teacher text spilling into the second pupil span (28, 37)
    with pytest.raises(ValueError, match=r"pupil.*28.*37"):
        tape_segments(two_ask_spans(), text, fake_record())


def test_holdout_row_refuses_a_teacher_index():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    with pytest.raises(ValueError, match="not a pupil"):
        holdout_row(rec, 1, "typed")


def test_voiced_rows_voice_exactly_each_typed_rows_typed_turns():
    rows = training_rows("stem", two_ask_spans(), base(), typed)
    vrows = voiced_rows("stem", two_ask_spans(), base(), voiced, "D", "vdddd")
    assert [v["pairs_with"] for v in vrows] == ["mask", "complement"]
    assert [v["id"] for v in vrows] == ["stem#mask-D", "stem#complement-D"]
    for v, r in zip(vrows, rows[:2], strict=True):
        assert v["kind"] == "voiced" and v["stem"] == "stem" and v["voice"] == "D"
        assert v["typed_turns"] == []
        assert v["voiced_turns"] == r["typed_turns"]
        for k, s in enumerate(v["spans"]):
            if k in v["voiced_turns"]:
                assert s["rendering"] == "D"
            else:
                assert s["rendering"] == "tape"


def test_voiced_turns_have_the_tts_length_and_answers_keep_the_tape():
    spoken = training_rows("stem", two_ask_spans(), base(), typed)[2]
    for v in voiced_rows("stem", two_ask_spans(), base(), voiced, "D", "vdddd"):
        for k, (s, t) in enumerate(zip(v["spans"], spoken["spans"], strict=True)):
            if k in v["voiced_turns"]:
                words = two_ask_spans()[k].words
                assert s["end"] - s["start"] == 2 * len(words) + GAP_FRAMES
                assert list(v["pupil"][s["start"]]) == [5] * 8  # len("vdddd")
            if s["party"] == "teacher":
                assert (
                    v["text"][s["start"] : s["end"]]
                    == spoken["text"][t["start"] : t["end"]]
                )
                assert (
                    v["own"][s["start"] : s["end"]]
                    == spoken["own"][t["start"] : t["end"]]
                )


def test_voiced_rows_refuse_past_the_context():
    def long_voiced(words, voice):
        return [[1] * 8] * 4000

    with pytest.raises(ValueError, match="over moshi"):
        voiced_rows("stem", two_ask_spans(), base(), long_voiced, "D", "v")


def test_add_rendering_adds_a_label_to_every_turn_and_keeps_the_rest():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    out = add_rendering(rec, two_ask_spans(), voiced, {"D": "vdddd"})
    assert out["base"] == rec["base"] and out["pupil_turns"] == rec["pupil_turns"]
    for k in ("0", "2"):
        assert set(out["renderings"][k]) == {"typed", "A", "B", "C", "D"}
        for lab in ("typed", "A", "B", "C"):
            assert out["renderings"][k][lab] == rec["renderings"][k][lab]
        assert out["renderings"][k]["D"]["pupil"][0] == [5] * 8
    assert set(rec["renderings"]["0"]) == {"typed", "A", "B", "C"}  # input untouched
    row, _ = holdout_row(out, 2, "D")
    assert row["spans"][2]["rendering"] == "D"


@pytest.mark.parametrize("label", ["A", "typed", "tape"])
def test_add_rendering_refuses_a_label_that_exists(label):
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    with pytest.raises(ValueError, match="already"):
        add_rendering(rec, two_ask_spans(), voiced, {label: "vx"})


def test_add_rendering_refuses_other_spans():
    rec = holdout_record("stem", two_ask_spans(), base(), typed, voiced, VOICES)
    with pytest.raises(ValueError, match="pupil turns"):
        add_rendering(rec, two_ask_spans()[:2], voiced, {"D": "vx"})


def test_check_same_tape_accepts_the_round_trip_and_refuses_a_change():
    import json

    rebuilt = training_rows("stem", two_ask_spans(), base(), typed)[2]
    stored = json.loads(json.dumps(rebuilt))
    check_same_tape(rebuilt, stored)
    stored["pupil"][0] = [0] * 8
    with pytest.raises(ValueError, match="pupil"):
        check_same_tape(rebuilt, stored)
