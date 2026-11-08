import pytest

from dual.consistency import (
    AnswerPair,
    answer_pairs,
    flat_indices,
    longest_stem,
    spoken_by_stem,
    tape_teachers,
    voiced_teachers,
)
from dual.constants import PAD
from dual.frames import spoken_segment, tape_segment, to_record, typed_segment
from dual.render import assemble
from dual.turns import TurnSpan

OWN = [[7] * 8]
PUP = [[9] * 8]
# teacher, pupil (5 frames on tape), teacher, pupil (7), teacher
PARTS = (
    ("teacher", [PAD, PAD, 1, 40, 41, 2]),
    ("pupil", [PAD] * 5),
    ("teacher", [PAD, PAD, PAD, 1, 50, 51, 52, 2]),
    ("pupil", [PAD] * 7),
    ("teacher", [PAD, PAD, PAD, 1, 60, 2]),
)
TYPED = typed_segment([11, 12, 13])  # 3 tokens + 6 gap frames = 9 frames


VOICED = spoken_segment([[5] * 8] * 4)  # 4 TTS frames + 6 gap = 10 frames


def tape_parts(parts=PARTS):
    spans, base, start = [], [], 0
    for i, (party, text) in enumerate(parts):
        n = len(text)
        spans.append(TurnSpan(i, party, (f"w{i}",), start, start, start + n))
        base.append(tape_segment(text, OWN * n, PUP * n, scored=party == "teacher"))
        start += n
    return spans, base


def make_rows(typed_turns, parts=PARTS, stem="s"):
    spans, base = tape_parts(parts)
    swaps = {k: ("typed", TYPED) for k in typed_turns}
    typed = to_record(assemble(f"{stem}#mask", spans, base, swaps))
    typed |= {"kind": "mask", "stem": stem, "typed_turns": list(typed_turns)}
    spoken = to_record(assemble(f"{stem}#spoken", spans, base, {}))
    spoken |= {"kind": "spoken", "stem": stem, "typed_turns": []}
    return typed, spoken


def make_voiced(
    voiced_turns, parts=PARTS, stem="s", kind="mask", label="D", seg=VOICED
):
    spans, base = tape_parts(parts)
    swaps = {k: (label, seg) for k in voiced_turns}
    row = to_record(assemble(f"{stem}#{kind}-{label}", spans, base, swaps))
    row |= {
        "kind": "voiced",
        "stem": stem,
        "typed_turns": [],
        "pairs_with": kind,
        "voice": label,
        "voiced_turns": list(voiced_turns),
    }
    return row


def test_one_typed_turn_pairs_its_answer_by_offset():
    typed, spoken = make_rows([1])
    # typed spans: [0,6) [6,15) [15,23) ...; spoken: [0,6) [6,11) [11,19) ...
    assert answer_pairs(typed, spoken) == [
        AnswerPair(turn=1, typed=(15, 23), spoken=(11, 19), content=(3, 4, 5, 6, 7))
    ]


def test_two_typed_turns_shift_the_later_answer_by_both():
    typed, spoken = make_rows([1, 3])
    pairs = answer_pairs(typed, spoken)
    assert [p.turn for p in pairs] == [1, 3]
    assert pairs[1] == AnswerPair(
        turn=3, typed=(32, 38), spoken=(26, 32), content=(3, 4, 5)
    )


def test_paired_frames_carry_the_same_target_text():
    typed, spoken = make_rows([1, 3])
    student, teacher = flat_indices(answer_pairs(typed, spoken))
    assert student == [18, 19, 20, 21, 22, 35, 36, 37]
    assert teacher == [14, 15, 16, 17, 18, 29, 30, 31]
    assert [typed["text"][i] for i in student] == [spoken["text"][i] for i in teacher]
    assert PAD not in [typed["text"][i] for i in student]


def test_no_typed_turn_no_pairs():
    typed, spoken = make_rows([])
    assert answer_pairs(typed, spoken) == []
    assert flat_indices([]) == ([], [])


def test_a_typed_turn_that_ends_the_dialogue_has_no_answer():
    parts = (*PARTS, ("pupil", [PAD] * 4))
    typed, spoken = make_rows([1, 5], parts=parts)
    assert [p.turn for p in answer_pairs(typed, spoken)] == [1]


def test_refuses_a_row_of_another_stem():
    typed, _ = make_rows([1])
    _, other = make_rows([], stem="t")
    with pytest.raises(ValueError, match="stem"):
        answer_pairs(typed, other)


def test_refuses_a_teacher_row_that_is_not_the_spoken_kind():
    typed, _ = make_rows([1])
    with pytest.raises(ValueError, match="spoken"):
        answer_pairs(typed, typed)


def test_refuses_a_typed_turn_the_row_does_not_type():
    typed, spoken = make_rows([1])
    typed["typed_turns"] = [3]  # span 3 is tape in this row
    with pytest.raises(ValueError, match="not typed"):
        answer_pairs(typed, spoken)


def test_refuses_answers_whose_text_differs():
    typed, spoken = make_rows([1])
    spoken["text"][14] = 99  # inside the spoken answer to turn 1
    with pytest.raises(ValueError, match="text differs"):
        answer_pairs(typed, spoken)


def test_refuses_rows_with_different_turn_counts():
    typed, spoken = make_rows([1])
    spoken["spans"] = spoken["spans"][:-1]
    with pytest.raises(ValueError, match="spans"):
        answer_pairs(typed, spoken)


def test_spoken_by_stem_maps_each_stem_to_its_spoken_row():
    a_typed, a_spoken = make_rows([1], stem="a")
    b_typed, b_spoken = make_rows([3], stem="b")
    got = spoken_by_stem([a_typed, a_spoken, b_typed, b_spoken])
    assert got == {"a": a_spoken, "b": b_spoken}


def test_spoken_by_stem_refuses_a_typed_stem_without_its_spoken_row():
    a_typed, _ = make_rows([1], stem="a")
    with pytest.raises(ValueError, match="no spoken row"):
        spoken_by_stem([a_typed])


def test_spoken_by_stem_refuses_two_spoken_rows_for_one_stem():
    _, a_spoken = make_rows([], stem="a")
    with pytest.raises(ValueError, match="two spoken rows"):
        spoken_by_stem([a_spoken, dict(a_spoken)])


def test_longest_stem_is_the_stem_of_the_longest_row_first_by_name_on_ties():
    rows = [
        {"stem": "b", "frames": 10},
        {"stem": "a", "frames": 30},
        {"stem": "c", "frames": 30},
    ]
    assert longest_stem(rows) == "a"
    with pytest.raises(ValueError):
        longest_stem([])


def test_a_voiced_teacher_pairs_by_offset_after_a_longer_pupil_turn():
    typed, _ = make_rows([1])
    voiced = make_voiced([1])  # turn 1: 10 frames (tape 5, typed 9)
    assert answer_pairs(typed, voiced) == [
        AnswerPair(turn=1, typed=(15, 23), spoken=(16, 24), content=(3, 4, 5, 6, 7))
    ]


def test_a_voiced_teacher_pairs_like_the_tape_teacher_frame_for_frame():
    typed, spoken = make_rows([1, 3])
    voiced = make_voiced([1, 3])
    by_tape, by_voice = answer_pairs(typed, spoken), answer_pairs(typed, voiced)
    assert [(p.turn, p.content) for p in by_tape] == [
        (p.turn, p.content) for p in by_voice
    ]
    s_idx, t_idx = flat_indices(by_voice)
    assert [typed["text"][i] for i in s_idx] == [voiced["text"][i] for i in t_idx]


def test_refuses_a_voiced_teacher_of_other_turns():
    typed, _ = make_rows([1])
    with pytest.raises(ValueError, match="voices"):
        answer_pairs(typed, make_voiced([3]))


def test_refuses_a_voiced_teacher_whose_turn_is_the_tape():
    typed, _ = make_rows([1])
    bad = make_voiced([1]) | {"voice": "E"}  # spans say D
    with pytest.raises(ValueError, match="not E"):
        answer_pairs(typed, bad)


def test_tape_teachers_map_each_typed_row_to_its_stems_spoken_row():
    s_typed, s_spoken = make_rows([1])
    t_typed, t_spoken = make_rows([1, 3], stem="t")
    out = tape_teachers([s_typed, s_spoken, t_typed, t_spoken])
    assert out == {"s#mask": s_spoken, "t#mask": t_spoken}


def test_voiced_teachers_map_each_typed_row_to_its_kinds_voiced_row():
    s_typed, s_spoken = make_rows([1])
    v = make_voiced([1])
    other = make_voiced([3], kind="complement")
    assert voiced_teachers([s_typed, s_spoken], [v, other]) == {"s#mask": v}


def test_voiced_teachers_refuse_a_missing_row_two_rows_and_two_voices():
    s_typed, s_spoken = make_rows([1])
    with pytest.raises(ValueError, match="no voiced row"):
        voiced_teachers([s_typed, s_spoken], [make_voiced([3], kind="complement")])
    with pytest.raises(ValueError, match="two voiced rows"):
        voiced_teachers([s_typed], [make_voiced([1]), make_voiced([1])])
    with pytest.raises(ValueError, match="one teacher voice"):
        voiced_teachers(
            [s_typed],
            [make_voiced([1]), make_voiced([3], kind="complement", label="E")],
        )
    with pytest.raises(ValueError, match="not a voiced row"):
        voiced_teachers([s_typed], [s_spoken])
    with pytest.raises(ValueError, match="voices"):
        voiced_teachers([s_typed], [make_voiced([3])])


def test_refuses_a_voiced_teacher_that_also_voices_an_untyped_turn():
    typed, _ = make_rows([1])
    bad = make_voiced([1, 3]) | {
        "voiced_turns": [1]
    }  # metadata says 1, spans voice 1 and 3
    with pytest.raises(ValueError, match="pupil turn 3 is D, not tape"):
        answer_pairs(typed, bad)
