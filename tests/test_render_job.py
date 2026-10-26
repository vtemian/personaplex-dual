from dual.jobs.render import stable_seed


def test_stable_seed_is_deterministic_and_32_bit() -> None:
    key = (("the", "same", "words"), "vctk/p225_023.wav")
    assert stable_seed(*key) == stable_seed(*key)
    assert 0 <= stable_seed(*key) < 2**32


def test_stable_seed_separates_keys() -> None:
    a = stable_seed(("hello",), "A")
    assert a != stable_seed(("hello",), "B")
    assert a != stable_seed(("hello", "there"), "A")


def test_stable_seed_is_pinned() -> None:
    # sha256-based, so it survives Python's per-process str hash randomisation
    assert stable_seed(("a", "b"), "v") == 2890411864


def _record():
    from dual.constants import PAD
    from dual.render import holdout_record, tape_segments
    from dual.turns import TurnSpan

    n = 60
    spans = [
        TurnSpan(0, "pupil", ("a", "b"), 3, 0, 10),
        TurnSpan(1, "teacher", ("c", "d"), 14, 10, 28),
        TurnSpan(2, "pupil", ("e",), 32, 28, 37),
        TurnSpan(3, "teacher", ("f", "g"), 41, 37, 60),
    ]
    text = [PAD] * n
    for f in list(range(10, 28)) + list(range(37, 60)):
        text[f] = 500 + f
    rec = {"audio": [(7,) * 8] * n, "user_audio": [(9,) * 8] * n}
    segs = tape_segments(spans, text, rec)
    voices = {"A": "va", "B": "vbb", "C": "vccc"}
    return holdout_record(
        "stem",
        spans,
        segs,
        lambda w: [100 + len(x) for x in w],
        lambda w, v: [[len(v)] * 8] * (2 * len(w)),
        voices,
    ), voices


def test_validate_holdout_counts_rows_and_windows() -> None:
    from dual.jobs.render import validate_holdout

    rec, voices = _record()
    rows, max_frames, min_window = validate_holdout([rec], tuple(voices))
    assert rows == 2 * 5  # two pupil turns x (typed, A, B, C, tape)
    assert max_frames >= 60
    assert min_window == 18  # the shorter teacher answer, turn 1


def test_validate_holdout_raises_on_a_missing_answer() -> None:
    import pytest

    from dual.jobs.render import validate_holdout

    rec, voices = _record()
    rec["pupil_turns"] = [0, 2, 3]  # turn 3 is a teacher turn
    with pytest.raises(ValueError):
        validate_holdout([rec], tuple(voices))
