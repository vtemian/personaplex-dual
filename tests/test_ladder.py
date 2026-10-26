import pytest

from dual.ladder import RUNGS, rung

STEMS = [f"dlg-{i:05d}" for i in range(117)]


def test_rungs_have_the_ceil_sizes_and_nest():
    sizes = {d: rung(STEMS, d) for d in RUNGS}
    assert [len(sizes[d]) for d in (8, 4, 2, 1)] == [15, 30, 59, 117]
    assert set(sizes[8]) <= set(sizes[4]) <= set(sizes[2]) <= set(sizes[1])
    assert sizes[1] == sorted(STEMS)


def test_rungs_are_sorted_and_independent_of_input_order():
    assert rung(STEMS[::-1], 4) == rung(STEMS, 4)
    assert rung(STEMS, 4) == sorted(rung(STEMS, 4))


def test_a_bad_denominator_raises():
    with pytest.raises(ValueError, match="denominator 3"):
        rung(STEMS, 3)
