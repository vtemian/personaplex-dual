import json

import pytest

from dual.constants import CODEBOOKS
from dual.sidebyside import (
    RENDERINGS,
    SWAP_RENDERINGS,
    Item,
    StepInput,
    answer_head,
    answer_words,
    batch_cap,
    item_of,
    pack,
    pick_donors,
    same_words,
    schedule,
    step_input,
    swap_record,
)

T = 12


def make_row():
    # distinct codes per frame and codebook so every shift is visible
    return {
        "text": [100 + f for f in range(T)],
        "own": [[1000 + 10 * f + k for k in range(CODEBOOKS)] for f in range(T)],
        "pupil": [[5000 + 10 * f + k for k in range(CODEBOOKS)] for f in range(T)],
    }


def test_renderings():
    assert RENDERINGS == ("typed", "A", "B")


def test_item_of_and_total():
    it = item_of("s", 2, "A", (10, 25))
    assert it == Item("s", 2, "A", 10, 15)
    assert it.total == 25


@pytest.mark.parametrize("answer", [(0, 5), (-1, 5), (5, 5), (5, 4)])
def test_item_of_refuses(answer):
    with pytest.raises(ValueError):
        item_of("s", 0, "A", answer)


def test_batch_cap_rounds_up():
    a = Item("a", 0, "A", 40, 10)  # 50
    b = Item("b", 0, "A", 60, 10)  # 70
    assert batch_cap([a, b]) == 128
    assert batch_cap([a], align=64) == 64
    assert batch_cap([Item("c", 0, "A", 32, 32)]) == 64
    assert batch_cap([a, b], align=10) == 70


def test_batch_cap_empty():
    with pytest.raises(ValueError):
        batch_cap([])


def items(totals):
    return [Item(f"s{i:02d}", 0, "A", t - 5, 5) for i, t in enumerate(totals)]


def test_pack_every_item_once_and_sorted():
    its = items([300, 50, 120, 60, 64, 200, 10])
    out = pack(its, budget=400, max_batch=3)
    flat = [i for b in out for i in b]
    assert sorted(flat, key=lambda i: i.stem) == sorted(its, key=lambda i: i.stem)
    assert [i.total for i in flat] == sorted(i.total for i in its)
    for b in out:
        assert len(b) <= 3
        assert len(b) * batch_cap(b) <= 400


def test_pack_greedy_boundaries():
    # totals 60,60,60,60 -> cap 64 each; budget 128 -> two per batch
    out = pack(items([60, 60, 60, 60]), budget=128, max_batch=8)
    assert [len(b) for b in out] == [2, 2]
    out = pack(items([60, 60, 60]), budget=1000, max_batch=2)
    assert [len(b) for b in out] == [2, 1]


def test_pack_cap_grows_with_longest():
    # 64-cap items then a 200 (cap 256): 2*256 > 400 so it opens a new batch
    out = pack(items([60, 60, 200]), budget=400, max_batch=8)
    assert [len(b) for b in out] == [2, 1]


def test_pack_tiebreak_deterministic():
    a = Item("b", 1, "B", 10, 5)
    b = Item("b", 1, "A", 10, 5)
    c = Item("a", 2, "typed", 10, 5)
    out = pack([a, b, c], budget=1000, max_batch=8)
    assert out == [[c, b, a]]
    assert pack([c, a, b], 1000, 8) == out


def test_pack_errors():
    with pytest.raises(ValueError):
        pack(items([100]), budget=64, max_batch=1)
    with pytest.raises(ValueError):
        pack(items([10]), budget=0, max_batch=1)
    with pytest.raises(ValueError):
        pack(items([10]), budget=100, max_batch=0)
    assert pack([], 100, 2) == []


def test_step_zero():
    row = make_row()
    s = step_input(row, 4, 0)
    assert isinstance(s, StepInput)
    assert s.pupil == tuple(row["pupil"][0])
    assert s.text == 100
    # f = 0 - d_k: delay 0 -> frame 0 forced; delay 1 -> f=-1 uses frame 0 codes
    assert s.own == tuple(row["own"][0])


def test_pupil_shift():
    row = make_row()
    assert step_input(row, 4, 1).pupil == tuple(row["pupil"][0])
    assert step_input(row, 4, 7).pupil == tuple(row["pupil"][6])


def test_mid_prefix_forced():
    row = make_row()
    s = step_input(row, 6, 3)
    assert s.text == 103
    assert s.own[0] == row["own"][3][0]
    assert s.own[1] == row["own"][2][1]
    assert s.own[7] == row["own"][2][7]


def test_boundary_at_prefix():
    row = make_row()
    p = 6
    s = step_input(row, p, p)
    assert s.text is None
    assert s.own[0] is None
    for k in range(1, CODEBOOKS):
        assert s.own[k] == row["own"][p - 1][k]
    after = step_input(row, p, p + 1)
    assert after.text is None
    assert after.own == (None,) * CODEBOOKS


def test_step_before_boundary_all_forced():
    row = make_row()
    s = step_input(row, 6, 5)
    assert s.text == 105
    assert all(v is not None for v in s.own)


@pytest.mark.parametrize("prefix,step", [(4, -1), (4, T), (0, 1), (T + 1, 1)])
def test_step_input_errors(prefix, step):
    with pytest.raises(ValueError):
        step_input(make_row(), prefix, step)


VOCAB = ["so", "the", "slope", "is", "two"]


def decode(ids):
    return " ".join(VOCAB[i - 4] for i in ids)


def test_answer_words():
    assert answer_words([0, 3, 3, 6, 0, 8, 2], decode) == "slope two"
    assert answer_words([0, 3, 1, 2], decode) == ""


def test_same_words():
    assert same_words("The slope, is TWO.", "the slope is two")
    assert not same_words("the slope", "the slope is")


def test_swap_renderings():
    assert SWAP_RENDERINGS == ("typed", "A")


def test_answer_head():
    assert answer_head("Minus Three, because it is odd.") == "minus three"
    assert answer_head("  Yes! It's  four ") == "yes its four"
    assert answer_head(", only a tail") == ""


def test_pick_donors_first_different_head_cyclically():
    order = ["a", "b", "c", "d"]
    heads = {}
    for stem, h in zip(order, ["x", "x", "y", "x"], strict=True):
        heads[(stem, 2)] = h
    got = pick_donors(order, heads, [2])
    assert got == {("a", 2): "c", ("b", 2): "c", ("c", 2): "d", ("d", 2): "c"}


def test_pick_donors_skips_stems_without_the_turn_and_other_turns():
    order = ["a", "b", "c"]
    heads = {("a", 2): "x", ("c", 2): "y", ("a", 6): "x", ("b", 6): "z"}
    got = pick_donors(order, heads, [2, 6])
    assert got == {("a", 2): "c", ("c", 2): "a", ("a", 6): "b", ("b", 6): "a"}


def test_pick_donors_ignores_turns_not_asked():
    assert pick_donors(["a", "b"], {("a", 4): "x", ("b", 4): "y"}, [2]) == {}


def test_pick_donors_raises_without_a_different_head():
    with pytest.raises(ValueError):
        pick_donors(["a", "b"], {("a", 2): "x", ("b", 2): "x"}, [2])


def make_record(stem, tag):
    return {
        "id": stem,
        "pupil_turns": [2, 6],
        "base": {"keep": stem},
        "renderings": {
            "2": {"typed": {"text": [tag]}, "A": {"text": [tag, 1]}},
            "6": {"typed": {"text": [tag + 10]}},
        },
    }


def test_swap_record_takes_the_donors_rendering_without_mutating():
    rec, don = make_record("a", 1), make_record("b", 2)
    before = json.dumps(rec, sort_keys=True), json.dumps(don, sort_keys=True)
    out = swap_record(rec, 2, don)
    assert out["renderings"]["2"] == don["renderings"]["2"]
    assert out["renderings"]["6"] == rec["renderings"]["6"]
    assert out["id"] == "a" and out["base"] == rec["base"]
    assert (json.dumps(rec, sort_keys=True), json.dumps(don, sort_keys=True)) == before
    out["renderings"]["2"]["typed"]["text"].append(99)
    assert don["renderings"]["2"]["typed"]["text"] == [2]


def test_swap_record_refuses_non_pupil_turn():
    rec, don = make_record("a", 1), make_record("b", 2)
    with pytest.raises(ValueError):
        swap_record(rec, 3, don)
    don["pupil_turns"] = [6]
    with pytest.raises(ValueError):
        swap_record(rec, 2, don)


def test_schedule_true_batches_first_and_unchanged():
    true = [Item(f"s{i}", 2, "typed", 100 + 50 * i, 30) for i in range(10)]
    swap = [Item(f"s{i}", 2, "typed-swap", 100 + 50 * i, 30) for i in range(10)]
    got = schedule(true, swap, 1000, 4)
    want = pack(true, 1000, 4)
    assert got[: len(want)] == want
    assert got[len(want) :] == pack(swap, 1000, 4)
    for batch in got:
        assert len({it.rendering.endswith("-swap") for it in batch}) == 1
    assert schedule([], swap, 1000, 4) == pack(swap, 1000, 4)
