import hashlib

import pytest

from dual.holdout import mask_bits, ordered, split


def test_split_holds_out_a_tenth_rounded_up_and_is_stable():
    stems = [f"dlg-{i:03d}" for i in range(130)]
    held, train = split(stems)
    assert len(held) == 13 and len(train) == 117
    assert set(held) | set(train) == set(stems) and not set(held) & set(train)
    assert split(list(reversed(stems))) == (held, train)


def test_split_refuses_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        split(["a", "a"])


def test_ordered_is_a_permutation_keyed_by_salt():
    stems = ["a", "b", "c", "d"]
    assert sorted(ordered(stems, "x")) == stems
    assert ordered(stems, "x") == ordered(stems, "x")


def test_mask_bits_is_stable_and_bit_per_turn():
    bits = mask_bits("dlg-00005-000", 5)
    assert len(bits) == 5 and set(bits) <= {0, 1}
    assert bits == mask_bits("dlg-00005-000", 5)
    assert mask_bits("dlg-00005-000", 3) == bits[:3]


def _digest(stem, salt):
    return int(hashlib.sha256(f"{stem}:{salt}".encode()).hexdigest(), 16)


def test_ordered_matches_independent_sha256_golden():
    stems = ["alpha", "beta", "gamma", "delta", "dlg-001"]
    want = sorted(stems, key=lambda s: (int(hashlib.sha256(f"{s}:x".encode()).hexdigest(), 16), s))
    assert ordered(stems, "x") == want


def test_mask_bits_are_lsb_first_bits_of_sha256_mask():
    s = "dlg-00005-000"
    digest = _digest(s, "mask")
    assert mask_bits(s, 5) == tuple((digest >> k) & 1 for k in range(5))


def test_holdout_count_is_a_ceil():
    assert len(split([f"s{i}" for i in range(131)])[0]) == 14
    assert len(split(["a"])[0]) == 1
