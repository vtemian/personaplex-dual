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
