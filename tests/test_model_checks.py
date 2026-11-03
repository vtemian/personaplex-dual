import pytest

from dual.constants import AUDIO_CARD, CONTEXT, PAD, SILENCE, SINE, TEXT_CARD
from dual.model import (
    Base,
    adapter_meta,
    adapter_params,
    adapter_setup,
    check_adapter_meta,
    check_row,
)


def row(t: int = 4) -> dict:
    return {"text": [PAD] * t, "own": [list(SILENCE)] * t, "pupil": [list(SINE)] * t}


def test_a_good_row_passes():
    check_row(row())
    check_row(row(CONTEXT))


def test_too_long_names_the_context():
    with pytest.raises(ValueError, match="CONTEXT"):
        check_row(row(CONTEXT + 1))


@pytest.mark.parametrize("bad", [-1, TEXT_CARD])
def test_text_id_out_of_range_names_text(bad):
    r = row()
    r["text"][2] = bad
    with pytest.raises(ValueError, match="text"):
        check_row(r)


@pytest.mark.parametrize("stream", ["own", "pupil"])
@pytest.mark.parametrize("bad", [-1, AUDIO_CARD])
def test_audio_id_out_of_range_names_the_stream(stream, bad):
    r = row()
    r[stream][1][3] = bad
    with pytest.raises(ValueError, match=stream):
        check_row(r)


def test_ragged_streams_are_refused():
    r = row()
    r["own"] = r["own"][:-1]
    with pytest.raises(ValueError, match="own"):
        check_row(r)


META = {"base": "personaplex", "rank": 16, "scaling": 2.0, "params": 34_078_720}


def test_matching_meta_passes():
    check_adapter_meta(dict(META), dict(META))


@pytest.mark.parametrize(
    "key,val", [("base", "moshiko"), ("rank", 8), ("scaling", 1.0), ("params", 1)]
)
def test_meta_mismatch_names_the_field(key, val):
    got = {**META, key: val}
    with pytest.raises(ValueError, match=key):
        check_adapter_meta(got, dict(META))


def test_missing_meta_is_refused():
    with pytest.raises(ValueError, match="rank"):
        check_adapter_meta({"base": "personaplex"}, dict(META))


@pytest.mark.parametrize(
    "rank,params", [(4, 8_519_680), (16, 34_078_720), (64, 136_314_880)]
)
def test_adapter_params_scale_with_rank(rank, params):
    assert adapter_params(rank) == params


def test_adapter_setup_reads_a_ladder_1_meta():
    assert adapter_setup(dict(META)) == (16, 2.0)


@pytest.mark.parametrize("key", ["rank", "scaling"])
def test_adapter_setup_refuses_a_meta_missing_a_key(key):
    meta = {k: v for k, v in META.items() if k != key}
    with pytest.raises(ValueError, match=key):
        adapter_setup(meta)


def test_adapter_meta_reads_the_base_setup():
    base = Base("personaplex", None, None, None, rank=64, scaling=0.5)
    assert adapter_meta(base) == {
        "base": "personaplex",
        "rank": 64,
        "scaling": 0.5,
        "params": 136_314_880,
    }


def test_adapter_meta_raises_without_an_adapter():
    with pytest.raises(RuntimeError, match="adapter"):
        adapter_meta(Base("personaplex", None, None, None))


def test_check_adapter_meta_refuses_a_rank_mismatch():
    want = adapter_meta(Base("personaplex", None, None, None, rank=64, scaling=0.5))
    with pytest.raises(ValueError, match="rank"):
        check_adapter_meta(dict(META), want)
