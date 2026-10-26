import itertools

import pytest

from dual.constants import CUE, END, PAD, TURN
from dual.turns import onsets_of, spans, text_row, typed_tokens

SRC = "ask a b\nsay c d\npause 2\nask e\nsay f g\n"
FRAMES = 60


def record() -> dict:
    return {
        "id": "dlg-test",
        "frames": FRAMES,
        "words": [
            [["a", 3], ["b", 5]],
            [["c", 14], ["d", 17]],
            [["e", 32]],
            [["f", 41], ["g", 44]],
        ],
    }


def fake_pieces(word: str) -> list[str]:
    return ["▁" + word[0], *word[1:]]


def fake_id(piece: str) -> int:
    return 1000 + sum(ord(ch) for ch in piece)


def fake_ids(text: str) -> list[int]:
    return [fake_id(p) for w in text.split() for p in fake_pieces(w)]


def test_spans_parties_bounds_words_and_onsets():
    out = spans(SRC, record())
    assert [s.party for s in out] == ["pupil", "teacher", "pupil", "teacher"]
    assert [s.start for s in out] == [0, 10, 28, 37]
    assert [s.end for s in out] == [10, 28, 37, 60]
    assert [s.words for s in out] == [("a", "b"), ("c", "d"), ("e",), ("f", "g")]
    assert [s.onset for s in out] == [3, 14, 32, 41]
    assert [s.index for s in out] == [0, 1, 2, 3]


def test_spans_tile_the_tape_contiguously():
    out = spans(SRC, record())
    assert out[0].start == 0
    for left, right in itertools.pairwise(out):
        assert left.end == right.start
    assert out[-1].end == FRAMES


def test_spans_refuse_an_empty_span():
    rec = record()
    rec["words"][1] = [["c", 3], ["d", 4]]  # span 0 would end at -1
    with pytest.raises(ValueError, match="empty"):
        spans(SRC, rec)


def test_spans_refuse_a_miscounted_record():
    rec = record()
    rec["words"] = rec["words"][:-1]
    with pytest.raises(ValueError, match="spoken lines"):
        spans(SRC, rec)


def test_spans_refuse_a_tape_too_short():
    rec = record()
    rec["frames"] = 30
    with pytest.raises(ValueError):
        spans(SRC, rec)


def test_onsets_of_reads_pairs():
    assert onsets_of(record())[1] == [("c", 14), ("d", 17)]


def test_text_row_lays_the_teacher_words_and_pads_the_pupil():
    row = text_row(SRC, record(), fake_pieces, fake_id)
    assert len(row) == FRAMES
    assert row[13] == TURN
    assert row[14] == fake_id("▁c")
    assert row[16] == CUE  # the frame before d
    assert row[17] == fake_id("▁d")
    # every pupil frame (0..9 and 28..36) is PAD
    assert all(t == PAD for t in row[0:10] + row[28:37])
    assert row[40] == TURN
    assert row[41] == fake_id("▁f")
    assert row[44] == fake_id("▁g")
    assert row[45] == END


def test_text_row_refuses_text_past_the_tape():
    rec = record()
    rec["frames"] = 45  # END of g lands at frame 45
    with pytest.raises(ValueError, match="past the tape"):
        text_row(SRC, rec, fake_pieces, fake_id)


def test_typed_tokens_punctuates_the_last_word():
    assert typed_tokens(["so", "what"], fake_ids) == fake_ids("so what.")
    assert typed_tokens(["so", "what?"], fake_ids) == fake_ids("so what?")


def test_typed_tokens_refuses_an_empty_list():
    with pytest.raises(ValueError, match="no tokens"):
        typed_tokens([], fake_ids)


def _spm_path() -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(
            "kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model"
        )
    except Exception:
        return None
    return path if isinstance(path, str) else None


@pytest.mark.skipif(_spm_path() is None, reason="SPM not in the HF cache")
def test_typed_tokens_with_the_real_spm():
    import sentencepiece

    spm = sentencepiece.SentencePieceProcessor(model_file=_spm_path())
    ids = typed_tokens(["two", "since", "the", "curve"], spm.EncodeAsIds)
    assert 4 <= len(ids) <= 10
    assert not any(i in (0, 1, 2, 3) for i in ids)
