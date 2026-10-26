import pytest

from dual.constants import GAP_FRAMES, MIMI_SILENCE, PAD, SILENCE, SINE
from dual.frames import (
    Segment,
    Span,
    concat,
    from_record,
    spoken_segment,
    tape_segment,
    to_record,
    typed_segment,
)


def test_typed_segment_is_tokens_then_gap_on_silence_and_sine():
    seg = typed_segment([10, 11, 12])
    assert seg.frames == 3 + GAP_FRAMES
    assert seg.text == (10, 11, 12) + (PAD,) * GAP_FRAMES
    assert set(seg.own) == {SILENCE}
    assert set(seg.pupil) == {SINE}
    assert seg.scored is False


def test_typed_segment_refuses_no_tokens():
    with pytest.raises(ValueError, match="no tokens"):
        typed_segment([])


def test_spoken_segment_is_codes_then_mimi_silence_under_pad():
    codes = [[1] * 8, [2] * 8]
    seg = spoken_segment(codes)
    assert seg.frames == 2 + GAP_FRAMES
    assert seg.text == (PAD,) * (2 + GAP_FRAMES)
    assert seg.pupil[:2] == ((1,) * 8, (2,) * 8)
    assert set(seg.pupil[2:]) == {MIMI_SILENCE}
    assert set(seg.own) == {SILENCE}


def test_segment_refuses_ragged_streams():
    with pytest.raises(ValueError, match="frames"):
        Segment(text=(PAD, PAD), own=(SILENCE,), pupil=(SINE, SINE), scored=False)
    with pytest.raises(ValueError, match="codebooks"):
        Segment(text=(PAD,), own=((1, 2),), pupil=(SINE,), scored=False)


def test_concat_records_spans_and_scored_mask():
    a = tape_segment([PAD, 5], [SILENCE, SILENCE], [SINE, SINE], scored=True)
    b = typed_segment([7])
    row = concat("dlg-1", [("teacher", "tape", a), ("pupil", "typed", b)])
    assert row.frames == 2 + 1 + GAP_FRAMES
    assert row.scored == (1, 1) + (0,) * (1 + GAP_FRAMES)
    assert row.spans == (
        Span("teacher", "tape", 0, 2),
        Span("pupil", "typed", 2, 2 + 1 + GAP_FRAMES),
    )
    assert row.text[:3] == (PAD, 5, 7)


def test_record_round_trip():
    row = concat("dlg-1", [("pupil", "typed", typed_segment([7, 8]))])
    assert from_record(to_record(row)) == row
    assert to_record(row)["frames"] == row.frames
