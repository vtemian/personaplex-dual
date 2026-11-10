import json

import pytest

from dual.constants import GAP_FRAMES, PAD, SILENCE, SINE
from dual.engine import (
    AUDIO,
    BACKLOG_KEEP,
    FRAME_BYTES,
    FRAME_SAMPLES,
    KINDS,
    MAX_BACKLOG,
    PROMPT_SILENCE_FRAMES,
    READY,
    REPLAY_VARIANTS,
    STATUS,
    STATUS_EVERY,
    TEXT,
    TYPED,
    Counters,
    PcmBuffer,
    Timing,
    burst,
    first_class_counts,
    is_word,
    pack,
    piece_text,
    replay_summary,
    session_name,
    step_row,
    system_prompt_frames,
    trim_backlog,
    typed_words,
    unpack,
    wrap_system,
)


def test_frame_sizes():
    assert FRAME_SAMPLES == 1920
    assert FRAME_BYTES == 3840


def test_kinds_are_the_wire():
    assert (READY, AUDIO, TEXT, TYPED, STATUS) == (0, 1, 2, 3, 4)
    assert KINDS == (0, 1, 2, 3, 4)


def test_pack_unpack_round_trip():
    msg = pack(TEXT, "hi \u2581there".encode())
    assert msg[0] == TEXT
    assert unpack(msg) == (TEXT, "hi \u2581there".encode())
    assert unpack(pack(READY)) == (READY, b"")


def test_pack_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="kind 9"):
        pack(9, b"")


def test_unpack_refuses_empty_and_unknown():
    with pytest.raises(ValueError, match="empty"):
        unpack(b"")
    with pytest.raises(ValueError, match="kind 7"):
        unpack(bytes([7, 1, 2]))


def test_pcm_buffer_cuts_whole_frames_and_keeps_the_rest():
    buf = PcmBuffer()
    assert buf.push(b"\x01\x00" * 1000) == []
    assert buf.pending == 2000
    out = buf.push(b"\x02\x00" * 1000)
    assert len(out) == 1 and len(out[0]) == FRAME_BYTES
    assert out[0][:2000] == b"\x01\x00" * 1000
    assert buf.pending == 4000 - FRAME_BYTES
    out = buf.push(b"\x03\x00" * 4000)
    assert len(out) == 2
    assert buf.pending == (4000 - FRAME_BYTES + 8000) - 2 * FRAME_BYTES


def test_pcm_buffer_refuses_half_a_sample():
    with pytest.raises(ValueError, match="not whole int16"):
        PcmBuffer().push(b"\x00\x00\x00")


def test_trim_backlog_rule():
    assert (MAX_BACKLOG, BACKLOG_KEEP) == (25, 2)
    assert trim_backlog(0) == 0
    assert trim_backlog(25) == 0
    assert trim_backlog(26) == 24
    assert trim_backlog(100) == 98
    assert trim_backlog(5, limit=4, keep=1) == 4


def test_trim_backlog_refuses_a_bad_rule():
    with pytest.raises(ValueError, match="bad backlog rule"):
        trim_backlog(1, limit=1, keep=2)
    with pytest.raises(ValueError, match="bad backlog rule"):
        trim_backlog(1, limit=3, keep=-1)


def test_typed_words_splits_and_refuses_empty():
    assert typed_words("  what is  a derivative? ") == [
        "what",
        "is",
        "a",
        "derivative?",
    ]
    with pytest.raises(ValueError, match="no words"):
        typed_words("   ")


def test_burst_is_the_typed_triple_frame_by_frame():
    frames = burst([11, 12, 13])
    assert len(frames) == 3 + GAP_FRAMES
    assert [f[0] for f in frames] == [11, 12, 13] + [PAD] * GAP_FRAMES
    assert all(f[1] == SILENCE for f in frames)
    assert all(f[2] == SINE for f in frames)
    with pytest.raises(ValueError, match="no tokens"):
        burst([])


def test_is_word_and_piece_text():
    assert not any(is_word(i) for i in range(4))
    assert is_word(4) and is_word(31999)
    assert piece_text("\u2581hello") == " hello"
    assert piece_text("world") == "world"


def test_timing_summary():
    t = Timing()
    assert t.summary() == {"n": 0}
    for s in (0.010, 0.030, 0.020, 0.100):
        t.add(s)
    got = t.summary()
    assert got["n"] == 4
    assert got["mean"] == pytest.approx(40.0)
    assert got["p50"] == pytest.approx(30.0)
    assert got["p90"] == pytest.approx(100.0)
    assert got["max"] == pytest.approx(100.0)


def test_counters_status_is_a_status_message():
    c = Counters()
    c.frames, c.heard, c.typed, c.dropped = 50, 48, 1, 3
    c.hear.add(0.08)
    kind, payload = unpack(c.status())
    assert kind == STATUS
    body = json.loads(payload)
    assert body["frames"] == 50 and body["heard"] == 48
    assert body["typed"] == 1 and body["dropped"] == 3
    assert body["hear_ms"]["n"] == 1 and body["burst_ms"] == {"n": 0}
    assert STATUS_EVERY == 25


def test_step_row():
    row = step_row(7, "typed", 123, 88.456)
    assert row == {"i": 7, "kind": "typed", "text": 123, "ms": 88.46}
    assert step_row(0, "opening", 5, 1.0)["kind"] == "opening"
    with pytest.raises(ValueError, match="step kind"):
        step_row(0, "sing", 1, 1.0)


def test_session_name_is_utc_seconds():
    assert session_name(0.0) == "19700101-000000"
    assert len(session_name()) == 15


def test_replay_summary_counts_classes_per_variant():
    rows = [
        {"stem": "a", "variant": "typed", "class": "answer"},
        {"stem": "a", "variant": "spoken", "class": "answer"},
        {"stem": "b", "variant": "typed", "class": "echo"},
        {"stem": "b", "variant": "spoken", "class": "ignore"},
    ]
    got = replay_summary(rows)
    assert REPLAY_VARIANTS == ("typed", "spoken")
    assert got["dialogues"] == 2
    assert got["classes"]["typed"] == {"answer": 1, "echo": 1, "ignore": 0}
    assert got["classes"]["spoken"] == {"answer": 1, "echo": 0, "ignore": 1}
    with pytest.raises(ValueError, match="variant"):
        replay_summary([{"stem": "a", "variant": "sung", "class": "answer"}])


def test_wrap_system_is_personaplexs():
    assert wrap_system("  hi there ") == "<system> hi there <system>"
    assert wrap_system("<system> hi <system>\n") == "<system> hi <system>"
    assert wrap_system("<system> hi") == "<system> <system> hi <system>"
    for blank in ("", "  \n"):
        with pytest.raises(ValueError, match="blank"):
            wrap_system(blank)


def test_system_prompt_frames_order():
    voice = [[i] * 8 for i in (5, 6)]
    got = system_prompt_frames(voice, [40, 41, 42])
    sil = PROMPT_SILENCE_FRAMES
    assert sil == 6
    want = (
        [(PAD, (5,) * 8, SINE), (PAD, (6,) * 8, SINE)]
        + [(PAD, SILENCE, SINE)] * sil
        + [(t, SILENCE, SINE) for t in (40, 41, 42)]
        + [(PAD, SILENCE, SINE)] * sil
    )
    assert got == want
    assert len(system_prompt_frames(voice, [40], silence=2)) == 2 + 2 + 1 + 2


def test_system_prompt_frames_voice_or_text_alone_keep_both_silences():
    text_only = system_prompt_frames([], [40])
    assert (
        text_only
        == [(PAD, SILENCE, SINE)] * 6
        + [(40, SILENCE, SINE)]
        + [(PAD, SILENCE, SINE)] * 6
    )
    voice_only = system_prompt_frames([[1] * 8], [])
    assert len(voice_only) == 1 + 6 + 6
    assert voice_only[0] == (PAD, (1,) * 8, SINE)


def test_system_prompt_frames_refuse_bad_input():
    with pytest.raises(ValueError, match="empty"):
        system_prompt_frames([], [])
    with pytest.raises(ValueError, match="8"):
        system_prompt_frames([[1] * 7], [])
    with pytest.raises(ValueError, match="2047"):
        system_prompt_frames([[2048] + [0] * 7], [])
    with pytest.raises(ValueError, match="2047"):
        system_prompt_frames([[-1] + [0] * 7], [])


def test_prompt_is_a_step_kind():
    assert step_row(3, "prompt", 40, 1.0)["kind"] == "prompt"


def test_first_class_counts_skip_none():
    rows = [
        {"variant": "typed", "first_class": "answer"},
        {"variant": "typed", "first_class": None},
        {"variant": "spoken", "first_class": "echo"},
        {"variant": "spoken", "first_class": "echo"},
    ]
    got = first_class_counts(rows)
    assert got["typed"] == {"answer": 1, "echo": 0, "ignore": 0}
    assert got["spoken"] == {"answer": 0, "echo": 2, "ignore": 0}
    with pytest.raises(ValueError, match="variant"):
        first_class_counts([{"variant": "sung", "first_class": "echo"}])
