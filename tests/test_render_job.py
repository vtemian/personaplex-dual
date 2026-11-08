import pytest

from dual.jobs.render import VOICES, parse_args, parse_voices, stable_seed


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


def test_parse_voices_reads_labels_and_refuses_bar_voices_and_clashes() -> None:
    assert parse_voices(["D1=vctk/p226_023.wav", "D2=vctk/p280_023.wav"], VOICES) == {
        "D1": "vctk/p226_023.wav",
        "D2": "vctk/p280_023.wav",
    }
    for bad, why in [
        (["A=vctk/p226_023.wav"], "already"),
        (["D=vctk/p225_023.wav"], "already voices"),
        (["typed=x"], "label"),
        (["d=x"], "label"),
        (["D"], "LABEL=VOICE"),
        (["D=x", "D=y"], "twice"),
        (["D=x", "E=x"], "twice"),
    ]:
        with pytest.raises(ValueError, match=why):
            parse_voices(bad, VOICES)


def test_flags_default_to_render_1s_behaviour(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["render", "--name", "render-x"])
    a = parse_args()
    assert (a.source, a.add_voice, a.teacher_voice, a.limit) == (None, [], None, 0)


def test_teacher_voice_must_be_an_added_label_and_needs_from(monkeypatch) -> None:
    base = ["render", "--name", "r"]
    for extra in (
        ["--add-voice", "D=x"],  # no --from
        ["--from", "render-1", "--teacher-voice", "D"],  # D not added
    ):
        monkeypatch.setattr("sys.argv", [*base, *extra])
        with pytest.raises(SystemExit):
            parse_args()
    monkeypatch.setattr(
        "sys.argv",
        [*base, "--from", "render-1", "--add-voice", "D=x", "--teacher-voice", "D"],
    )
    a = parse_args()
    assert (a.source, a.add_voice, a.teacher_voice) == ("render-1", ["D=x"], "D")


# --- extend_render on fakes (no TTS, no GPU) ---------------------------------------

D_VOICE = "vctk/p226_023.wav"


def _fake_world(tmp_path, monkeypatch):
    """A render-1 written with the real pure functions on a two-pupil-turn dialogue, and
    the job's loaders replaced by fakes. Returns the voiced-call log."""
    import json
    import types

    from dual.constants import PAD
    from dual.holdout import mask_bits
    from dual.jobs import render as job
    from dual.render import holdout_record, tape_segments, training_rows
    from dual.turns import TurnSpan

    stem = next(
        f"s{i}" for i in range(99) if mask_bits(f"s{i}", 2) == (1, 0)
    )  # mask types turn 0, complement types turn 2
    sp = [
        TurnSpan(0, "pupil", ("a", "b"), 3, 0, 10),
        TurnSpan(1, "teacher", ("c", "d"), 14, 10, 28),
        TurnSpan(2, "pupil", ("e",), 32, 28, 37),
        TurnSpan(3, "teacher", ("f", "g"), 41, 37, 60),
    ]
    text = [PAD] * 60
    for f in list(range(10, 28)) + list(range(37, 60)):
        text[f] = 500 + f
    rec = {"id": stem, "audio": [(7,) * 8] * 60, "user_audio": [(9,) * 8] * 60}
    segs = tape_segments(sp, text, rec)

    def typed(w):
        return [100 + len(x) for x in w]

    def voiced(w, v):
        return [[len(v)] * 8] * (2 * len(w))

    run1 = tmp_path / "runs" / "render-1" / "render"
    run1.mkdir(parents=True)
    rows = training_rows(stem, sp, segs, typed)
    (run1 / "rows-train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run1 / "holdout.jsonl").write_text(
        json.dumps(holdout_record(stem, sp, segs, typed, voiced, job.VOICES)) + "\n"
    )
    sha = tmp_path / "SHA"
    sha.write_text("abc\n")
    vfile = tmp_path / "voices" / f"{D_VOICE}{job.VOICE_SUFFIX}"
    vfile.parent.mkdir(parents=True)
    vfile.write_text("x")
    (run1 / "stems.json").write_text(
        json.dumps(
            {
                "held": [stem],
                "train": [stem],
                "voices": job.VOICES,
                "chalkboard_sha": "abc",
            }
        )
    )
    calls: list[tuple[tuple[str, ...], str]] = []

    def voiced_d(w, v):
        calls.append((tuple(w), v))
        return [[1] * 8] * (5 * len(w))

    eng = types.SimpleNamespace(
        pieces=None,
        spm=types.SimpleNamespace(PieceToId=None),
        typed=typed,
        voiced=voiced_d,
        cache={},
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(job, "SHA_FILE", sha)
    monkeypatch.setattr(job, "load_engines", lambda data: eng)
    monkeypatch.setattr(
        job, "load_dialogues", lambda d: ({stem: rec}, {stem: "x"}, [stem])
    )
    monkeypatch.setattr(job, "spans", lambda src, r: sp)
    monkeypatch.setattr(job, "text_row", lambda *a: text)
    return stem, calls


def _args(**kw):
    import argparse

    base = {
        "name": "render-2",
        "limit": 0,
        "source": "render-1",
        "add_voice": [f"D={D_VOICE}"],
        "teacher_voice": "D",
    }
    return argparse.Namespace(**{**base, **kw})


def test_extend_render_copies_rows_and_writes_teacher_and_holdout(
    tmp_path, monkeypatch
) -> None:
    import json

    from dual.consistency import answer_pairs
    from dual.jobs.render import extend_render

    stem, calls = _fake_world(tmp_path, monkeypatch)
    extend_render(_args(), tmp_path)
    r1, r2 = (tmp_path / "runs" / n / "render" for n in ("render-1", "render-2"))
    assert (r2 / "rows-train.jsonl").read_bytes() == (
        r1 / "rows-train.jsonl"
    ).read_bytes()
    stems = json.loads((r2 / "stems.json").read_text())
    assert stems["voices"]["D"] == D_VOICE and set(stems["voices"]) == {
        "A",
        "B",
        "C",
        "D",
    }
    assert (stems["from"], stems["teacher_voice"]) == ("render-1", "D")
    assert stems["train"] == [stem]

    rows = {
        r["kind"]: r
        for r in map(json.loads, (r1 / "rows-train.jsonl").read_text().splitlines())
    }
    teach = [
        json.loads(x) for x in (r2 / "rows-teacher.jsonl").read_text().splitlines()
    ]
    assert [(t["kind"], t["pairs_with"], t["voice"]) for t in teach] == [
        ("voiced", "mask", "D"),
        ("voiced", "complement", "D"),
    ]
    assert [t["voiced_turns"] for t in teach] == [[0], [2]]
    for t in teach:
        assert len(answer_pairs(rows[t["pairs_with"]], t)) == len(
            answer_pairs(rows[t["pairs_with"]], rows["spoken"])
        )
        k = t["voiced_turns"][0]
        assert t["spans"][k]["rendering"] == "D"
    # the voice NAME reaches the TTS, never the label
    assert calls and {v for _, v in calls} == {D_VOICE}

    held = json.loads((r2 / "holdout.jsonl").read_text())
    old = json.loads((r1 / "holdout.jsonl").read_text())
    for k, per in held["renderings"].items():
        assert set(per) == {"typed", "A", "B", "C", "D"}
        assert {x: per[x] for x in ("typed", "A", "B", "C")} == old["renderings"][k]
    assert held["base"] == old["base"]


def test_extend_render_without_teacher_voice_writes_no_teacher_rows(
    tmp_path, monkeypatch
) -> None:
    from dual.jobs.render import extend_render

    _fake_world(tmp_path, monkeypatch)
    extend_render(_args(teacher_voice=None), tmp_path)
    assert not (
        tmp_path / "runs" / "render-2" / "render" / "rows-teacher.jsonl"
    ).exists()


def test_extend_render_refuses_a_sha_mismatch_itself_and_a_changed_tape(
    tmp_path, monkeypatch
) -> None:
    import json

    from dual.jobs import render as job

    _fake_world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="itself"):
        job.extend_render(_args(name="render-1"), tmp_path)
    (tmp_path / "SHA").write_text("zzz\n")
    with pytest.raises(SystemExit, match="chalkboard"):
        job.extend_render(_args(), tmp_path)
    (tmp_path / "SHA").write_text("abc\n")
    path = tmp_path / "runs" / "render-1" / "render" / "rows-train.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    for r in rows:
        if r["kind"] == "spoken":
            r["text"][11] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="rebuilt tape differs"):
        job.extend_render(_args(), tmp_path)
    # a render that failed part way leaves no stems.json: it never looks complete
    assert not (tmp_path / "runs" / "render-2" / "render" / "stems.json").exists()


def test_extend_render_refuses_a_missing_voice_file_before_any_tts(
    tmp_path, monkeypatch
) -> None:
    from dual.jobs import render as job

    _, calls = _fake_world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="no voice file"):
        job.extend_render(_args(add_voice=["D=vctk/p999_023.wav"]), tmp_path)
    assert calls == []
