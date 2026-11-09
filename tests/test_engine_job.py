import asyncio
import contextlib
import json
import pathlib
import sys
import time
import types

import pytest
import torch
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from dual.constants import GAP_FRAMES, PAD, SILENCE, SINE
from dual.engine import (
    AUDIO,
    FRAME_BYTES,
    READY,
    STATUS,
    TEXT,
    TYPED,
    Counters,
    pack,
    unpack,
)
from dual.jobs.engine import (
    CLIENT,
    DEP_Q,
    Engine,
    Session,
    make_app,
    new_state,
    parse_args,
    pcm_to_tensor,
    shutdown,
    tensor_to_pcm,
    wav_of,
)


def test_defaults_are_the_checkpoint_and_the_server():
    a = parse_args(["--name", "engine-x"])
    assert (a.base, a.run, a.epoch) == ("personaplex", "cons1-pp-l3", 3)
    assert (a.host, a.port, a.replay) == ("0.0.0.0", 8998, 0)
    assert a.max_minutes == 180.0 and not a.eager and not a.bare
    assert (a.temp, a.temp_text) == (0.0, 0.0)
    assert a.render == "render-1" and a.mem_cap_gib == 64.0
    assert DEP_Q == 8


def test_flags_are_parsed_and_checked():
    a = parse_args(
        ["--name", "r", "--replay", "13", "--eager", "--epoch", "2", "--bare"]
    )
    assert a.replay == 13 and a.eager and a.epoch == 2 and a.bare
    for bad in (
        ["--replay", "-1"],
        ["--max-minutes", "0"],
        ["--port", "0"],
        ["--temp", "-1"],
    ):
        with pytest.raises(SystemExit):
            parse_args(["--name", "r", *bad])


def test_pcm_round_trip_on_cpu():
    samples = torch.linspace(-1.0, 1.0, 1920).view(1, 1, -1)
    pcm = tensor_to_pcm(samples)
    assert len(pcm) == FRAME_BYTES
    back = pcm_to_tensor(pcm, "cpu")
    assert back.shape == (1, 1, 1920)
    assert torch.allclose(back, samples, atol=1 / 32767)
    with pytest.raises(ValueError, match="not one frame"):
        pcm_to_tensor(pcm[:-2], "cpu")


def test_wav_of_joins_frames():
    assert wav_of([]).shape == (0,)
    one = tensor_to_pcm(torch.full((1, 1, 1920), 0.5))
    wav = wav_of([one, one])
    assert wav.shape == (3840,)
    assert abs(float(wav[0]) - 0.5) < 1e-4


def test_client_page_is_next_to_the_package():
    assert pathlib.Path("src/dual/client.html").resolve() == CLIENT
    assert CLIENT.exists()


class _StubGen:
    """LMGen's surface: step() returns None for the first two steps, then [1, 9, 1]."""

    def __init__(self, hook):
        self.hook = hook
        self.n = 0
        self.seen: list[tuple[list[int], list[int] | None]] = []

    def streaming(self, batch):
        return contextlib.nullcontext()

    def step(self, pupil, depformer_replace_tokens=None):
        token = torch.tensor([7])  # what the model "sampled"
        self.hook(token)
        own = depformer_replace_tokens
        self.seen.append(
            (pupil.view(-1).tolist(), None if own is None else own.view(-1).tolist())
        )
        self.n += 1
        if self.n <= 2:
            return None
        return torch.cat([token.view(1, 1, 1), torch.full((1, 8, 1), 5)], dim=1)


class _StubMimi:
    def streaming(self, batch):
        return contextlib.nullcontext()

    def encode(self, pcm):
        return torch.full((1, 8, 1), 9)

    def decode(self, codes):
        return torch.zeros(1, 1, 1920)


def _engine(monkeypatch):
    # `from moshi.models.lm import LMGen` inside Engine.__init__ resolves to the stub
    # on a laptop without moshi: real module objects in sys.modules, parents included.
    def fake_lmgen(lm, **kw):
        return _StubGen(kw["on_text_hook"])

    moshi = types.ModuleType("moshi")
    models = types.ModuleType("moshi.models")
    lm = types.ModuleType("moshi.models.lm")
    lm.LMGen = fake_lmgen
    moshi.models, models.lm = models, lm
    for name, mod in (
        ("moshi", moshi),
        ("moshi.models", models),
        ("moshi.models.lm", lm),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    base = types.SimpleNamespace(lm=None)
    return Engine(base, _StubMimi(), "cpu", 0.0, 0.0)


def test_engine_steps_hear_and_type_on_cpu(monkeypatch):
    e = _engine(monkeypatch)
    with pytest.raises(RuntimeError, match="open"):
        e.step_codes(e.quiet)
    e.open()
    text, voice = e.hear(b"\x00" * FRAME_BYTES)
    assert text == 7 and voice is None  # delays not filled yet
    e.hear(b"\x00" * FRAME_BYTES)
    text, voice = e.hear(b"\x00" * FRAME_BYTES)
    assert voice is not None and len(voice) == FRAME_BYTES
    ids = e.type([11, 12])
    assert ids == [11, 12] + [PAD] * GAP_FRAMES  # forced, not sampled
    gen = e.lm_gen
    assert gen.seen[-1] == (list(SINE), list(SILENCE))
    assert e.counters.frames == 3 + 2 + GAP_FRAMES
    assert e.counters.heard == 3 and e.counters.typed == 1
    assert e.counters.hear.summary()["n"] == 3 and e.counters.burst.summary()["n"] == 1
    e.close()
    with pytest.raises(RuntimeError, match="open"):
        e.step_codes(e.quiet)


class FakeEngine:
    """Engine's surface for the session tests: no torch work, deterministic."""

    def __init__(self):
        self.counters = Counters()
        self.opened = 0
        self.closed = 0
        self.typed: list[list[int]] = []

    def open(self):
        self.opened += 1
        self.counters = Counters()

    def close(self):
        self.closed += 1

    def hear(self, frame):
        self.counters.frames += 1
        self.counters.heard += 1
        self.counters.hear.add(0.01)
        # a word every third frame; a voice frame from the second on
        text = 100 + self.counters.frames if self.counters.frames % 3 == 0 else PAD
        voice = (
            None
            if self.counters.frames < 2
            else bytes([self.counters.frames % 256]) * FRAME_BYTES
        )
        return text, voice

    def type(self, tokens):
        self.typed.append(list(tokens))
        ids = list(tokens) + [PAD] * GAP_FRAMES
        self.counters.frames += len(ids)
        self.counters.typed += 1
        self.counters.burst.add(0.05)
        return ids


class FakeSpm:
    # sentencepiece's own method names; the N rules are not enabled, so no noqa
    def EncodeAsIds(self, s):
        return [10 + i for i, _ in enumerate(s.split())]

    def IdToPiece(self, i):
        return f"\u2581w{i}"

    def DecodeIds(self, ids):
        return " ".join(f"w{i}" for i in ids)


def _drain(loop, outbox):
    out = []
    loop.run_until_complete(asyncio.sleep(0))
    while not outbox.empty():
        out.append(outbox.get_nowait())
    return out


def test_session_runs_hear_and_typed_and_saves(tmp_path):
    loop = asyncio.new_event_loop()
    try:
        eng, spm = FakeEngine(), FakeSpm()
        s = Session(eng, spm, tmp_path, loop)
        s.feed(pack(AUDIO, b"\x01\x00" * 1920 * 3))  # three mic frames at once
        s.feed(pack(TYPED, b"what is a derivative"))
        s.feed(
            pack(AUDIO, b"\x02\x00" * 1920)
        )  # the frame after the typed turn is dropped (D3)
        with pytest.raises(ValueError, match="server-only"):
            s.feed(pack(TEXT, b"x"))
        s.stop()
        s.run()  # the worker, inline
        msgs = _drain(loop, s.outbox)
        kinds = [unpack(m)[0] for m in msgs if m is not None]
        assert kinds[0] == READY
        assert AUDIO in kinds and TEXT in kinds and STATUS in kinds
        assert msgs[-1] is None
        assert eng.typed == [[10, 11, 12, 13]]  # typed_tokens: one sentence, punctuated
        assert eng.counters.frames == 3 + 4 + GAP_FRAMES
        assert eng.counters.dropped == 1
        assert eng.opened == 1 and eng.closed == 1
        s.save()
        d = s.dir
        assert (d / "steps.jsonl").exists() and (d / "pupil.wav").exists()
        assert (d / "moshi.wav").exists()
        meta = json.loads((d / "meta.json").read_text())
        assert meta["typed"][0]["words"] == ["what", "is", "a", "derivative"]
        assert meta["typed"][0]["at"] == 3
        assert meta["frames"] == 3 + 4 + GAP_FRAMES and meta["error"] is None
        assert meta["typed"][0]["dropped"] == 1
        rows = [json.loads(ln) for ln in (d / "steps.jsonl").read_text().splitlines()]
        assert [r["kind"] for r in rows] == ["hear"] * 3 + ["typed"] * 4 + [
            "gap"
        ] * GAP_FRAMES
    finally:
        loop.close()


def test_session_drops_the_backlog_but_never_a_typed_turn(tmp_path):
    loop = asyncio.new_event_loop()
    try:
        eng = FakeEngine()
        s = Session(eng, FakeSpm(), tmp_path, loop)
        for _ in range(30):
            s.inbox.put(("hear", b"\x00" * FRAME_BYTES))
        s.inbox.put(("type", "hi"))
        s.stop()
        s.run()
        # 30 queued: the first hear sees 29 mic frames behind it (the typed turn and
        # the stop do not count) -> drops 27, keeps 2; the typed turn survives
        assert eng.counters.dropped == 27
        assert eng.counters.heard == 3
        assert eng.typed == [[10]]
    finally:
        loop.close()


def test_session_records_an_engine_error(tmp_path):
    class Broken(FakeEngine):
        def hear(self, frame):
            raise RuntimeError("cuda went away")

    loop = asyncio.new_event_loop()
    try:
        eng = Broken()
        s = Session(eng, FakeSpm(), tmp_path, loop)
        s.inbox.put(("hear", b"\x00" * FRAME_BYTES))
        s.run()
        assert s.error == "RuntimeError('cuda went away')"
        assert eng.closed == 1
        assert _drain(loop, s.outbox)[-1] is None
    finally:
        loop.close()


def test_http_health_page_and_a_websocket_session(tmp_path):
    async def go():
        state = new_state("engine-test", tmp_path)
        async with TestClient(TestServer(make_app(state))) as client:
            r = await client.get("/health")
            body = await r.json()
            assert body["ready"] is False and body["busy"] is False
            r = await client.get("/")
            assert r.status == 200 and "AudioWorklet" in await r.text()
            r = await client.get("/ws")
            assert r.status == 503
            state["engine"], state["spm"] = FakeEngine(), FakeSpm()
            async with client.ws_connect("/ws") as ws:
                first = await ws.receive_bytes()
                assert unpack(first)[0] == READY
                await ws.send_bytes(pack(AUDIO, b"\x00" * FRAME_BYTES * 3))
                await ws.send_bytes(pack(TYPED, b"hello there"))
                got = []
                for _ in range(3):
                    got.append(unpack(await ws.receive_bytes())[0])
                assert AUDIO in got or TEXT in got or STATUS in got
                r = await client.get("/health")
                assert (await r.json())["busy"] is True
            await asyncio.sleep(1.5)
            r = await client.get("/health")
            body = await r.json()
            assert body["busy"] is False and body["sessions"] == 1
        dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(dirs) == 1 and (dirs[0] / "meta.json").exists()

    asyncio.run(go())


def test_websocket_refuses_a_cross_site_origin(tmp_path):
    from aiohttp import WSServerHandshakeError

    async def go():
        state = new_state("engine-origin", tmp_path)
        state["engine"], state["spm"] = FakeEngine(), FakeSpm()
        async with TestClient(TestServer(make_app(state))) as client:
            with pytest.raises(WSServerHandshakeError) as err:
                await client.ws_connect(
                    "/ws", headers={"Origin": "http://evil.example"}
                )
            assert err.value.status == 403
            assert state["lock"].locked() is False
            port = client.port
            for origin in (
                f"http://localhost:{port}",
                f"http://127.0.0.1:{port}",
                f"http://{client.host}:{port}",
            ):
                async with client.ws_connect("/ws", headers={"Origin": origin}) as ws:
                    assert unpack(await ws.receive_bytes())[0] == READY
                await asyncio.sleep(0.3)
            async with client.ws_connect("/ws") as ws:  # no Origin: not a browser
                assert unpack(await ws.receive_bytes())[0] == READY

    asyncio.run(go())


def test_pump_survives_a_dropped_client():
    from dual.jobs.engine import pump

    class DeadWs:
        closed = False

        def __init__(self):
            self.sent = 0

        async def send_bytes(self, message):
            self.sent += 1
            raise ConnectionResetError("client gone")

        async def close(self):
            raise ConnectionResetError("client gone")

    async def go():
        session = types.SimpleNamespace(outbox=asyncio.Queue())
        for m in (b"a", b"b", b"c", None):
            session.outbox.put_nowait(m)
        ws = DeadWs()
        await pump(session, ws)
        assert ws.sent == 1  # nothing sent after the first failure
        assert session.outbox.empty()

    asyncio.run(go())


def test_session_posts_the_sentinel_when_close_raises(tmp_path):
    class BadClose(FakeEngine):
        def close(self):
            raise RuntimeError("close failed")

    loop = asyncio.new_event_loop()
    try:
        s = Session(BadClose(), FakeSpm(), tmp_path, loop)
        s.stop()
        s.run()
        assert _drain(loop, s.outbox)[-1] is None
        assert s.error is not None and "close failed" in s.error
    finally:
        loop.close()


def test_session_refuses_blank_and_oversized_typed_text(tmp_path, capsys):
    from dual.jobs.engine import MAX_TYPED_WORDS

    assert MAX_TYPED_WORDS == 60
    loop = asyncio.new_event_loop()
    try:
        eng = FakeEngine()
        s = Session(eng, FakeSpm(), tmp_path, loop)
        s.feed(pack(TYPED, b"   \n\t "))
        s.feed(pack(TYPED, " ".join(["word"] * 61).encode()))
        s.feed(pack(AUDIO, b"\x01\x00" * 1920))  # still heard afterwards
        s.stop()
        s.run()
        assert eng.typed == [] and eng.counters.heard == 1 and s.error is None
        assert "refused" in capsys.readouterr().out
        s2 = Session(FakeEngine(), FakeSpm(), tmp_path, loop)
        s2.feed(pack(TYPED, " ".join(["word"] * 60).encode()))  # at the cap: fine
        s2.stop()
        s2.run()
        assert len(s2.engine.typed) == 1
    finally:
        loop.close()


class _StubRunner:
    def __init__(self):
        self.cleaned = 0

    async def cleanup(self):
        self.cleaned += 1


def test_shutdown_saves_the_open_session_and_closes_the_socket(tmp_path):
    async def go():
        state = new_state("engine-test", tmp_path)
        state["engine"], state["spm"] = FakeEngine(), FakeSpm()
        runner = _StubRunner()
        async with (
            TestClient(TestServer(make_app(state))) as client,
            client.ws_connect("/ws") as ws,
        ):
            assert unpack(await ws.receive_bytes())[0] == READY
            await ws.send_bytes(pack(AUDIO, b"\x00" * FRAME_BYTES * 3))
            assert len(state["open"]) == 1
            await shutdown(state, runner)
            assert runner.cleaned == 1
            assert state["sessions"] == 1 and not state["open"]
            dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
            assert len(dirs) == 1 and (dirs[0] / "meta.json").exists()
            while True:  # the client sees the socket close
                msg = await asyncio.wait_for(ws.receive(), 5)
                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                    break

    asyncio.run(go())


def test_shutdown_with_no_session_just_cleans_up(tmp_path):
    async def go():
        state = new_state("engine-test", tmp_path)
        runner = _StubRunner()
        await shutdown(state, runner)
        assert runner.cleaned == 1 and state["sessions"] == 0

    asyncio.run(go())


def test_websocket_refuses_new_sessions_while_closing(tmp_path):
    from aiohttp import WSServerHandshakeError

    async def go():
        state = new_state("engine-closing", tmp_path)
        state["engine"], state["spm"] = FakeEngine(), FakeSpm()
        state["closing"] = True
        async with TestClient(TestServer(make_app(state))) as client:
            with pytest.raises(WSServerHandshakeError) as err:
                await client.ws_connect("/ws")
            assert err.value.status == 503
            assert state["lock"].locked() is False

    asyncio.run(go())


def test_shutdown_gives_up_after_its_budget(tmp_path):
    class Stuck:
        def stop(self):
            pass

    async def go():
        state = new_state("engine-budget", tmp_path)
        state["open"].add(Stuck())
        runner = _StubRunner()
        t0 = time.monotonic()
        await shutdown(state, runner, budget=0.2)
        assert time.monotonic() - t0 < 1.0
        assert runner.cleaned == 1
        assert state["closing"] is True

    asyncio.run(go())


def test_engine_close_clears_its_state_when_the_stack_raises(monkeypatch):
    e = _engine(monkeypatch)

    class Boom:
        def close(self):
            raise RuntimeError("exit failed")

    e._stack = Boom()
    with pytest.raises(RuntimeError, match="exit failed"):
        e.close()
    assert not e.is_open
    e.open()  # not stuck "already open"
    e.close()
