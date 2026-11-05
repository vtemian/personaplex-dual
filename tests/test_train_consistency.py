import functools
import json
import types

import pytest
import torch
import torch.nn.functional as F

import dual.jobs.train
import dual.model
from dual.constants import SINE
from dual.jobs.train import build_teachers, losses, main
from dual.model import Forward
from test_consistency import make_rows

V = 64


class FakeLM(torch.nn.Module):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(c))


def fake_forward(lm, codes):
    """Text logits peaked on each frame's own text id, tilted by how many typed (SINE)
    pupil frames came before: typed and spoken renderings differ iff lm.c != 0."""
    text = codes[:, 0]
    b, t = text.shape
    seen = torch.cumsum((codes[:, 9] == SINE[0]).float(), dim=-1)
    tilt = torch.arange(V, dtype=torch.float32) / V
    logits = 3.0 * F.one_hot(text % V, V).float() + lm.c * seen[..., None] * tilt
    return Forward(
        text_logits=logits,
        text_mask=torch.ones(b, t, dtype=torch.bool),
        audio_logits=torch.zeros(b, 8, t, 2048),
        audio_mask=torch.ones(b, 8, t, dtype=torch.bool),
    )


@pytest.fixture(autouse=True)
def cpu_model(monkeypatch):
    monkeypatch.setattr(dual.model, "forward", fake_forward)
    real = dual.model.codes_from_row
    monkeypatch.setattr(
        dual.model, "codes_from_row", functools.partial(real, device="cpu")
    )


def test_teachers_cover_the_typed_row_only_at_its_answer_content_frames():
    typed, spoken = make_rows([1])
    teachers = build_teachers(FakeLM(1.0), [spoken, typed], {"s": spoken})
    assert list(teachers) == [1]
    idx, lp = teachers[1]
    assert idx.tolist() == [18, 19, 20, 21, 22]
    assert lp.shape == (5, V)
    assert not lp.requires_grad


def test_zero_when_typed_and_spoken_agree():
    typed, spoken = make_rows([1, 3])
    lm = FakeLM(0.0)
    teachers = build_teachers(lm, [typed], {"s": spoken})
    _, _, cons = losses(lm, [typed], teachers, "bar")
    assert float(cons.detach()) == 0.0


def test_positive_and_trainable_when_they_differ():
    typed, spoken = make_rows([1, 3])
    lm = FakeLM(1.0)
    teachers = build_teachers(lm, [typed], {"s": spoken})
    _, _, cons = losses(lm, [typed], teachers, "bar")
    assert float(cons.detach()) > 0
    cons.backward()
    assert lm.c.grad is not None and float(lm.c.grad) != 0.0


def test_the_lm_losses_do_not_change_with_the_term():
    typed, spoken = make_rows([1])
    lm = FakeLM(1.0)
    base = losses(lm, [typed, spoken])
    with_term = losses(
        lm, [typed, spoken], build_teachers(lm, [typed, spoken], {"s": spoken})
    )
    assert torch.equal(base[0], with_term[0]) and torch.equal(base[1], with_term[1])
    assert float(base[2]) == 0.0


def test_teacher_frames_line_up_inside_a_padded_batch():
    # the short typed row is padded to the long one's width; indices stay its own
    typed, spoken = make_rows([1])
    long_typed, long_spoken = make_rows([1, 3], stem="t")
    lm = FakeLM(0.0)
    batch = [typed, long_typed]
    teachers = build_teachers(lm, batch, {"s": spoken, "t": long_spoken})
    _, _, cons = losses(lm, batch, teachers, "bar")
    assert float(cons.detach()) == 0.0


# main() end to end on CPU: the fake forward above, a fake base, the real sparks track
# (no job env: it reports nothing but still checks the metric names) and a tiny render.


@pytest.fixture
def job(tmp_path, monkeypatch):
    for var in ("SPARKS_RUN_ID", "SPARKS_PROMETHEUS_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    render = tmp_path / "runs" / "render-x" / "render"
    render.mkdir(parents=True)
    s_typed, s_spoken = make_rows([1])
    t_typed, t_spoken = make_rows([1, 3], stem="t")
    rows = [s_typed, s_spoken, t_typed, t_spoken]
    (render / "stems.json").write_text(json.dumps({"train": ["s", "t"]}))
    (render / "rows-train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (render / "holdout.jsonl").write_text(json.dumps({"base": s_spoken}) + "\n")
    lm = FakeLM(1.0)
    monkeypatch.setattr(
        dual.model, "load_base", lambda name: types.SimpleNamespace(lm=lm)
    )
    monkeypatch.setattr(dual.model, "attach_lora", lambda base, rank, scaling: 0)
    monkeypatch.setattr(dual.model, "load_blob_adapters", lambda base, blob: None)

    def save(base, path, extra=None, resume=None):
        torch.save({"run": extra, "resume": resume}, path)

    monkeypatch.setattr(dual.model, "save_adapters", save)
    calls = []
    real = dual.jobs.train.build_teachers

    def counted(*a):
        calls.append(len(a[1]))
        return real(*a)

    monkeypatch.setattr(dual.jobs.train, "build_teachers", counted)

    def run(*flags):
        argv = ["train", "--name", "n", "--base", "b", "--render", "render-x"]
        argv += ["--frames-per-batch", "40"]  # one row a batch: 4 steps an epoch
        monkeypatch.setattr("sys.argv", [*argv, *flags])
        main()
        return tmp_path / "runs" / "n" / "train"

    return types.SimpleNamespace(run=run, calls=calls, lm=lm)


def test_main_smoke_runs_the_term_and_writes_no_checkpoint(job, capsys):
    out = job.run("--consistency", "10", "--max-steps", "2", "--epochs", "1")
    log = capsys.readouterr().out
    assert "typed rows 2, answer pairs 3, content frames 13" in log
    assert "smoke: 2 steps, peak mem 0.0 GiB; no checkpoint written" in log
    assert job.calls  # the teacher ran
    assert list(out.glob("*.pt*")) == []
    assert job.lm.c.grad is not None


def test_main_at_zero_never_builds_a_teacher(job, capsys):
    out = job.run("--max-steps", "2", "--epochs", "1")
    log = capsys.readouterr().out
    assert "consistency 0.0 (bar); typed rows 2;" not in log
    assert "consistency 0.0 (bar); typed rows 2\n" in log
    assert " cons 0.00000 (0 frames)" in log
    assert "smoke: 2 steps" in log
    assert job.calls == []
    assert list(out.glob("*.pt*")) == []


def test_main_full_epoch_saves_the_recipe_and_resume_keeps_it(job, capsys):
    out = job.run("--consistency", "10", "--epochs", "1", "--only-stem", "longest")
    log = capsys.readouterr().out
    assert "2 dialogues, 2 rows, 1 held, 2 steps/epoch" in log  # the longest stem
    assert "mean cons" in log and "held text" in log
    blob = torch.load(out / "model-epoch-01.pt", weights_only=True)
    assert blob["run"]["consistency"] == 10.0
    assert blob["run"]["consistency_dir"] == "bar"
    assert blob["run"]["only_stem"] == "longest"
    with pytest.raises(SystemExit, match="consistency"):
        job.run("--consistency", "3", "--epochs", "2", "--resume")


def test_main_smoke_past_the_last_step_still_writes_no_checkpoint(job, capsys):
    out = job.run("--consistency", "10", "--max-steps", "9", "--epochs", "2")
    assert "smoke: 8 steps" in capsys.readouterr().out
    assert list(out.glob("*.pt*")) == []
