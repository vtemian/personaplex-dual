import functools

import pytest
import torch
import torch.nn.functional as F

import dual.model
from dual.constants import SINE
from dual.jobs.train import build_teachers, losses
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
