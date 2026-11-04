import os
from types import SimpleNamespace

import pytest
import torch

from dual.jobs import measure, train
from dual.model import MEM_CAP_GIB, cap_gpu_fraction, cap_gpu_memory

GIB = 2**30


def test_fraction_is_cap_over_total():
    assert cap_gpu_fraction(64.0, 128 * GIB) == 0.5


def test_fraction_allows_the_whole_card():
    assert cap_gpu_fraction(128.0, 128 * GIB) == 1.0


@pytest.mark.parametrize("cap", [0.0, -1.0])
def test_fraction_refuses_a_cap_that_is_not_positive(cap):
    with pytest.raises(ValueError):
        cap_gpu_fraction(cap, 128 * GIB)


def test_fraction_refuses_a_cap_over_the_card():
    with pytest.raises(ValueError):
        cap_gpu_fraction(129.0, 128 * GIB)


def test_default_cap_is_64_gib():
    assert MEM_CAP_GIB == 64.0


@pytest.mark.parametrize("job", [train, measure])
def test_both_jobs_default_the_flag(job, monkeypatch):
    monkeypatch.setattr("sys.argv", ["j", "--name", "n", "--base", "b"])
    assert job.parse_args().mem_cap_gib == MEM_CAP_GIB


@pytest.mark.parametrize("job", [train, measure])
def test_both_jobs_take_the_flag(job, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["j", "--name", "n", "--base", "b", "--mem-cap-gib", "0"]
    )
    assert job.parse_args().mem_cap_gib == 0.0


def test_no_cuda_sets_env_and_no_fraction(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a)
    )
    assert cap_gpu_memory(64.0) == 0.0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert calls == []


def test_env_is_not_overwritten(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cap_gpu_memory(64.0)
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"


def test_cuda_sets_the_fraction_on_device_zero(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda i: SimpleNamespace(total_memory=128 * GIB),
    )
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a)
    )
    assert cap_gpu_memory(64.0) == 0.5
    assert calls == [(0.5, 0)]
    out = capsys.readouterr().out
    assert "64" in out and "0.5" in out


def test_zero_disables_the_cap_but_sets_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a)
    )
    assert cap_gpu_memory(0.0) == 0.0
    assert calls == []
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
