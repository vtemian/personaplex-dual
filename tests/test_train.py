import random

from dual.jobs.train import batches


def rows(*frames):
    return [{"id": i, "frames": f} for i, f in enumerate(frames)]


def test_every_row_lands_in_exactly_one_batch():
    rs = rows(100, 2765, 700, 650, 900, 1200, 30)
    out = batches(rs, random.Random(0), limit=4000)
    assert sorted(r["id"] for b in out for r in b) == list(range(len(rs)))


def test_a_padded_batch_stays_under_the_limit_unless_one_row_is_over():
    rs = rows(*[500 + 37 * i for i in range(40)], 5000)
    for b in batches(rs, random.Random(1), limit=4000):
        width = max(r["frames"] for r in b)
        assert len(b) * width <= 4000 or len(b) == 1


def test_batches_group_similar_lengths():
    rs = rows(100, 110, 3000, 120)
    out = batches(rs, random.Random(0), limit=4000)
    assert sorted(sorted(r["frames"] for r in b) for b in out) == [
        [100, 110, 120],
        [3000],
    ]


def test_the_seed_fixes_the_order():
    rs = rows(*[100 * i + 50 for i in range(30)])
    a = batches(rs, random.Random(3), limit=1000)
    b = batches(rs, random.Random(3), limit=1000)
    assert [[r["id"] for r in x] for x in a] == [[r["id"] for r in x] for x in b]


def test_latest_checkpoint_is_the_highest_epoch(tmp_path):
    from dual.jobs.train import latest_checkpoint

    assert latest_checkpoint(tmp_path) is None
    for n in (1, 2, 10):
        (tmp_path / f"model-epoch-{n:02d}.pt").write_bytes(b"")
    (tmp_path / "model-epoch-11.pt.tmp").write_bytes(b"")
    epoch, path = latest_checkpoint(tmp_path)
    assert (epoch, path.name) == (10, "model-epoch-10.pt")


def test_stale_guard_refuses_checkpoints_without_resume(tmp_path):
    import pytest

    from dual.jobs.train import refuse_stale

    refuse_stale(tmp_path, resume=False)
    (tmp_path / "model-epoch-01.pt").write_bytes(b"")
    refuse_stale(tmp_path, resume=True)
    with pytest.raises(SystemExit, match="--resume"):
        refuse_stale(tmp_path, resume=False)


def test_rng_state_round_trips_through_a_weights_only_file(tmp_path):
    import random

    import torch

    from dual.jobs.train import capture_rng, restore_rng

    rng = random.Random(5)
    state = capture_rng(rng)
    want = (rng.random(), random.random(), float(torch.rand(1)))
    f = tmp_path / "s.pt"
    torch.save({"rng": state}, f)
    restore_rng(rng, torch.load(f, weights_only=True)["rng"])
    got = (rng.random(), random.random(), float(torch.rand(1)))
    assert got == want


def test_optimizer_state_loads_weights_only_and_resumes(tmp_path):
    import random

    import torch

    from dual.jobs.train import capture_rng

    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([w], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    w.sum().backward()
    opt.step()
    f = tmp_path / "model-epoch-01.pt"
    torch.save(
        {
            "state": {"w": w.detach()},
            "run": {"epoch": 1},
            "resume": {"opt": opt.state_dict(), "rng": capture_rng(random.Random(0))},
        },
        f,
    )
    blob = torch.load(f, map_location="cpu", weights_only=True)
    opt2 = torch.optim.AdamW([w], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)
    opt2.load_state_dict(blob["resume"]["opt"])
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
