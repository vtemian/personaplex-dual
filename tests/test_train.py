import random

import pytest

from dual.jobs.train import (
    batches,
    check_resume_consistency,
    check_resume_lr,
    parse_args,
    select_rows,
    widest_first,
)


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


def test_recipe_flags_default_to_ladder_1(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--name", "n", "--base", "b"])
    a = parse_args()
    assert (a.rank, a.scaling, a.lr) == (16, 2.0, 2e-5)


def test_recipe_flags_are_parsed(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--name",
            "n",
            "--base",
            "b",
            "--rank",
            "64",
            "--scaling",
            "0.5",
            "--lr",
            "1e-4",
        ],
    )
    a = parse_args()
    assert (a.rank, a.scaling, a.lr) == (64, 0.5, 1e-4)


def test_resume_accepts_the_same_lr_and_a_ladder_1_checkpoint():
    check_resume_lr({"lr": 1e-4}, 1e-4)
    check_resume_lr({"epoch": 1}, 2e-5)  # missing key = the old default


def test_resume_refuses_a_different_lr():
    with pytest.raises(ValueError, match="lr"):
        check_resume_lr({"lr": 2e-5}, 1e-4)
    with pytest.raises(ValueError, match="lr"):
        check_resume_lr({}, 1e-4)


def test_consistency_flags_default_to_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--name", "n", "--base", "b"])
    a = parse_args()
    assert (a.consistency, a.consistency_dir) == (0.0, "bar")
    assert (a.only_stem, a.max_steps, a.widest_first) == (None, 0, False)


def test_consistency_flags_are_parsed(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--name",
            "n",
            "--base",
            "b",
            "--consistency",
            "10",
            "--consistency-dir",
            "forward",
            "--only-stem",
            "longest",
            "--max-steps",
            "3",
            "--widest-first",
        ],
    )
    a = parse_args()
    assert (a.consistency, a.consistency_dir) == (10.0, "forward")
    assert (a.only_stem, a.max_steps, a.widest_first) == ("longest", 3, True)


def test_a_negative_weight_is_refused(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train", "--name", "n", "--base", "b", "--consistency", "-1"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_resume_accepts_the_same_consistency_and_an_older_checkpoint():
    check_resume_consistency(
        {"consistency": 10.0, "consistency_dir": "bar"}, 10.0, "bar"
    )
    check_resume_consistency({"epoch": 1}, 0.0, "bar")  # predates the flag: off


def test_resume_refuses_another_consistency_setup():
    with pytest.raises(ValueError, match="consistency"):
        check_resume_consistency({"consistency": 10.0}, 3.0, "bar")
    with pytest.raises(ValueError, match="consistency"):
        check_resume_consistency({"consistency": 10.0}, 10.0, "forward")


def test_select_rows_keeps_one_stem_or_the_longest():
    rs = [
        {"stem": "a", "frames": 5},
        {"stem": "b", "frames": 9},
        {"stem": "b", "frames": 2},
    ]
    assert select_rows(rs, None) == rs
    assert select_rows(rs, "a") == [rs[0]]
    assert select_rows(rs, "longest") == [rs[1], rs[2]]
    with pytest.raises(ValueError, match="no rows"):
        select_rows(rs, "zzz")


def test_widest_first_orders_batches_by_padded_frames():
    bs = [[{"frames": 10}] * 3, [{"frames": 100}], [{"frames": 40}] * 2]
    assert [len(b) * max(r["frames"] for r in b) for b in widest_first(bs)] == [
        100,
        80,
        30,
    ]


def test_teacher_defaults_to_the_tape(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--name", "n", "--base", "b"])
    assert parse_args().teacher == "tape"


def test_teacher_voiced_is_parsed_and_needs_the_term(monkeypatch):
    argv = ["train", "--name", "n", "--base", "b", "--teacher", "voiced"]
    monkeypatch.setattr("sys.argv", [*argv, "--consistency", "3"])
    assert parse_args().teacher == "voiced"
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        parse_args()


def test_resume_refuses_another_teacher():
    meta = {"consistency": 3.0, "consistency_dir": "bar"}
    check_resume_consistency(meta, 3.0, "bar", "tape")  # predates B15: the tape
    with pytest.raises(ValueError, match="teacher"):
        check_resume_consistency(meta, 3.0, "bar", "voiced")
    check_resume_consistency(meta | {"teacher": "voiced"}, 3.0, "bar", "voiced")
