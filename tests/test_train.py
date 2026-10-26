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
