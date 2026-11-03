"""Seed noise and the "did the recipe move the ratio" verdict (B6.0.2). Pure: no torch.

`python -m dual.noise runs/ --seeds A B --variants C D ...` reads the per-turn lines
`runs/measure/<label>/epoch-NN.jsonl` and prints the noise line (d, R0, T) and a row
per run. The median and p90 are `kl.bar`'s (`np.median`, `np.percentile(.., 90)`),
recomputed here because `dual.kl` imports torch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

Lines = Sequence[Mapping[str, Any]]
Interval = tuple[float, float, float]
Verdict = Literal["down", "up", "not moved"]


def _col(lines: Lines, key: str) -> Any:
    return np.asarray([x[key]["content"] for x in lines], dtype=float)


def _div(num: float, den: float) -> float:
    out = num / den if den != 0 else float("nan")
    if not np.isfinite(out):
        raise ValueError(f"non-finite ratio: {num} / {den}")
    return out


def ratio(lines: Lines) -> float:
    """Median content gap over median content floor."""
    if not lines:
        raise ValueError("no per-turn lines")
    return _div(
        float(np.median(_col(lines, "gap"))), float(np.median(_col(lines, "floor")))
    )


def p90_ratio(lines: Lines) -> float:
    return _div(
        float(np.percentile(_col(lines, "gap"), 90)),
        float(np.percentile(_col(lines, "floor"), 90)),
    )


def _passes(lines: Lines) -> bool:
    g, f = _col(lines, "gap"), _col(lines, "floor")
    return bool(
        np.median(g) <= np.median(f) and np.percentile(g, 90) <= np.percentile(f, 90)
    )


def _in_range(by_epoch: Mapping[int, Lines], max_epoch: int) -> list[int]:
    epochs = sorted(e for e in by_epoch if 1 <= e <= max_epoch and by_epoch[e])
    if not epochs:
        raise ValueError(f"no measured epoch in 1..{max_epoch}")
    return epochs


def pick(by_epoch: Mapping[int, Lines], max_epoch: int = 3) -> tuple[int, float]:
    """`report.best`'s rule over epochs 1..max_epoch: a passing epoch (median and p90)
    over one that does not, then the smallest median gap / median floor."""
    epochs = _in_range(by_epoch, max_epoch)
    e = min(epochs, key=lambda e: (not _passes(by_epoch[e]), ratio(by_epoch[e])))
    return e, ratio(by_epoch[e])


def lowest_gap(by_epoch: Mapping[int, Lines], max_epoch: int = 3) -> tuple[int, float]:
    """The ratio at the epoch with the lowest median content gap."""
    epochs = _in_range(by_epoch, max_epoch)
    e = min(epochs, key=lambda e: float(np.median(_col(by_epoch[e], "gap"))))
    return e, ratio(by_epoch[e])


def _by_stem(lines: Lines) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for x in lines:
        out.setdefault(str(x["stem"]), []).append(x)
    return out


def paired_ci(a: Lines, b: Lines, draws: int = 2000, seed: int = 0) -> Interval:
    """(ratio(a) - ratio(b), lo, hi): the 95% stem bootstrap of the difference. Each
    draw resamples the stems with replacement, the same stems on both sides."""
    sa, sb = _by_stem(a), _by_stem(b)
    if sa.keys() != sb.keys():
        missing = sorted(sa.keys() ^ sb.keys())
        raise ValueError(f"stems on one side only: {', '.join(missing)}")
    stems = sorted(sa)
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for i in range(draws):
        pick_ = [stems[j] for j in rng.integers(0, len(stems), len(stems))]
        ra = ratio([x for s in pick_ for x in sa[s]])
        rb = ratio([x for s in pick_ for x in sb[s]])
        deltas[i] = ra - rb
    q: Any = np.percentile(deltas, [2.5, 97.5])
    return ratio(a) - ratio(b), float(q[0]), float(q[1])


def _excludes_zero(ci: Interval) -> int:
    _, lo, hi = ci
    return 1 if lo > 0 else -1 if hi < 0 else 0


def verdict(
    r: float,
    r_s0: float,
    r_s1: float,
    ci_s0: Interval,
    ci_s1: Interval,
    threshold: float,
) -> Verdict:
    """Moved only if |R - R0| >= T and both paired intervals (vs seed 0, vs seed 1)
    exclude 0 with the same sign, which is also the sign of R - R0.

    Requiring the interval sign to match sign(R - R0) is an addition to B6.0.2 (stricter
    than the plan: an interval excluding 0 on the wrong side of R0 is "not moved")."""
    r0 = (r_s0 + r_s1) / 2
    s0, s1 = _excludes_zero(ci_s0), _excludes_zero(ci_s1)
    if abs(r - r0) < threshold or s0 == 0 or s0 != s1:
        return "not moved"
    if (s0 > 0) != (r > r0):
        return "not moved"
    return "up" if s0 > 0 else "down"


def read_epochs(root: pathlib.Path, label: str) -> dict[int, list[dict[str, Any]]]:
    """`root/measure/<label>/epoch-NN.jsonl` as {epoch: per-turn lines}; empty files are
    skipped."""
    d = root / "measure" / label
    if not d.is_dir():
        raise FileNotFoundError(f"no per-turn lines for {label!r}: {d}")
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(d.glob("epoch-*.jsonl")):
        m = re.fullmatch(r"epoch-(\d+)\.jsonl", path.name)
        if not m:
            continue
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if rows:
            out[int(m.group(1))] = rows
    return out


def _ci(ci: Interval) -> str:
    return f"{ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}]"


def report(
    runs: Mapping[str, Mapping[int, Lines]],
    seeds: Sequence[str],
    variants: Sequence[str],
    max_epoch: int = 3,
) -> str:
    if len(seeds) != 2:
        raise ValueError("--seeds takes exactly two labels")
    picked = {k: pick(v, max_epoch) for k, v in runs.items()}
    lines = {k: runs[k][picked[k][0]] for k in runs}
    r_s0, r_s1 = picked[seeds[0]][1], picked[seeds[1]][1]
    d = abs(r_s0 - r_s1)
    r0 = (r_s0 + r_s1) / 2
    t = max(2 * d, 0.5)
    out = [
        f"Seed noise: d = {d:.2f}, R0 = {r0:.2f}, T = max(2d, 0.5) = {t:.2f}",
        "",
        "| run | R (epoch) | p90 ratio | R_gap (epoch) | R - R0 | "
        f"95% CI vs {seeds[0]} | 95% CI vs {seeds[1]} | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in [*seeds, *variants]:
        e, r = picked[label]
        eg, rg = lowest_gap(runs[label], max_epoch)
        c0 = paired_ci(lines[label], lines[seeds[0]])
        c1 = paired_ci(lines[label], lines[seeds[1]])
        is_seed = label in seeds
        cell0 = "-" if label == seeds[0] else _ci(c0)
        cell1 = "-" if label == seeds[1] else _ci(c1)
        v = "-" if is_seed else verdict(r, r_s0, r_s1, c0, c1, t)
        out.append(
            f"| {label} | {r:.2f} ({e}) | {p90_ratio(lines[label]):.2f} | "
            f"{rg:.2f} ({eg}) | {r - r0:+.2f} | {cell0} | {cell1} | {v} |"
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dual.noise")
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("--seeds", nargs=2, required=True, metavar="LABEL")
    p.add_argument("--variants", nargs="*", default=[], metavar="LABEL")
    p.add_argument("--max-epoch", type=int, default=3)
    args = p.parse_args(argv)
    labels = [*args.seeds, *args.variants]
    runs = {k: read_epochs(args.root, k) for k in dict.fromkeys(labels)}
    print(report(runs, args.seeds, args.variants, args.max_epoch))


if __name__ == "__main__":
    main()
