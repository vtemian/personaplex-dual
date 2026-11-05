"""The bar by position in the answer (B8.1). Pure: no torch.

A frame's position is its index in the answer window minus `f0`, the first frame whose
target text is not PAD. `lead` (position < 0) is the silence before the tutor's first
word. `python -m dual.position runs/measure/<label>/epoch-NN.jsonl ...` reads the
measure job's per-turn lines (with `content_mask` and `per_frame`) and prints markdown.
The median and p90 are `kl.bar`'s (`np.median`, `np.percentile(.., 90)`), recomputed
here because `dual.kl` imports torch.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Lines = Sequence[Mapping[str, Any]]

BUCKETS = ("lead", "0", "1", "2", "3-5", "6-10", "11-25", "26+")
MODES = ("content", "all")
CUTS: tuple[int | None, ...] = (None, 0, 1, 2, 3, 6, 11)
_EDGES = ((1, "0"), (2, "1"), (3, "2"), (6, "3-5"), (11, "6-10"), (26, "11-25"))


def bucket(pos: int) -> str:
    if pos < 0:
        return "lead"
    for upper, name in _EDGES:
        if pos < upper:
            return name
    return "26+"


def positions(content_mask: Sequence[int]) -> list[int]:
    """Each frame's index minus that of the first content frame."""
    try:
        f0 = next(i for i, m in enumerate(content_mask) if m)
    except StopIteration:
        raise ValueError("no content frame in the answer window") from None
    return [i - f0 for i in range(len(content_mask))]


def _frames(
    per_frame: Sequence[float], content_mask: Sequence[int], content_only: bool
) -> list[tuple[int, float]]:
    """(position, KL) for the frames counted in the mode."""
    if len(per_frame) != len(content_mask):
        raise ValueError(
            f"per_frame has {len(per_frame)} frames, content_mask {len(content_mask)}"
        )
    pos = positions(content_mask)
    return [
        (p, float(v))
        for p, v, m in zip(pos, per_frame, content_mask, strict=True)
        if m or not content_only
    ]


def turn_means(
    per_frame: Sequence[float], content_mask: Sequence[int], *, content_only: bool
) -> dict[str, float | None]:
    """One turn's mean KL per bucket; None where the bucket has no frames."""
    by: dict[str, list[float]] = {b: [] for b in BUCKETS}
    for p, v in _frames(per_frame, content_mask, content_only):
        by[bucket(p)].append(v)
    return {b: float(np.mean(v)) if v else None for b, v in by.items()}


def _content_only(mode: str) -> bool:
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}, got {mode!r}")
    return mode == "content"


def _median(xs: Sequence[float]) -> float | None:
    return float(np.median(np.asarray(xs, dtype=float))) if xs else None


def _div(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def bucket_table(lines: Lines, mode: str) -> list[dict[str, Any]]:
    """One row per bucket: turns and frames counted, median over turns of the per-turn
    bucket mean for gap, floor and door, their ratio, and each bucket's share of the
    pooled gap and floor KL mass in the mode."""
    only = _content_only(mode)
    if not lines:
        raise ValueError("no lines")
    means: dict[str, dict[str, list[float]]] = {
        k: {b: [] for b in BUCKETS} for k in ("gap", "floor", "door")
    }
    frames = dict.fromkeys(BUCKETS, 0)
    mass = {k: dict.fromkeys(BUCKETS, 0.0) for k in ("gap", "floor")}
    for x in lines:
        mask = x["content_mask"]
        for k in means:
            for b, m in turn_means(x["per_frame"][k], mask, content_only=only).items():
                if m is not None:
                    means[k][b].append(m)
        for p, _ in _frames(x["per_frame"]["gap"], mask, only):
            frames[bucket(p)] += 1
        for k in mass:
            for p, v in _frames(x["per_frame"][k], mask, only):
                mass[k][bucket(p)] += v
    total = {k: sum(v.values()) for k, v in mass.items()}
    rows = []
    for b in BUCKETS:
        mg, mf = _median(means["gap"][b]), _median(means["floor"][b])
        rows.append(
            {
                "bucket": b,
                "turns": len(means["gap"][b]),
                "frames": frames[b],
                "median_gap": mg,
                "median_floor": mf,
                "median_door": _median(means["door"][b]),
                "ratio": _div(mg, mf),
                "gap_share": mass["gap"][b] / total["gap"],
                "floor_share": mass["floor"][b] / total["floor"],
            }
        )
    return rows


def bar_from(lines: Lines, min_pos: int | None, mode: str) -> dict[str, Any]:
    """The bar on per-turn means over the frames in the mode at position >= min_pos
    (None: no cut, so `all` mode includes the lead). Turns with no such frame are
    skipped; `turns` counts those that remain."""
    only = _content_only(mode)
    cols: dict[str, list[float]] = {"gap": [], "floor": []}
    used = 0
    for x in lines:
        kept = {
            k: [
                v
                for p, v in _frames(x["per_frame"][k], x["content_mask"], only)
                if min_pos is None or p >= min_pos
            ]
            for k in cols
        }
        if not kept["gap"]:
            continue
        used += len(kept["gap"])
        for k in cols:
            cols[k].append(float(np.mean(kept[k])))
    out: dict[str, Any] = {"turns": len(cols["gap"]), "frames": used}
    if not cols["gap"]:
        return out | dict.fromkeys(
            ("median_gap", "median_floor", "p90_gap", "p90_floor", "ratio")
        )
    g, f = np.asarray(cols["gap"]), np.asarray(cols["floor"])
    out |= {
        "median_gap": float(np.median(g)),
        "median_floor": float(np.median(f)),
        "p90_gap": float(np.percentile(g, 90)),
        "p90_floor": float(np.percentile(f, 90)),
    }
    out["ratio"] = _div(out["median_gap"], out["median_floor"])
    return out


def _kl(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def _r(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def _scalar_check(lines: Lines, key: str) -> str:
    g = _median([x["gap"][key] for x in lines])
    f = _median([x["floor"][key] for x in lines])
    return f"{key} {_kl(g)} / {_kl(f)} = {_r(_div(g, f))}"


def render(lines: Lines, title: str) -> str:
    stems = len({x["stem"] for x in lines})
    out = [
        f"### {title}",
        "",
        f"{len(lines)} turns, {stems} stems. Per-turn scalars, median gap / median "
        f"floor: {_scalar_check(lines, 'content')}; "
        f"{_scalar_check(lines, 'all_frames')}.",
    ]
    for mode in MODES:
        rows = [
            [
                r["bucket"],
                str(r["turns"]),
                str(r["frames"]),
                _kl(r["median_gap"]),
                _kl(r["median_floor"]),
                _r(r["ratio"]),
                _kl(r["median_door"]),
                _pct(r["gap_share"]),
                _pct(r["floor_share"]),
            ]
            for r in bucket_table(lines, mode)
        ]
        head = ["bucket", "turns", "frames", "median gap", "median floor", "ratio"]
        head += ["median door", "gap share", "floor share"]
        out += ["", f"By position, {mode} frames:", "", *_table(head, rows)]
    rows = []
    for mode in MODES:
        for cut in CUTS:
            b = bar_from(lines, cut, mode)
            rows.append(
                [
                    mode,
                    "all" if cut is None else str(cut),
                    str(b["turns"]),
                    str(b["frames"]),
                    _kl(b["median_gap"]),
                    _kl(b["median_floor"]),
                    _r(b["ratio"]),
                    _kl(b["p90_gap"]),
                    _kl(b["p90_floor"]),
                ]
            )
    head = ["mode", "from position", "turns", "frames", "median gap", "median floor"]
    head += ["ratio", "p90 gap", "p90 floor"]
    return "\n".join([*out, "", "Bar from position:", "", *_table(head, rows)])


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m dual.position epoch-NN.jsonl ...", file=sys.stderr)
        raise SystemExit(2)
    for a in args:
        path = pathlib.Path(a)
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        print(render(lines, f"{path.resolve().parent.name}/{path.name}"))
        print()


if __name__ == "__main__":
    main()
