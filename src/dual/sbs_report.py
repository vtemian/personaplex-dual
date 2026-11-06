"""Side-by-side report: what a model says after typed vs spoken pupil turns. Pure.

`python -m dual.sbs_report rows.jsonl [--judgments j.jsonl] [--models a,b]` prints
the pair statistics, the reader's judgment counts if given, and one block per turn.
`--quality` adds the answer-quality and first-word tables.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import pathlib
import statistics
from collections.abc import Sequence
from typing import Any

from dual.constants import PAD
from dual.probe_text import words_of

PAIRS = (("typed", "A"), ("A", "B"), ("typed", "B"))
RENDERINGS = ("typed", "A", "B")
STATS_COLUMNS = (
    "model",
    "pair",
    "n",
    "both silent",
    "one silent",
    "exact",
    "exact rate",
    "median similarity",
    "similarity >= 0.8",
)


def load(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def similarity(a: str, b: str) -> float:
    wa, wb = words_of(a), words_of(b)
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb).ratio()


def _index(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    out: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for r in rows:
        if r["rendering"] not in RENDERINGS:
            continue  # swap renderings belong to the quality tables only
        key = (r["model"], r["stem"], int(r["turn"]), r["rendering"])
        if key in out:
            raise ValueError(f"duplicate row {key}")
        out[key] = r
    return out


def pair_stats(
    rows: Sequence[dict[str, Any]], model: str, p: str, q: str
) -> dict[str, Any]:
    idx = _index(rows)
    answers = [
        (idx[(m, s, t, p)]["answer"], idx[(m, s, t, q)]["answer"])
        for (m, s, t, rd) in idx
        if m == model and rd == p and (m, s, t, q) in idx
    ]
    # Two silent answers are not an agreement in what was said: counted apart, and
    # the exact and similarity figures are over the pairs where at least one spoke.
    said = [(a, b) for a, b in answers if words_of(a) or words_of(b)]
    sims = [similarity(a, b) for a, b in said]
    exact = sum(words_of(a) == words_of(b) for a, b in said)
    return {
        "model": model,
        "p": p,
        "q": q,
        "n": len(answers),
        "both_empty": len(answers) - len(said),
        "one_empty": sum(not words_of(a) or not words_of(b) for a, b in said),
        "said": len(said),
        "exact": exact,
        "exact_rate": exact / len(said) if said else None,
        "median_sim": statistics.median(sims) if sims else None,
        "sim_ge_0_8": sum(s >= 0.8 for s in sims),
        "empty_p": sum(not words_of(a) for a, _ in answers),
        "empty_q": sum(not words_of(b) for _, b in answers),
    }


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def stats_table(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> str:
    body = []
    for m in models:
        for p, q in PAIRS:
            s = pair_stats(rows, m, p, q)
            body.append(
                [
                    m,
                    f"{p} vs {q}",
                    str(s["n"]),
                    str(s["both_empty"]),
                    str(s["one_empty"]),
                    str(s["exact"]),
                    _fmt(s["exact_rate"]),
                    _fmt(s["median_sim"]),
                    str(s["sim_ge_0_8"]),
                ]
            )
    return _table(STATS_COLUMNS, body)


def _cell(answer: str) -> str:
    if not words_of(answer):
        return "(silence)"
    return " ".join(answer.replace("|", "\\|").split())


def turn_blocks(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> str:
    idx = _index(rows)
    turns = sorted({(s, t) for (_, s, t, _) in idx})
    blocks = []
    for stem, turn in turns:
        first = next(r for (m, s, t, _), r in idx.items() if (s, t) == (stem, turn))
        body = [
            [rd]
            + [
                _cell(idx[(m, stem, turn, rd)]["answer"])
                if (m, stem, turn, rd) in idx
                else ""
                for m in models
            ]
            for rd in RENDERINGS
        ]
        blocks.append(
            f"### {stem} turn {turn}\n\n"
            f"**Pupil:** {first['pupil']}\n\n"
            f"**Tape tutor:** {first['reference']}\n\n"
            + _table(["rendering", *models], body)
        )
    return "\n\n".join(blocks)


def onset(ids: Sequence[int]) -> int | None:
    """First answer-window frame with a non-PAD id (`<s>` counts); None if all PAD."""
    return next((i for i, t in enumerate(ids) if int(t) != PAD), None)


def _rendering_order(rd: str) -> tuple[int, str]:
    return (RENDERINGS.index(rd), "") if rd in RENDERINGS else (len(RENDERINGS), rd)


def _repetitive(answer: str) -> bool:
    w = words_of(answer)
    grams = collections.Counter(tuple(w[i : i + 4]) for i in range(len(w) - 3))
    return any(c >= 3 for c in grams.values())


def quality_table(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> str:
    body = []
    for m in models:
        mine = [r for r in rows if r["model"] == m]
        for rd in sorted({r["rendering"] for r in mine}, key=_rendering_order):
            rs = [r for r in mine if r["rendering"] == rd]
            spoke = [r for r in rs if words_of(r["answer"])]
            ons = [onset(r["ids"]) for r in spoke]
            counts = [sum(o == k for o in ons) for k in range(4)]
            other = sum(o is None or o > 3 for o in ons)
            med = (
                statistics.median(len(words_of(r["answer"])) for r in spoke)
                if spoke
                else None
            )
            body.append(
                [
                    m,
                    rd,
                    str(len(rs)),
                    str(len(rs) - len(spoke)),
                    *map(str, counts),
                    str(other),
                    "-" if med is None else f"{med:g}",
                    str(sum(_repetitive(r["answer"]) for r in spoke)),
                    str(
                        sum(
                            not r["answer"].strip().endswith((".", "?", "!"))
                            for r in spoke
                        )
                    ),
                ]
            )
    head = ["model", "rendering", "n", "silent", "onset 0", "onset 1", "onset 2"]
    head += ["onset 3", "onset other", "median words", "repetitive", "unfinished"]
    return _table(head, body)


def first_word_table(rows: Sequence[dict[str, Any]], models: Sequence[str]) -> str:
    idx = _index(rows)
    body = []
    for m in models:
        for p, q in PAIRS:
            firsts = [
                (
                    words_of(idx[(m, s, t, p)]["answer"]),
                    words_of(idx[(m, s, t, q)]["answer"]),
                )
                for (mm, s, t, rd) in idx
                if mm == m and rd == p and (m, s, t, q) in idx
            ]
            both = [(a[0], b[0]) for a, b in firsts if a and b]
            body.append(
                [m, f"{p} vs {q}", str(len(both)), str(sum(a == b for a, b in both))]
            )
    return _table(["model", "pair", "both spoke", "same first word"], body)


def judgments_table(judgments: Sequence[dict[str, Any]], models: Sequence[str]) -> str:
    body = []
    for m in models:
        js = [j for j in judgments if j["model"] == m]
        body.append(
            [
                m,
                str(len(js)),
                str(sum(j["typed_vs_A"] == "differs" for j in js)),
                str(sum(j["A_vs_B"] == "differs" for j in js)),
            ]
        )
    return _table(["model", "turns", "typed vs A differs", "A vs B differs"], body)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dual.sbs_report")
    ap.add_argument("rows", type=pathlib.Path)
    ap.add_argument("--judgments", type=pathlib.Path)
    ap.add_argument("--models")
    ap.add_argument("--quality", action="store_true")
    a = ap.parse_args(argv)
    rows = load(a.rows)
    models = (
        a.models.split(",")
        if a.models
        else list(dict.fromkeys(r["model"] for r in rows))
    )
    parts = [stats_table(rows, models)]
    if a.judgments:
        parts.append(judgments_table(load(a.judgments), models))
    if a.quality:
        parts += [quality_table(rows, models), first_word_table(rows, models)]
    parts.append(turn_blocks(rows, models))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
