"""Side-by-side report: what a model says after typed vs spoken pupil turns. Pure.

`python -m dual.sbs_report rows.jsonl [--judgments j.jsonl] [--models a,b]` prints
the pair statistics, the reader's judgment counts if given, and one block per turn.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import statistics
from collections.abc import Sequence
from typing import Any

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
        "exact_rate": exact / len(said) if said else 0.0,
        "median_sim": statistics.median(sims) if sims else 0.0,
        "sim_ge_0_8": sum(s >= 0.8 for s in sims),
        "empty_p": sum(not words_of(a) for a, _ in answers),
        "empty_q": sum(not words_of(b) for _, b in answers),
    }


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


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
                    f"{s['exact_rate']:.2f}",
                    f"{s['median_sim']:.2f}",
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
    parts.append(turn_blocks(rows, models))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
