"""combine8.py REPO OUT [label=path:src ...]: the eight-checkpoint pool input from committed answer files.
Six B12 checkpoints exactly as B12's combined6 (qual4 mapping); extra specs appended (B13's two)."""
import json, sys
R = sys.argv[1] + "/docs/results/"; out = sys.argv[2]
L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
s1 = L(R + "side-by-side-1/answers.jsonl"); s2 = L(R + "side-by-side-2/answers.jsonl")
s3 = L(R + "side-by-side-3/answers.jsonl"); c2 = L(R + "consistency-2/answers.jsonl")
rows = ([dict(r, model="base-e2") for r in s1 if r["model"] == "trained"]
        + [dict(r, model="base-e2") for r in s2 if r["model"] == "baseline-e2" and r["rendering"].endswith("-swap")]
        + [dict(r, model="l10-e1") for r in s3 if r["model"] == "cons1-e1"]
        + [dict(r, model="l10-e2") for r in s2 if r["model"] == "cons1-e2"]
        + [r for r in c2])
for sp in sys.argv[3:]:
    lab, rest = sp.split("=", 1); p, src = rest.rsplit(":", 1)
    got = [dict(r, model=lab) for r in L(p) if r["model"] == src]; assert got, sp; rows += got
with open(out, "w") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
import collections; print(collections.Counter(r["model"] for r in rows))
