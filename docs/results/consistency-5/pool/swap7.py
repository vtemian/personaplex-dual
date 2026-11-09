"""swap7.py JUDGEDIR [B14 judge-labels.jsonl]: consistency-4's swap6.py for 13 checkpoints (1,352 swap items):
swap-follow per model, paired McNemar, the D9 swap criterion for voiced1 (v1-e3 >= base-e3 - 5), agreement with B14."""
import json, sys, pathlib, collections, math
d = pathlib.Path(sys.argv[1]); L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
lab = {}
for p in sorted((d / "in").glob("swap-*")): lab.update({x["id"]: x for x in L(p / "labels.jsonl")})
sk = L(d / "swap-key.jsonl"); assert len(lab) == len(sk) == 1352, (len(lab), len(sk))
fol = {}
for k in sk:
    t = lab[k["id"]]["takes_up"]; fol[(k["model"], k["stem"], k["turn"], k["rendering"], k["condition"])] = (t == k["played"], t)
models = sorted({k[0] for k in fol})
tot = lambda m: sum(x[0] for k, x in fol.items() if k[0] == m)
for m in models:
    v = [x for k, x in fol.items() if k[0] == m]
    print(m, "followed", tot(m), "/", len(v), "| true", sum(a for k, (a, _) in fol.items() if k[0] == m and k[4] == "true"),
          "swap", sum(a for k, (a, _) in fol.items() if k[0] == m and k[4] == "swap"), "| labels", dict(collections.Counter(t for _, t in v)))
def exact(b_, c_):
    n = b_ + c_; return min(1, 2 * sum(math.comb(n, i) for i in range(0, min(b_, c_) + 1)) / 2 ** n) if n else 1
def disc(a, b):
    ks = [k[1:] for k in fol if k[0] == a]
    return sum(fol[(a,) + k][0] and not fol[(b,) + k][0] for k in ks), sum(not fol[(a,) + k][0] and fol[(b,) + k][0] for k in ks)
for a, b in [("base-e3", "v1-e3"), ("l3-e3", "v1-e3"), ("base-e3", "l3-e3"), ("base-s1-e3", "l3s1-e3"), ("l3-e1", "v1-e1"), ("base-e1", "v1-e1")]:
    b_, c_ = disc(a, b); print("mcnemar", a, "only-followed", b_, "|", b, "only-followed", c_, "| exact p", round(exact(b_, c_), 3))
print(f"D9 swap criterion: v1-e3 {tot('v1-e3')} >= base-e3 {tot('base-e3')} - 5 = {tot('base-e3') - 5}: {tot('v1-e3') >= tot('base-e3') - 5}")
if len(sys.argv) > 2:
    old = [r for r in L(sys.argv[2]) if r["type"] == "swap"]
    ag = sum(fol[(r["model"], r["stem"], r["turn"], r["rendering"], r["condition"])][0] == (r["takes_up"] == r["played"]) for r in old)
    print("agreement with B14 on followed-vs-not:", ag, "/", len(old))
    per = collections.defaultdict(lambda: [0, 0])
    for r in old:
        key = (r["model"], r["stem"], r["turn"], r["rendering"], r["condition"]); per[r["model"]][0] += r["takes_up"] == r["played"]; per[r["model"]][1] += fol[key][0]
    print("B14 vs B15 followed counts:", dict(per))
