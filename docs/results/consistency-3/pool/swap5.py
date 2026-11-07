"""swap5.py JUDGEDIR: swap-follow per model, paired McNemar between models, agreement with B12 labels."""
import json, sys, pathlib, collections, itertools, math
d = pathlib.Path(sys.argv[1]); L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
lab = {}
for p in sorted((d / "in").glob("swap-*")): lab.update({x["id"]: x for x in L(p / "labels.jsonl")})
sk = L(d / "swap-key.jsonl"); assert len(lab) == len(sk) == 832
fol = {}
for k in sk:
    t = lab[k["id"]]["takes_up"]; fol[(k["model"], k["stem"], k["turn"], k["rendering"], k["condition"])] = (t == k["played"], t)
models = sorted({k[0] for k in fol})
for m in models:
    v = [x for k, x in fol.items() if k[0] == m]; tr = [x for k, x in fol.items() if k[0] == m and k[4] == "true"]
    print(m, "followed", sum(a for a, _ in v), "/", len(v), "| true", sum(a for a, _ in tr), "swap", sum(a for k, (a, _) in fol.items() if k[0] == m and k[4] == "swap"),
          "| labels", dict(collections.Counter(t for _, t in v)))
def mcnemar(a, b):
    ks = [k[1:] for k in fol if k[0] == a]; b_ = sum(fol[(a,) + k][0] and not fol[(b,) + k][0] for k in ks); c_ = sum(not fol[(a,) + k][0] and fol[(b,) + k][0] for k in ks)
    n = b_ + c_; p = min(1, 2 * sum(math.comb(n, i) for i in range(0, min(b_, c_) + 1)) / 2 ** n) if n else 1
    return b_, c_, p
for a, b in [("base-e3", "l3-e3"), ("base-e2", "l3-e3"), ("base-e2", "base-e3"), ("base-e1", "l3-e1"), ("base-e2", "l3-e2"), ("l3-e2", "l3-e3")]:
    print("mcnemar", a, "only", b, "only, exact p:", a, b, mcnemar(a, b))
# agreement with B12 on the six shared checkpoints (followed vs not, and full label)
if len(sys.argv) > 2:
    old = [r for r in L(sys.argv[2]) if r["type"] == "swap"]
    ag = ag2 = n = 0
    for r in old:
        key = (r["model"], r["stem"], r["turn"], r["rendering"], r["condition"]); 
        new_t = fol[key][1]; old_f = r["takes_up"] == r["played"]
        # P/Q orientation differs between pools: compare as followed/own/both/neither
        n += 1; ag += (fol[key][0] == old_f)
        def cat(t, kk): return "played" if t == kk["played"] else "own" if t == kk.get("own_turn") else t
        ag2 += 0
    print("agreement with B12 on followed-vs-not:", ag, "/", n)
    per = collections.defaultdict(lambda: [0, 0])
    for r in old:
        key = (r["model"], r["stem"], r["turn"], r["rendering"], r["condition"]); per[r["model"]][0] += r["takes_up"] == r["played"]; per[r["model"]][1] += fol[key][0]
    print("B12 vs B13 followed counts:", dict(per))
