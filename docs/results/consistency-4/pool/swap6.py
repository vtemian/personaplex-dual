"""swap6.py JUDGEDIR [B13 judge-labels.jsonl]: swap-follow per model (11 checkpoints), paired McNemar,
pooled 208-item lambda3-vs-lambda0 epoch-3 test across seeds, agreement with B13's labels."""
import json, sys, pathlib, collections, math
d = pathlib.Path(sys.argv[1]); L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
lab = {}
for p in sorted((d / "in").glob("swap-*")): lab.update({x["id"]: x for x in L(p / "labels.jsonl")})
sk = L(d / "swap-key.jsonl"); assert len(lab) == len(sk) == 1144, (len(lab), len(sk))
fol = {}
for k in sk:
    t = lab[k["id"]]["takes_up"]; fol[(k["model"], k["stem"], k["turn"], k["rendering"], k["condition"])] = (t == k["played"], t)
models = sorted({k[0] for k in fol})
for m in models:
    v = [x for k, x in fol.items() if k[0] == m]; tr = [x for k, x in fol.items() if k[0] == m and k[4] == "true"]
    print(m, "followed", sum(a for a, _ in v), "/", len(v), "| true", sum(a for a, _ in tr), "swap", sum(a for k, (a, _) in fol.items() if k[0] == m and k[4] == "swap"),
          "| labels", dict(collections.Counter(t for _, t in v)))
def exact(b_, c_):
    n = b_ + c_; return min(1, 2 * sum(math.comb(n, i) for i in range(0, min(b_, c_) + 1)) / 2 ** n) if n else 1
def disc(a, b):
    ks = [k[1:] for k in fol if k[0] == a]
    return sum(fol[(a,) + k][0] and not fol[(b,) + k][0] for k in ks), sum(not fol[(a,) + k][0] and fol[(b,) + k][0] for k in ks)
for a, b in [("base-e3", "l3-e3"), ("base-s1-e3", "l3s1-e3"), ("base-s1-e3", "l3s1-e2"), ("base-e3", "base-s1-e3"), ("l3-e3", "l3s1-e3"),
             ("l3s1-e2", "l3s1-e3"), ("base-e2", "l3-e3"), ("base-e2", "base-e3")]:
    b_, c_ = disc(a, b); print("mcnemar", a, "only-followed", b_, "|", b, "only-followed", c_, "| exact p", round(exact(b_, c_), 3))
b0, c0 = disc("base-e3", "l3-e3"); b1, c1 = disc("base-s1-e3", "l3s1-e3")
B, C = b0 + b1, c0 + c1
print(f"POOLED 208 (lambda0 e3 vs lambda3 e3, paired within seed): lambda0-only {B}, lambda3-only {C}, exact p {exact(B, C):.3f}")
n = B + C
if n:
    diff = (B - C); se = math.sqrt(B + C - (B - C) ** 2 / 208)
    print(f"  diff {diff} items of 208, Wald 95% [{diff - 1.96*se:+.1f}, {diff + 1.96*se:+.1f}]")
tot = lambda m: sum(x[0] for k, x in fol.items() if k[0] == m)
print("pooled followed: lambda0 e3", tot("base-e3") + tot("base-s1-e3"), "/208; lambda3 e3", tot("l3-e3") + tot("l3s1-e3"), "/208")
if len(sys.argv) > 2:
    old = [r for r in L(sys.argv[2]) if r["type"] == "swap"]
    ag = sum(fol[(r["model"], r["stem"], r["turn"], r["rendering"], r["condition"])][0] == (r["takes_up"] == r["played"]) for r in old)
    print("agreement with B13 on followed-vs-not:", ag, "/", len(old))
    per = collections.defaultdict(lambda: [0, 0])
    for r in old:
        key = (r["model"], r["stem"], r["turn"], r["rendering"], r["condition"]); per[r["model"]][0] += r["takes_up"] == r["played"]; per[r["model"]][1] += fol[key][0]
    print("B13 vs B14 followed counts:", dict(per))
