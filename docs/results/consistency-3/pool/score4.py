"""score.py JUDGE2DIR: unblind labels, print tables, write unblinded labels."""
import json, sys, pathlib, collections
d = pathlib.Path(sys.argv[1])
def L(p): return [json.loads(l) for l in open(p) if l.strip()]
def labels(prefix, n):
    out = {}
    for j in range(1, n + 1):
        items = L(d / f"{prefix}-{j}.jsonl"); lab = {x["id"]: x for x in L(d / "in" / f"{prefix}-{j}" / "labels.jsonl")}
        missing = [x["id"] for x in items if x["id"] not in lab]
        assert not missing, (prefix, j, missing[:5])
        out.update(lab)
    return out
pk = L(d / "pairs-key.jsonl"); ak = L(d / "answers-key.jsonl"); sk = L(d / "swap-key.jsonl")
cnt = lambda p: len(list((d / "in").glob(p + "-*")))
pl, al, sl = labels("pairs", cnt("pairs")), labels("answers", cnt("answers")), labels("swap", cnt("swap"))
un = []
print("## pairs: differs / n (all turns; turns 2-8)")
c = collections.defaultdict(lambda: [0, 0, 0, 0])
for k in pk:
    lab = pl[k["id"]]; un.append({"type": "pair", **k, **{x: lab[x] for x in lab if x != "id"}})
    v = c[(k["model"], k["kind"])]; dif = lab["substance"] == "differs"
    v[0] += dif; v[1] += 1
    if k["turn"] > 0: v[2] += dif; v[3] += 1
for key in sorted(c): print(key, f"{c[key][0]}/{c[key][1]}", f"{c[key][2]}/{c[key][3]}")
print("## answers")
c = collections.defaultdict(collections.Counter)
for k in ak:
    lab = al[k["id"]]; un.append({"type": "answer", **k, **{x: lab[x] for x in lab if x != "id"}})
    for t in ("all", "t0" if k["turn"] == 0 else "t2-8"):
        g = c[(k["model"], k["rendering"], t)]
        g["n"] += 1; g["engages"] += lab["engages"] == "yes"; g["matches_tape"] += lab["matches_tape"] == "yes"
        g["well_formed"] += lab["well_formed"] == "yes"; g["v_" + lab["verdict"]] += 1
for key in sorted(c): print(key, dict(c[key]))
print("## swap: follows played / own / both / neither")
c = collections.defaultdict(collections.Counter)
for k in sk:
    lab = sl[k["id"]]; t = lab["takes_up"]; un.append({"type": "swap", **k, **{x: lab[x] for x in lab if x != "id"}})
    res = "played" if t == k["played"] else "own" if t == k["own_turn"] else t
    if k["condition"] == "true" and res == "played": res = "played"
    c[(k["model"], k["condition"], k["rendering"])][res] += 1
for key in sorted(c): print(key, dict(c[key]))
with open(d / "unblinded.jsonl", "w") as f: f.writelines(json.dumps(x) + "\n" for x in un)
