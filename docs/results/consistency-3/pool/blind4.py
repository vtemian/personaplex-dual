"""blind4.py OUTDIR SPEC...: blind3.py generalised to N models. SPEC = label=answers.jsonl:source_model
(a label may repeat, e.g. base-e2 = B7 true rows + B10 swap rows). Same item formats, rubrics, seed 13;
chunks of ~98 items per judge."""
import json, random, sys, pathlib
out = pathlib.Path(sys.argv[1]); out.mkdir(exist_ok=True)
SPECS = sys.argv[2:]
def load(p, src, model):
    return [dict(r, model=model) for r in map(json.loads, open(p)) if r["model"] == src]
rows = []
for sp in SPECS:
    lab, rest = sp.split("=", 1); path, src = rest.rsplit(":", 1)
    got = load(path, src, lab); assert got, sp; rows += got
idx = {(r["model"], r["stem"], r["turn"], r["rendering"]): r for r in rows}
assert len(idx) == len(rows), "duplicate rows"
MODELS = tuple(dict.fromkeys(sp.split("=", 1)[0] for sp in SPECS))
for m in MODELS:
    n = sum(1 for r in rows if r["model"] == m)
    assert n == 247, (m, n)  # 195 true + 52 swap
base = [r for r in rows if r["model"] == MODELS[0] and not r["rendering"].endswith("-swap")]
ref = {(r["stem"], r["turn"]): r["reference"] for r in base}
pup = {(r["stem"], r["turn"]): r["pupil"] for r in base if r["rendering"] == "typed"}
for r in rows:  # every model saw the same tape
    if not r["rendering"].endswith("-swap"):
        assert r["reference"] == ref[(r["stem"], r["turn"])], r["stem"]
def prev(stem, turn):
    return "(start of the lesson)" if turn == 0 else ref[(stem, turn - 2)]
rng = random.Random(13)
turns = sorted({(r["stem"], r["turn"]) for r in base})
assert len(turns) == 65
pairs = [(m, s, t, p, q) for m in MODELS for s, t in turns for p, q in (("typed", "A"), ("A", "B"))]
rng.shuffle(pairs)
pin, pkey = [], []
for i, (m, s, t, p, q) in enumerate(pairs):
    a, b = idx[(m, s, t, p)], idx[(m, s, t, q)]
    if rng.random() < 0.5: a, b = b, a
    pid = f"p{i:03d}"
    pin.append({"id": pid, "previous_tutor_turn": prev(s, t), "pupil_turn": pup[(s, t)],
                "tape_tutor_answer": ref[(s, t)], "answer_X": a["answer"] or "(silence)",
                "answer_Y": b["answer"] or "(silence)"})
    pkey.append({"id": pid, "model": m, "stem": s, "turn": t, "kind": f"{p}_vs_{q}", "x": a["rendering"], "y": b["rendering"]})
ans = [(m, s, t, r) for m in MODELS for s, t in turns for r in ("typed", "A", "B")]
rng.shuffle(ans)
ain, akey = [], []
for i, (m, s, t, r) in enumerate(ans):
    aid = f"a{i:03d}"
    ain.append({"id": aid, "previous_tutor_turn": prev(s, t), "pupil_turn": pup[(s, t)],
                "tape_tutor_answer": ref[(s, t)], "answer": idx[(m, s, t, r)]["answer"] or "(silence)"})
    akey.append({"id": aid, "model": m, "stem": s, "turn": t, "rendering": r})
sw = []
for m in MODELS:
    for s, t in turns:
        if t not in (2, 6): continue
        for r in ("typed", "A"):
            sw.append((m, s, t, r, "true")); sw.append((m, s, t, f"{r}-swap", "swap"))
rng.shuffle(sw)
sin, skey = [], []
for i, (m, s, t, r, cond) in enumerate(sw):
    row = idx[(m, s, t, r)]
    if cond == "swap":
        played, other = row["pupil"], row["true_pupil"]
        assert other == pup[(s, t)]
    else:
        played, other = row["pupil"], idx[(m, s, t, f"{r}-swap")]["pupil"]
    assert played != other
    if rng.random() < 0.5: P, Q, played_is = played, other, "P"
    else: P, Q, played_is = other, played, "Q"
    sid = f"s{i:03d}"
    sin.append({"id": sid, "previous_tutor_turn": prev(s, t), "pupil_turn_P": P, "pupil_turn_Q": Q,
                "answer": row["answer"] or "(silence)"})
    skey.append({"id": sid, "model": m, "stem": s, "turn": t, "rendering": r, "condition": cond,
                 "played": played_is, "own_turn": played_is if cond == "true" else ("Q" if played_is == "P" else "P")})
def split(name, items, n):
    k = -(-len(items) // n)
    for j in range(n):
        d = out / "in" / f"{name}-{j+1}"; d.mkdir(parents=True, exist_ok=True)
        chunk = items[j*k:(j+1)*k]
        for p in (out / f"{name}-{j+1}.jsonl", d / "items.jsonl"):
            with open(p, "w") as f: f.writelines(json.dumps(x) + "\n" for x in chunk)
        rub = {"pairs": "rubric-pair.txt", "answers": "rubric-answer.txt", "swap": "rubric-swap.txt"}[name]
        (d / "rubric.txt").write_text((pathlib.Path(__file__).parent / rub).read_text())
split("pairs", pin, -(-len(pin)//98)); split("answers", ain, -(-len(ain)//98)); split("swap", sin, -(-len(sin)//105))
print("chunks", -(-len(pin)//98), -(-len(ain)//98), -(-len(sin)//105))
for name, k in (("pairs-key", pkey), ("answers-key", akey), ("swap-key", skey)):
    with open(out / f"{name}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in k)
print(len(pin), len(ain), len(sin))
