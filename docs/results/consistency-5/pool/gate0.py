"""GATE-0 of plan 2026-11-08-voice-d-teacher.md D3, from the screen's summary.json."""
import json, sys
s = json.load(open(sys.argv[1]))
s = s[-1] if isinstance(s, list) else s
print("epoch", s["epoch"])
c = s["content"]
gap, floor = c["median_gap"], c["median_floor"]
print(f"reproduction: gap {gap:.5f} floor {floor:.5f} R {gap/floor:.3f} (B12 e3: .0146/.0078, R 1.87 +-0.03)")
ok_rep = abs(gap / 0.014583110809326172 - 1) <= 0.02 and abs(floor / 0.0078058624640107155 - 1) <= 0.02
print("  within 2%:", ok_rep, f"(gap {gap/0.014583110809326172-1:+.2%}, floor {floor/0.0078058624640107155-1:+.2%})")
tape = s["tape_floor_median"] / floor
door = s["door_median"] / floor
print(f"tape||A/floor {tape:.3f}   C||A/floor (door) {door:.3f}")
rows = []
for L in ("D1", "D2", "D3"):
    r = s[f"floor_{L}_median"] / floor
    rb = s[f"floor_{L}_b_median"] / floor
    sl = s[f"gap_{L}_median"] / gap
    adm = max(r, rb) <= 1.3 and 0.7 <= sl <= 1.3
    lever = r <= tape - 0.3
    rows.append((L, r, rb, sl, adm, lever))
    print(f"{L}: r_L {r:.3f} r_L_b {rb:.3f} s_L {sl:.3f} admissible {adm} lever {lever}")
adm = sorted([x for x in rows if x[4]], key=lambda x: x[1])
if adm:
    best = adm[0]
    tied = [x for x in adm if x[1] - best[1] <= 0.05]
    best = min(tied, key=lambda x: x[2])
    print("PICK", best[0], "lever check", best[5])
else:
    print("NO ADMISSIBLE CANDIDATE")
