"""epoch_ci.py RUNS: consistency-4's epoch_ci.py (its l3 s0/s1 rows unchanged) plus voiced1-pp-l3:
per-epoch R, gap/floor, p90 ratio, same-epoch paired CIs vs both lambda-0 seeds and vs cons1-pp-l3."""
import sys, pathlib
from dual import noise as n
root = pathlib.Path(sys.argv[1])
s0 = n.read_epochs(root, "personaplex-ladder-personaplex-1"); s1 = n.read_epochs(root, "personaplex-ladder2-pp-s1")
print("seed ratios by epoch", {e: round(n.ratio(s0[e]), 2) for e in (1, 2, 3)}, {e: round(n.ratio(s1[e]), 2) for e in (1, 2, 3)})
for lab in ("personaplex-cons1-pp-l3", "personaplex-cons1-pp-l3-s1"):
    v = n.read_epochs(root, lab)
    for e in (1, 2, 3):
        a, b = n.paired_ci(v[e], s0[e]), n.paired_ci(v[e], s1[e])
        print(lab, f"e{e}: R {n.ratio(v[e]):.2f} gap {n.median_gap(v[e]):.4f} floor {n.median_floor(v[e]):.4f} p90 {n.p90_ratio(v[e]):.2f}",
              "| vs s0 same epoch", n._ci(a), "| vs s1 same epoch", n._ci(b))
l3 = n.read_epochs(root, "personaplex-cons1-pp-l3")
v = n.read_epochs(root, "personaplex-voiced1-pp-l3")
for e in (1, 2, 3):
    a, b, c = n.paired_ci(v[e], s0[e]), n.paired_ci(v[e], s1[e]), n.paired_ci(v[e], l3[e])
    print("personaplex-voiced1-pp-l3", f"e{e}: R {n.ratio(v[e]):.2f} gap {n.median_gap(v[e]):.4f} floor {n.median_floor(v[e]):.4f} p90 {n.p90_ratio(v[e]):.2f}",
          "| vs s0 same epoch", n._ci(a), "| vs s1 same epoch", n._ci(b), "| vs cons1-pp-l3 same epoch", n._ci(c))
