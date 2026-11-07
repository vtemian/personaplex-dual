"""epoch_ci.py RUNS: per-epoch R, median gap/floor and same-epoch paired CIs for l3 s0/s1 vs both lambda-0 seeds."""
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
