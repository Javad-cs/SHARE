import csv
from collections import defaultdict
from pathlib import Path

PATH = Path(
    "robustness/baseline_concentration/"
    "baseline_concentration_summary.csv"
)

vals = defaultdict(list)

with PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    vals[r["cluster"]].append(float(r["top1_share_all_pct"]))

print("MEAN MODEL-SPECIFIC TOP-1 BASELINE SHARE")

for cluster in sorted(vals):
    x = vals[cluster]
    assert len(x) == 4, (cluster, len(x))
    mean = sum(x) / len(x)
    print(f"{cluster}: {mean:.3f}%")

expected = {
    "pilaf_cl": 53.750,
    "mantou_cl": 34.375,
    "samosa_cl": 94.375,
    "meatball_cl": 38.125,
}

assert set(vals.keys()) == set(expected.keys()), (set(vals.keys()), set(expected.keys()))

for cluster, target in expected.items():
    got = sum(vals[cluster]) / len(vals[cluster])
    assert abs(got - target) < 1e-9, (cluster, got, target)

print("\nALL ASSERTIONS PASSED")
