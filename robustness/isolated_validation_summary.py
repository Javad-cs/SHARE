#!/usr/bin/env python3

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
VALIDATION = ROOT / "results/analysis/all_validation.csv"
EXCLUSIONS = ROOT / "notes/cue_exclusions.json"
OUTDIR = ROOT / "robustness/isolated_validation"

EXPECTED_SHA256 = {
    VALIDATION: "1e3f5db53166b9a56d64ec1009f78ed7d2ec2b723ef82db1f2835748c26890da",
    EXCLUSIONS: "94d3acc5528edf3a88dc46b4421870e96c39430dfc9ab16c5a0c360a74e4f73a",
}

CUE_ORDER = ["flag", "currency", "sign", "native", "endonym"]
MODEL_ORDER = ["gemini", "gpt41", "qwen3vl", "gemma3"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Verify submitted files.
for path, expected in EXPECTED_SHA256.items():
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )
    print(f"HASH OK: {path}")


with EXCLUSIONS.open("r", encoding="utf-8") as f:
    rules = json.load(f)["exclusions"]

normalized_rules = []
for r in rules:
    nr = {
        k: str(v).strip().lower()
        for k, v in r.items()
    }
    normalized_rules.append(nr)


def is_excluded(row):
    for rule in normalized_rules:
        if row["model"] != rule["model"]:
            continue
        if row["country"] != rule["country"]:
            continue
        if row["validation_type"] != rule["cue"]:
            continue
        if (
            "cluster_id" in rule
            and row["cluster_id"] != rule["cluster_id"]
        ):
            continue
        return True
    return False


rows = []

with VALIDATION.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        for col in [
            "validation_type",
            "cluster_id",
            "model",
            "country",
        ]:
            row[col] = row[col].strip().lower()

        row["excluded"] = is_excluded(row)
        rows.append(row)


if len(rows) != 480:
    raise SystemExit(
        f"Expected 480 validation rows; found {len(rows)}"
    )


overall = defaultdict(lambda: {"n": 0, "pass": 0})
by_model = defaultdict(lambda: {"n": 0, "pass": 0})

# Only flag/currency have country-identification confidence.
country_conf = defaultdict(list)

for row in rows:
    cue = row["validation_type"]
    model = row["model"]
    passed = not row["excluded"]

    overall[cue]["n"] += 1
    overall[cue]["pass"] += int(passed)

    by_model[(cue, model)]["n"] += 1
    by_model[(cue, model)]["pass"] += int(passed)

    if cue in {"flag", "currency"}:
        x = row.get("confidence_score", "").strip()
        if x:
            country_conf[cue].append(float(x))


excluded_count = sum(r["excluded"] for r in rows)

if excluded_count != 65:
    raise SystemExit(
        f"Expected 65 excluded validation rows; found {excluded_count}"
    )

OUTDIR.mkdir(parents=True, exist_ok=True)


# Save overall table.
with (OUTDIR / "isolated_validation_overall.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    w = csv.writer(f)
    w.writerow(["cue", "n", "passed", "failed", "pass_rate_pct"])

    for cue in CUE_ORDER:
        n = overall[cue]["n"]
        p = overall[cue]["pass"]
        w.writerow([
            cue,
            n,
            p,
            n - p,
            f"{100*p/n:.3f}",
        ])


# Save model-level table.
with (OUTDIR / "isolated_validation_by_model.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    w = csv.writer(f)
    w.writerow([
        "cue",
        "model",
        "n",
        "passed",
        "failed",
        "pass_rate_pct",
    ])

    for cue in CUE_ORDER:
        for model in MODEL_ORDER:
            d = by_model[(cue, model)]
            n = d["n"]
            p = d["pass"]

            w.writerow([
                cue,
                model,
                n,
                p,
                n - p,
                f"{100*p/n:.3f}",
            ])


print()
print("ISOLATED-CUE VALIDATION SUMMARY")
print("================================")
print(f"Rows: {len(rows)}")
print(f"Submitted exclusion rules: {len(rules)}")
print(f"Excluded validation rows: {excluded_count}")

print()
print("Overall pass rates:")
for cue in CUE_ORDER:
    n = overall[cue]["n"]
    p = overall[cue]["pass"]
    print(
        f"  {cue:<8} {p:>2}/{n} = {100*p/n:5.1f}%"
    )

print()
print("Pass rates by model:")
print(
    f"{'cue':<10}"
    f"{'gemini':>10}"
    f"{'gpt41':>10}"
    f"{'qwen3vl':>10}"
    f"{'gemma3':>10}"
)

for cue in CUE_ORDER:
    line = f"{cue:<10}"

    for model in MODEL_ORDER:
        d = by_model[(cue, model)]
        rate = 100 * d["pass"] / d["n"]
        line += f"{rate:>10.1f}"

    print(line)


print()
print("Country-identification confidence:")
for cue in ["flag", "currency"]:
    vals = country_conf[cue]
    mean = sum(vals) / len(vals)
    print(
        f"  {cue:<8} n={len(vals)}, "
        f"mean={mean:.1f}, "
        f"min={min(vals):.1f}, "
        f"max={max(vals):.1f}"
    )

print()
print(
    "NOTE: confidence_score here is the self-reported "
    "confidence for country identification in the isolated "
    "flag/currency validation task. It is not food confidence."
)

print(
    "NOTE: sign, native, and endonym validation rows do not "
    "contain confidence_score values."
)

print(
    "NOTE: validation criteria differ by cue type. "
    "Endonyms are validated for dish-family recognizability, "
    "not country identification."
)

print()
print(f"Outputs written to: {OUTDIR}")
