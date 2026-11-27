#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()

RESULTS = ROOT / "results/analysis/all_results.csv"
ALIASES = ROOT / "notes/aliases.json"
OUTDIR = ROOT / "robustness/baseline_concentration"

EXPECTED_SHA256 = {
    RESULTS: "f270847e31750f4622193dd56392c79bb68b60aff94234a9afb9b23b574a9d19",
    ALIASES: "3602e82e30b253bbea729a189137e8bf9692b097f144c997b9e32e96b99da2d3",
}

MODELS = ["gemini", "gpt41", "qwen3vl", "gemma3"]
CLUSTERS = ["pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------
# 1. Verify immutable submitted inputs
# -------------------------------------------------

for path, expected in EXPECTED_SHA256.items():
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")

    actual = sha256(path)

    if actual != expected:
        raise SystemExit(
            f"HASH MISMATCH: {path}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}\n"
            "Stopping without writing outputs."
        )

    print(f"HASH OK: {path}")


# -------------------------------------------------
# 2. Build exact alias lookup used by submitted R
# -------------------------------------------------

with ALIASES.open("r", encoding="utf-8") as f:
    raw_aliases = json.load(f)

lookup = {}

for canonical, variants in raw_aliases.items():
    if canonical.startswith("_"):
        continue

    lookup[canonical.strip().lower()] = canonical

    for variant in variants:
        lookup[str(variant).strip().lower()] = canonical


def normalize_country(name):
    """
    Mirrors normalize_country() in submitted clash_report_csv.R:
    blank / unrecognized -> 'other'
    """
    if name is None:
        return "other"

    key = str(name).strip().lower()

    if not key:
        return "other"

    return lookup.get(key, "other")


# -------------------------------------------------
# 3. Load ONLY the exact neutral-English baseline
# -------------------------------------------------

groups = {
    (model, cluster): {
        "requested": 0,
        "successful": 0,
        "preds": [],
    }
    for model in MODELS
    for cluster in CLUSTERS
}

required_cols = {
    "cluster_id",
    "model",
    "visual_cue",
    "visual_country",
    "textual_cue_type",
    "textual_country",
    "experiment",
    "success",
    "primary_country",
}

total_rows = 0
baseline_rows = 0

with RESULTS.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    missing = required_cols - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(
            f"all_results.csv missing required columns: {sorted(missing)}"
        )

    for row in reader:
        total_rows += 1

        model = row["model"].strip().lower()
        cluster = row["cluster_id"].strip().lower()

        is_baseline = (
            model in MODELS
            and cluster in CLUSTERS
            and row["visual_cue"].strip().lower() == "neutral"
            and row["visual_country"].strip().lower() == "none"
            and row["experiment"].strip().lower() == "country_classification"
            and row["textual_cue_type"].strip().lower() == "en"
            and row["textual_country"].strip().lower() == "none"
        )

        if not is_baseline:
            continue

        baseline_rows += 1
        g = groups[(model, cluster)]
        g["requested"] += 1

        if row["success"].strip().lower() != "true":
            continue

        g["successful"] += 1
        g["preds"].append(
            normalize_country(row["primary_country"])
        )


if total_rows != 231040:
    raise SystemExit(
        f"Expected 231040 rows in all_results.csv, found {total_rows}"
    )

if baseline_rows != 640:
    raise SystemExit(
        f"Expected 640 neutral-English baseline rows "
        f"(4 models × 4 clusters × 40 images), found {baseline_rows}"
    )


# -------------------------------------------------
# 4. Summaries
# -------------------------------------------------

def entropy_bits(counter):
    total = sum(counter.values())

    if total == 0:
        return float("nan")

    result = 0.0

    for n in counter.values():
        p = n / total
        result -= p * math.log2(p)

    return result


summary_rows = []
distribution_rows = []

for model in MODELS:
    for cluster in CLUSTERS:
        g = groups[(model, cluster)]

        requested = g["requested"]
        successful = g["successful"]
        preds = g["preds"]

        counts_all = Counter(preds)

        # This exactly mirrors the submitted baseline logic:
        # region-level predictions are not eligible to define baseline country.
        counts_submitted = Counter(
            p for p in preds if p != "region"
        )

        region_count = counts_all.get("region", 0)
        other_count = counts_all.get("other", 0)

        if counts_submitted:
            submitted_top_count = max(counts_submitted.values())
            submitted_baselines = sorted(
                label
                for label, n in counts_submitted.items()
                if n == submitted_top_count
            )
            submitted_n = sum(counts_submitted.values())
            submitted_top_share = (
                submitted_top_count / submitted_n
            )
        else:
            submitted_top_count = 0
            submitted_baselines = []
            submitted_n = 0
            submitted_top_share = float("nan")

        ranked = sorted(
            counts_all.items(),
            key=lambda x: (-x[1], x[0])
        )

        top1_label = ranked[0][0] if ranked else ""
        top1_count = ranked[0][1] if ranked else 0

        top2_label = ranked[1][0] if len(ranked) > 1 else ""
        top2_count = ranked[1][1] if len(ranked) > 1 else 0

        top1_share = (
            top1_count / successful
            if successful else float("nan")
        )

        top2_share = (
            (top1_count + top2_count) / successful
            if successful else float("nan")
        )

        ent = entropy_bits(counts_all)

        summary_rows.append({
            "model": model,
            "cluster": cluster,
            "requested": requested,
            "successful": successful,
            "region_count": region_count,
            "other_count": other_count,
            "n_distinct_outputs": len(counts_all),
            "top1_label_all": top1_label,
            "top1_count_all": top1_count,
            "top1_share_all_pct": 100 * top1_share,
            "top2_label_all": top2_label,
            "top2_count_all": top2_count,
            "top2_cumulative_share_all_pct": 100 * top2_share,
            "entropy_bits_all": ent,
            "submitted_baseline": "|".join(submitted_baselines),
            "submitted_baseline_top_count": submitted_top_count,
            "submitted_nonregion_n": submitted_n,
            "submitted_baseline_share_pct": (
                100 * submitted_top_share
            ),
        })

        for label, count in ranked:
            distribution_rows.append({
                "model": model,
                "cluster": cluster,
                "normalized_country": label,
                "count": count,
                "share_of_success_pct": (
                    100 * count / successful
                    if successful else float("nan")
                ),
            })


# -------------------------------------------------
# 5. Write rebuttal-only derived outputs
# -------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)

summary_path = OUTDIR / "baseline_concentration_summary.csv"

with summary_path.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
    fieldnames = list(summary_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in summary_rows:
        out = row.copy()

        for k in [
            "top1_share_all_pct",
            "top2_cumulative_share_all_pct",
            "entropy_bits_all",
            "submitted_baseline_share_pct",
        ]:
            value = out[k]

            if isinstance(value, float) and math.isnan(value):
                out[k] = ""
            else:
                out[k] = f"{value:.3f}"

        writer.writerow(out)


dist_path = OUTDIR / "baseline_full_distribution.csv"

with dist_path.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:
    fieldnames = list(distribution_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for row in distribution_rows:
        out = row.copy()
        out["share_of_success_pct"] = (
            f"{out['share_of_success_pct']:.3f}"
        )
        writer.writerow(out)


# -------------------------------------------------
# 6. Human-readable report
# -------------------------------------------------

lines = []

lines.append("BASELINE ATTRIBUTION CONCENTRATION")
lines.append("==================================")
lines.append("")
lines.append(
    "Condition: neutral visual cue + English prompt "
    "(exact submitted baseline condition)"
)
lines.append(
    "Submitted baseline country excludes normalized "
    "'region' responses, matching clash_report_csv.R."
)
lines.append("")
lines.append(
    f"{'model':<9} {'cluster':<12} "
    f"{'n':>3} {'baseline':<18} "
    f"{'base%':>6} {'top1(all)%':>10} "
    f"{'top2%':>7} {'H(bits)':>8} {'#labels':>7}"
)
lines.append("-" * 92)

for row in summary_rows:
    lines.append(
        f"{row['model']:<9} "
        f"{row['cluster']:<12} "
        f"{row['successful']:>3} "
        f"{row['submitted_baseline']:<18} "
        f"{row['submitted_baseline_share_pct']:>6.1f} "
        f"{row['top1_share_all_pct']:>10.1f} "
        f"{row['top2_cumulative_share_all_pct']:>7.1f} "
        f"{row['entropy_bits_all']:>8.3f} "
        f"{row['n_distinct_outputs']:>7}"
    )

lines.append("")
lines.append("FULL NORMALIZED DISTRIBUTIONS")
lines.append("=============================")

for model in MODELS:
    for cluster in CLUSTERS:
        rows = [
            r
            for r in distribution_rows
            if r["model"] == model
            and r["cluster"] == cluster
        ]

        parts = [
            f"{r['normalized_country']}="
            f"{r['count']} "
            f"({r['share_of_success_pct']:.1f}%)"
            for r in rows
        ]

        lines.append(
            f"{model} / {cluster}: "
            + ", ".join(parts)
        )

lines.append("")
lines.append(
    "Interpretation note: lower entropy / higher top-1 share "
    "means a more concentrated neutral-English attribution "
    "distribution. This is descriptive; it does not by itself "
    "establish that baseline concentration causes later "
    "cue-conflict behavior."
)

report = "\n".join(lines) + "\n"

report_path = OUTDIR / "baseline_concentration_report.txt"
report_path.write_text(report, encoding="utf-8")

print()
print(report)

print(f"Summary CSV: {summary_path}")
print(f"Full distribution CSV: {dist_path}")
print(f"Report: {report_path}")
print()
print("[DONE] Submitted files were only read; all outputs are under robustness/.")
