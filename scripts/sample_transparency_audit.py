#!/usr/bin/env python3

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path(
    "results/analysis/transparency/transparency_flips.csv"
)
OUTPUT_DIR = Path(
    "results/analysis/transparency/human_audit"
)

SAMPLE_SIZE = 362
RANDOM_SEED = 20260802
ALLOWED_LABELS = {"acknowledged", "omitted", "unclear"}

AUDIT_CSV = OUTPUT_DIR / "human_audit_362.csv"
AUDIT_TXT = OUTPUT_DIR / "human_audit_362.txt"
KEY_CSV = OUTPUT_DIR / "human_audit_362_key.csv"
ALLOCATION_CSV = OUTPUT_DIR / "human_audit_362_allocation.csv"


def read_population():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "id",
            "model",
            "cluster",
            "vcue",
            "vcountry",
            "image",
            "baseline_pred",
            "condition_pred",
            "after_reasoning",
            "judge_label",
            "judge_model",
            "judge_error",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Missing required columns: {sorted(missing)}"
            )

        rows = []

        for row in reader:
            label = row["judge_label"].strip().lower()

            if label not in ALLOWED_LABELS:
                continue

            if row["judge_error"].strip():
                continue

            row["judge_label"] = label
            rows.append(row)

    return rows


def allocate_proportionally(strata, sample_size):
    """
    Proportional allocation using the largest-remainder method.

    For each stratum:
        exact allocation = sample_size * stratum_size / population_size

    We first assign the floor of each allocation, then distribute the
    remaining places to strata with the largest fractional remainders.
    """
    population_size = sum(len(rows) for rows in strata.values())

    exact = {
        key: sample_size * len(rows) / population_size
        for key, rows in strata.items()
    }

    allocation = {
        key: math.floor(value)
        for key, value in exact.items()
    }

    remaining = sample_size - sum(allocation.values())

    ranked = sorted(
        strata,
        key=lambda key: (
            -(exact[key] - allocation[key]),
            key,
        ),
    )

    for key in ranked[:remaining]:
        allocation[key] += 1

    if sum(allocation.values()) != sample_size:
        raise RuntimeError("Allocation does not sum to sample size.")

    for key, n_selected in allocation.items():
        if n_selected > len(strata[key]):
            raise RuntimeError(
                f"Allocation exceeds stratum size for {key}."
            )

    return allocation, exact


def main():
    population = read_population()

    if len(population) != 6249:
        raise SystemExit(
            f"Expected 6,249 eligible rows, found {len(population)}."
        )

    strata = defaultdict(list)

    for row in population:
        key = (
            row["model"].strip(),
            row["cluster"].strip(),
            row["vcue"].strip(),
        )
        strata[key].append(row)

    allocation, exact_allocation = allocate_proportionally(
        strata,
        SAMPLE_SIZE,
    )

    rng = random.Random(RANDOM_SEED)
    selected = []

    for key in sorted(strata):
        stratum_rows = list(strata[key])
        rng.shuffle(stratum_rows)

        n_selected = allocation[key]

        for row in stratum_rows[:n_selected]:
            copied = dict(row)
            copied["_stratum"] = "|".join(key)
            copied["_stratum_population"] = len(strata[key])
            copied["_stratum_sample"] = n_selected
            copied["_sampling_weight"] = (
                len(strata[key]) / n_selected
                if n_selected > 0
                else ""
            )
            selected.append(copied)

    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected {SAMPLE_SIZE} selected rows, got {len(selected)}."
        )

    # Randomize annotation order independently of the stratum order.
    rng.shuffle(selected)

    for index, row in enumerate(selected, start=1):
        row["audit_id"] = f"AUDIT-{index:03d}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Human-facing CSV: Claude labels and experimental metadata are hidden.
    with AUDIT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        fieldnames = [
            "audit_id",
            "inserted_country",
            "inserted_cue",
            "reasoning",
            "human_label",
            "human_notes",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in selected:
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "inserted_country": row["vcountry"],
                    "inserted_cue": row["vcue"],
                    "reasoning": row["after_reasoning"],
                    "human_label": "",
                    "human_notes": "",
                }
            )

    # Readable annotation document.
    with AUDIT_TXT.open("w", encoding="utf-8") as f:
        f.write("TRANSPARENCY JUDGMENT HUMAN AUDIT\n")
        f.write("=" * 42 + "\n\n")
        f.write("Labels:\n")
        f.write("  A = ACKNOWLEDGED\n")
        f.write("  O = OMITTED\n")
        f.write("  U = UNCLEAR\n\n")
        f.write(
            "Assign the label independently, without consulting "
            "the automated label.\n\n"
        )

        for row in selected:
            reasoning = row["after_reasoning"].strip()

            f.write(f"{row['audit_id']}\n")
            f.write(
                f"Inserted cue: {row['vcountry']} "
                f"{row['vcue']}\n"
            )
            f.write("Reasoning:\n")
            f.write(reasoning + "\n")
            f.write("Human label (A/O/U): ______\n")
            f.write("Notes: ________________________________\n")
            f.write("\n" + "-" * 72 + "\n\n")

    # Hidden key: retain original metadata and automated labels.
    key_fields = [
        "audit_id",
        "id",
        "model",
        "cluster",
        "vcue",
        "vcountry",
        "image",
        "baseline_pred",
        "condition_pred",
        "after_reasoning",
        "judge_label",
        "judge_model",
        "judge_error",
        "_stratum",
        "_stratum_population",
        "_stratum_sample",
        "_sampling_weight",
    ]

    with KEY_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=key_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(selected)

    # Document the exact allocation.
    with ALLOCATION_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        fieldnames = [
            "model",
            "cluster",
            "vcue",
            "population_n",
            "exact_proportional_allocation",
            "sample_n",
            "sampling_fraction",
            "sampling_weight",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(strata):
            model, cluster, cue = key
            population_n = len(strata[key])
            sample_n = allocation[key]

            writer.writerow(
                {
                    "model": model,
                    "cluster": cluster,
                    "vcue": cue,
                    "population_n": population_n,
                    "exact_proportional_allocation": (
                        f"{exact_allocation[key]:.6f}"
                    ),
                    "sample_n": sample_n,
                    "sampling_fraction": (
                        f"{sample_n / population_n:.8f}"
                        if sample_n
                        else "0"
                    ),
                    "sampling_weight": (
                        f"{population_n / sample_n:.8f}"
                        if sample_n
                        else ""
                    ),
                }
            )

    label_counts = Counter(
        row["judge_label"] for row in selected
    )
    model_counts = Counter(row["model"] for row in selected)
    cluster_counts = Counter(row["cluster"] for row in selected)
    cue_counts = Counter(row["vcue"] for row in selected)

    print(f"Population:       {len(population)}")
    print(f"Sample size:      {len(selected)}")
    print(f"Random seed:      {RANDOM_SEED}")
    print(f"Non-empty strata: {len(strata)}")
    print()

    print("Hidden Claude-label composition:")
    for label in ("acknowledged", "omitted", "unclear"):
        print(f"  {label:<13} {label_counts[label]}")

    print("\nSample by model:")
    for key, value in sorted(model_counts.items()):
        print(f"  {key:<12} {value}")

    print("\nSample by cluster:")
    for key, value in sorted(cluster_counts.items()):
        print(f"  {key:<14} {value}")

    print("\nSample by cue:")
    for key, value in sorted(cue_counts.items()):
        print(f"  {key:<10} {value}")

    zero_allocations = [
        key for key, value in allocation.items()
        if value == 0
    ]

    if zero_allocations:
        print("\nStrata receiving zero samples under proportional allocation:")
        for model, cluster, cue in sorted(zero_allocations):
            print(
                f"  {model} / {cluster} / {cue} "
                f"(population={len(strata[(model, cluster, cue)])})"
            )

    print("\nCreated:")
    print(f"  {AUDIT_CSV}")
    print(f"  {AUDIT_TXT}")
    print(f"  {KEY_CSV}")
    print(f"  {ALLOCATION_CSV}")


if __name__ == "__main__":
    main()
