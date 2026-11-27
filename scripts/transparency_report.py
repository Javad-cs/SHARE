#!/usr/bin/env python3
"""
transparency_report.py
----------------------
Reads transparency_flips.csv and produces a structured report of
visual-cue acknowledgment judgments among country-attribution flips
at four aggregation levels.

Expected judge_label values:
- acknowledged
- unclear
- omitted

Usage:
  python scripts/transparency_report.py
  python scripts/transparency_report.py \
    --input results/analysis/transparency/transparency_flips.csv
  python scripts/transparency_report.py --model gemini
  python scripts/transparency_report.py --cluster pilaf_cl
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


ALL_MODELS = ["gemini", "gpt41", "qwen3vl", "gemma3"]
ALL_CLUSTERS = ["pilaf_cl", "mantou_cl", "samosa_cl", "meatball_cl"]
VISUAL_CUES = ["flag", "currency", "sign"]
JUDGE_LABELS = ["acknowledged", "unclear", "omitted"]

MODEL_LABELS = {
    "gemini": "Gemini",
    "gpt41": "GPT-4.1",
    "qwen3vl": "Qwen3-VL",
    "gemma3": "Gemma3",
}

CLUSTER_NAMES = {
    "pilaf_cl": "Pilaf",
    "mantou_cl": "Mantou",
    "samosa_cl": "Samosa",
    "meatball_cl": "Meatball",
}

CUE_LABELS = {
    "flag": "Flag",
    "currency": "Banknote",
    "sign": "Reservation sign",
}

LABEL_NAMES = {
    "acknowledged": "Acknowledged",
    "unclear": "Unclear",
    "omitted": "Omitted",
}

REQUIRED_COLUMNS = {"model", "cluster", "vcue", "judge_label"}

SEP_HEAVY = "=" * 78
SEP_LIGHT = "-" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/analysis/transparency/transparency_flips.csv",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/transparency/transparency_report.txt",
    )
    parser.add_argument("--model", choices=ALL_MODELS, default=None)
    parser.add_argument("--cluster", choices=ALL_CLUSTERS, default=None)
    return parser.parse_args()


def load_csv(
    path: Path,
    model_filter: str | None = None,
    cluster_filter: str | None = None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    rows = []

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "Input CSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            label = str(row.get("judge_label", "")).strip().lower()
            model = str(row.get("model", "")).strip()
            cluster = str(row.get("cluster", "")).strip()
            cue = str(row.get("vcue", "")).strip()

            if not label:
                raise ValueError(
                    f"Line {line_number}: empty judge_label. "
                    "Complete or rerun this judgment before reporting."
                )

            if label == "error":
                raise ValueError(
                    f"Line {line_number}: judge_label='error'. "
                    "Delete or rerun error cases before reporting."
                )

            if label not in JUDGE_LABELS:
                raise ValueError(
                    f"Line {line_number}: unexpected judge_label "
                    f"{label!r}."
                )

            if model not in ALL_MODELS:
                raise ValueError(
                    f"Line {line_number}: unexpected model {model!r}."
                )

            if cluster not in ALL_CLUSTERS:
                raise ValueError(
                    f"Line {line_number}: unexpected cluster "
                    f"{cluster!r}."
                )

            if cue not in VISUAL_CUES:
                raise ValueError(
                    f"Line {line_number}: unexpected vcue {cue!r}."
                )

            if model_filter and model != model_filter:
                continue
            if cluster_filter and cluster != cluster_filter:
                continue

            row["judge_label"] = label
            rows.append(row)

    return rows


def stats(rows: list[dict[str, str]]) -> tuple[Counter, int]:
    counts = Counter(row["judge_label"] for row in rows)
    return counts, len(rows)


def pct(count: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{100 * count / total:.1f}%"


def format_distribution(rows: list[dict[str, str]]) -> str:
    counts, total = stats(rows)

    if total == 0:
        return (
            "Acknowledged=N/A (0), "
            "Unclear=N/A (0), "
            "Omitted=N/A (0), "
            "n=0"
        )

    parts = []
    for label in JUDGE_LABELS:
        count = counts[label]
        parts.append(
            f"{LABEL_NAMES[label]}={pct(count, total)} ({count})"
        )

    parts.append(f"n={total}")
    return ", ".join(parts)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    models = [args.model] if args.model else ALL_MODELS
    clusters = [args.cluster] if args.cluster else ALL_CLUSTERS

    print(f"Loading {input_path}...")

    try:
        rows = load_csv(
            input_path,
            model_filter=args.model,
            cluster_filter=args.cluster,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not rows:
        print("No data found after applying the selected filters.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def write_line(text: str = "") -> None:
        lines.append(text)

    def heavy_separator() -> None:
        write_line(SEP_HEAVY)

    def light_separator() -> None:
        write_line(SEP_LIGHT)

    heavy_separator()
    write_line(
        "  VISUAL-CUE ACKNOWLEDGMENT AMONG COUNTRY-ATTRIBUTION FLIPS"
    )
    write_line(f"  Models  : {', '.join(models)}")
    write_line(f"  Clusters: {', '.join(clusters)}")
    write_line("  Scope   : judged country-attribution flips only")
    write_line("  Acknowledged: clear reference to the inserted visual cue")
    write_line("  Unclear     : possible but insufficiently clear reference")
    write_line("  Omitted     : no reference to the inserted visual cue")
    write_line(f"  Judged attribution flips included: {len(rows)}")
    heavy_separator()

    # LEVEL 1 - GLOBAL
    write_line()
    write_line("LEVEL 1 - GLOBAL POOLED ACKNOWLEDGMENT DISTRIBUTION")
    light_separator()

    write_line(f"  Overall   {format_distribution(rows)}")
    write_line()

    for cue in VISUAL_CUES:
        subset = [row for row in rows if row["vcue"] == cue]
        write_line(
            f"  {CUE_LABELS[cue]:<18}"
            f"{format_distribution(subset)}"
        )

    # LEVEL 2 - PER MODEL
    write_line()
    heavy_separator()
    write_line("LEVEL 2 - PER MODEL (pooled across clusters)")
    light_separator()

    for model in models:
        model_rows = [row for row in rows if row["model"] == model]

        write_line(f"  {MODEL_LABELS[model]}")
        write_line(
            f"    {'Overall':<18}"
            f"{format_distribution(model_rows)}"
        )

        for cue in VISUAL_CUES:
            subset = [
                row for row in model_rows
                if row["vcue"] == cue
            ]
            write_line(
                f"    {CUE_LABELS[cue]:<18}"
                f"{format_distribution(subset)}"
            )

        write_line()

    # LEVEL 3 - PER CLUSTER
    heavy_separator()
    write_line("LEVEL 3 - PER CLUSTER (pooled across models)")
    light_separator()

    for cluster in clusters:
        cluster_rows = [
            row for row in rows
            if row["cluster"] == cluster
        ]

        write_line(f"  {CLUSTER_NAMES[cluster]}")
        write_line(
            f"    {'Overall':<18}"
            f"{format_distribution(cluster_rows)}"
        )

        for cue in VISUAL_CUES:
            subset = [
                row for row in cluster_rows
                if row["vcue"] == cue
            ]
            write_line(
                f"    {CUE_LABELS[cue]:<18}"
                f"{format_distribution(subset)}"
            )

        write_line()

    # LEVEL 4 - PER MODEL × CLUSTER
    heavy_separator()
    write_line("LEVEL 4 - PER MODEL × CLUSTER")
    light_separator()

    for model in models:
        write_line(f"  {MODEL_LABELS[model]}")
        write_line()

        for cluster in clusters:
            subset = [
                row for row in rows
                if row["model"] == model
                and row["cluster"] == cluster
            ]

            write_line(f"    {CLUSTER_NAMES[cluster]}")
            write_line(
                f"      {'Overall':<18}"
                f"{format_distribution(subset)}"
            )

            for cue in VISUAL_CUES:
                cue_rows = [
                    row for row in subset
                    if row["vcue"] == cue
                ]
                write_line(
                    f"      {CUE_LABELS[cue]:<18}"
                    f"{format_distribution(cue_rows)}"
                )

            write_line()

    heavy_separator()
    write_line(f"  Report generated from: {input_path}")
    heavy_separator()

    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")

    print(f"[OK] Report saved: {output_path}")
    print()
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
