#!/usr/bin/env python3
"""
Build one canonical long-form CSV from all experiment JSON records.

Source:
    results/<cluster>/<model>/<visual_cue>/<visual_country>/
        country_classification/<textual_cue_type>/<textual_country>/<image>.json

Output:
    results/analysis/all_results.csv

This script:
  - reads experiment JSONs only;
  - excludes cue_validation and textual_validation;
  - validates path metadata against JSON metadata;
  - detects duplicate experiment identities;
  - preserves failed records;
  - writes atomically;
  - never modifies source JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "image",
    "cluster_id",
    "model",
    "visual_cue",
    "visual_country",
    "textual_cue_type",
    "textual_country",
    "experiment",
    "success",
    "raw",
    "error",
    "food_name",
    "primary_country",
    "confidence_score_for_food",
    "confidence_score_for_country",
    "reasoning",
    "repair_note",
    "source_json",
]

IDENTITY_FIELDS = [
    "image",
    "cluster_id",
    "model",
    "visual_cue",
    "visual_country",
    "textual_cue_type",
    "textual_country",
    "experiment",
]

EXPECTED_JSON_FIELDS = set(FIELDNAMES) - {"source_json"}


class BuildError(RuntimeError):
    """Raised when the source results are inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one canonical CSV from experiment JSON files."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root experiment-results directory.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/all_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run all checks without writing the CSV.",
    )
    return parser.parse_args()


def canonical(value: Any) -> str:
    """Return a stable string representation for identity checking."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def csv_value(field: str, value: Any) -> Any:
    """Normalise values before writing them to CSV."""
    if value is None:
        return ""

    if field == "success":
        if isinstance(value, bool):
            return "True" if value else "False"

        lowered = str(value).strip().lower()
        if lowered == "true":
            return "True"
        if lowered == "false":
            return "False"

        raise BuildError(f"Invalid success value: {value!r}")

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def discover_json_files(results_root: Path) -> list[Path]:
    """
    Discover only experiment JSONs with the exact expected path structure.

    Expected relative path:
      cluster/model/visual_cue/visual_country/country_classification/
      textual_cue_type/textual_country/file.json
    """
    files: list[Path] = []

    for path in results_root.rglob("*.json"):
        relative = path.relative_to(results_root)
        parts = relative.parts

        if "cue_validation" in parts or "textual_validation" in parts:
            continue

        if len(parts) != 8:
            continue

        if parts[4] != "country_classification":
            continue

        files.append(path)

    return sorted(files, key=lambda p: str(p.relative_to(results_root)))


def expected_metadata_from_path(
    json_path: Path,
    results_root: Path,
) -> dict[str, str]:
    relative = json_path.relative_to(results_root)

    (
        cluster_id,
        model,
        visual_cue,
        visual_country,
        experiment,
        textual_cue_type,
        textual_country,
        filename,
    ) = relative.parts

    return {
        "cluster_id": cluster_id,
        "model": model,
        "visual_cue": visual_cue,
        "visual_country": visual_country,
        "experiment": experiment,
        "textual_cue_type": textual_cue_type,
        "textual_country": textual_country,
        "image_stem": Path(filename).stem,
    }


def load_record(
    json_path: Path,
    results_root: Path,
) -> tuple[dict[str, Any], set[str]]:
    try:
        with json_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise BuildError(
            f"Invalid JSON: {json_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise BuildError(
            f"Could not read {json_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise BuildError(
            f"Expected a JSON object, got {type(data).__name__}: {json_path}"
        )

    expected = expected_metadata_from_path(json_path, results_root)

    for field in [
        "cluster_id",
        "model",
        "visual_cue",
        "visual_country",
        "experiment",
        "textual_cue_type",
        "textual_country",
    ]:
        actual_value = canonical(data.get(field, ""))
        expected_value = expected[field]

        if actual_value != expected_value:
            raise BuildError(
                f"Path/JSON mismatch in {json_path}\n"
                f"  field:    {field}\n"
                f"  path:     {expected_value!r}\n"
                f"  JSON:     {actual_value!r}"
            )

    image = data.get("image", "")
    if not image:
        raise BuildError(f"Missing image field: {json_path}")

    if Path(str(image)).stem != expected["image_stem"]:
        raise BuildError(
            f"Image/path mismatch in {json_path}\n"
            f"  JSON image: {image!r}\n"
            f"  file stem:  {expected['image_stem']!r}"
        )

    if "success" not in data:
        raise BuildError(f"Missing success field: {json_path}")

    extra_fields = set(data) - EXPECTED_JSON_FIELDS

    row = {
        field: csv_value(field, data.get(field, ""))
        for field in FIELDNAMES
        if field != "source_json"
    }

    row["source_json"] = json_path.relative_to(results_root.parent).as_posix()

    return row, extra_fields


def identity_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(canonical(row.get(field, "")) for field in IDENTITY_FIELDS)


def write_csv_atomically(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            writer = csv.DictWriter(
                temp_file,
                fieldnames=FIELDNAMES,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)

            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Verify the temporary output before replacing the final CSV.
        with temp_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != FIELDNAMES:
                raise BuildError(
                    "Temporary CSV header verification failed."
                )

            written_rows = sum(1 for _ in reader)

        if written_rows != len(rows):
            raise BuildError(
                f"Temporary CSV row-count mismatch: "
                f"expected {len(rows)}, found {written_rows}"
            )

        os.replace(temp_path, output_path)
        temp_path = None

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_dir)
    output_path = Path(args.output)

    if not results_root.is_dir():
        raise BuildError(
            f"Results directory does not exist: {results_root}"
        )

    json_files = discover_json_files(results_root)

    if not json_files:
        raise BuildError(
            f"No experiment JSON files found under {results_root}"
        )

    rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, ...], Path] = {}
    extra_field_counts: Counter[str] = Counter()

    success_count = 0
    failure_count = 0

    model_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()

    for json_path in json_files:
        row, extra_fields = load_record(json_path, results_root)

        key = identity_key(row)

        if key in seen:
            raise BuildError(
                "Duplicate experiment identity detected:\n"
                f"  first:  {seen[key]}\n"
                f"  second: {json_path}\n"
                f"  key:    {key}"
            )

        seen[key] = json_path
        rows.append(row)

        for field in extra_fields:
            extra_field_counts[field] += 1

        if row["success"] == "True":
            success_count += 1
        else:
            failure_count += 1

        model_counts[str(row["model"])] += 1
        cluster_counts[str(row["cluster_id"])] += 1

    rows.sort(
        key=lambda row: tuple(
            canonical(row.get(field, ""))
            for field in [
                "cluster_id",
                "model",
                "visual_cue",
                "visual_country",
                "textual_cue_type",
                "textual_country",
                "image",
            ]
        )
    )

    print("=" * 72)
    print("ALL RESULTS CSV BUILD")
    print("=" * 72)
    print(f"Experiment JSON files: {len(json_files)}")
    print(f"Unique CSV rows:        {len(rows)}")
    print(f"Successful records:     {success_count}")
    print(f"Failed records:         {failure_count}")

    print("\nRows by cluster:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  {cluster:<20} {count}")

    print("\nRows by model:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model:<20} {count}")

    if extra_field_counts:
        print("\nExtra JSON fields not included as CSV columns:")
        for field, count in sorted(extra_field_counts.items()):
            print(f"  {field:<30} {count}")
    else:
        print("\nExtra JSON fields: none")

    if args.check_only:
        print("\nCHECK ONLY: no CSV was written.")
        return

    write_csv_atomically(rows, output_path)

    print(f"\nWrote: {output_path}")
    print(f"Rows:  {len(rows)}")
    print("Source JSON files modified: 0")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except BuildError as exc:
        raise SystemExit(f"SAFETY STOP: {exc}")
