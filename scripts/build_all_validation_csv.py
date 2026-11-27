#!/usr/bin/env python3

import csv
import json
from pathlib import Path

RESULTS_ROOT = Path("results")
OUTPUT = RESULTS_ROOT / "analysis" / "all_validation.csv"

COLUMNS = [
    "validation_group",
    "validation_type",
    "cluster_id",
    "model",
    "country",
    "country_code",
    "input_text",
    "success",
    "error",
    "primary_country",
    "confidence_score",
    "reasoning",
    "sign_text",
    "language_name",
    "english_translation",
    "english_name",
    "dish_description",
    "raw",
    "source_path",
]


def clean(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


rows = []
invalid = []

for path in sorted(RESULTS_ROOT.rglob("*.json")):
    rel = path.relative_to(RESULTS_ROOT)
    parts = rel.parts

    # Expected:
    # cluster/model/cue_validation/type/country.json
    # cluster/model/textual_validation/type/country.json
    if len(parts) != 5:
        continue

    cluster_id, model, validation_dir, validation_type, filename = parts

    if validation_dir == "cue_validation":
        validation_group = "cue"
        if validation_type not in {"flag", "currency", "sign"}:
            continue
    elif validation_dir == "textual_validation":
        validation_group = "textual"
        if validation_type not in {"endonym", "native"}:
            continue
    else:
        continue

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON value is not an object")
    except Exception as exc:
        invalid.append((path, str(exc)))
        continue

    row = {
        "validation_group": validation_group,
        "validation_type": validation_type,
        "cluster_id": data.get("cluster_id", cluster_id),
        "model": data.get("model", model),
        "country": data.get("country", Path(filename).stem),
        "country_code": data.get("country_code", ""),
        "input_text": data.get("input_text", ""),
        "success": data.get("success", ""),
        "error": data.get("error", ""),
        "primary_country": data.get("primary_country", ""),
        "confidence_score": data.get("confidence_score", ""),
        "reasoning": data.get("reasoning", ""),
        "sign_text": data.get("sign_text", ""),
        "language_name": data.get("language_name", ""),
        "english_translation": data.get("english_translation", ""),
        "english_name": data.get("english_name", ""),
        "dish_description": data.get("dish_description", ""),
        "raw": data.get("raw", ""),
        "source_path": str(path),
    }

    rows.append({key: clean(row.get(key, "")) for key in COLUMNS})

rows.sort(
    key=lambda row: (
        row["validation_group"],
        row["validation_type"],
        row["cluster_id"],
        row["model"],
        row["country"],
    )
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created: {OUTPUT}")
print(f"Rows written: {len(rows)}")
print(f"Invalid JSON files: {len(invalid)}")

if invalid:
    print("\nInvalid files:")
    for path, error in invalid:
        print(f"  {path}: {error}")
