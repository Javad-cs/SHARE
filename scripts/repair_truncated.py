import json
import re
from pathlib import Path


def try_repair(data: dict) -> dict | None:
    """
    Repair a result dict where success=False due to truncated JSON in raw field.
    Extracts attribution fields from the partial raw string.
    Returns repaired dict or None if unrecoverable.
    """
    raw = data.get("raw", "")

    match = re.match(
        r'\s*\{\s*'
        r'"food_name"\s*:\s*"([^"]+)"\s*,\s*'
        r'"confidence_score_for_food"\s*:\s*(\d+)\s*,\s*'
        r'"primary_country"\s*:\s*"([^"]+)"\s*,\s*'
        r'"confidence_score_for_country"\s*:\s*(\d+)\s*,\s*'
        r'"reasoning"\s*:\s*"',
        raw,
        re.DOTALL
    )
    if not match:
        return None

    food_name       = match.group(1)
    conf_food       = int(match.group(2))
    primary_country = match.group(3)
    conf_country    = int(match.group(4))
    reasoning       = "[TRUNCATED: repetition loop in native script generation]"

    repaired = {
        **data,  # keep all original metadata exactly
        "success": True,
        "error": "",
        "food_name": food_name,
        "primary_country": primary_country,
        "confidence_score_for_food": conf_food,
        "confidence_score_for_country": conf_country,
        "reasoning": reasoning,
        "repair_note": "Primary attribution fields recovered. Reasoning field truncated due to Qwen3VL repetition loop in Cyrillic script generation. Excluded from transparency analysis."
    }
    return repaired


def repair_all(results_root="results", model_filter="qwen3vl", dry_run=True):
    results_root = Path(results_root)

    repaired = []
    unrecoverable = []

    for f in sorted(results_root.rglob("*.json")):
        if model_filter and model_filter not in str(f):
            continue

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  Wrapper itself malformed, skipping: {f}")
            continue

        # Skip already successful or already repaired
        if data.get("success") or data.get("repair_note"):
            continue

        # Only attempt repair on truncation errors
        if "Unterminated string" not in data.get("error", ""):
            continue

        repaired_data = try_repair(data)

        if repaired_data:
            repaired.append(f)
            if not dry_run:
                f.write_text(
                    json.dumps(repaired_data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
        else:
            unrecoverable.append(f)
            print(f"  Unrecoverable: {f}")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Repair Summary")
    print("=" * 50)
    print(f"Repaired:      {len(repaired)}")
    print(f"Unrecoverable: {len(unrecoverable)}")
    if dry_run and repaired:
        print(f"\nFiles that would be repaired:")
        for f in repaired[:10]:
            print(f"  {f}")
        if len(repaired) > 10:
            print(f"  ... and {len(repaired) - 10} more")

    return repaired, unrecoverable


if __name__ == "__main__":
    # Always dry run first
    # repair_all(results_root="results", model_filter="qwen3vl", dry_run=True)

    # Once satisfied uncomment:
    repair_all(results_root="results", model_filter="qwen3vl", dry_run=False)