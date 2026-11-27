#!/usr/bin/env python3
"""
normalize_raw_strings.py
------------------------
Step 1: Collect unmapped raw strings using fast audit logic.
Step 2: Call Claude API in parallel (one string per call) to normalize.
Step 3: Save mapping to results/analysis/normalization_map.json for review.

Usage:
  python scripts/normalize_raw_strings.py
  python scripts/normalize_raw_strings.py --workers 8
"""

import json
import argparse
import anthropic
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--aliases",     default="notes/aliases.json")
    parser.add_argument("--output",      default="results/analysis/normalization_map.json")
    parser.add_argument("--workers",     type=int, default=8)
    return parser.parse_args()

# ─────────────────────────────────────────────
# FAST AUDIT — same logic as normalize_country_names.py
# ─────────────────────────────────────────────

def build_lookup(aliases_path):
    with open(aliases_path) as f:
        data = json.load(f)
    lookup = {}
    for canonical, variants in data.items():
        if canonical.startswith("_"):
            continue
        for variant in variants:
            lookup[variant.strip().lower()] = canonical
    return lookup

def collect_unmapped(results_dir, lookup):
    unmapped = Counter()
    results_path = Path(results_dir)

    for json_file in sorted(results_path.rglob("*.json")):
        if ("cue_validation" in json_file.parts or
                "textual_validation" in json_file.parts):
            continue
        try:
            d = json.loads(json_file.read_text())
        except Exception:
            continue
        if not d.get("success", False):
            continue
        raw = (d.get("primary_country") or "").strip()
        if not raw:
            continue
        if raw.lower() not in lookup:
            unmapped[raw] += 1

    return unmapped

# ─────────────────────────────────────────────
# CLAUDE API
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a country name normalizer. Your task is to convert any string that represents a country name in any language, script, or format into its canonical English country name.

Rules:
- Reply with ONLY the canonical English country name
- If the string refers to a city, region, or sub-national entity, reply with "CITY"
- If the string refers to a multi-country region or is ambiguous, reply with "REGION"
- If the string is a refusal or expresses uncertainty, reply with "UNKNOWN"
- If you cannot determine what country it refers to, reply with "UNKNOWN"
- Never explain your answer. Never add punctuation. Just the canonical name or one of the four special values."""

def normalize_one(raw_string, client):
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_string}]
        )
        canonical = message.content[0].text.strip().lower()
        return raw_string, canonical
    except Exception as e:
        print(f"  [ERROR] '{raw_string}': {e}")
        return raw_string, "error"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    lookup = build_lookup(args.aliases)

    print("Scanning results for unmapped strings...")
    unmapped = collect_unmapped(args.results_dir, lookup)

    if not unmapped:
        print("No unmapped strings found — aliases.json is complete!")
        return

    print(f"Found {len(unmapped)} unique unmapped strings "
          f"({sum(unmapped.values())} total occurrences)\n")
    print("Top unmapped by frequency:")
    for raw, count in unmapped.most_common(20):
        print(f"  {count:>5}x  {repr(raw)}")
    print()

    # load existing map to avoid re-calling
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_map = {}
    if output_path.exists():
        existing_map = json.loads(output_path.read_text())
        print(f"Loaded {len(existing_map)} existing mappings from {output_path}")

    to_process = [raw for raw in unmapped if raw not in existing_map]

    if not to_process:
        print("All strings already mapped.")
        return

    print(f"Calling Claude API for {len(to_process)} strings "
          f"(workers={args.workers})...\n")

    client  = anthropic.Anthropic()
    results = dict(existing_map)
    done    = 0
    total   = len(to_process)
    errors  = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(normalize_one, raw, client): raw
            for raw in to_process
        }
        for future in as_completed(futures):
            raw, canonical = future.result()
            results[raw]   = canonical
            done          += 1
            if canonical == "error":
                errors += 1
            print(f"  [{done}/{total}] {repr(raw)} -> {canonical}")
            if done % 20 == 0:
                output_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2))

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2))

    print(f"\nDone. {total} strings processed, {errors} errors.")
    print(f"Mapping saved to: {output_path}")
    print(f"\nReview the map, then run:")
    print(f"  python scripts/apply_normalization_map.py --dry-run")

if __name__ == "__main__":
    main()