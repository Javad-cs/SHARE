#!/usr/bin/env python3
"""
apply_normalization_map.py
--------------------------
Step 2: Takes normalization_map.json (reviewed by human),
        merges into aliases.json, grouped by canonical country.

Shows preview first, asks for confirmation before writing.

Usage:
  python scripts/apply_normalization_map.py
  python scripts/apply_normalization_map.py --dry-run
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",     default="results/analysis/normalization_map.json")
    parser.add_argument("--aliases", default="notes/aliases.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, do not write to aliases.json")
    return parser.parse_args()

def main():
    args = parse_args()

    # load normalization map
    map_path = Path(args.map)
    if not map_path.exists():
        print(f"[ERROR] Map file not found: {map_path}")
        print("Run normalize_raw_strings.py first.")
        return

    norm_map = json.loads(map_path.read_text())
    print(f"Loaded {len(norm_map)} mappings from {map_path}\n")

    # load existing aliases
    aliases_path = Path(args.aliases)
    with open(aliases_path) as f:
        aliases = json.load(f)

    # build reverse lookup of what's already in aliases
    existing = set()
    for canonical, variants in aliases.items():
        if canonical.startswith("_"):
            continue
        for v in variants:
            existing.add(v.strip().lower())

    # group new mappings by canonical
    to_add    = defaultdict(list)   # canonical → [new variants]
    skipped   = []                  # already in aliases
    unknown   = []                  # claude returned "unknown"
    errors    = []                  # claude returned "error"
    vague_new = []                  # claude returned "vague"

    for raw, canonical in norm_map.items():
        if canonical in ("unknown", "UNKNOWN"):
            unknown.append(raw)
            continue
        if canonical == "error":
            errors.append(raw)
            continue
        if canonical in ("vague", "REGION"):
            vague_new.append(raw)
            continue
        if canonical == "CITY":
            unknown.append(raw)  # cities go to unknown — don't map them
            continue
        if raw.strip().lower() in existing:
            skipped.append((raw, canonical))
            continue
        to_add[canonical].append(raw)

    # ── preview ──
    sep = "─" * 60
    print(sep)
    print("PREVIEW — changes to aliases.json")
    print(sep)

    if to_add:
        print(f"\n✓ NEW ALIASES TO ADD ({sum(len(v) for v in to_add.values())} strings"
              f" across {len(to_add)} canonical countries):\n")
        for canonical in sorted(to_add.keys()):
            variants = to_add[canonical]
            # check if canonical already exists in aliases
            exists = canonical in aliases
            status = "(existing key)" if exists else "(NEW country key)"
            print(f"  {canonical} {status}:")
            for v in sorted(variants):
                print(f"    + {repr(v)}")
    else:
        print("\n✓ No new aliases to add.")

    if vague_new:
        print(f"\n─ VAGUE ({len(vague_new)}) — will be added under 'vague':")
        for v in sorted(vague_new):
            if v.strip().lower() not in existing:
                print(f"    + {repr(v)}")

    if unknown:
        print(f"\n? UNKNOWN ({len(unknown)}) — not recognized as countries, skipped:")
        for v in sorted(unknown):
            print(f"    {repr(v)}")

    if errors:
        print(f"\n⚠ ERRORS ({len(errors)}) — API call failed, needs retry:")
        for v in sorted(errors):
            print(f"    {repr(v)}")

    if skipped:
        print(f"\n○ ALREADY IN ALIASES ({len(skipped)}) — skipped:")
        for raw, canonical in sorted(skipped):
            print(f"    {repr(raw)} → {canonical}")

    print(f"\n{sep}")

    if args.dry_run:
        print("DRY RUN — no changes written.")
        return

    if not to_add and not vague_new:
        print("Nothing to write.")
        return

    # confirm
    answer = input("\nWrite these changes to aliases.json? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted. No changes written.")
        return

    # apply changes
    for canonical in sorted(to_add.keys()):
        variants = to_add[canonical]
        if canonical not in aliases:
            # new country key — insert before "other" and "vague"
            new_aliases = {}
            for k, v in aliases.items():
                if k in ("other", "vague", "_comment"):
                    continue
                new_aliases[k] = v
            new_aliases[canonical] = variants
            # re-add special keys at end
            for k in ("vague", "other"):
                if k in aliases:
                    new_aliases[k] = aliases[k]
            if "_comment" in aliases:
                new_aliases = {"_comment": aliases["_comment"], **new_aliases}
            aliases = new_aliases
        else:
            # add to existing key, avoiding duplicates
            existing_variants = set(v.strip().lower()
                                    for v in aliases[canonical])
            for v in variants:
                if v.strip().lower() not in existing_variants:
                    aliases[canonical].append(v)

    # add new vague entries
    if vague_new:
        if "vague" not in aliases:
            aliases["vague"] = []
        existing_vague = set(v.strip().lower() for v in aliases["vague"])
        for v in vague_new:
            if v.strip().lower() not in existing_vague:
                aliases["vague"].append(v)

    # write back
    aliases_path.write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2))

    total_added = sum(len(v) for v in to_add.values()) + len(vague_new)
    print(f"\n[OK] Written {total_added} new aliases to {aliases_path}")
    print(f"     Run the audit to verify:")
    print(f"     python scripts/normalize_country_names.py --audit")

if __name__ == "__main__":
    main()