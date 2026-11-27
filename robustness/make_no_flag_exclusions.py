#!/usr/bin/env python3

import json
from pathlib import Path

MODELS = ["gemini", "gpt41", "qwen3vl", "gemma3"]

work = Path(__file__).resolve().parent
root = work.parent

original_path = root / "notes" / "cue_exclusions.json"
common_path = work / "common_valid_exclusions.json"
clusters_path = root / "notes" / "clusters.json"

with original_path.open(encoding="utf-8") as f:
    original = json.load(f)["exclusions"]

with common_path.open(encoding="utf-8") as f:
    common = json.load(f)["exclusions"]

with clusters_path.open(encoding="utf-8") as f:
    clusters = json.load(f)["clusters"]

# Derive countries from the submitted cluster definition rather than
# hard-coding them.
countries = sorted({
    country["country"].strip().lower()
    for cluster in clusters
    for country in cluster["countries"]
})

if len(original) != 27:
    raise RuntimeError(
        f"Expected 27 submitted exclusion rules, found {len(original)}"
    )

if len(common) != 44:
    raise RuntimeError(
        f"Expected 44 common-valid rules, found {len(common)}"
    )

if len(countries) != 10:
    raise RuntimeError(
        f"Expected 10 unique countries, found {len(countries)}: {countries}"
    )

# Flags were validated for all target countries in the submitted setup.
# Fail rather than silently proceeding if that assumption is false.
existing_flag_rules = [
    x for x in original
    if x["cue"].strip().lower() == "flag"
]

if existing_flag_rules:
    raise RuntimeError(
        "Unexpected pre-existing flag exclusions in submitted file: "
        f"{existing_flag_rules}"
    )


def canonical_key(x):
    cluster = x.get("cluster_id")
    if cluster is not None:
        cluster = cluster.strip().lower()
        if cluster == "":
            cluster = None

    return (
        x["model"].strip().lower(),
        x["country"].strip().lower(),
        x["cue"].strip().lower(),
        cluster,
    )


def add_flag_exclusions(base):
    out = [dict(x) for x in base]
    seen = {canonical_key(x) for x in out}

    for model in MODELS:
        for country in countries:
            item = {
                "model": model,
                "country": country,
                "cue": "flag",
            }

            key = canonical_key(item)
            if key not in seen:
                out.append(item)
                seen.add(key)

    return out


variants = {
    "no_flags_original_exclusions.json": add_flag_exclusions(original),
    "no_flags_common_exclusions.json": add_flag_exclusions(common),
}

for filename, entries in variants.items():
    path = work / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump({"exclusions": entries}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"{filename}: {len(entries)} rules")

print(f"Countries ({len(countries)}): {', '.join(countries)}")
