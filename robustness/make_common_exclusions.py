#!/usr/bin/env python3

import json
from pathlib import Path

MODELS = ["gemini", "gpt41", "qwen3vl", "gemma3"]

work_dir = Path(__file__).resolve().parent
root = work_dir.parent

src = root / "notes" / "cue_exclusions.json"
dst = work_dir / "common_valid_exclusions.json"

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

entries = data["exclusions"]

# Structural sanity check for the submitted exclusion file.
if len(entries) != 27:
    raise RuntimeError(
        f"Expected 27 submitted exclusion rules, found {len(entries)}"
    )

# Ignore model identity and collect every cue-country(-cluster)
# combination that was invalid for AT LEAST ONE model.
#
# Missing cluster_id means the exclusion was global across clusters.
invalid_scopes = set()

for e in entries:
    cue = e["cue"].strip().lower()
    country = e["country"].strip().lower()

    cluster = e.get("cluster_id")
    if cluster is not None:
        cluster = cluster.strip().lower()
        if cluster == "":
            cluster = None

    invalid_scopes.add((cue, country, cluster))

# For a common-valid test set, if one model considers a cue invalid,
# that cue must be removed for ALL models in the same scope.
common_entries = []

for cue, country, cluster in sorted(
    invalid_scopes,
    key=lambda x: (x[2] or "", x[0], x[1])
):
    for model in MODELS:
        item = {
            "model": model,
            "country": country,
            "cue": cue,
        }

        if cluster is not None:
            item["cluster_id"] = cluster

        common_entries.append(item)

# With the exact submitted file we expect:
# 27 model-specific rules -> 11 unique invalid scopes -> 44
# rules after propagating each scope to all four models.
if len(invalid_scopes) != 11:
    raise RuntimeError(
        f"Expected 11 unique invalid scopes, found {len(invalid_scopes)}"
    )

if len(common_entries) != 44:
    raise RuntimeError(
        f"Expected 44 common-set rules, found {len(common_entries)}"
    )

with dst.open("w", encoding="utf-8") as f:
    json.dump(
        {"exclusions": common_entries},
        f,
        indent=2,
        ensure_ascii=False,
    )
    f.write("\n")

print(f"Original exclusion rules: {len(entries)}")
print(f"Unique invalid scopes:    {len(invalid_scopes)}")
print(f"Common-set rules:         {len(common_entries)}")
print(f"Wrote: {dst}")
