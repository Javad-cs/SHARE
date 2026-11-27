#!/usr/bin/env python3

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

SOURCE = Path("results/analysis/all_results.csv")
OUTDIR = Path("robustness/repeat_stability")
MANIFEST = OUTDIR / "sample_manifest.csv"

MODELS = ["gpt41", "gemini"]
N_PER_MODEL = 300
SEED = 4160

OUTDIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------- Load submitted results ----------
with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

required = {
    "image", "cluster_id", "model",
    "visual_cue", "visual_country",
    "textual_cue_type", "textual_country",
    "experiment", "success",
    "primary_country", "source_json",
}
missing = required - set(fieldnames or [])
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")

# ---------- Verify population ----------
populations = {}

for model in MODELS:
    m = [r for r in rows if r["model"] == model]

    if len(m) != 57760:
        raise SystemExit(
            f"{model}: expected 57,760 rows, found {len(m)}"
        )

    unsuccessful = [
        r for r in m
        if r["success"].strip().lower() not in {"true", "1"}
    ]
    if unsuccessful:
        raise SystemExit(
            f"{model}: found {len(unsuccessful)} unsuccessful rows; "
            "stop and inspect before sampling."
        )

    source_ids = [r["source_json"].strip() for r in m]

    if any(not x for x in source_ids):
        raise SystemExit(f"{model}: missing source_json values")

    if len(set(source_ids)) != len(source_ids):
        raise SystemExit(f"{model}: source_json is not unique")

    # Canonical ordering before seeded sampling.
    m = sorted(m, key=lambda r: r["source_json"])
    populations[model] = m

# ---------- Sample ----------
#
# Use a separate RNG initialized with the same predeclared seed
# for each model. The population ordering is canonical.
#
selected = []

for model in MODELS:
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(populations[model])), N_PER_MODEL))

    for j, idx in enumerate(indices, start=1):
        r = dict(populations[model][idx])

        r["stability_sample_id"] = f"{model}-{j:03d}"
        r["population_index"] = str(idx)
        r["sampling_seed"] = str(SEED)
        r["sampling_n"] = str(N_PER_MODEL)
        r["population_n"] = str(len(populations[model]))

        selected.append(r)

if len(selected) != 600:
    raise RuntimeError(f"Expected 600 sampled rows, got {len(selected)}")

if len({(r["model"], r["source_json"]) for r in selected}) != 600:
    raise RuntimeError("Sample contains duplicate model-condition pairs")

# ---------- Write frozen manifest ----------
extra = [
    "stability_sample_id",
    "population_index",
    "sampling_seed",
    "sampling_n",
    "population_n",
]

manifest_fields = extra + fieldnames

with MANIFEST.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_fields)
    writer.writeheader()
    writer.writerows(selected)

# ---------- Hashes ----------
source_hash = sha256(SOURCE)
manifest_hash = sha256(MANIFEST)

# ---------- Summary ----------
print("SOURCE SHA256 :", source_hash)
print("MANIFEST SHA256:", manifest_hash)

for model in MODELS:
    s = [r for r in selected if r["model"] == model]

    print(f"\n{model}: {len(s)} sampled")
    print("clusters:", dict(Counter(r["cluster_id"] for r in s)))
    print("visual cues:", dict(Counter(r["visual_cue"] for r in s)))
    print("textual cues:", dict(Counter(r["textual_cue_type"] for r in s)))

print("\nWrote:")
print(" ", MANIFEST)
