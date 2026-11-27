import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from scripts.normalize_country_names import normalize

FILES = {
    "results/analysis/transparency/transparency_flips.csv":
        "c7be091cf0fa7771127e3d2237746b3b7de406ee60763b1c0929c74e364ffc8a",
    "scripts/normalize_country_names.py":
        "6e697228a7c8cc4fb22d7af772595260bd49c9f6b752fd7e1f67fe45c77a4187",
    "notes/aliases.json":
        "3602e82e30b253bbea729a189137e8bf9692b097f144c997b9e32e96b99da2d3",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("VERIFYING SUBMITTED INPUTS")
for path, expected in FILES.items():
    actual = sha256(path)
    assert actual == expected, (path, actual, expected)
    print("[OK]", path)

PATH = Path("results/analysis/transparency/transparency_flips.csv")

stats = defaultdict(lambda: {"n": 0, "omit": 0})

with PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"\nTotal transparency flips in submitted file: {len(rows)}")
assert len(rows) == 6249

for r in rows:
    cue_target = normalize(r["vcountry"])

    # condition_pred in transparency_flips.csv is already canonicalized
    # by the submitted pipeline.
    prediction = r["condition_pred"].strip().lower()

    # Exact canonical equality only.
    # Example: "north korea" != "south korea".
    if prediction != cue_target:
        continue

    cue = r["vcue"]
    model = r["model"]

    stats[(cue, "ALL")]["n"] += 1
    stats[(cue, model)]["n"] += 1

    if r["judge_label"].strip().lower() == "omitted":
        stats[(cue, "ALL")]["omit"] += 1
        stats[(cue, model)]["omit"] += 1


def show(cue, model="ALL"):
    x = stats[(cue, model)]
    rate = 100 * x["omit"] / x["n"]
    print(
        f"{cue:8s} {model:8s}: "
        f"{x['omit']}/{x['n']} omitted = {rate:.3f}%"
    )


print("\nTARGET-COUNTRY FLIPS BY CUE")
for cue in ["flag", "currency", "sign"]:
    show(cue)

print("\nRESERVATION SIGNS BY MODEL")
for model in ["gemini", "gpt41", "qwen3vl", "gemma3"]:
    show("sign", model)

total = sum(
    stats[(cue, "ALL")]["n"]
    for cue in ["flag", "currency", "sign"]
)

print(f"\nTOTAL TARGET-COUNTRY FLIPS: {total}")

# Regression checks: these must all hold.
assert stats[("flag", "ALL")] == {"n": 2934, "omit": 0}
assert stats[("currency", "ALL")] == {"n": 1662, "omit": 13}
assert stats[("sign", "ALL")] == {"n": 834, "omit": 225}

assert stats[("sign", "gemini")] == {"n": 98, "omit": 62}
assert stats[("sign", "gpt41")] == {"n": 278, "omit": 11}
assert stats[("sign", "qwen3vl")] == {"n": 243, "omit": 27}
assert stats[("sign", "gemma3")] == {"n": 215, "omit": 125}

assert total == 5430

# Explicit safeguard against the earlier Korea issue.
assert normalize("Korea") == "south korea"
assert normalize("North Korea") == "north korea"
assert normalize("Korea") != normalize("North Korea")

print("\nALL ASSERTIONS PASSED")
