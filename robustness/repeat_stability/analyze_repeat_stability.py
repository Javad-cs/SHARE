import json
from pathlib import Path
from collections import defaultdict
from scripts.normalize_country_names import normalize

ROOT = Path("robustness/repeat_stability/raw_calls")
groups = defaultdict(dict)

for p in ROOT.rglob("*.json"):
    r = json.loads(p.read_text(encoding="utf-8"))
    groups[(r["model"], r["stability_sample_id"])][r["repeat_label"]] = r

for model, expected_same in [("gpt41", 291), ("gemini", 294)]:
    pairs = []
    for (m, sid), pair in groups.items():
        if m != model:
            continue
        assert "A" in pair and "B" in pair, (model, sid, "missing A/B")
        a, b = pair["A"], pair["B"]
        assert a.get("success") and b.get("success"), (model, sid, "failed call")
        pairs.append((sid, a, b))

    assert len(pairs) == 300, (model, len(pairs))

    same = 0
    mismatches = []
    for sid, a, b in pairs:
        na = normalize(a.get("primary_country", ""))
        nb = normalize(b.get("primary_country", ""))
        if na == nb:
            same += 1
        else:
            mismatches.append(
                (sid, a.get("primary_country"), b.get("primary_country"), na, nb)
            )

    print(f"\n{model}: {same}/300 = {100*same/300:.2f}%")
    print("Normalized disagreements:")
    for x in mismatches:
        print(" ", x)

    assert same == expected_same, (model, same, expected_same)

print("\nALL ASSERTIONS PASSED")
