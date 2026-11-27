#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/robustness"

cd "$ROOT"

echo "=========================================="
echo "1. VERIFYING SUBMITTED INPUTS"
echo "=========================================="

check_sha256 () {
    local file="$1"
    local expected="$2"

    if [[ ! -f "$file" ]]; then
        echo "[ERROR] Missing file: $file"
        exit 1
    fi

    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"

    if [[ "$actual" != "$expected" ]]; then
        echo "[ERROR] SHA256 mismatch: $file"
        echo "expected: $expected"
        echo "actual:   $actual"
        exit 1
    fi

    echo "[OK] $file"
}

check_sha256 \
  "results/analysis/all_results.csv" \
  "f270847e31750f4622193dd56392c79bb68b60aff94234a9afb9b23b574a9d19"

check_sha256 \
  "notes/cue_exclusions.json" \
  "94d3acc5528edf3a88dc46b4421870e96c39430dfc9ab16c5a0c360a74e4f73a"

check_sha256 \
  "notes/aliases.json" \
  "3602e82e30b253bbea729a189137e8bf9692b097f144c997b9e32e96b99da2d3"

echo
echo "=========================================="
echo "2. REGENERATING COMMON-VALID POLICY"
echo "=========================================="

python3 "$WORK/make_common_exclusions.py"

echo
echo "=========================================="
echo "3. TRIAL-LEVEL + BASE-IMAGE BOOTSTRAP"
echo "=========================================="

Rscript "$WORK/conflict_trial_bootstrap.R" \
  --results "results/analysis/all_results.csv" \
  --aliases "notes/aliases.json" \
  --original-exclusions "notes/cue_exclusions.json" \
  --common-exclusions "$WORK/common_valid_exclusions.json" \
  --output-dir "$WORK/trial_bootstrap" \
  --bootstrap 5000 \
  --seed 4160

echo
echo "=========================================="
echo "4. KEY OUTPUT"
echo "=========================================="

cat "$WORK/trial_bootstrap/trial_bootstrap_report.txt"

echo
echo "[DONE] Original submission files were read only."
