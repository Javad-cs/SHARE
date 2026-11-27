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
  "scripts/clash_report_csv.R" \
  "2e908eac043e99f985477957303561922b13985eade4daf71918a28b45e58921"

echo
echo "=========================================="
echo "2. GENERATING NO-FLAG EXCLUSION POLICIES"
echo "=========================================="

# Regenerate common-valid directly from submitted exclusions first.
python3 "$WORK/make_common_exclusions.py"
python3 "$WORK/make_no_flag_exclusions.py"

mkdir -p \
    "$WORK/output_no_flags_original" \
    "$WORK/output_no_flags_common"

echo
echo "=========================================="
echo "3. NO FLAGS + ORIGINAL FILTERING"
echo "=========================================="

Rscript scripts/clash_report_csv.R \
    --results-csv results/analysis/all_results.csv \
    --exclusions "$WORK/no_flags_original_exclusions.json" \
    --aliases notes/aliases.json \
    --clusters-json notes/clusters.json \
    --output-dir "$WORK/output_no_flags_original"

echo
echo "=========================================="
echo "4. NO FLAGS + COMMON-VALID FILTERING"
echo "=========================================="

Rscript scripts/clash_report_csv.R \
    --results-csv results/analysis/all_results.csv \
    --exclusions "$WORK/no_flags_common_exclusions.json" \
    --aliases notes/aliases.json \
    --clusters-json notes/clusters.json \
    --output-dir "$WORK/output_no_flags_common"

echo
echo "=========================================="
echo "5. SUMMARY"
echo "=========================================="

for mode in no_flags_original no_flags_common; do
    report="$WORK/output_${mode}/clash_report.txt"

    echo
    echo "---------- $mode ----------"

    grep -m1 "Total classified cells:" "$report"
    grep -m1 "Visual-cue country most frequent" "$report"
    grep -m1 "Textual-cue country most frequent" "$report"
    grep -m1 "Visual-textual tie" "$report"
    grep -m1 "Neutral-baseline country most frequent" "$report"
    grep -m1 "Region-level response most frequent" "$report"
    grep -m1 "Other country most frequent" "$report"
done

echo
echo "[DONE] Submitted files were only read, never modified."
