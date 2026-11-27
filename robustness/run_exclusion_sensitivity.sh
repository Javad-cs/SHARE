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
        echo "[ERROR] SHA256 mismatch:"
        echo "  file:     $file"
        echo "  expected: $expected"
        echo "  actual:   $actual"
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

check_sha256 \
  "notes/clusters.json" \
  "998df666b89b3dd5df51bf515523c8f04eac979fc885f2644b1ba25ddac4ef3e"

check_sha256 \
  "scripts/clash_report_csv.R" \
  "2e908eac043e99f985477957303561922b13985eade4daf71918a28b45e58921"

echo
echo "=========================================="
echo "2. BUILDING COMMON-VALID EXCLUSION FILE"
echo "=========================================="

python3 "$WORK/make_common_exclusions.py"

COMMON_EXCLUSIONS="$WORK/common_valid_exclusions.json"

# For the all-condition analysis we deliberately pass a path that
# DOES NOT EXIST. The submitted R function treats a nonexistent
# exclusion file as zero exclusions.
NO_EXCLUSIONS="$WORK/__NO_EXCLUSIONS_FILE__.json"

if [[ -e "$NO_EXCLUSIONS" ]]; then
    echo "[ERROR] $NO_EXCLUSIONS unexpectedly exists."
    exit 1
fi

mkdir -p \
    "$WORK/output_original" \
    "$WORK/output_common" \
    "$WORK/output_all"

echo
echo "=========================================="
echo "3. REPRODUCING SUBMITTED ANALYSIS"
echo "=========================================="

Rscript scripts/clash_report_csv.R \
    --results-csv results/analysis/all_results.csv \
    --exclusions notes/cue_exclusions.json \
    --aliases notes/aliases.json \
    --clusters-json notes/clusters.json \
    --output-dir "$WORK/output_original"

SUBMITTED_REPORT="results/analysis/heatmaps/clash/clash_report.txt"
REPRO_REPORT="$WORK/output_original/clash_report.txt"

echo
echo "Comparing reproduced report with submitted report..."

# The generated timestamp is expected to differ.
if ! diff -u \
    <(grep -v "Report generated:" "$SUBMITTED_REPORT") \
    <(grep -v "Report generated:" "$REPRO_REPORT")
then
    echo
    echo "[ERROR] Reproduction does NOT exactly match submitted analysis."
    echo "Stopping before any sensitivity analyses."
    exit 1
fi

echo "[OK] Exact submitted analysis reproduced."

echo
echo "=========================================="
echo "4. COMMON-VALID INTERSECTION"
echo "=========================================="

Rscript scripts/clash_report_csv.R \
    --results-csv results/analysis/all_results.csv \
    --exclusions "$COMMON_EXCLUSIONS" \
    --aliases notes/aliases.json \
    --clusters-json notes/clusters.json \
    --output-dir "$WORK/output_common"

echo
echo "=========================================="
echo "5. ALL CONDITIONS / NO EXCLUSIONS"
echo "=========================================="

Rscript scripts/clash_report_csv.R \
    --results-csv results/analysis/all_results.csv \
    --exclusions "$NO_EXCLUSIONS" \
    --aliases notes/aliases.json \
    --clusters-json notes/clusters.json \
    --output-dir "$WORK/output_all"

echo
echo "=========================================="
echo "6. SUMMARY"
echo "=========================================="

for mode in original common all; do
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
echo "[DONE] No submitted files were modified."
echo "Outputs are under:"
echo "  $WORK/output_original"
echo "  $WORK/output_common"
echo "  $WORK/output_all"
