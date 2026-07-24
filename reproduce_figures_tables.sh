#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR_DEFAULT="$SCRIPT_DIR/data"
DATA_DIR_MERGED="$SCRIPT_DIR/data/merged"
FIG_DIR_DEFAULT="$SCRIPT_DIR/figures"
FIG_DIR_MERGED="$SCRIPT_DIR/figures/merged"
H200_NEW_FILE="${H200_NEW_FILE:-$SCRIPT_DIR/data/new/h200/hbm3e_results.csv}"
H200_MERGED_FILE="${H200_MERGED_FILE:-$DATA_DIR_MERGED/HBM3E_measurements.csv}"
# A representative FPGA-only base CSV, written by scripts/fpga/substitute.sh but
# never by the H200 substitution. Its presence marks a real FPGA-merged dataset
# (an H200-only substitution leaves data/merged with just HBM3E_measurements.csv).
FPGA_MERGED_FILE="${FPGA_MERGED_FILE:-$DATA_DIR_MERGED/all_idd_measurements.csv}"

DRY_RUN=0
SKIP_H200=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-h200)
      SKIP_H200=1
      shift
      ;;
    *)
      echo "usage: $0 [--dry-run] [--skip-h200]" >&2
      exit 2
      ;;
  esac
done

has_substituted_h200() {
  [[ -f "$H200_MERGED_FILE" ]] || return 1
  python3 - "$H200_MERGED_FILE" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as csv_file:
    rows = list(csv.DictReader(csv_file))
raise SystemExit(0 if len(rows) >= 4 else 1)
PY
}

FIGURE_SCRIPTS=(
  scripts/figures/figure2.py
  scripts/figures/figure3.py
  scripts/figures/figure4.py
  scripts/figures/figure5.py
  scripts/figures/figure6.py
  scripts/figures/figure7.py
  scripts/figures/figure8.py
  scripts/figures/figure9.py
  scripts/figures/figure10.py
  scripts/figures/figure11.py
  scripts/figures/figure12.py
  scripts/figures/figure13.py
  scripts/figures/figure14.py
  scripts/figures/figure15.py
  scripts/figures/figure16.py
  scripts/figures/figure17.py
)

MERGED_FIGURE_SCRIPTS=(
  scripts/figures/figure2.py
  scripts/figures/figure3.py
  scripts/figures/figure4.py
  scripts/figures/figure5.py
  scripts/figures/figure6.py
  scripts/figures/figure7.py
  scripts/figures/figure8.py
  scripts/figures/figure9.py
  scripts/figures/figure10.py
  scripts/figures/figure11.py
  scripts/figures/figure12.py
  scripts/figures/figure13.py
  scripts/figures/figure14.py
  scripts/figures/figure15.py
  scripts/figures/figure16.py
)

MERGED_FIGURE_SCRIPTS_SKIP_H200=(
  scripts/figures/figure2.py
  scripts/figures/figure3.py
  scripts/figures/figure4.py
  scripts/figures/figure5.py
  scripts/figures/figure6.py
  scripts/figures/figure7.py
  scripts/figures/figure8.py
  scripts/figures/figure9.py
  scripts/figures/figure10.py
  scripts/figures/figure11.py
  scripts/figures/figure12.py
  scripts/figures/figure13.py
  scripts/figures/figure14.py
  scripts/figures/figure15.py
)

TABLE_SCRIPTS=(
  scripts/tables/table3.py
)

run_script() {
  local data_dir="$1"
  local fig_dir="$2"
  local out_dir="$3"
  local script="$4"

  echo "[run] DATA_DIR=$data_dir FIG_DIR=$fig_dir OUT_DIR=$out_dir python3 $script"
  if [[ "$DRY_RUN" == 1 ]]; then
    return
  fi

  DATA_DIR="$data_dir" FIG_DIR="$fig_dir" OUT_DIR="$out_dir" python3 "$SCRIPT_DIR/$script"
}

run_set() {
  local label="$1"
  local data_dir="$2"
  local fig_dir="$3"
  local figure_list_name="$4"
  local -n figure_scripts="$figure_list_name"

  echo "== Reproducing $label figures/tables =="
  mkdir -p "$fig_dir"
  for script in "${figure_scripts[@]}"; do
    run_script "$data_dir" "$fig_dir" "$fig_dir" "$script"
  done
  for script in "${TABLE_SCRIPTS[@]}"; do
    run_script "$data_dir" "$fig_dir" "$fig_dir" "$script"
  done
}

run_set "released-data" "$DATA_DIR_DEFAULT" "$FIG_DIR_DEFAULT" FIGURE_SCRIPTS

# The merged set is released data with your freshly measured FPGA chip
# (scripts/fpga/substitute.sh) and, unless --skip-h200, your fresh H200 GPU
# (scripts/h200/substitute.sh) substituted in. Each substitution is optional:
# we report whatever is missing and reproduce whatever is available, rather
# than erroring out.
have_fpga_merged=0
have_h200_merged=0
[[ -f "$FPGA_MERGED_FILE" ]] && have_fpga_merged=1
if [[ "$SKIP_H200" == 0 ]] && has_substituted_h200; then
  have_h200_merged=1
fi

if [[ "$have_fpga_merged" == 0 ]]; then
  echo "== Missing FPGA merged data: $FPGA_MERGED_FILE =="
  echo "   run run_hbm2_characterization.sh, then scripts/fpga/substitute.sh to build it"
fi
if [[ "$SKIP_H200" == 0 && "$have_h200_merged" == 0 ]]; then
  echo "== Missing H200 merged data: $H200_MERGED_FILE =="
  echo "   run run_h200_characterization.sh, then scripts/h200/substitute.sh to build it"
fi

if [[ "$have_fpga_merged" == 0 ]]; then
  echo "== Skipping merged-data figures: no merged dataset to plot from =="
elif [[ "$have_h200_merged" == 1 ]]; then
  run_set "merged-data" "$DATA_DIR_MERGED" "$FIG_DIR_MERGED" MERGED_FIGURE_SCRIPTS
else
  echo "== Reproducing merged-data figures without H200-dependent figure16 =="
  run_set "merged-data" "$DATA_DIR_MERGED" "$FIG_DIR_MERGED" MERGED_FIGURE_SCRIPTS_SKIP_H200
fi

echo "done"