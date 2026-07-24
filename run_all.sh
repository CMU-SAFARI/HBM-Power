#!/usr/bin/env bash
# End-to-end artifact reproduction, in dependency order:
#   1. HBM2 (FPGA) characterization       -> data/new/fpga/*.csv                  (~8 hours)
#   2. Substitute the fresh FPGA chip      -> data/merged/*.csv
#   3. H200 (HBM3E) characterization       -> data/new/h200/hbm3e_results.csv
#   4. Substitute the fresh H200 GPU       -> data/merged/HBM3E_measurements.csv  (4 GPUs)
#   5. Power-model simulation (DRAMPower)  -> workload traces + HBM2_runner
#   6. Figure & table generation           -> figures/*.pdf, figures/merged/*, table3.tex
#
# Step 2 must run before step 4: the FPGA substitution rewrites data/merged from
# scratch and would wipe the merged H200 CSV.
#
# Each step is checkpointed. If a step fails (e.g. a transient vast.ai error),
# fix the cause and re-run with --continue to resume from the first unfinished
# step -- already-completed steps, including the ~8-hour FPGA run, are skipped.
#
# The vast.ai key (from HotCRP) is required only while the H200 step is still
# pending; export it before a fresh run:
#   export VAST_API_KEY=<key-from-HotCRP>
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT_FILE="${CHECKPOINT_FILE:-$SCRIPT_DIR/temp/run_all_checkpoint.txt}"

CONTINUE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --continue) CONTINUE=1; shift ;;
    -h|--help)  echo "usage: $0 [--continue]"; exit 0 ;;
    *) echo "usage: $0 [--continue]" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$CHECKPOINT_FILE")"
if [[ "$CONTINUE" == 0 || ! -f "$CHECKPOINT_FILE" ]]; then
  : > "$CHECKPOINT_FILE"   # fresh run (or nothing to resume from): start clean
fi

is_done()   { grep -qxF "$1" "$CHECKPOINT_FILE"; }
mark_done() { printf '%s\n' "$1" >> "$CHECKPOINT_FILE"; }

# The vast.ai key is only needed for the (still-pending) H200 step.
if ! is_done h200 && [[ -z "${VAST_API_KEY:-}" ]]; then
  echo "error: VAST_API_KEY is not set (needed for the H200 step)." >&2
  echo "Export the vast.ai key we provide via HotCRP before running:" >&2
  echo "  export VAST_API_KEY=<key-from-HotCRP>" >&2
  exit 1
fi

trap 'echo "step failed; fix the cause and resume with: ./run_all.sh --continue" >&2' ERR

run_step() {
  local id="$1" desc="$2"; shift 2
  if is_done "$id"; then
    echo "== [skip] $desc (already completed) =="
    return
  fi
  echo "== $desc =="
  "$@"
  mark_done "$id"
}

# On --continue, hand --resume to the FPGA driver so an interrupted 8-hour run
# picks up from its own internal checkpoint instead of restarting.
if [[ "$CONTINUE" == 1 ]]; then
  run_step fpga "[1/6] HBM2 (FPGA) power characterization (~8 hours)" ./run_hbm2_characterization.sh --resume
else
  run_step fpga "[1/6] HBM2 (FPGA) power characterization (~8 hours)" ./run_hbm2_characterization.sh
fi
run_step fpga_substitute "[2/6] Substituting the fresh FPGA chip into data/merged" scripts/fpga/substitute.sh
run_step h200            "[3/6] H200 (HBM3E) power characterization"                ./run_h200_characterization.sh
run_step h200_substitute "[4/6] Substituting the fresh H200 GPU into data/merged"  scripts/h200/substitute.sh
run_step power_model     "[5/6] Power-model simulation (DRAMPower)"                 ./run_power_model.sh
run_step figures         "[6/6] Figure & table generation"                         ./reproduce_figures_tables.sh

echo "all done"
