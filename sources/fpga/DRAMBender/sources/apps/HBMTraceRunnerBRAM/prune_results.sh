#!/bin/bash
# Remove the unnecessary local result CSVs under results/, keeping only the clean
# zeros-vs-random comparison set: files whose data-init label is "zeros" or
# "rand1" (i.e. names ending in _data_zeros_bram.csv or _data_rand1_bram.csv).
#
# Everything else under results/ is deleted: the older seeded passes
# (_data_00000000_bram.csv, _data_7a41c47d_bram.csv) and the un-labelled legacy
# runs (llm_*_bram.csv, pc0_bram.csv).
#
# This only touches results/ in the current directory -- nothing on the FPGAs.
#
# DRY RUN by default: lists what would be deleted. Pass --apply (or -f) to delete.
#   ./prune_results.sh           # preview
#   ./prune_results.sh --apply   # actually delete

set -euo pipefail

RESULTS_DIR="results"

APPLY=0
case "${1:-}" in
  --apply|-f) APPLY=1 ;;
  ""        ) APPLY=0 ;;
  *) echo "usage: $0 [--apply|-f]" >&2; exit 2 ;;
esac

if [ ! -d "$RESULTS_DIR" ]; then
  echo "no '$RESULTS_DIR/' directory here -- run from the app dir" >&2
  exit 1
fi

# Everything under results/ that is NOT a kept (zeros / rand1) CSV.
mapfile -d '' VICTIMS < <(
  find "$RESULTS_DIR" -type f -name '*.csv' \
    ! -name '*_data_zeros_bram.csv' \
    ! -name '*_data_rand1_bram.csv' \
    -print0
)

if [ "${#VICTIMS[@]}" -eq 0 ]; then
  echo "Nothing to remove -- results/ already holds only zeros/rand1 CSVs."
  exit 0
fi

printf '%s\n' "${VICTIMS[@]}"
echo "----"
echo "${#VICTIMS[@]} file(s) match for removal."

if [ "$APPLY" -eq 1 ]; then
  rm -f -- "${VICTIMS[@]}"
  echo "Deleted ${#VICTIMS[@]} file(s)."
else
  echo "Dry run -- re-run with --apply to delete."
fi
