#!/usr/bin/env bash
# Component 3: power-model simulation (DRAMPower).
#
# Rebuilds the power-model machinery from source:
#   1. regenerates the LLaMA workload command traces (scripts/traces/llm_traces.py), and
#   2. builds the DRAMPower HBM2_runner (sources/drampower).
#
# The figure/table step (reproduce_figures_tables.sh) invokes HBM2_runner on the
# committed workload traces to predict HBM2/HBM3E power; run this only to rebuild
# those inputs from source. Runtime a few minutes. No hardware or remote access.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== Regenerating workload traces =="
python3 "$SCRIPT_DIR/scripts/traces/llm_traces.py"

echo "== Building DRAMPower engine =="
ENGINE="$SCRIPT_DIR/sources/drampower"
BUILD="$ENGINE/build"
cmake -S "$ENGINE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DDRAMPOWER_BUILD_TESTS=OFF -DDRAMPOWER_BUILD_BENCHMARKS=OFF -DDRAMPOWER_BUILD_CLI=ON
cmake --build "$BUILD" --target HBM2_runner -j
echo "done"
