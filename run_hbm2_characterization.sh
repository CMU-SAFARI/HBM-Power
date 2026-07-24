#!/usr/bin/env bash
# Component 1: HBM2 (FPGA) power characterization.
#
# Thin wrapper around scripts/fpga/run_fpga_ae.sh. Reprograms our FPGA and runs
# the single-chip IDD/structural/trace measurements over SSH, writing the
# sanitized per-sample CSVs to data/new/fpga/*.csv.
#
# Runtime ~8 hours. Requires the FPGA SSH key pair we provide via HotCRP
# (./aevaluator1, ./aevaluator1.pub). Pass --resume to continue an interrupted
# run. See scripts/fpga/README.md for details and connection overrides.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/fpga/run_fpga_ae.sh" "$@"
