#!/usr/bin/env bash
# Component 2: H200 (HBM3E) power characterization.
#
# Thin wrapper around scripts/h200/vast_run.py. Rents the cheapest qualifying
# 1x NVIDIA H200 on vast.ai, runs the memory-power benchmark, fetches the
# results to data/new/h200/hbm3e_results.csv, and always destroys the instance.
#
# Runtime a few minutes (~$0.30-0.50). Requires VAST_API_KEY, which we provide
# via HotCRP. Pass --dry-run for a free check (confirms the key and target
# offer, rents nothing). See scripts/h200/README.md for details.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/scripts/h200/vast_run.py" "$@"
