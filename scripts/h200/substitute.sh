#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$ARTIFACT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/merged}"
NEW_FILE="${NEW_FILE:-$DATA_DIR/new/h200/hbm3e_results.csv}"
SAMPLE="${SAMPLE:-vast_h200}"

python3 "$SCRIPT_DIR/substitute_h200.py" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --new-file "$NEW_FILE" \
  --sample "$SAMPLE"