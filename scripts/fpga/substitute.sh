#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR="${DATA_DIR:-$ARTIFACT_DIR/data}"
NEW_DIR="${NEW_DIR:-$DATA_DIR/new/fpga}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/merged}"
CHIP_ID="${CHIP_ID:-0}"

python3 "$SCRIPT_DIR/substitute_chip0.py" \
	--data-dir "$DATA_DIR" \
	--new-dir "$NEW_DIR" \
	--output-dir "$OUTPUT_DIR" \
	--chip-id "$CHIP_ID"
