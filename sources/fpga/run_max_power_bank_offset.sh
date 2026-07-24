#!/bin/bash
# Test 14: Max power loop -- bank-offset test (Fig 11).
#
# Bank groups 0 and 2, trying all 4 offsets (0, 1, 2, 3) within the bank group.
# Offset k means banks (4*0 + k) and (4*2 + k), i.e. banks k and 8+k.
# Each point is run with --invert-col1 (inv1), which the extractor requires.
# Data pattern: 00000000 FFFFFFFF FFFFFFFF 00000000 FFFFFFFF 00000000 00000000 FFFFFFFF
#
# test_select=14 -> results/max_power_bank_offset_fixed_reset_full_ipp/
#   consumed by: extract_bank_offset.py (inv1 files) -> bank_offset_measurements.csv
#
# Usage: ./run_max_power_bank_offset.sh [--warmup] [--high] [path_to_binary]
#
# Duration knobs (env, default = paper config):
#   SWEEP_S    per-point measurement duration  (default 90)
#   WARMUP_S   optional warm-up duration        (default 900; only with --warmup)
#

set +e

# Parse optional flags first, then positional args
DO_WARMUP=0
USE_HIGH=0
while [[ "$1" == --* ]]; do
    case "$1" in
        --warmup) DO_WARMUP=1; shift ;;
        --high) USE_HIGH=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BINARY="${1:-./SoftMC_rdwr}"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: measurement binary '$BINARY' not found or not executable (cwd: $(pwd))." >&2
    echo "       Run from the Power_structural_variation dir on the FPGA, or pass the binary path." >&2
    exit 1
fi

TEST_SELECT=14
ACCESS_PATTERN=11  # max_power_loop
PATTERN=7          # 32-byte pattern
if [[ $USE_HIGH -eq 1 ]]; then
    NUM_EXCLUDED=8
    EXCLUDE_LIST="0 1 2 3 4 5 6 7"
    CH_LABEL="8-15"
else
    NUM_EXCLUDED=8
    EXCLUDE_LIST="8 9 10 11 12 13 14 15"
    CH_LABEL="0-7"
fi
DURATION="${SWEEP_S:-90}"
WARMUP_DURATION="${WARMUP_S:-900}"

PAT="--pattern-32b 0x00000000 0xFFFFFFFF 0xFFFFFFFF 0x00000000 0xFFFFFFFF 0x00000000 0x00000000 0xFFFFFFFF"

echo "=============================================="
echo "  Max Power Loop: Bank Offset Test"
echo "  Bank groups 0 and 2, offsets 0-3"
echo "  Channels: $CH_LABEL"
echo "=============================================="

if [ "$DO_WARMUP" -eq 1 ]; then
    echo "=== Warmup: max_power_loop on channels $CH_LABEL for ${WARMUP_DURATION}s ==="
    $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST \
        --bg 0 2 --banks 0 0 0 0 --rows 0 0 --cols 21 10 \
        --fixed-duration $WARMUP_DURATION $PAT
    echo "=== Warmup complete ==="
fi

for offset in 0 1 2 3; do
    echo "--- Offset $offset -> banks ($offset, $((8 + offset))) ---"
    $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST \
        --bg 0 2 --banks $offset $offset $offset $offset --rows 0 0 --cols 21 10 \
        --fixed-duration $DURATION $PAT --invert-col1
done

echo ""
echo "=== Bank offset test complete ==="
