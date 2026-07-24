#!/bin/bash
# Test 15: Max power loop -- bank-group pair test with inversion variants (Fig 10).
#
# Tests all C(4,2) = 6 pairs of bank groups, each with two variations:
#   1. Same data pattern on both columns (no inversion)   -> inv0 (unused by Fig 10)
#   2. Inverted data pattern on col1 (--invert-col1)      -> inv1 (used by Fig 10)
# Bank group i contains banks 4*i .. 4*i+3. Offset 0 within each group.
# Data pattern: 00000000 FFFFFFFF FFFFFFFF 00000000 FFFFFFFF 00000000 00000000 FFFFFFFF
#
# test_select=15 -> results/max_power_bg_invert_fixed_reset_full_ipp/
#   consumed by: extract_bank_group.py (inv1 files) -> bank_group_measurements.csv
#
# Usage: ./run_max_power_bg_invert.sh [--warmup] [--high] [path_to_binary]
#
# Duration knobs (env, default = paper config):
#   SWEEP_S    per-point measurement duration  (default 90)
#   WARMUP_S   optional warm-up duration        (default 1800; only with --warmup)
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

TEST_SELECT=15
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
WARMUP_DURATION="${WARMUP_S:-1800}"

PAT="--pattern-32b 0x00000000 0xFFFFFFFF 0xFFFFFFFF 0x00000000 0xFFFFFFFF 0x00000000 0x00000000 0xFFFFFFFF"

echo "=============================================="
echo "  Max Power Loop: Bank Group Inversion Test"
echo "  All 6 BG pairs x 2 inversion variants"
echo "  Channels: $CH_LABEL"
echo "=============================================="

if [ "$DO_WARMUP" -eq 1 ]; then
    echo "=== Warmup: max_power_loop on channels $CH_LABEL for ${WARMUP_DURATION}s ==="
    $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST \
        --bg 0 1 --banks 0 0 0 0 --rows 0 0 --cols 21 10 \
        --fixed-duration $WARMUP_DURATION $PAT
    echo "=== Warmup complete ==="
fi

for bg0 in 0 1 2 3; do
    for bg1 in $(seq $((bg0 + 1)) 3); do
        echo "--- BG pair ($bg0, $bg1), no inversion ---"
        $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST \
            --bg $bg0 $bg1 --banks 0 0 0 0 --rows 0 0 --cols 0 1 \
            --fixed-duration $DURATION $PAT

        echo "--- BG pair ($bg0, $bg1), with inversion ---"
        $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST \
            --bg $bg0 $bg1 --banks 0 0 0 0 --rows 0 0 --cols 0 1 \
            --fixed-duration $DURATION $PAT --invert-col1
    done
done

echo ""
echo "=== Bank group inversion test complete ==="
