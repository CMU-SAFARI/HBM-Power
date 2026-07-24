#!/bin/bash
#
# IDD4W temperature-dependence measurement (Fig 7).
#
# IDD4W (continuous writes) on channels 0-7, PC0, preceded by an IDD4R-full
# warm-up (skip with --no-warmup). Data pattern 6 (8B 00, 8B 55, 8B FF, 8B AA).
#
# test_select=7 -> results/temperature_dependence_fixed_reset_full_ipp/
#
# Usage: ./run_idd4w_test.sh [--no-warmup] [--high] [path_to_binary]
#
# Duration knobs (env, default = paper config):
#   WARMUP_S   warm-up duration in seconds   (default 1800)
#   MEASURE_S  IDD4W measurement duration    (default 1800)
#

set +e

# Parse options
DO_WARMUP=1
USE_HIGH=0
while [[ "$1" == --* ]]; do
    case "$1" in
        --no-warmup) DO_WARMUP=0; shift ;;
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

# Data pattern (6 = 8B 0x00, 8B 0x55, 8B 0xFF, 8B 0xAA)
PATTERN=6

# Test select (7 = temperature dependence)
TEST_SELECT=7

# Warmup: IDD4R-full access pattern (4) with --use-full
WARMUP_ACCESS_PATTERN=4
UF="--use-full"
WARMUP_DURATION="${WARMUP_S:-1800}"

# IDD4W access pattern (1) and duration
IDD4W_ACCESS_PATTERN=1
IDD4W_DURATION="${MEASURE_S:-1800}"

# Exclude opposite half to keep only the target channels
NUM_EXCLUDED=8
if [[ $USE_HIGH -eq 1 ]]; then
    EXCLUDE_LIST="0 1 2 3 4 5 6 7"
    CH_LABEL="8-15"
else
    EXCLUDE_LIST="8 9 10 11 12 13 14 15"
    CH_LABEL="0-7"
fi

echo "=============================================="
echo "  IDD4W Temperature Dependence Test"
echo "=============================================="
echo "Binary:   $BINARY"
echo "Pattern:  $PATTERN"
echo "Channels: $CH_LABEL"
echo "Results:  results/temperature_dependence"
echo ""

# =====================================================================
# WARMUP: Run IDD4R-full on the active channels
# =====================================================================
if [[ $DO_WARMUP -eq 1 ]]; then
    echo "====== WARMUP: IDD4R-full, channels $CH_LABEL, ${WARMUP_DURATION}s ======"
    echo "Running: $BINARY $TEST_SELECT $WARMUP_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $WARMUP_DURATION $UF"
    $BINARY $TEST_SELECT $WARMUP_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $WARMUP_DURATION $UF
    echo "Warmup complete."
    echo ""
else
    echo "====== WARMUP: Skipped (--no-warmup) ======"
    echo ""
fi

# =====================================================================
# IDD4W MEASUREMENT
# =====================================================================
echo "====== IDD4W MEASUREMENT: channels $CH_LABEL, ${IDD4W_DURATION}s ======"
for PC in 0; do
    echo "  --- PC$PC ---"
    echo "  Running: $BINARY $TEST_SELECT $IDD4W_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD4W_DURATION --pc $PC"
    $BINARY $TEST_SELECT $IDD4W_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD4W_DURATION --pc $PC
done
echo "IDD4W measurement complete."
echo ""

echo "=============================================="
echo "  IDD4W temperature dependence test complete."
echo "=============================================="
