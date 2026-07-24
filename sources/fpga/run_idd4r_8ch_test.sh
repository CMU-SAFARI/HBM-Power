#!/bin/bash
#
# IDD4R-full 8-channel temperature-dependence measurement (Fig 7).
#
# IDD4R-full (--use-full) on channels 0-7, PC0. This loop is itself the warm
# read pattern, so there is no separate warm-up phase -- run it FIRST in the
# chip campaign to heat the die for the subsequent IDD loops.
#
# test_select=7 -> results/temperature_dependence_fixed_reset_full_ipp/
#
# NOTE (build spec 3.4): --use-full makes the current binary emit the label
# "idd4r_16bank" in the filename, whereas the released standardizer keys on
# "idd4r_full". The single-chip standardizer (standardize_single_chip.py)
# accepts either name, so this stays faithful to the original script.
#
# Usage: ./run_idd4r_8ch_test.sh [--high] [path_to_binary]
#
# Duration knob (env, default = paper config):
#   MEASURE_S  IDD4R-full measurement duration  (default 1800)
#

set +e

# Parse options
USE_HIGH=0
while [[ "$1" == --* ]]; do
    case "$1" in
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

# IDD4R-full access pattern (4) with --use-full
ACCESS_PATTERN=4
UF="--use-full"

MEASURE_DURATION="${MEASURE_S:-1800}"  # 30 minutes

# Exclude opposite half to keep only the target channels
if [[ $USE_HIGH -eq 1 ]]; then
    NUM_EXCLUDED=8
    EXCLUDE_LIST="0 1 2 3 4 5 6 7"
    CH_LABEL="8-15"
else
    NUM_EXCLUDED=8
    EXCLUDE_LIST="8 9 10 11 12 13 14 15"
    CH_LABEL="0-7"
fi

echo "=============================================="
echo "  IDD4R-full 8-Channel ($CH_LABEL) Temperature Test"
echo "=============================================="
echo "Binary:   $BINARY"
echo "Pattern:  $PATTERN"
echo "Channels: $CH_LABEL"
echo "Results:  results/temperature_dependence"
echo ""

# =====================================================================
# IDD4R-full MEASUREMENT
# =====================================================================
echo "====== IDD4R-full MEASUREMENT: channels $CH_LABEL, ${MEASURE_DURATION}s ======"
for PC in 0; do
    echo "  --- PC$PC ---"
    echo "  Running: $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $MEASURE_DURATION --pc $PC $UF"
    $BINARY $TEST_SELECT $ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $MEASURE_DURATION --pc $PC $UF
done
echo "IDD4R-full 8-channel measurement complete."
echo ""

echo "=============================================="
echo "  IDD4R-full 8-channel test complete."
echo "=============================================="
