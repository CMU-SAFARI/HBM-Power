#!/bin/bash
#
# IDD3N1 temperature-dependence measurement (Fig 5).
#
# IDD3N1 (one row open in one bank / active standby) on channels 0-7, PC0,
# preceded by an IDD4R-full warm-up (skip with --no-warmup).
#
# test_select=7 -> results/temperature_dependence_fixed_reset_full_ipp/
#
# Usage: ./run_idd3n1_test.sh [--no-warmup] [--high] [path_to_binary]
#
# Duration knobs (env, default = paper config):
#   WARMUP_S   warm-up duration in seconds     (default 1800)
#   MEASURE_S  IDD3N1 measurement duration      (default 1800)
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

# Data pattern (2 = 0x0055ffaa)
PATTERN=2

# Test select (7 = temperature dependence)
TEST_SELECT=7

# Warmup: IDD4R-full access pattern (4) with --use-full
WARMUP_ACCESS_PATTERN=4
WARMUP_PATTERN=6  # multi_00_55_ff_aa
UF="--use-full"
WARMUP_DURATION="${WARMUP_S:-1800}"  # 15 minutes

# IDD3N1 access pattern (3) and duration
IDD3N1_ACCESS_PATTERN=3
IDD3N1_DURATION="${MEASURE_S:-1800}"  # 15 minutes

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
echo "  IDD3N1 Temperature Dependence Test"
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
    echo "Running: $BINARY $TEST_SELECT $WARMUP_ACCESS_PATTERN $WARMUP_PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $WARMUP_DURATION $UF"
    $BINARY $TEST_SELECT $WARMUP_ACCESS_PATTERN $WARMUP_PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $WARMUP_DURATION $UF
    echo "Warmup complete."
    echo ""
else
    echo "====== WARMUP: Skipped (--no-warmup) ======"
    echo ""
fi

# =====================================================================
# IDD3N1 MEASUREMENT
# =====================================================================
echo "====== IDD3N1 MEASUREMENT: channels $CH_LABEL, ${IDD3N1_DURATION}s ======"
for PC in 0; do
    echo "  --- PC$PC ---"
    echo "  Running: $BINARY $TEST_SELECT $IDD3N1_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD3N1_DURATION --pc $PC"
    $BINARY $TEST_SELECT $IDD3N1_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD3N1_DURATION --pc $PC
done
echo "IDD3N1 measurement complete."
echo ""

echo "=============================================="
echo "  IDD3N1 temperature dependence test complete."
echo "=============================================="
