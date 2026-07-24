#!/bin/bash
#
# No-HBM idle-offset baseline (required by ALL figures 2-13).
#
# Trimmed copy of run_idd2_test.sh: a single-channel (channel 0 only) IDD2 run
# with the HBM core absent, capturing the FPGA board's own current draw. Meant
# to run on the XCU55_no_hbm bitstream. No warm-up.
#
# test_select=7 -> results/temperature_dependence_fixed_reset_full_ipp/
#   emitted file: hbm_idd2_0055ffaa_0_pc0_..._dur<NOHBM_S>s.csv
#   consumed by:  extract_no_hbm.py -> no_hbm_idd2_measurements.csv
#
# Usage: ./run_no_hbm.sh [--no-warmup] [path_to_binary]
#
# Duration knob (env, default = paper config):
#   NOHBM_S  baseline measurement duration in seconds  (default 90)
#

set +e

# Parse options (accepted for interface parity; this baseline never warms up
# and always uses channel 0).
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

# IDD2 access pattern (2) and duration
IDD2_ACCESS_PATTERN=2
IDD2_DURATION="${NOHBM_S:-90}"  # 90 seconds

# Exclude all channels except 0
NUM_EXCLUDED=15
EXCLUDE_LIST="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
CH_LABEL="0"

echo "=============================================="
echo "  No-HBM Idle-Offset Baseline (IDD2, channel 0)"
echo "=============================================="
echo "Binary:   $BINARY"
echo "Pattern:  $PATTERN"
echo "Channels: $CH_LABEL"
echo "Results:  results/temperature_dependence"
echo ""

# =====================================================================
# WARMUP: never run for the no-HBM baseline
# =====================================================================
echo "====== WARMUP: Skipped ======"
echo ""

# =====================================================================
# IDD2 MEASUREMENT (channel 0 only, no other HBM active)
# =====================================================================
echo "====== IDD2 MEASUREMENT: channel $CH_LABEL, ${IDD2_DURATION}s ======"
for PC in 0; do
    echo "  --- PC$PC ---"
    echo "  Running: $BINARY $TEST_SELECT $IDD2_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD2_DURATION --pc $PC"
    $BINARY $TEST_SELECT $IDD2_ACCESS_PATTERN $PATTERN $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $IDD2_DURATION --pc $PC
done
echo "IDD2 baseline measurement complete."
echo ""

echo "=============================================="
echo "  No-HBM baseline test complete."
echo "=============================================="
