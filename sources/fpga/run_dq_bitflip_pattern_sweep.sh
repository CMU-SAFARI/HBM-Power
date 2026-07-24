#!/bin/bash
# DQ bit-flip pattern sweep -- IDD4R-full with 32 128-bit-periodic patterns (Fig 12).
#
# Pattern k (k=0..31): first 64 bits have k LSBs set, second 64 bits have k MSBs
# cleared. The 128-bit pattern repeats to fill 256 bits. Uses BGs 0 and 1.
# Each point: SWEEP_S seconds, first 8 channels (exclude 8-15), PC0, --use-full.
#
# test_select=10 -> results/bitflip_variation_fixed_reset_full_ipp/
#   emitted:   hbm_idd4r_full_32b_..._dur<SWEEP_S>s.csv  (128-bit-periodic)
#   consumed:  extract_bitflip.py (IDD4R_full rows) -> bitflip_measurements.csv
#
# Usage: ./run_dq_bitflip_pattern_sweep.sh [--no-warmup] [--high] [path_to_binary]
#
# Duration knobs (env, default = paper config):
#   SWEEP_S    per-point measurement duration  (default 90)
#   WARMUP_S   warm-up duration                 (default 900)
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

DURATION="${SWEEP_S:-90}"
WARMUP_DURATION="${WARMUP_S:-900}"  # 15 minutes

# Channel exclusion
NUM_EXCLUDED=8
if [[ $USE_HIGH -eq 1 ]]; then
    EXCLUDE_LIST="0 1 2 3 4 5 6 7"
    CH_LABEL="8-15"
else
    EXCLUDE_LIST="8 9 10 11 12 13 14 15"
    CH_LABEL="0-7"
fi

echo "=============================================="
echo "  DQ Bit-flip Pattern Sweep (IDD4R-full)"
echo "  Channels: $CH_LABEL   32 patterns x ${DURATION}s"
echo "=============================================="

# =====================================================================
# WARMUP: Run IDD4R-full to raise temperature
# =====================================================================
if [[ $DO_WARMUP -eq 1 ]]; then
    echo "====== WARMUP: IDD4R-full, channels $CH_LABEL, ${WARMUP_DURATION}s ======"
    $BINARY 10 4 6 $NUM_EXCLUDED $EXCLUDE_LIST --fixed-duration $WARMUP_DURATION --use-full
    echo "Warmup complete."
    echo ""
else
    echo "====== WARMUP: Skipped (--no-warmup) ======"
    echo ""
fi

for k in $(seq 0 31); do
    # First 64 bits: k LSBs set
    first_64_lo=$(( (1 << k) - 1 ))
    first_64_hi=0

    # Second 64 bits: k MSBs cleared (starting from all 1s)
    second_64_lo=$(( 0xFFFFFFFF ))
    second_64_hi=$(( 0xFFFFFFFF >> k ))

    # Format as 0x hex words
    w0=$(printf "0x%08x" $first_64_lo)
    w1=$(printf "0x%08x" $first_64_hi)
    w2=$(printf "0x%08x" $second_64_lo)
    w3=$(printf "0x%08x" $second_64_hi)
    # 128-bit periodic: repeat
    w4=$w0
    w5=$w1
    w6=$w2
    w7=$w3

    echo "========================================"
    echo "Running pattern sweep test k=$k ($((k+1))/32)"
    echo "  Pattern: $w0 $w1 $w2 $w3 (128-bit, repeated)"
    echo "========================================"
    $BINARY 10 4 7 $NUM_EXCLUDED $EXCLUDE_LIST \
        --bg 0 1 \
        --fixed-duration $DURATION \
        --use-full \
        --pattern-32b $w0 $w1 $w2 $w3 $w4 $w5 $w6 $w7
    echo "Test k=$k complete."
    echo ""
done

echo "All pattern sweep tests complete."
