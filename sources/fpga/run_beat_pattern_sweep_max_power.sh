#!/bin/bash
# Beat-pattern sweep over 4-bit patterns for the max_power_loop access pattern
# (Figs 13 and 15, measured side).
#
# A 4-bit pattern (0-15) controls whether each of the 4 data beats is
# "normal" or "flipped":
#   normal:  0x00000000 0xFFFFFFFF  (beat = 0x00000000FFFFFFFF)
#   flipped: 0xFFFFFFFF 0x00000000  (beat = 0xFFFFFFFF00000000)
# Each pattern runs twice: once without --invert-col1 (inv0) and once with it
# (inv1). Uses bank groups 0 and 2, bank offset 0.
#
# test_select=24 -> results/beat_pattern_variation_fixed_reset_full_ipp/
#   consumed by: extract_beat_pattern_combined.py -> beat_pattern_combined_measurements.csv
#                (and derived beat_pattern_perpattern.csv for Fig 15)
#
# Usage: ./run_beat_pattern_sweep_max_power.sh [--no-warmup] [--high] [path_to_binary]
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
echo "  Beat Pattern Sweep (max_power_loop)"
echo "=============================================="
echo "Binary:            $BINARY"
echo "Channels:          $CH_LABEL"
echo "Bank groups:       0, 2 (offset 0)"
echo "Warmup duration:   ${WARMUP_DURATION}s"
echo "Measure duration:  ${DURATION}s per test"
echo "Number of patterns: 16 (x2 inv variants = 32 tests)"
echo ""

# =====================================================================
# WARMUP: Run max_power_loop to raise temperature
# =====================================================================
if [[ $DO_WARMUP -eq 1 ]]; then
    echo "====== WARMUP: max_power_loop, channels $CH_LABEL, ${WARMUP_DURATION}s ======"
    WARMUP_PAT="--pattern-32b 0x00000000 0xFFFFFFFF 0xFFFFFFFF 0x00000000 0xFFFFFFFF 0x00000000 0x00000000 0xFFFFFFFF"
    $BINARY 24 11 7 $NUM_EXCLUDED $EXCLUDE_LIST \
        --bg 0 2 --banks 0 0 0 0 --rows 0 0 --cols 21 10 \
        --fixed-duration $WARMUP_DURATION $WARMUP_PAT
    echo "Warmup complete."
    echo ""
else
    echo "====== WARMUP: Skipped (--no-warmup) ======"
    echo ""
fi

# =====================================================================
# BEAT PATTERN SWEEP: 16 patterns x 2 inv variants = 32 tests
# =====================================================================
for p in $(seq 0 15); do
    # Generate 8 x 32-bit words from the 4-bit beat pattern
    WORDS=()
    for beat in 0 1 2 3; do
        bit=$(( (p >> beat) & 1 ))
        if [[ $bit -eq 0 ]]; then
            # normal beat: 0x00000000 0xFFFFFFFF
            WORDS+=("0x00000000" "0xFFFFFFFF")
        else
            # flipped beat: 0xFFFFFFFF 0x00000000
            WORDS+=("0xFFFFFFFF" "0x00000000")
        fi
    done

    PATTERN_BINARY=$(printf "%04b" "$p")
    echo "========================================"
    echo "Beat pattern p=$p (0b${PATTERN_BINARY}): ${WORDS[*]}"
    echo "  Beat 0: ${WORDS[0]} ${WORDS[1]}"
    echo "  Beat 1: ${WORDS[2]} ${WORDS[3]}"
    echo "  Beat 2: ${WORDS[4]} ${WORDS[5]}"
    echo "  Beat 3: ${WORDS[6]} ${WORDS[7]}"
    echo "========================================"

    for INV in 0 1; do
        if [[ $INV -eq 0 ]]; then
            INV_FLAG=""
            INV_LABEL="inv0"
        else
            INV_FLAG="--invert-col1"
            INV_LABEL="inv1"
        fi

        echo "  Running p=$p ${INV_LABEL} ($((p * 2 + INV + 1))/32)..."
        $BINARY 24 11 7 $NUM_EXCLUDED $EXCLUDE_LIST \
            --bg 0 2 --banks 0 0 0 0 --rows 0 0 --cols 21 10 \
            --fixed-duration $DURATION \
            --pattern-32b ${WORDS[0]} ${WORDS[1]} ${WORDS[2]} ${WORDS[3]} \
                          ${WORDS[4]} ${WORDS[5]} ${WORDS[6]} ${WORDS[7]} \
            $INV_FLAG
        echo "  Test p=$p ${INV_LABEL} complete."
        echo ""
    done
done

echo "All beat pattern sweep tests complete."
