#!/bin/bash
#
# AE BRAM trace-runner sweep (Fig 14 + Table 3, single chip).
#
# Runs ON the FPGA host from the vendored HBMTraceRunnerBRAM app directory
# (build ./HBMTraceRunnerBRAM there with `make`). Replays each of the 14 AE
# workloads for TRACE_S seconds while sampling HBM power, once per data-init
# scheme (zeros, then per-word random seed 1). Each run writes
# results/<set>_<trace>_data_{zeros,rand1}_bram.csv.
#
# This is the AE-subset restriction of the upstream run_bram_local.sh:
#   * trace set 2 (the 6 Ramulator microbenchmarks) -- ENABLED here (it is
#     commented out upstream),
#   * trace set 3 (the 8-point LLaMa3.1-8B pc0 ctx1024 batch sweep) -- ENABLED,
#   * trace set 1 (the 1-rank A100 trace) -- DROPPED (not one of the 14 plotted
#     workloads).
# The 14 traces here are exactly figure14.py MICRO + table3.py WORKLOADS.
#
# Downstream: copy_results.sh --trace pulls results/*_bram.csv back, then
# aggregate_trace_ground_truth.py maps them to data/ground_truth_*.csv.
#
# Usage:  ./run_trace_sweep.sh [--channels low|high] [--llm-only|--micro-only]
#                              [path_to_binary]
#
#   --channels low   stack0 / chip0, channels 0-7   (default; bram_tracer_chip0)
#   --channels high  stack1 / chip1, channels 8-15  (bram_tracer_chip1)
#   --llm-only       replay only the 8 LLaMa batch points
#   --micro-only     replay only the 6 microbenchmarks
#   path_to_binary   trace-runner binary            (default ./HBMTraceRunnerBRAM)
#
# Duration knob (env, default = paper config):
#   TRACE_S   per-trace replay duration in seconds   (default 90)
#
# Trace inputs (env-overridable):
#   TRACES_BASE  root of the trace tree, relative to the app dir (default
#                ../../../../traces = the artifact's sources/fpga/traces, with
#                micro/ + llm/ subdirs; for a private_bsc_traces checkout set
#                TRACES_BASE=../../../../cmd_traces/rebuttal ORIG_IN=original_traces
#                LLAMA_IN=llama8Bshort_runner_pc0)
#
# A single failed replay does not abort the rest of the sweep (set +e), so a
# transient failure on one workload still lets the others produce data.

set +e

# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
CHANNELS="low"
RUN_MICRO=1
RUN_LLM=1
while [[ "$1" == --* ]]; do
    case "$1" in
        --channels) CHANNELS="$2"; shift 2 ;;
        --channels=*) CHANNELS="${1#*=}"; shift ;;
        --llm-only)   RUN_MICRO=0; RUN_LLM=1; shift ;;
        --micro-only) RUN_MICRO=1; RUN_LLM=0; shift ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$CHANNELS" in
    low|high) ;;
    *) echo "ERROR: --channels must be 'low' or 'high' (got '$CHANNELS')." >&2; exit 2 ;;
esac

BINARY="${1:-./HBMTraceRunnerBRAM}"
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: trace-runner binary '$BINARY' not found or not executable (cwd: $(pwd))." >&2
    echo "       Run from the HBMTraceRunnerBRAM app dir on the FPGA (build it with 'make')," >&2
    echo "       or pass the binary path." >&2
    exit 1
fi

# --------------------------------------------------------------------------- #
# Config (matches upstream run_bram_local.sh)
# --------------------------------------------------------------------------- #
TRACES_BASE="${TRACES_BASE:-../../../../traces}"

# Trace set 2: original Ramulator microbenchmarks (flat single-channel CSVs).
# ORIG_IN/LLAMA_IN are the input subdirs under TRACES_BASE; ORIG_DIR/LLAMA_SET
# stay the OUTPUT filename prefixes (the released ground-truth test_name keys
# that aggregate_trace_ground_truth.py matches on) -- do not rename those.
ORIG_IN="${ORIG_IN:-micro}"
ORIG_DIR=original_traces
ORIG_TRACES="act_hammer_ramulator interleaved_ramulator streaming_all_banks_4cyc_ramulator streaming_reads_4cyc_ramulator wr_rd_turnaround_4cyc_ramulator ws_bg0_bg2_ramulator"

# Trace set 3: llama8Bshort pc0-only ctx1024 batch sweep (8 points). The pc0 set
# is the one the released ground truth uses; its heaviest bank fits the deployed
# 131072-depth tracer bitstream for every ctx1024 trace (incl. bs64/bs128), so
# no truncation is needed. The set label is the ground-truth test_name prefix --
# keep it as-is (aggregate_trace_ground_truth.py maps LLM names by identity).
LLAMA_IN="${LLAMA_IN:-llm}"
LLAMA_SET=llama8Bshort_runner_pc0
LLAMA_TRACES="bs1_ctx1024 bs2_ctx1024 bs4_ctx1024 bs8_ctx1024 bs16_ctx1024 bs32_ctx1024 bs64_ctx1024 bs128_ctx1024"

# Per-trace replay duration (s). The app loops the trace forever (--iterations 0);
# `timeout -s INT` sends a graceful stop at the deadline, -k 10 hard-kills 10 s later.
DURATION="${TRACE_S:-90}"

# Idle slots after the trace's closing PREs before the loop wraps (loop-seam
# timing guarantee; 64 slots ~107 ns > tRP). Also the app default.
TAIL_GAP=64

# Random-init seed. Fixed so the data is identical across FPGAs and reproducible;
# the seed is recorded in each CSV name (rand<seed>).
INIT_SEED=1

mkdir -p results

echo "=============================================="
echo "  AE BRAM trace sweep (Fig 14 + Table 3)"
echo "=============================================="
echo "Binary:    $BINARY"
echo "Channels:  $CHANNELS"
echo "Duration:  ${DURATION}s per trace"
echo "Sets:      micro=$RUN_MICRO llm=$RUN_LLM"
echo "Traces base: $TRACES_BASE"
echo ""

# Replay one trace. $1 = CSV path, $2 = output CSV basename (under results/).
# Reads the current data-init scheme from $INIT_ARGS (set per mode below).
run_trace() {
    if [[ ! -f "$1" ]]; then
        echo "  !! trace not found, skipping: $1" >&2
        return 1
    fi
    echo "=== ${2} (${INIT_ARGS}) ==="
    timeout -k 10 -s INT "${DURATION}" "$BINARY" \
        --csv "${1}" \
        --channels "${CHANNELS}" \
        ${INIT_ARGS} \
        --tail-gap "${TAIL_GAP}" \
        --output "results/${2}"
}

for mode in zeros random; do
    if [ "$mode" = "random" ]; then
        INIT_ARGS="--init random --init-seed ${INIT_SEED}"; label="rand${INIT_SEED}"
    else
        INIT_ARGS="--init zeros --wr-pattern 00000000";      label="zeros"
    fi

    # Trace set 2: original Ramulator microbenchmarks.
    if [ "$RUN_MICRO" -eq 1 ]; then
        for t in $ORIG_TRACES; do
            run_trace "${TRACES_BASE}/${ORIG_IN}/${t}.csv" \
                      "${ORIG_DIR}_${t}_data_${label}_bram.csv"
        done
    fi

    # Trace set 3: llama8Bshort pc0-only ctx1024 batch sweep.
    if [ "$RUN_LLM" -eq 1 ]; then
        for t in $LLAMA_TRACES; do
            run_trace "${TRACES_BASE}/${LLAMA_IN}/${t}.csv" \
                      "${LLAMA_SET}_${t}_data_${label}_bram.csv"
        done
    fi
done

echo ""
echo "=============================================="
echo "  run_trace_sweep.sh: sweep complete."
echo "  Results in $(pwd)/results/*_bram.csv"
echo "=============================================="
