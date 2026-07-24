#!/bin/bash
# Per-FPGA BRAM trace-runner sweep. Executed ON each FPGA by run_on_infra_bram.sh
# (after git checkout/pull + make). Replays each selected trace for DURATION
# seconds while sampling HBM power, once per data-init scheme: the trace footprint
# is pre-written with zeros, then with per-word random data. Results land in
# results/<set>_<trace>_data_{zeros,rand<seed>}_bram.csv.
#
# This file holds the actual run list and is committed, so every FPGA runs the
# identical sequence after pulling. run_on_infra_bram.sh only orchestrates
# ssh/tmux/git/build and then `bash run_bram_local.sh` -- this keeps that
# orchestrator's tmux send-keys payload short. (Inlining the whole sequence there
# produced a single command line that exceeded the terminal's 4096-byte canonical
# input limit and got silently truncated mid-command.)
#
# Run from the app dir (HBMTraceRunnerBRAM); all trace paths are relative to it.

set +e   # a single failed run must not abort the rest of the sweep

# All trace sets live under cmd_traces/rebuttal/ (moved there in the "Reorder
# CSVs + cmd_traces for llama8Bshort" commit). TRACES_BASE is relative to the
# app dir (four levels above the repo root's cmd_traces/).
TRACES_BASE=../../../../cmd_traces/rebuttal

# Trace set 1: the 1-rank A100 HBM2 command trace (cmd_hbm2_40gb-1rank.csv)
# split into one runnable trace per pseudo-channel; with a single rank both PCs
# live under stack0.
#
# We replay the *_trunc set: each per-PC trace is truncated so no replay bank
# (timestamp mod 4) exceeds the deployed bitstream's BANK_DEPTH (65536 entries),
# since the full 1-rank traces need ~69k/bank. The full (untruncated) traces
# live in cmd_traces/llm_hbm2_40gb_1rank/ -- switch TRACE_DIR to that once a
# bitstream built with the larger BANK_DEPTH (see bram_replay.v) is deployed.
TRACE_DIR=llm_hbm2_40gb_1rank_trunc
TRACES="stack0/pc0 stack0/pc1"

# Trace set 2: original Ramulator microbenchmark traces (flat CSVs under
# ${TRACES_BASE}/${ORIG_DIR}/). Single channel/PC (channel 0, pc0) and all fit
# the 65536-entry replay banks, so they run as-is on the deployed bitstream (no
# truncation needed). Each already ends with all banks precharged, so it loops
# cleanly. DISABLED below -- uncomment the loop to re-enable.
ORIG_DIR=original_traces
ORIG_TRACES="act_hammer_ramulator interleaved_ramulator streaming_all_banks_4cyc_ramulator streaming_reads_4cyc_ramulator wr_rd_turnaround_4cyc_ramulator ws_bg0_bg2_ramulator"

# Trace set 3: llama8Bshort LLM traces -- a batch-size sweep at context length
# 1024 (flat single-channel CSVs, channel 0). These are the RUNNER-FORMAT traces
# produced by cmd_traces/convert_llama8Bshort.py from the raw Ramulator dumps in
# ../llama8Bshort/ (the raw files are NOT directly runnable: wrong column order +
# comma-space-quoted command names); the converter also appends closing PREs so
# every bank ends precharged for clean looping.
#
# The replay engine splits each trace into 4 banks by (timestamp mod 4); the
# heaviest bank must fit the bitstream's BANK_DEPTH. Heaviest-bank entry counts
# for ctx1024 (common set, incl. closing PREs):
#   bs1 42425  bs2 42855  bs4 43830  bs8 45720  bs16 49420  bs32 56819
#   bs64 71739  bs128 101535
# all fit the current 131072 design; bs64_ctx1024 and bs128_ctx1024 exceed the
# old 65536 bitstream (drop those two if that's what is deployed).
#
# Two sets of each trace are available (see convert_llama8Bshort.py):
#   LLAMA_DIR     -- "common": both pseudo-channels interleaved, as recorded.
#   LLAMA_PC0_DIR -- "pc0-only": only the commands targeting PC0 (PC1's slots
#                    become idle NOPs); its heaviest bank is smaller, so it also
#                    fits 131072 for every ctx1024 trace.
# LLAMA_SETS selects which to replay (space-separated; add $LLAMA_DIR for common).
LLAMA_DIR=llama8Bshort_runner
LLAMA_PC0_DIR=llama8Bshort_runner_pc0
LLAMA_SETS="$LLAMA_PC0_DIR"
LLAMA_TRACES="bs1_ctx1024 bs2_ctx1024 bs4_ctx1024 bs8_ctx1024 bs16_ctx1024 bs32_ctx1024 bs64_ctx1024 bs128_ctx1024"

# Per-FPGA run duration (seconds). The app loops the trace forever
# (--iterations 0); `timeout -s INT` sends a graceful stop at the deadline, and
# -k 10 hard-kills it 10 s later if it hasn't exited.
DURATION=90

# Idle slots appended after the trace's closing PREs before the loop wraps to the
# next iteration's opening ACTs. At 600 M slots/s (~1.67 ns/slot) 64 slots is
# ~107 ns, comfortably above tRP (~14 ns), so each loop seam leaves every bank
# precharged long enough before it is re-activated. (64 is also the app default;
# set explicitly here to document the loop-seam timing guarantee.)
TAIL_GAP=64

# Data-init schemes to test. Both pre-write the trace's read/write footprint over
# the PCIe stream path before the measured loop; they differ only in the stored
# data, so the zeros-vs-random comparison is clean:
#   zeros  -> every footprint address holds 0
#   random -> every footprint address holds a different random 256-bit word
# The random pass is seeded (fixed below) so the data is identical across every
# FPGA and reproducible across re-runs; the seed is recorded in each CSV name.
INIT_SEED=1

# Replay one trace. $1 = CSV path, $2 = output CSV basename (written under results/).
# Reads the current data-init scheme from $INIT_ARGS (set per mode below).
run_trace() {
    echo "=== ${2} (${INIT_ARGS}) ==="
    timeout -k 10 -s INT "${DURATION}" ./HBMTraceRunnerBRAM \
        --csv "${1}" \
        --channels low \
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

    # Trace set 1: 1-rank A100 HBM2, one run per pseudo-channel.
    for t in $TRACES; do
        name=${t//\//_}                  # stack0/pc0 -> stack0_pc0
        run_trace "${TRACES_BASE}/${TRACE_DIR}/${t}.csv" "${TRACE_DIR}_${name}_data_${label}_bram.csv"
    done

    # Trace set 2: original Ramulator microbenchmarks -- DISABLED (uncomment to re-enable).
    # for t in $ORIG_TRACES; do
    #     run_trace "${TRACES_BASE}/${ORIG_DIR}/${t}.csv" "${ORIG_DIR}_${t}_data_${label}_bram.csv"
    # done

    # Trace set 3: llama8Bshort, for each selected set (common and/or pc0-only).
    for lset in $LLAMA_SETS; do
        for t in $LLAMA_TRACES; do
            run_trace "${TRACES_BASE}/${lset}/${t}.csv" "${lset}_${t}_data_${label}_bram.csv"
        done
    done
done

echo "=== run_bram_local.sh: sweep complete ==="
