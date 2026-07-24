#!/bin/bash
# Fill the one missing measurement: the ws_bg0_bg2 Ramulator microbenchmark with
# per-word random data. In the full run (run_on_infra_bram.sh) this was the last
# command in the chain and got dropped because the single send-keys line exceeded
# the 4096-byte TTY canonical-mode limit, so its CSV is the only one missing the
# rand1 pass. This script replays just that one trace, mirroring the settings of
# run_on_infra_bram.sh exactly so the result is comparable.
#
# Output (per FPGA, remote then collected via copy_results.sh):
#   results/original_traces_ws_bg0_bg2_ramulator_data_rand1_bram.csv

# Keep in sync with run_on_infra_bram.sh
FPGAS="7 10 42 56 57 58 59 65 66 67 68 69 70 71 72 73 74 75 76 77"

ORIG_DIR=original_traces
TRACE=ws_bg0_bg2_ramulator

DURATION=90        # per-FPGA run duration (s); timeout -s INT stops gracefully
TAIL_GAP=64        # idle slots at the loop seam (matches run_on_infra_bram.sh)
INIT_SEED=1        # random-data seed; reproduces the rand1 label

# Single run: random data init, seeded, same flags as the full sweep.
RUN="timeout -k 10 -s INT ${DURATION} ./HBMTraceRunnerBRAM"
RUN+=" --csv ../../../../cmd_traces/${ORIG_DIR}/${TRACE}.csv"
RUN+=" --channels low"
RUN+=" --init random --init-seed ${INIT_SEED}"
RUN+=" --tail-gap ${TAIL_GAP}"
RUN+=" --output results/${ORIG_DIR}_${TRACE}_data_rand${INIT_SEED}_bram.csv"

REMOTE="cd private_bsc/DRAMBender/sources/apps/HBMTraceRunnerBRAM"
REMOTE+=" && git stash; git pull && make clean && make && ${RUN}"
REMOTE+=" ; tmux kill-session -t power_experiment"

# Well under 4096 bytes (single command), so send-keys is safe here.
for i in $FPGAS; do
  ssh sgalanopoulo@safari-fpga${i}.ethz.ch \
    "tmux new-session -d -s power_experiment && tmux send-keys -t power_experiment '${REMOTE}' Enter" &
done
wait
