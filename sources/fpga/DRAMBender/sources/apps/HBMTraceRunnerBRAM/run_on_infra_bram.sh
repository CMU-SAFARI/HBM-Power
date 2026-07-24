#!/bin/bash
# Launch the BRAM trace runner on every FPGA host. For each FPGA: stash local
# changes, switch to the bram-replay branch, pull, rebuild, then run the committed
# run_bram_local.sh -- which holds the actual per-trace run list and writes each
# power CSV to results/<set>_<trace>_data_{zeros,rand<seed>}_bram.csv.
#
# Why the run list lives in run_bram_local.sh (committed + pulled onto each FPGA)
# rather than inline here: this orchestrator delivers its command to the remote
# tmux session via a single `send-keys`, and a terminal in canonical mode caps an
# input line at ~4096 bytes (the kernel TTY N_TTY/MAX_CANON limit). The old inline
# sequence was ~5.3 KB, so it was silently truncated mid-command (you'd see
# "option '--csv' requires an argument" / a half-written path). Keeping the
# payload short and fixed-size avoids that no matter how many traces are added.

# FPGAs to use
FPGAS="7 10 42 56 57 58 59 65 66 67 68 69 70 71 72 73 74 75 76 77"
# FPGAS="7"

# App directory on each FPGA (relative to the ssh login home).
APP_DIR="private_bsc/DRAMBender/sources/apps/HBMTraceRunnerBRAM"

# Short, fixed-size remote command: prep the checkout, build, run the committed
# sweep script, then tear down the tmux session. `make ... && bash ...` so a
# failed build skips the runs; the trailing `;` always kills the session.
REMOTE="cd ${APP_DIR}"
REMOTE+=" && git stash; git checkout bram-replay; git pull && make clean && make && bash run_bram_local.sh"
REMOTE+="; tmux kill-session -t power_experiment"

for i in $FPGAS; do
  ssh sgalanopoulo@safari-fpga${i}.ethz.ch \
    "tmux new-session -d -s power_experiment && tmux send-keys -t power_experiment '${REMOTE}' Enter" &
done
wait
