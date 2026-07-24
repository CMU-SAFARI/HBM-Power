#!/bin/bash
# Kill the 'power_experiment' tmux session on every FPGA. Useful to stop runs
# early, or to clean up sessions left behind when a run chain was truncated
# before its trailing 'tmux kill-session' executed.

# Keep in sync with run_on_infra_bram.sh
FPGAS="7 10 42 56 57 58 59 65 66 67 68 69 70 71 72 73 74 75 76 77"

SESSION=power_experiment

for i in $FPGAS; do
  ssh sgalanopoulo@safari-fpga${i}.ethz.ch \
    "tmux kill-session -t ${SESSION} 2>/dev/null \
       && echo 'fpga${i}: killed' \
       || echo 'fpga${i}: no ${SESSION} session'" &
done
wait
