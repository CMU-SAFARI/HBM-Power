#!/bin/bash
# Pull the BRAM trace-runner power CSVs from every FPGA into results/<host>/.

# FPGAs to collect from (keep in sync with run_on_infra_bram.sh)
FPGAS="7 10 42 56 57 58 59 65 66 67 68 69 70 71 72 73 74 75 76 77"

mkdir -p results
for i in $FPGAS; do
  mkdir -p results/safari-fpga${i}
  rsync -az \
    --include="*" \
    sgalanopoulo@safari-fpga${i}.ethz.ch:/home/sgalanopoulo/private_bsc/DRAMBender/sources/apps/HBMTraceRunnerBRAM/results/ \
    results/safari-fpga${i}/ &
done
wait
