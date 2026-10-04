#!/bin/bash
# Build the Ayna engine and regenerate the outputs of every case study under case_studies/.
#   ./run_all.sh
# Set PYTHON to choose the interpreter (default: python3).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDRAMPOWER_BUILD_CLI=ON
cmake --build build --target HBM2_runner HBM3_runner -j

# 1. HBM2: measured power vs DRAMSim3, FGDRAM HBM2 and Ayna (+ per-workload MAPE table)
$PY case_studies/hbm2_model_comparison/plot_measured_vs_models.py
$PY case_studies/hbm2_model_comparison/mape_summary.py
# 2. HBM2: single-toggle vs Ayna's data-pattern model on the 32 beat patterns
$PY case_studies/hbm2_data_pattern_dependence/plot_toggle_models.py
# 3. HBM3E: Ayna vs measured NVIDIA H200 memory power
$PY case_studies/hbm3e_validation/plot_model_vs_h200.py
# 4. HBM4: power and energy per bit across the JEDEC speed bins
$PY case_studies/hbm4_case_study/sweep_hbm4_power.py
$PY case_studies/hbm4_case_study/plot_hbm3e_vs_hbm4.py
