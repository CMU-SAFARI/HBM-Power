# Reproducing the Paper

This tutorial regenerates every modeling result of the paper: Figs. 20 to 23
and the MAPE numbers quoted alongside them. You will run each case study, see
which numbers to look for in its output, and compare them with the paper.

It assumes you have completed [Getting started](getting-started.md).

## 1. Install the Python dependencies

The case-study scripts use NumPy, pandas, Matplotlib and seaborn:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run everything at once

```bash
./run_all.sh
```

The script builds both runners (if needed) and then runs the six case-study
scripts in order. The whole run takes a few minutes. Each script writes its
outputs next to itself in `case_studies/<study>/`. `run_all.sh` uses
`python3` by default; set `PYTHON=/path/to/python` to choose a different
interpreter.

The rest of this tutorial goes through the case studies one at a time, so you
know what each output shows and which numbers should match.

## 3. HBM2: measured power vs three models (Fig. 20)

```bash
python3 case_studies/hbm2_model_comparison/plot_measured_vs_models.py
python3 case_studies/hbm2_model_comparison/mape_summary.py
```

The first script runs `HBM2_runner` for every model, data pattern and workload,
prints a table of measured and predicted power, and writes
`model_comparison_absolute.{pdf,png}`. The second prints the per-workload MAPE
summary:

```text
DRAMSim3     all-0s  min=11.0 max=23.8 avg=21.6   random min=21.8 max=48.9 avg=41.6
FGDRAM HBM2  all-0s  min=10.9 max=26.7 avg=22.6   random min= 9.3 max=28.0 avg=20.1
Ayna         all-0s  min= 6.6 max=11.3 avg= 8.1   random min= 4.8 max=16.3 avg= 7.6
```

Ayna's average MAPE is 8.1% with all-0s data and 7.6% with random data, as
reported in the paper. Compare `model_comparison_absolute.pdf` with Fig. 20 of
the paper.

## 4. HBM2: data-pattern dependence (Fig. 21)

```bash
python3 case_studies/hbm2_data_pattern_dependence/plot_toggle_models.py
```

This study does not use the engine. It fits two data-pattern models to the
measured read current of 32 beat patterns and prints:

```text
MAPE of read data-movement energy per bit (pJ/bit axis): baseline 11.44%  improved 2.34%
```

The single-toggle baseline (FGDRAM) has an 11.4% MAPE. Ayna's model, which
treats toggling on the DQ, TSV and bank-group buses separately, has a 2.3% MAPE.
The case study's README describes both models in detail.

## 5. HBM3E: validation against NVIDIA H200 (Fig. 22)

```bash
python3 case_studies/hbm3e_validation/plot_model_vs_h200.py
```

```text
HBM3E model   1.1 V: total  242.5 W = idle 31.6 + active  210.9 W
HBM3E empirical (H200, n=3): active 209.8 [207.0-212.2] | idle 38.4 | total 248.2 [241.7-255.2]
```

At the nominal 1.1 V, Ayna predicts 242.5 W of HBM3E power for a random-read
microbenchmark (242.3 W in the paper). The three measured H200 GPUs report 241.7 W to 255.2 W. The
script writes the two subfigures `model_vs_empirical_hbm3e_read.pdf` and
`model_vs_empirical_hbm3e_idle_refresh.pdf`.

## 6. HBM4: read energy across speed bins (Fig. 23)

Run the sweep first. The plot script reuses its CSV.

```bash
python3 case_studies/hbm4_case_study/sweep_hbm4_power.py
python3 case_studies/hbm4_case_study/plot_hbm3e_vs_hbm4.py
```

The sweep writes `hbm4_power_sweep.csv`, and the plot script writes
`hbm3_vs_hbm4_power.csv` and `hbm3_vs_hbm4_pjbit.{pdf,png}` (Fig. 23). The plot
script prints the device power of every curve per speed bin and the measured
H200 reference, including:

```text
 6.4 GT/s | HBM4  622.6 W (1.05/0.9)  612.7 (1.05/0.7)  585.1 (1.0/0.7) | HBM3  646.6 (1.1V)
H200 measured HBM3E read energy: 6.864 pJ/bit (range 6.688-7.053); model HBM3E 1.10 V @ 6.4 Gbps: 6.730 pJ/bit
```

`hbm3_vs_hbm4_power.csv` holds the energy per bit behind Fig. 23: at 6.4 Gbps,
HBM4 reads use 3.7%, 5.3% and 9.5% less energy per bit than HBM3E for the
three HBM4 voltage points.

## What you reproduced

| Paper result | Script | Check |
|---|---|---|
| Fig. 20 and Ayna's 8.1% / 7.6% MAPE | `hbm2_model_comparison/` | MAPE printout, `model_comparison_absolute.pdf` |
| Fig. 21, 2.3% vs 11.4% MAPE | `hbm2_data_pattern_dependence/` | MAPE printout |
| Fig. 22, HBM3E vs H200 | `hbm3e_validation/` | model vs empirical printout |
| Fig. 23, HBM3E vs HBM4 energy per bit | `hbm4_case_study/` | Power printout, `hbm3_vs_hbm4_power.csv` |

[Case studies](../reference/case-studies.md) lists every input and output, and
[Where the IDD values come from](../explanation/idd-derivation.md) explains how
the configurations behind these results were derived.

## Disclaimer

The HBM3E and HBM4 results (Figs. 22 and 23) come out up to 0.1% above the
paper's numbers. In this updated version of the power model, the HBM3 engine
charges the active background per open bank, from `IDD3N1` (one bank) to
`IDD3N16` (all banks), as the HBM2 engine does. 