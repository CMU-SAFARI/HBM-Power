# HBM2 model comparison (Section 6, Fig. 20)

Measured HBM2 power across the tested chips vs the power predicted by DRAMSim3, FGDRAM HBM2 and
Ayna, for 14 workloads (6 Ramulator microbenchmarks + the 8-point Llama3.1-8B decode batch-size
sweep) and two data patterns (all-0s, random). Each panel ends with an `Average` group: per data
pattern, the distribution across chips of each chip's power averaged over that panel's workloads
(chips measured on every workload of the panel), with the model markers at the mean prediction
over those workloads.

```bash
python3 plot_measured_vs_models.py   # -> model_comparison_absolute.{pdf,png}   (Fig. 20)
python3 mape_summary.py              # -> table_mape_summary.tex + the MAPE numbers quoted in Section 6
python3 plot_error_boxplots.py       # -> model_comparison_error_models.{pdf,png} (per-workload MAPE boxes)
```

Build the engine first (see [Build the engine](../../docs/how-to/build-the-engine.md)).

## What the scripts do

For each (model, data pattern, workload) the engine (`build/bin/HBM2_runner ORG TIMING IDD
TRACE`) is run once on the workload's command trace with the model's IDD config; the reported
average power is one prediction. It is compared with the per-chip measured power of that
workload (the ground truth). MAPE of a workload = mean over chips of |predicted - measured| /
measured.

| model | all-0s config | random config |
|---|---|---|
| DRAMSim3 | `IDD_dramsim3` | `IDD_dramsim3` (no data-pattern term) |
| FGDRAM HBM2 | `IDD_oconnor_notoggle` | `IDD_oconnor_50toggle` |
| Ayna | `IDD_ours_allzeros` | `IDD_ours_random` |

The IDD configs are in `configs/`;
[Where the IDD values come from](../../docs/explanation/idd-derivation.md) explains their
derivation. Every IDD config
turns the data-pattern model off (`"datapattern": {"enabled": false}`), so IDD4R/IDD4W stay fixed
per config: the random configs already hold the measured random-data read current.

## Inputs

```
configs/HBM2_organization.json        one HBM2 stack: 8 channels x 2 pseudo-channels, 4 BG x 4 banks
configs/HBM2_1.2Gbps_timing_BL4.json  1200 MT/s timing with nBL = 4 
configs/IDD_*.json                    IDD sets of the three models
traces/<workload>.csv                 14 command traces (6 microbenchmarks, bs{1..128}_ctx1024)
data/ground_truth_allzeros.csv        per-chip measured VDD power, all-0s (test_name, power_vdd_avg)
data/ground_truth_random.csv          per-chip measured VDD power, random
data/llama_single_chip/               re-measurement of the Llama traces on one chip at a time
                                      (17 chips, both data patterns); the chip-0 rows replace the ground-truth Llama
                                      rows, which were taken with both stacks of a board powered
```

The scripts load `data/ground_truth_*.csv` and swap in the Llama rows from
`data/llama_single_chip/` (`load_gt`). Ground truth per workload: 35 chips (all-0s
microbenchmarks), 18 chips (random microbenchmarks), 17 chips (Llama, both patterns).

