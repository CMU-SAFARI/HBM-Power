# Case Studies

Each directory under `case_studies/` reproduces one result of the paper and is
self-contained: its configurations, traces and data live inside it, duplicated
where two case studies share a file. Every script can be run from any working
directory, and it writes its outputs next to itself. All scripts use the
runners in `build/bin/`. `run_all.sh` runs them all in order.

| Directory | Paper | Engine | Main output |
|---|---|---|---|
| [`hbm2_model_comparison/`](#hbm2_model_comparison) | Section 6, Fig. 20 | `HBM2_runner` | `model_comparison_absolute.pdf` |
| [`hbm2_data_pattern_dependence/`](#hbm2_data_pattern_dependence) | Section 6.1, Fig. 21 | none | `model_vs_fgdram_single_toggle.pdf` |
| [`hbm3e_validation/`](#hbm3e_validation) | Section 6.2, Fig. 22 | `HBM3_runner` | `model_vs_empirical_hbm3e_{read,idle_refresh}.pdf` |
| [`hbm4_case_study/`](#hbm4_case_study) | Section 6.3, Fig. 23 | `HBM3_runner` | `hbm3_vs_hbm4_pjbit.pdf` |

## hbm2_model_comparison

Measured HBM2 power across tested chips vs. the power predicted by DRAMSim3,
FGDRAM HBM2 and Ayna, for 14 workloads (six microbenchmarks and an 8-point
Llama3.1-8B decode batch-size sweep) and two data patterns.

| Script | Output |
|---|---|
| `plot_measured_vs_models.py` | `model_comparison_absolute.{pdf,png}` (Fig. 20) and a table of measured vs predicted power |
| `mape_summary.py` | `table_mape_summary.tex` and the MAPE numbers quoted in the paper |
| `plot_error_boxplots.py` | `model_comparison_error_models.{pdf,png}`, per-workload error boxes (not in the paper) |

| Input | Contents |
|---|---|
| `configs/` | Organization, timing, and the IDD set of each model |
| `traces/` | 14 command traces |
| `data/ground_truth_{allzeros,random}.csv` | Per-chip measured VDD power (`chip_id,test_name,power_vdd_avg`) |
| `data/llama_single_chip/` | Re-measurement of the Llama traces with one chip powered at a time. These rows replace the Llama rows of the ground truth. |

Model to configuration mapping:

| Model | all-0s | random |
|---|---|---|
| DRAMSim3 | `IDD_dramsim3` | `IDD_dramsim3` |
| FGDRAM HBM2 | `IDD_oconnor_notoggle` | `IDD_oconnor_50toggle` |
| Ayna | `IDD_ours_allzeros` | `IDD_ours_random` |

MAPE of a workload is the mean over chips of |predicted − measured| / measured.
The ground truth covers 35 chips for all-0s microbenchmarks, 18 for random
microbenchmarks, and 17 for the Llama workloads.

## hbm2_data_pattern_dependence

Fits two data-pattern models to the measured read current of 32 beat patterns
and compares them:

- **Baseline (FGDRAM single toggle):** floor + I/O at 1.60 pJ/bit × DQ toggle
  + one on-die toggle term. R² ≈ 0.41.
- **Ayna:** floor + DQ toggle + TSV toggle + bank-group burst-to-burst toggle,
  plus the additional terms described in the case study's README. R² ≈ 0.97.

The script also prints the coefficients of the model used by the device
configurations in `config/`.

| Script | Output |
|---|---|
| `plot_toggle_models.py` | `model_vs_fgdram_single_toggle.{pdf,png}` (Fig. 21), R² and MAPE of both models |

| Input | Contents |
|---|---|
| `data/beat_pattern_perpattern.csv` | `beat_pattern, col1_inverted, mean_idd_mA, n_chips` for 32 patterns |
| `configs/IDD_ours_allzeros.json` | Read background `IDD3N1` (a copy of the `hbm2_model_comparison` config) |

The bank-group bus wire order is fitted (best of 24 permutations, giving
(1, 3, 0, 2)). Set `WIRE_ORDER = "linear"` in the script to use the fixed order
(1, 0, 3, 2) instead.

## hbm3e_validation

Device-level HBM3E power predicted by Ayna at three VDD values (1.067, 1.10 and
1.177 V, the JEDEC minimum, nominal and maximum) vs. the memory power measured
on three NVIDIA H200 GPUs, for a random-read microbenchmark and for idle with
refresh.

| Script | Output |
|---|---|
| `plot_model_vs_h200.py` | `model_vs_empirical_hbm3e_read.{pdf,png}` and `model_vs_empirical_hbm3e_idle_refresh.{pdf,png}` (Fig. 22) |

| Input | Contents |
|---|---|
| `configs/` | Organization, 6.4 Gbps timing, power with `datapattern` block |
| `traces/hbm3_random_read_4rpa_ref.csv` | Random reads, 4 per activate, `REFA` every tREFI = 3.9 µs, tRFC = 350 ns |
| `traces/hbm3_baseline_nop_ref.csv` | NOP + `REFA` idle trace |
| `data/emp_runs_tuning_summary.csv`, `data/emp_quick_*.csv` | `nvidia-smi` memory power of the three H200 systems |

Method: per pseudo-channel, the read trace runs with all knobs at 0.5 and the
idle trace with all knobs at 0. Both results are multiplied by 192
pseudo-channels (H200).

## hbm4_case_study

HBM4 read energy per bit across the JEDEC HBM4 speed bins (4.8 to 8.0 Gbps) at
three voltage points, compared with HBM3E (4.8 to 6.4 Gbps).

| Script | Output |
|---|---|
| `sweep_hbm4_power.py` | `hbm4_power_sweep.csv`, `hbm4_power_vs_datarate.{pdf,png}` |
| `plot_hbm3e_vs_hbm4.py` | `hbm3_vs_hbm4_pjbit.{pdf,png}` (Fig. 23), `hbm3_vs_hbm4_power.{csv,pdf,png}` |
| `tracegen.py` | Library: per-speed-bin traces into `traces/generated/` |

Run `sweep_hbm4_power.py` first. The plot script reads its CSV and checks that
its own HBM4 builder reproduces it.

| Input | Contents |
|---|---|
| `configs/HBM4_organization.json` | 512-pseudo-channel HBM4 device (8 stacks × 64 PCs), 2 bank groups × 8 banks |
| `configs/HBM3_16Gb_8hi_organization.json` | HBM3E bank organization (4 bank groups × 4 banks) for the HBM3E runs (a copy of the HBM3E validation config) |
| `configs/HBM3_6400_power_datapattern.json` | Rails and `extrapolation` block (a copy of the HBM3E validation config) |
| `traces/` | Source traces for `tracegen.py` |
| `data/` | H200 measurements (a copy of the HBM3E validation data), used for the measured HBM3E marker |

The assumptions are explained in [HBM4 power prediction](../explanation/hbm4-power-prediction.md).
