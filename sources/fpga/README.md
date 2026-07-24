# FPGA measurement scripts (single-chip HBM2-power AE)

Self-contained recreations of the in-scope `run_*.sh` measurement scripts for
the `Power_structural_variation` DRAM Bender app (provided in
[`DRAMBender/`](DRAMBender/) together with the API it builds against and the
prebuilt bitstreams), plus the single-chip standardizer. These are the scripts
that actually drive `SoftMC_rdwr` on the FPGA and turn the raw CSVs into the
figure inputs.

You normally do **not** run these by hand — the one-command driver
[`../../scripts/fpga/run_fpga_ae.sh`](../../scripts/fpga/run_fpga_ae.sh) reprograms the board, stages these
scripts onto the FPGA, runs them, copies the results back, and standardizes them
into the sanitized `data/new/fpga/*.csv`. See its `--help`. This directory documents the
pieces it orchestrates.

## Measurement scripts

Each script runs **on the FPGA host** from the `Power_structural_variation`
directory (where `./SoftMC_rdwr` and the HBM live). It takes optional flags then
an optional binary path: `run_xxx.sh [--flags] [path_to_binary]`.

| Script | test-select → raw subdir | Figure(s) | Notes |
|---|---|---|---|
| `run_no_hbm.sh` | 7 → `temperature_dependence` | baseline (all) | channel-0 IDD2, no warm-up; run on `XCU55_no_hbm` |
| `run_idd4r_8ch_test.sh` | 7 → `temperature_dependence` | 7 | IDD4R-full; run **first** to heat the die (no warm-up phase) |
| `run_idd2_test.sh` | 7 → `temperature_dependence` | 2, 3, 4 | warm-up + IDD2 |
| `run_idd0_test.sh` | 7 → `temperature_dependence` | 6 | warm-up + IDD0 |
| `run_idd3n1_test.sh` | 7 → `temperature_dependence` | 5 | warm-up + IDD3N1 |
| `run_idd3n16_test.sh` | 7 → `temperature_dependence` | 5 | warm-up + IDD3N16 |
| `run_idd4w_test.sh` | 7 → `temperature_dependence` | 7 | warm-up + IDD4W |
| `run_idd5_test.sh` | 7 → `temperature_dependence` | 8 | warm-up + IDD5B |
| `run_max_power_bg_invert.sh` | 15 → `max_power_bg_invert` | 10 | 6 BG pairs × {inv0, inv1} |
| `run_max_power_bank_offset.sh` | 14 → `max_power_bank_offset` | 11 | offsets 0–3, inv1 |
| `run_dq_bitflip_pattern_sweep.sh` | 10 → `bitflip_variation` | 12 | 32 DQ patterns, IDD4R-full |
| `run_beat_pattern_sweep_max_power.sh` | 24 → `beat_pattern_variation` | 13, 15 | 16 beat patterns × {inv0, inv1} |

Flags: `--no-warmup` (IDD loops), `--warmup` (max-power loops, off by default),
`--high` (chip 1 / channels 8–15). Fig 9 (current-vs-temperature slope) reuses
all seven test-7 loops.

### Duration knobs (env)

Defaults reproduce the paper configuration; override for faster shakeout runs:

| Env | Applies to | Default |
|---|---|---|
| `WARMUP_S`  | every warm-up phase | 1800 (900 for the DQ/beat/offset sweeps) |
| `MEASURE_S` | the long IDD loops (Figs 2–9) | 1800 |
| `SWEEP_S`   | per-point sweep duration (Figs 10–13/15) | 90 |
| `NOHBM_S`   | the no-HBM baseline | 90 |

> Cutting `MEASURE_S` too low starves Fig 9's temperature fit (it needs ≥10
> samples in >1 temperature bin per loop). Short runs are fine for plumbing.

## Trace scripts (Fig 14 + Table 3)

The two *trace-based* artifacts need real HBM2 power while **replaying DRAM
command traces** — a different app (`HBMTraceRunnerBRAM`) and a different
bitstream (`bram_tracer_chipN`) than the IDD/structural `SoftMC_rdwr` above. The
driver runs this as **Phase 2** (last, since it programs a different bitstream).

| Script | Runs where | Role |
|---|---|---|
| `run_trace_sweep.sh` | **on the FPGA**, in the `HBMTraceRunnerBRAM` app dir | replays the 14 AE workloads (6 microbenchmarks + the 8-point LLaMa3.1-8B decode sweep) for `TRACE_S` s each, once zeros-init and once random-init; writes `results/<set>_<trace>_data_{zeros,rand1}_bram.csv` |
| `aggregate_trace_ground_truth.py` | **locally** | maps each `*_bram.csv` (last-`N`=10 steady-state VDD) to the ground-truth `test_name` and writes the sanitized `ground_truth_{allzeros,random}.csv` (Fig 14 / Table 3 ground truth) |

`run_trace_sweep.sh` is the AE-subset restriction of the upstream
`run_bram_local.sh` (provided beside the app in
`DRAMBender/sources/apps/HBMTraceRunnerBRAM/`): it **enables** the 6-microbenchmark
set (commented out upstream) and the 8-point LLM `pc0` sweep, and **drops** the
1-rank A100 trace (not one of the 14 plotted workloads). Flags:
`--channels low|high` (chip0/chip1), `--llm-only`, `--micro-only`; env `TRACE_S`
(per-trace duration, default 90) and `TRACES_BASE`.

`aggregate_trace_ground_truth.py` applies the trace ↔ `test_name` map. Modes:

- **replace** (default) — write only the AE chip's 14+14 rows, so Fig 14 / Table 3
  collapse to single points, exactly like the IDD figures on one chip. The
  released full-fleet files are copied to `data/released_backup/` first.
- **`--merge`** — upsert the AE chip's rows into the fleet file (drop this
  chip's old rows, keep every other chip), to overlay the chip within the
  paper's cross-stack boxes.

Config env: `RESULTS_DIR`, `DATA_DIR`, `CHIP_ID`, `LAST_N`; `-n/--dry-run`.

`scripts/fpga/copy_results.sh --trace` pulls the flat `*_bram.csv` off the FPGA into
`results/<fpga-label>/trace_runner/` (where the aggregator finds them).

## `standardize_single_chip.py`

Turns the copied raw CSVs into the sanitized `data/new/fpga/*.csv`. It **reuses** the
fleet extractors in the provided
`DRAMBender/sources/apps/Power_structural_variation/standardize/`
unchanged, overriding their config to one FPGA / one chip and repointing I/O at
this artifact tree:

```
<artifact>/results/safari-fpga<N>/<subdir>_fixed_reset_full_ipp/   (input, from copy_results.sh)
        → <artifact>/data/new/fpga/*.csv                           (output, the sanitized CSVs; default)
```

Produces: `all_idd_measurements.csv`, `no_hbm_idd2_measurements.csv`,
`bank_group_measurements.csv`, `bank_offset_measurements.csv`,
`bitflip_measurements.csv`, `beat_pattern_combined_measurements.csv`, and the
derived `beat_pattern_perpattern.csv` (Fig 15). Config env: `PRIV_STD_DIR`,
`RESULTS_DIR`, `OUTPUT_DIR`, `FPGA_NUM`, `CHIP`.
