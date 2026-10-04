# HBM3E validation (Section 6.2, Fig. 22): Ayna vs. measured H200 random-read power

`model_vs_empirical_hbm3e_read.pdf` and `model_vs_empirical_hbm3e_idle_refresh.pdf` — the two
subfigures of Fig. 22, each a standalone bar plot with its own y-axis: device-level DRAM power
for the random-read trace (subfigure a, left in the paper) and for the idle + refresh trace
(NOP + REFA; subfigure b, right), for Ayna at VDD 1.067 / 1.10 / 1.177 V beside the H200
measurement (min/max whisker over 3 units). Bars are named on the x axis, rotated 90 degrees,
two lines each ("Ayna" over its VDD).

This is the figure that exercises the **data-pattern energy model integrated into DRAMPower**
(HBM3, relative form). It needs `HBM3_runner`.

## Reproduce

Build the engine once from the repository root (see
[Build the engine](../../docs/how-to/build-the-engine.md)), then run the script:

```bash
cmake --preset release && cmake --build --preset release -j

python3 case_studies/hbm3e_validation/plot_model_vs_h200.py        # -> model_vs_empirical_hbm3e_{read,idle_refresh}.{png,pdf}
```

Needs Python with `matplotlib` and `seaborn`.

## Model (how the bars are computed)

Per pseudo-channel, `HBM3_runner` is run twice with the data-pattern toggle knobs:

- **total** — random-read trace with refresh (`hbm3_random_read_4rpa_ref.csv`: `hbm3_random_read_4rpa.csv` plus a REFA every
  tREFI = 3.9 us and a tRFC = 350 ns stall after each; 4.50 TB/s device read BW vs 4.52 measured), `--dq-rate=0.5 --tsv-rate=0.5
  --bg-rate=0.5` (uniform-random data on all three buses). Plotted directly against the measured benchmark power.
- **idle** — NOP + REFA baseline trace (`hbm3_baseline_nop_ref.csv`), all knobs `0.0`; refresh energy = (IDD5B - IDD3N16) x tRFC per REFA (tRFC 350 ns, tREFI 3.9 us = Ramulator 2.1 HBM3_16Gb_8hi)

Each is scaled by `N_PC = 192` pseudo-channels (HBM3E on H200, 2-SID); VDD is
overridden per bar (1.067 / 1.10 / 1.177 V = JEDEC HBM3 min / nominal / max; H200 operating point unknown). The model's `IDD4R` is scaled by the data-pattern
activity (relative form, coupling terms off — see
[the data-pattern model](../../docs/explanation/data-pattern-model.md) and, for the
coefficients and the coupling terms, [the data-pattern case study](../hbm2_data_pattern_dependence/README.md#coupling-included-vs-excluded)).

## Inputs

```
configs/HBM3_16Gb_8hi_organization.json                    HBM3E device organization (16Gb, 8-hi)
configs/HBM3_6400Mbps_timing.json                          6400 Mbps timing
configs/HBM3_6400_power_datapattern.json                   IDD rails (derivation: docs/explanation/idd-derivation.md) + datapattern block + extrapolation block for the HBM4 case study
traces/hbm3_random_read_4rpa.csv                           random-read trace (4 reads / activate), no refresh (kept for reference)
traces/hbm3_random_read_4rpa_ref.csv                       the same trace with REFA every tREFI and a tRFC stall (used for the figure)
traces/hbm3_baseline_nop_ref.csv                           NOP idle-baseline trace
data/emp_runs_tuning_summary.csv                           H200 sample 1 (tuning sweep; read_rand row)
data/emp_quick_145512.csv                                  H200 sample 2 (quick measure)
data/emp_quick_145735.csv                                  H200 sample 3 (quick measure)
```

Empirical aggregation: sample 1 uses `mem_w_steady` (total) and `mem_idle`; samples 2–3 use
`random_read_total` and `idle_baseline` (`mem_power_W`). The bar shows the 3-unit mean active
(total − idle) and mean idle; the whisker is min/max of total.
