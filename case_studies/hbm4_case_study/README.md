# HBM4 read-energy case study (Section 6.3, Fig. 23)

Forward prediction with Ayna: HBM4 read energy per bit across the JEDEC HBM4 speed bins
(4.8-8.0 Gbps/pin) for three core/I-O voltage points, against HBM3E (4.8-6.4 Gbps/pin). There is
no HBM4 silicon to measure; the rails are the HBM2-derived HBM3E rails of `../hbm3e_validation/`.

```bash
python3 sweep_hbm4_power.py       # -> hbm4_power_sweep.csv + hbm4_power_vs_datarate.{pdf,png}
python3 plot_hbm3e_vs_hbm4.py     # -> hbm3_vs_hbm4_pjbit.{pdf,png} (Fig. 23), hbm3_vs_hbm4_power.*
```

Run the sweep first: the plot script takes the canonical HBM4 curve (VDDC 1.05 V / VDDQ 0.9 V)
from `hbm4_power_sweep.csv` as-is and asserts that its own HBM4 builder reproduces it before
computing the two lower-voltage variants. Build the engine first
([Build the engine](../../docs/how-to/build-the-engine.md)).

## Method

Both standards run on one canonical **512-pseudo-channel device** (8 stacks x 64 PC; HBM3E is
forced to the same PC count so that the comparison isolates the per-PC standard and voltage
differences, not the interface width). Per speed bin:

* **Traces.** `tracegen.py` rebuilds the two traces in that bin's clock cycles: the random-read
  trace (4 reads per activate, uniform random data on all buses: `--dq-rate 0.5 --tsv-rate 0.5
  --bg-rate 0.5`) with an all-bank refresh every tREFI = 3.9 us and a tRFC = 350 ns stall after
  each, and a NOP + refresh idle trace. At 6.4 Gbps this reproduces
  `../hbm3e_validation/traces/hbm3_random_read_4rpa_ref.csv` exactly. The source trace addresses
  4 bank groups x 4 banks (the HBM3E organization); for the HBM4 runs, `tracegen.py` maps the same
  16 banks onto HBM4's 2 bank groups x 8 banks.
* **Power.** Device power = 512 x the engine's average power on the refresh-bearing read trace
  (standby, refresh and reads in one run, nothing added or double counted). Idle power (the
  NOP + refresh trace) is reported separately in the sweep CSV.
* **Energy per bit** = device power / the read bandwidth the trace achieves (reads x 256 bit over
  the trace duration, refresh stalls included; about 91 % of the peak bandwidth).
* **Rails.** Read from `configs/HBM3_6400_power_datapattern.json`, a copy of the HBM3E validation
  config. Its rails are derived from the HBM2 measurements, and its `extrapolation` block holds the
  quantities used here. IDD2N and the
  read/write currents (IDD4R, IDD4W) scale linearly with data rate relative to 6.4 Gbps, as measured
  on HBM2 (Section 5). IDD3N1 sits a fixed delta above IDD2N and IDD3N16 a fixed delta above IDD3N1
  (the measured IDD3N1 - IDD2N gap does not change with data rate, so the deltas are carried as
  absolute mA rather than re-scaled). The activate increment (IDD0 - IDD2N) and the refresh increment
  (IDD5B - IDD3N1) are fixed at 6.4 Gbps and follow their fixed-pacing sweeps' weak trends elsewhere
  (+0.009 % and +0.007 % per MT/s). At 6.4 Gbps the rails equal the HBM3E config exactly.
* **Voltages.** HBM4 (JESD270-4) has split rails: core VDDC powers everything on-die, the I/O rail
  VDDQ powers only the external DQ pins, so the engine applies VDDQ to the DQ component of the
  read current only (floor / TSV / bank-group terms of the data-pattern model stay on VDDC).
  Points: (VDDC 1.05 V, VDDQ 0.9 V) typical, (1.05, 0.7) minimum I/O rail, (1.00, 0.7) also a
  lower core rail. HBM3E uses its single 1.10 V rail (nominal VDD).
* **Timings.** HBM4: JESD270-4 absolute tRC/tRAS/tRP (48/33/15 ns) and nCCD_S/nCCD_L of 2/4 clock
  cycles; the remaining cycle counts are carried from the HBM3E 6400 bin at constant ns. HBM3E: the Ramulator 2.1 HBM3_6400 bin,
  scaled to each rate at constant ns. tCK = 4 / rate (four transfers per clock).
* **Data-pattern model.** Same relative-form coefficients for both standards, coupling terms
  off (see [the data-pattern model](../../docs/explanation/data-pattern-model.md) and
  [coupling included vs excluded](../hbm2_data_pattern_dependence/README.md#coupling-included-vs-excluded)).

## Result (Fig. 23)

Energy per bit rises with data rate for both standards (HBM4 at 1.05/0.9 V: 6.24 -> 6.73 pJ/bit
from 4.8 to 8.0 Gbps; HBM3E at 1.1 V: 6.52 -> 6.73 from 4.8 to 6.4 Gbps): standby current scales with data rate, so the
background energy does not amortize over the higher bandwidth. At the same 6.4 Gbps, HBM4 reads
consume 3.7 %, 5.3 % and 9.5 % less energy per bit than HBM3E for the three voltage points. HBM4 at
8.0 Gbps vs HBM3E at 6.4 Gbps: 0.0 %, -1.5 %, -6.0 %. Lower supply voltages, not the higher data
rate, are what make HBM4 reads more energy-efficient under these rails.

## Files

```
configs/HBM4_organization.json            canonical HBM4 device (8 stacks x 32 channels x 2 PC, 2 BG x 8 banks)
configs/HBM3_16Gb_8hi_organization.json   copy of ../hbm3e_validation/configs/ (4 BG x 4 banks), for the HBM3E runs
configs/HBM3_6400_power_datapattern.json  copy of ../hbm3e_validation/configs/ (rails + extrapolation block)
data/emp_*.csv                            copy of the H200 measurements in ../hbm3e_validation/data/
traces/hbm3_random_read_4rpa.csv          random-read trace without refresh (source for tracegen.py)
traces/hbm3_baseline_nop_ref.csv          NOP + refresh idle trace at 6.4 Gbps (reference)
traces/generated/                         per-speed-bin traces written by tracegen.py (ignored by git)
```

Assumptions and JEDEC facts: [HBM4 power prediction](../../docs/explanation/hbm4-power-prediction.md).
