# HBM4 Power Prediction

The HBM4 case study predicts HBM4 power by carrying the HBM3E rails, which are
themselves extrapolated from HBM2 measurements, over to HBM4's organization,
timings and voltages.

## HBM4 facts used (JESD270-4)

- **Speed bins:** 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0 Gbps/pin. Four
  transfers per clock, so tCK = 4 / rate (0.833 ns to 0.500 ns).
- **Interface:** 32 channels per stack × 2 pseudo-channels, 32 DQ per
  pseudo-channel. That is a 2048-bit stack interface, twice HBM3's. BL8, so a
  256-bit access per burst. 1 KB page per pseudo-channel.
- **Voltages:** core VDDC 1.05 V (range 1.018 to 1.124 V). I/O VDDQ 0.9, 0.8,
  0.75 or 0.7 V.
- **IDD-loop timings:** tRC 48 ns, tRAS 33 ns, tRP 15 ns.

## Method

- **Device.** One canonical device of 8 stacks × 64 pseudo-channels = 512
  pseudo-channels (`case_studies/hbm4_case_study/configs/HBM4_organization.json`). For the comparison,
  HBM3E is simulated with the same pseudo-channel count, so the comparison
  isolates per-pseudo-channel differences rather than interface width.
- **Rails.** Start from the HBM3E rails at 6.4 Gbps
  (`case_studies/hbm4_case_study/configs/HBM3_6400_power_datapattern.json`):
  - `IDD2N`, `IDD4R` and `IDD4W` scale linearly with data rate relative to
    6.4 Gbps, as the HBM2 data-rate sweep showed for standby and I/O current.
  - `IDD3N1` and `IDD3N16` stay a fixed number of mA above `IDD2N`. The measured
    gap does not change with data rate.
  - The activate (`IDD0 − IDD2N`) and refresh (`IDD5B − IDD3N1`) increments
    follow their measured fixed-pacing trends (+0.009% and +0.007% per MT/s).
  - At 6.4 Gbps, the rails equal the HBM3E configuration exactly.
- **Timings.** HBM4 uses the JESD270-4 absolute tRC, tRAS and tRP, and
  `nCCD_S` / `nCCD_L` of 2 / 4 clock cycles. The other parameters are carried
  over from the HBM3E 6.4 Gbps bin at constant nanoseconds. HBM3E uses the Ramulator 2 `HBM3_6400` bin scaled to each rate
  at constant nanoseconds.
- **Traces.** `tracegen.py` rebuilds the random-read trace (4 reads per
  activate) and the NOP + refresh idle trace in each bin's clock cycles. This
  keeps tREFI = 3.9 µs and tRFC = 350 ns constant in time. The trace's 16 banks
  are mapped onto HBM4's 2 bank groups × 8 banks.
- **Data pattern.** Uniform-random data (all knobs 0.5), relative form, the
  same as for HBM3E.
- **Split rails.** VDDC drives all on-die energy. VDDQ drives only the DQ share
  of read energy (see [the data-pattern model](data-pattern-model.md#the-dq-share-and-split-rails)).
  Writes stay on VDDC.
- **Energy per bit.** Device power divided by the read bandwidth the trace
  achieves, with refresh stalls included (about 91% of peak).

`config/HBM4_8000MTs/` holds this extrapolation at 8.0 Gbps, with VDDC 1.05 V
and VDDQ 0.9 V, ready to use with `HBM3_runner`.

## Results

Canonical 512-pseudo-channel device, random reads, VDDC 1.05 V / VDDQ 0.9 V
(`hbm4_power_sweep.csv`):

| Rate (Gbps) | Peak BW (TB/s) | Power (W) | Energy (pJ/bit) |
|---|---|---|---|
| 4.8 | 9.83 | 448 | 6.24 |
| 5.6 | 11.47 | 533 | 6.38 |
| 6.4 | 13.11 | 622 | 6.48 |
| 7.2 | 14.75 | 713 | 6.62 |
| 8.0 | 16.38 | 804 | 6.73 |

- Energy per bit rises with data rate for both standards: HBM4 from 6.24 to
  6.73 pJ/bit over 4.8–8.0 Gbps, HBM3E at 1.1 V from 6.52 to 6.73 pJ/bit over
  its in-spec 4.8–6.4 Gbps. Standby current scales
  with data rate, so the background energy does not amortize over the higher
  bandwidth.
- At the same 6.4 Gbps, HBM4 reads use 3.7%, 5.3% and 9.5% less energy per bit
  than HBM3E for the three HBM4 voltage points: (VDDC, VDDQ) = (1.05, 0.9),
  (1.05, 0.7) and (1.00, 0.7) V.
- The gain comes from the lower supply voltages, not from the higher data rate.
