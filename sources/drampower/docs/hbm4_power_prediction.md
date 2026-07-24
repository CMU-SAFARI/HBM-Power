# HBM4 random-read power prediction

Predicts HBM4 device power across the JEDEC speed bins using our data-pattern read/write
energy model (random data = toggle 0.5 on all three buses). Reuses the validated HBM3E
pipeline (reproduced in `figureC/`); see `docs/datapattern_model.md` for the model itself.

Run: `python3 figureD/make_figureD.py` → `figureD/hbm4_power_sweep.csv` +
`figureD/hbm4_power_vs_datarate.{png,pdf}`.

## HBM4 facts (JEDEC JESD270-4)
- **Speed bins:** 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0 Gbps/pin. fCK = rate/4
  (dataRate = 4 transfers/CK, DDR on WDQS@2×CK); tCK = 0.833 → 0.500 ns.
- **Interface:** 64 DQ/channel, 32 DQ/PC, 2 PC/channel, up to **32 channels/stack → 64 PC/stack
  = 2048-bit interface** (2× HBM3's 1024). BL8 (256-bit prefetch / 32 DQ). 1 KB page/PC.
- **Stacks:** 4/8/12/16-Hi (≥4 dies → 32 channels; extra dies add SIDs/banks/capacity).
  Density 3–16 Gb/channel, 16/32/48/64 banks/channel.
- **Voltages:** VDDC 1.05 V, VDDQ 0.9/0.8/0.75/0.7 V (split rails), VPP 1.8 V.
- **Timings (IDD loop, Table 101):** tRC 48, tRAS 33, tRP 15 ns.
- **No IDD currents:** Table 107 is an empty template — JEDEC defers actual currents to vendor
  datasheets. So power must be predicted from carried-over, calibrated rails (below).

## Method & assumptions
- **Canonical org:** 8 stacks × 64 PC/stack = **512 PC**; 32 DQ/PC. (Single org; data rate swept.)
- **Rails:** HBM3E-calibrated all-0s rails. **Dynamic** read/write current (above IDD3N1) scales
  **linearly with data rate**: `IDD4R(r)=34.6+(630.4−34.6)·r/6.4`, same for IDD4W from 505.1.
  Array rails (IDD0/2N/3N) are rate-independent. *Validation:* this reproduces the existing
  measured-ish H100 5.2-GT/s config (512.8 mA) to ~1%.
- **Data-pattern model:** relative/K-free, coupling OFF (HBM3 setting), all three knobs = 0.5
  (uniform-random data, Monte-Carlo-confirmed worst case).
- **Voltage:** JESD270-4 split rails — core **VDDC = 1.05 V** (typ; range 1.018–1.124) drives all
  core/array energy; I/O **VDDQ = 0.9 V** (typ; range 0.873–0.963) drives only the external DQ pins.
  The engine applies VDDQ to the DQ component of the read I/O current (IDD4R) alone — the data-pattern
  model decomposes the dynamic read current and the on-die floor/TSV/BG terms stay on VDDC; writes
  stay on VDDC for now. (The DQ pins are only ~14 % of the dynamic read energy at 0.5 toggling, so the
  split shaves ~1–2 % off device power vs a single 1.05 V rail.)
- **Per-PC → device:** `device = 512 × (active_PC + idle_PC)`. Active = engine on the
  random-read trace (4 reads/activate, near-peak); idle = NOP+REF standby floor.

## Results (canonical 8-stack device, random reads; VDDC = 1.05 V / VDDQ = 0.9 V split rails)
| Rate (Gbps) | tCK (ps) | Peak BW (TB/s) | Power (W) | pJ/bit |
|---|---|---|---|---|
| 4.8 | 833 | 9.8  | 448 | 5.69 |
| 5.2 | 769 | 10.7 | 482 | 5.65 |
| 5.6 | 714 | 11.5 | 516 | 5.62 |
| 6.0 | 667 | 12.3 | 550 | 5.59 |
| 6.4 | 625 | 13.1 | 583 | 5.56 |
| 6.8 | 588 | 13.9 | 618 | 5.55 |
| 7.2 | 556 | 14.8 | 652 | 5.53 |
| 7.6 | 526 | 15.6 | 686 | 5.51 |
| 8.0 | 500 | 16.4 | 719 | 5.49 |

- Power rises ~linearly with data rate (≈448→719 W over 4.8→8.0 Gbps): the dynamic I/O current
  scales with rate and the burst completes in fewer ns, so both factors push power up.
- Energy efficiency is ~flat (~5.5–5.7 pJ/bit), edging *down* with rate — peak BW grows slightly
  faster than power because the rate-independent array/idle share gets amortized over more bits.

## Caveats
- DRAM-die only (no PHY/controller); empirical GPU figures include the whole subsystem.
- HBM2-derived data-pattern coefficients over-predict HBM3E toggle sensitivity ~3× (DBI/scrambling
  → effective toggle ~0.15–0.21, not 0.5). The 0.5 numbers are an **upper bound** on data-pattern
  energy; a realistic-DBI run would use lower knobs.
- Linear data-rate scaling of the dynamic current is an extrapolation anchored on two points
  (HBM3E 6.4, H100 5.2 GT/s); vendor HBM4 datasheets, when available, should replace it.
- Refresh (IDD5B) energy is not modeled — idle = standby floor only.
