# Where the IDD Values Come From

Every IDD value provided with Ayna, in `config/` and in the case studies'
`configs/` directories, was derived from our HBM2 characterization data:
the standard IDD loops, the data-pattern loops, and the data-rate sweep
measured on real HBM2 chips. The measurements and the analysis scripts are
released with the paper's artifact at
<https://github.com/CMU-SAFARI/HBM-Power/tree/artifact>. This page explains the derivation
so that you can interpret the provided values and build comparable
configurations.

## Shared conventions

- **Chips.** The released data covers 36 chips (18 boards with two HBM2 stacks
  each). 
- **Last-minute mean.** A measurement point is the mean of the last 60
  one-second samples of a run. By then the chip is thermally settled, and the
  warm-up sample is excluded.
- **Board power-off current.** Each FPGA board draws current even with the HBM2
  stack powered off. The ground truth in the HBM2 comparison is raw board
  current, so every HBM2 case-study configuration includes this offset:
  821.3 mA, the mean over the 18 valid boards. The device configuration
  `HBM2_1200MTs` subtracts it.
- **Scope.** The HBM2 case-study values are per-stack rails at 1200 MT/s and VDD = 1.2 V,
  matching `HBM2_organization.json` (one 8-channel stack).
- **Data-rate trends.** Where values are moved to another data rate, they
  follow the lines fitted to the data-rate sweep (Fig. 13 of the paper).
  The standby, activate and refresh anchors at 1200 MT/s are taken from those
  lines.

## HBM2 case-study configurations

| Configuration | Standby and activate (`IDD0`, `IDD2N`, `IDD3N1`, `IDD3N16`) | Read and write (`IDD4R`, `IDD4W`) |
|---|---|---|
| Ayna, all-0s (`IDD_ours_allzeros`) | Data-rate sweep lines at 1200 MT/s plus the off-current. `IDD3N16` = `IDD3N1` + the measured `IDD3N16 − IDD3N1` gap. | `IDD4R`: all-0s read pattern (`0000`, no bus toggling). `IDD4W`: standard `IDD4W` × 0.602, the zero-toggle intercept of the IDD4W DQ bit-flip sweep. |
| Ayna, random (`IDD_ours_random`) | Same as all-0s | `IDD4R`: measured read current of the maximum-power (random) data-pattern loop, 3515.0 mA. `IDD4W`: an all-0s write current of 1913.9 mA × the random/all-0s read ratio 3515.0 / 2496.0. |
| DRAMSim3 (`IDD_dramsim3`) | DRAMSim3's published per-128-bit-channel currents at 2 Gbps, each scaled to 1200 MT/s by the ratio IDD(1200)/IDD(2000) of the same loop's measured data-rate line. Stack rail = background × 16 / 2 + (IDDx − background) × 8 / 2 + off-current. | Same recipe |
| FGDRAM HBM2 (`IDD_oconnor_*`) | Ayna's measured background. Activate from FGDRAM's 909 pJ per activation per channel (× 8 channels), solved through the engine's activation equation (+128.9 mA over `IDD2N`). | `IDD3N1 + K × pJ/bit` with FGDRAM's Table 3 energies: 1.51 pJ/bit with no toggling, 2.68 + 0.80 pJ/bit at 50% toggling, `K` = 512 mA per pJ/bit |

FGDRAM and DRAMSim3 thus differ from Ayna only in the quantities that their
published models specify. The shared background isolates the effect of each
model's read, write and activate energy.

## HBM3E extrapolation

`case_studies/hbm3e_validation/configs/HBM3_6400_power_datapattern.json` holds per-pseudo-channel rails
for HBM3E at 6400 MT/s, with static data and without the board offset. They
are extrapolated from the HBM2 measurements:

| Rail | Derivation |
|---|---|
| `IDD2N` | The IDD2N data-rate line evaluated at 6400 MT/s and re-levelled to the 1200 MT/s anchor; ÷ 8 pseudo-channels; × 36/32 for ECC |
| `IDD3N1`, `IDD3N16` | Fixed offsets above `IDD2N` (+0.72 mA and a further +2.12 mA). The measured `IDD3N1` and `IDD2N` lines are parallel, so the gap does not change with data rate. |
| `IDD0` | `IDD2N` + the activate increment: 18.88 mA at 1200 MT/s, scaled ×1.886 along the fixed-pacing IDD0 trend |
| `IDD5B` | `IDD3N1` + the refresh increment: 60.68 mA at 1200 MT/s, scaled ×1.541 along the fixed-pacing IDD5B trend |
| `IDD4R`, `IDD4W` | Their data-rate lines re-levelled to the all-0s anchors and evaluated at 6400 MT/s; ÷ 8; × 36/64 for DQ plus ECC lanes |

The file's `extrapolation` block records the fixed offsets and increments,
together with their relative slopes per MT/s, so that the
[HBM4 case study](hbm4-power-prediction.md) can re-scale the rails to other
data rates consistently.

The [HBM3E validation](../reference/case-studies.md#hbm3e_validation) compares
these rails, with no further tuning, against measured H200 power.

## Device configurations

The device configurations at the top of `config/` reuse these derivations:

| Configuration | Currents |
|---|---|
| `HBM2_1200MTs` | Ayna's all-0s HBM2 rails (`IDD_ours_allzeros`) with the 821.3 mA board power-off current subtracted, plus `IDD5B` = 624.7 mA per stack (the fixed-pacing refresh line of the data-rate sweep at 1200 MT/s, the same anchor the HBM3E refresh increment uses), then split per pseudo-channel: `rail / 8 − IDD2N / 16` |
| `HBM3E_6400MTs` | The HBM3E rails above, each lowered by half of their `IDD2N` (70.5 mA), so that standby is split over 16 pseudo-channels per HBM2 stack instead of 8 |
| `HBM4_8000MTs` | The HBM3E rails extrapolated to 8000 MT/s as in the [HBM4 case study](hbm4-power-prediction.md), each lowered by half of their `IDD2N` (88.1 mA) |

The split is explained in [How Ayna computes power](how-ayna-computes-power.md#one-pseudo-channel-at-a-time).
All three use static data as the reference, with the data-pattern coefficients
fitted to the HBM2 beat-pattern measurements (see
[The data-pattern energy model](data-pattern-model.md)).
