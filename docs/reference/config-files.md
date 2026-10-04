# Configuration Files

A run reads three JSON files (organization, timing, power) and optionally a
fourth (variation). The runners accept `//` and `/* */` comments. Fields whose
names start with `_` (`_comment`, `_note`, ...) are documentation and are
ignored.

**Defaults.** Every field you leave out, including a whole `timing`, `voltage`,
`IDD` or `datapattern` section, takes its value from a device configuration:
`config/HBM2_1200MTs` for `HBM2_runner` and `config/HBM3E_6400MTs` for
`HBM3_runner`. Empty files (`{}`) therefore reproduce those devices exactly.
The tables below list both sets of defaults. Fields that are not "used for
energy" describe the device but do not change the runner's result.

## Organization file

| Field | HBM2 default | HBM3 default | Used for energy | Description |
|---|---|---|---|---|
| `stacks` | 1 | 1 | no | Stacks in the device |
| `channels_per_stack` | 8 | 16 | no | Channels per stack |
| `pseudochannels_per_channel` | 2 | 2 | no | Pseudo-channels per channel |
| `bankgroups_per_pseudochannel` | 4 | 4 | yes | Bank groups per pseudo-channel |
| `banks_per_bankgroup` | 4 | 4 | yes | Banks per bank group |
| `rows` | 16384 | 16384 | no | Rows per bank |
| `columns` | 32 | 32 | no | Columns per row (bursts per row) |
| `width` | 64 | 32 | no | Pseudo-channel data width in bits |
| `dataRate` | 2 | 4 | yes | Transfers per clock cycle (2 for HBM2, 4 for HBM3/HBM4) |

Each pseudo-channel is simulated on its own, so the channel and stack counts
do not change the runner's result. They describe the device: to get device
power, add up the results of all its pseudo-channels.

## Timing file

All fields sit under a top-level `"timing"` object. Except for `tCK_ps`, values
are integer clock cycles.

| Field | HBM2 default | HBM3 default | Used for energy | Description |
|---|---|---|---|---|
| `tCK_ps` | 1666.7 | 625 | yes | Clock period in picoseconds |
| `nBL` | 4 | 8 | yes | Burst length in beats. The burst duration is `nBL / dataRate` cycles. |
| `nRCD` | 9 | 31 | no | ACT to read |
| `nRCDWR` | 6 | 15 | no | ACT to write |
| `nRAS` | 20 | 45 | yes | ACT to PRE |
| `nRP` | 9 | 26 | yes | PRE to ACT |
| `nRC` | 29 | 72 | no | ACT to ACT, same bank |
| `nRL`, `nWL` | 7, 7 | 20, 10 | `RDA`/`WRA` | Read and write latency. Set when an auto-precharge closes the bank. |
| `nCCD_S`, `nCCD_L` | 2, 2 | 2, 4 | no | Column-to-column delay, different / same bank group |
| `nWTR_S`, `nWTR_L` | 3, 6 | 7, 10 | no | Write-to-read turnaround, different / same bank group |
| `nWR` | 10 | 33 | `WRA` | Write recovery. Sets when a write auto-precharge closes the bank. |
| `nRFC` | 210 | 560 | yes | All-bank refresh cycle time (350 ns). 0 disables refresh energy. |

Other fields (`nCL`, `nFAW`, `nRRD_*`, `nRTP`, `nRTW`, `nREFI`, `nRFCpb`) are
accepted for completeness and ignored.

## Power file

```json
{
  "voltage": { "VDD": 1.1, "VDDQ": 0.9 },
  "IDD": {
    "IDD0": 106.1, "IDD2N": 70.5, "IDD3N1": 71.2, "IDD3N16": 73.3,
    "IDD4R": 560.7, "IDD4W": 389.5, "IDD5B": 164.7
  },
  "datapattern": { ... }
}
```

### `voltage`

| Field | HBM2 default | HBM3 default | Description |
|---|---|---|---|
| `VDD` | 1.2 | 1.1 | Core supply in V |
| `VDDQ` | — | equal to `VDD` | HBM3 only. I/O supply in V. Applied to the DQ-pin share of read energy. |

### `IDD`

All currents are in mA, for one pseudo-channel.

| Field | HBM2 default | HBM3 default | Description |
|---|---|---|---|
| `IDD0` | 28.544 | 106.1 | Activate-precharge current (one bank cycling at `tRC`) |
| `IDD2N` | 11.756 | 70.5 | Precharge standby |
| `IDD3N1` | 12.394 | 71.2 | Active standby, one bank open. Used as the read/write background. |
| `IDD3N16` | 14.281 | 73.3 | Active standby, all banks open |
| `IDD4R` | 202.769 | 560.7 | Burst read, static data |
| `IDD4W` | 137.406 | 389.5 | Burst write, static data |
| `IDD5B` | 66.331 | 164.7 | Burst refresh (back-to-back `REFA`). 0 disables refresh energy. |

### `datapattern` block

The data-pattern model is on unless the power file turns it off with
`"datapattern": { "enabled": false }`. A missing block, or missing fields in
it, take the defaults below, which are also the device configurations' values
and the same for both runners.

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Turn the model on or off. Off means fixed `IDD4R`/`IDD4W`. |
| `dq_rate`, `tsv_rate`, `bg_rate` | 0.5 | Toggle-rate knobs, 0 to 1 (0.5 is uniform-random data) |
| `ref_dq_rate`, `ref_tsv_rate`, `ref_bg_rate` | 0 | Activity at which the configured `IDD4R`/`IDD4W` were measured (relative form) |
| `K_mA_per_pJbit` | 0 | Greater than 0 selects the absolute form: `IDD4R_eff = IDD3N1 + K × pJ/bit`. 0 selects the relative form. |
| `floor_pJbit` | 3.1798 | pJ/bit with no toggling |
| `coef_T_DQ` | 1.0400 | pJ/bit per unit DQ toggle rate |
| `coef_T_2bit` | 1.4875 | pJ/bit per unit TSV toggle rate |
| `coef_busflip` | 1.4100 | pJ/bit per unit bank-group burst-to-burst toggle rate |
| `apply_to_writes` | `true` | Also scale `IDD4W` |

The coefficients are the values fitted to the HBM2 measurements. The HBM2
case-study configurations set `"enabled": false`, because they model each data
pattern with its own measured currents.

The equations are in [The data-pattern energy model](../explanation/data-pattern-model.md).

## Variation file

Optional fifth positional argument. It scales each bank's read and write current
to model structural (spatial) variation across bank groups and banks. This is
`config/HBM2_1200MTs/variation.json`:

```json
{
  "scaling_factors": {
    "bankgroup":       [1.0391, 0.9603, 1.0411, 0.9595],
    "bank":            [1.0259, 1.0253, 0.9763, 0.9725],
    "bankgroup_write": [1.0183, 0.9825, 1.0175, 0.9816],
    "bank_write":      [1.0028, 0.9998, 1.0006, 0.9969]
  }
}
```

| Field | Required | Description |
|---|---|---|
| `bankgroup` | yes | Read-current factor per bank group |
| `bank` | yes | Read-current factor per bank position within a bank group |
| `bankgroup_write` | no | Write-current factor per bank group. Defaults to `bankgroup`. |
| `bank_write` | no | Write-current factor per bank position. Defaults to `bank`. |

The read factor for flat bank `b` is
`bankgroup[b / banks_per_bankgroup] × bank[b % banks_per_bankgroup]`, and it
multiplies the bank's `IDD4R` (after the data-pattern model). The write factor
multiplies `IDD4W` the same way. Lists shorter than the count wrap around.
None of the case studies use a variation file.

## Provided configurations

### Device configurations

Each directory holds `organization.json`, `timing.json` and `power.json` for
one device. The currents describe one pseudo-channel of the DRAM device
only, with static data as the reference, and the data-pattern knobs default to
uniform-random data. Device power is the sum over all pseudo-channels, with
unused pseudo-channels run on an idle trace (see
[How Ayna computes power](../explanation/how-ayna-computes-power.md#one-pseudo-channel-at-a-time)).

| Directory | Runner | Device | Current scope |
|---|---|---|---|
| `config/HBM2_1200MTs/` | `HBM2_runner` | HBM2 stack, 8 channels × 2 pseudo-channels, 4 bank groups × 4 banks, 1200 MT/s, VDD 1.2 V | Per pseudo-channel |
| `config/HBM3E_6400MTs/` | `HBM3_runner` | HBM3E 16 Gb 8-high stack, 16 channels × 2 pseudo-channels, 4 bank groups × 4 banks, 6400 MT/s, VDD 1.1 V | Per pseudo-channel |
| `config/HBM4_8000MTs/` | `HBM3_runner` | HBM4 stack, 32 channels × 2 pseudo-channels, 2 bank groups × 8 banks, 8000 MT/s, VDDC 1.05 V / VDDQ 0.9 V | Per pseudo-channel |

`config/HBM2_1200MTs/` also holds `variation.json`, the measured bank-group and
bank variation of HBM2 read and write current (see
[How Ayna computes power](../explanation/how-ayna-computes-power.md#per-bank-variation)).
Pass it as the fifth argument of `HBM2_runner`.

### Case-study configurations

Each case study keeps the configurations behind its paper result in its own
`configs/` directory; see [Case studies](case-studies.md).

| File | Description |
|---|---|
| `hbm2_model_comparison/configs/HBM2_organization.json` | One HBM2 stack: 8 channels × 2 pseudo-channels, 4 bank groups × 4 banks |
| `hbm2_model_comparison/configs/HBM2_1.2Gbps_timing_BL4.json` | 1.2 Gbps (tCK 1666.7 ps), `nBL` = 4 |
| `hbm2_model_comparison/configs/IDD_ours_allzeros.json` | Ayna, all-0s data, board-level (includes the FPGA board's 821.3 mA power-off current) |
| `hbm2_model_comparison/configs/IDD_ours_random.json` | Ayna, random data, board-level |
| `hbm2_model_comparison/configs/IDD_dramsim3.json` | DRAMSim3's HBM2 currents rescaled to 1.2 Gbps, board-level |
| `hbm2_model_comparison/configs/IDD_oconnor_notoggle.json` | FGDRAM HBM2 (O'Connor et al.), no toggling, board-level |
| `hbm2_model_comparison/configs/IDD_oconnor_50toggle.json` | FGDRAM HBM2, 50% toggling, board-level |
| `hbm2_data_pattern_dependence/configs/IDD_ours_allzeros.json` | Copy of the Ayna all-0s config above |
| `hbm3e_validation/configs/HBM3_16Gb_8hi_organization.json` | HBM3 16 Gb die, 8-high stack, 16 channels × 2 pseudo-channels |
| `hbm3e_validation/configs/HBM3_6400Mbps_timing.json` | 6.4 Gbps (tCK 625 ps), from Ramulator 2 `HBM3_6400Mbps` |
| `hbm3e_validation/configs/HBM3_6400_power_datapattern.json` | Per-pseudo-channel HBM3E rails at 6.4 Gbps, the data-pattern settings of the paper, and an `extrapolation` block used by the HBM4 case study |
| `hbm4_case_study/configs/HBM3_6400_power_datapattern.json` | Copy of the HBM3E power config above |
| `hbm4_case_study/configs/HBM4_organization.json` | 8 stacks × 32 channels × 2 pseudo-channels, 2 bank groups × 8 banks |
| `hbm4_case_study/configs/HBM3_16Gb_8hi_organization.json` | Copy of the HBM3E organization above, for the HBM3E runs |

Paths are relative to `case_studies/`.
[Where the IDD values come from](../explanation/idd-derivation.md) explains how
the IDD values were derived.
