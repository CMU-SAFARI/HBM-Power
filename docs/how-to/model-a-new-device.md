# Model a New HBM Device or Speed Bin

A device in Ayna is three JSON files: organization, timing and power. This
guide shows how to add one, using an HBM3E device at a different speed bin as
the running example. The fields are defined in
[Configuration files](../reference/config-files.md).

## 1. Choose a runner

| Device | Runner |
|---|---|
| HBM2, HBM2E | `HBM2_runner` |
| HBM3, HBM3E, HBM4 | `HBM3_runner` | 

The HBM4 case study uses `HBM3_runner` with an HBM4 organization and HBM4
timings. At the level of detail the engine models, the per-pseudo-channel
command protocol is the same.

## 2. Write the organization file

Create a directory for the device, for example `config/HBM3E_5600MTs/`. Copy
`organization.json` from the closest device in `config/` and adjust the counts:

```json
{
    "stacks": 1,
    "channels_per_stack": 16,
    "pseudochannels_per_channel": 2,
    "bankgroups_per_pseudochannel": 4,
    "banks_per_bankgroup": 4,
    "rows": 16384,
    "columns": 32,
    "width": 32,
    "dataRate": 4
}
```

`bankgroups_per_pseudochannel`, `banks_per_bankgroup` and `dataRate` affect
energy. `dataRate` is transfers per clock: 2 for HBM2, and 4 for HBM3 and
HBM4. The other fields describe the device but do not change per-pseudo-channel
energy.

## 3. Write the timing file

All values except `tCK_ps` are in clock cycles. To move to a different speed
bin, keep each parameter constant in nanoseconds and convert it to cycles at the
new `tCK`:

```text
n_new = ceil(n_old × tCK_old / tCK_new)
```

Make sure `nBL` is the burst length in beats (8 for HBM3/HBM4 BL8). The engine
computes the burst duration as `nBL / dataRate` cycles. For refresh energy, set
`nRFC`.

`hbm3_timing_json()` in `case_studies/hbm4_case_study/plot_hbm3e_vs_hbm4.py`
does this scaling programmatically.

## 4. Write the power file

```json
{
    "voltage": { "VDD": 1.1, "VDDQ": 1.1 },
    "IDD": {
        "IDD0": 106.1, "IDD2N": 70.5, "IDD3N1": 71.2, "IDD3N16": 73.3,
        "IDD4R": 560.7, "IDD4W": 389.5, "IDD5B": 164.7
    },
    "datapattern": { "enabled": true,
                     "dq_rate": 0.5, "tsv_rate": 0.5, "bg_rate": 0.5 }
}
```

- **Units.** Currents are in mA and voltages in V.
- **Complete files.** Any field you leave out takes the value of
  `config/HBM3E_6400MTs` (`HBM3_runner`) or `config/HBM2_1200MTs`
  (`HBM2_runner`). Give every `IDD` and the energy-relevant timings for a new
  device, so that none of them silently come from the reference device.
- **Scope.** Give currents for one pseudo-channel, the convention of all
  device configurations in `config/`. 
- **Data pattern.** If `IDD4R`/`IDD4W` are measured with static data, keep the
  reference knobs (`ref_*_rate`) at 0. If they are datasheet values measured
  with random data, set the reference knobs to 0.5 so the model scales relative
  to that point.

If you have no measurements for the target device, scale an existing set.
[Where the IDD values come from](../explanation/idd-derivation.md#hbm3e-extrapolation)
describes how the HBM3E rails were extrapolated from HBM2 data, and
[HBM4 power prediction](../explanation/hbm4-power-prediction.md) describes how
the HBM4 rails were extrapolated from HBM3E.

## 5. Run and sanity-check

```bash
./build/bin/HBM3_runner my_org.json my_timing.json my_power.json \
    case_studies/hbm3e_validation/traces/hbm3_random_read_4rpa.csv
```

Check:

- The idle trace (`hbm3_baseline_nop_ref.csv`) with all knobs at 0 gives a
  little more than `IDD2N × VDD`, where the difference is refresh. For the
  HBM3E device configuration this is 86.8 mW against 70.5 mA × 1.1 V = 77.6 mW.
- Power grows when you raise the data-pattern knobs.

Note that the provided traces are in clock cycles. For a different `tCK`,
regenerate them so that refresh intervals stay constant in time, as
`case_studies/hbm4_case_study/tracegen.py` does.
