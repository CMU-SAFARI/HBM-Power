# How Ayna Computes Power

Ayna turns a command trace into energy the same way DRAMPower does. It counts
commands and the time banks spend open or closed, then charges each event or
interval a current above a background, multiplied by the supply voltage and the
duration. This page describes the HBM-specific parts of that calculation.

## One pseudo-channel at a time

An HBM pseudo-channel has its own command bus, banks and data bus, so the
runners simulate each one independently. Trace lines are grouped by
`(channel_id, pseudochannel_id)`. Each group is fed into a fresh HBM2 or HBM3
instance with `bankgroups × banks` banks. When the trace ends, the instance
reports its energy.

The runner then adds the energies of all pseudo-channels and divides by the
duration of the longest one, which gives the average power. A trace that drives
only pseudo-channel 0 therefore reports the power of one pseudo-channel.

The device configurations in `config/` (`HBM2_1200MTs`, `HBM3E_6400MTs`,
`HBM4_8000MTs`) hold per-pseudo-channel currents, so device power is the sum
over all pseudo-channels of the device: active ones run with their traffic,
unused ones with an idle trace (`NOP`s spanning the same time). For symmetric
traffic, simulate one pseudo-channel of each kind and multiply.

The per-pseudo-channel currents come from measurements of whole HBM2 stacks.
The standby current (`IDD2N`) flows in all 16 pseudo-channels of a stack,
whether or not they receive commands, so it is divided by 16. The measurement
loops drive 8 of the 16 pseudo-channels (one per channel), so every increment
above standby, from open banks, activates, reads, writes and refresh, is
divided by 8:

```text
rail_pc = rail_stack / 8 − IDD2N_stack / 16
```

The HBM3E and HBM4 device configurations apply the same split to the
extrapolated rails.

The case-study configurations (each case study's `configs/` directory) keep the conventions
of the paper. The HBM2 ones hold the board-level currents of the whole
measurement setup, including the FPGA board's power-off current (821.3 mA),
so that a trace run against them is directly comparable to the measured board
power. 

## Energy components

Currents are converted from mA to A, times from cycles to picoseconds, so
`V × A × ps` is energy in pJ. With `B` banks per pseudo-channel, `tBurst =
(nBL / dataRate) × tCK`, and the activation current

```text
I_act = (IDD0 × (tRP + tRAS) − IDD2N × tRP) / tRAS
```

recovered from the IDD0 loop, the components are:

| Component | Charged | Energy |
|---|---|---|
| Activate | per `ACT` | `VDD × (I_act − IDD3N1) × tRAS` |
| Precharge | per `PRE` / auto-precharge | 0: precharge current equals `IDD2N` |
| Precharged background | while all banks are closed | `VDD × IDD2N × t` |
| Active background | while at least one bank is open | From `VDD × IDD3N1 × t` (one bank open) to `VDD × IDD3N16 × t` (all open), see below |
| Read | per `RD`/`RDA` | `V_RD × (IDD4R − IDD3N1) × tBurst` |
| Write | per `WR`/`WRA` | `VDD × (IDD4W − IDD3N1) × tBurst` |
| Refresh | per `REFA` | `VDD × (IDD5B − IDD3N16) × tRFC`, see below |

Reads and writes are charged only for the current above the active background,
because the open bank is already being charged `IDD3N1`.

### Per-bank active background

The HBM2 measurements include `IDD3N16`, the standby current with all 16 banks
open. Both standards use it: the first open bank brings the pseudo-channel to
`IDD3N1`, and each additional open bank adds

```text
δ = (IDD3N16 − IDD3N1) / (B − 1)
```

for as long as it stays open. With all banks open, the total is exactly
`IDD3N16`.

### Refresh

Both runners charge refresh energy when `IDD5B` and `nRFC` are both non-zero.
A `REFA` keeps every bank of the pseudo-channel busy for `tRFC`, so the active
background is already charged during that time at `IDD3N16` (all banks open).
The refresh term adds the rest of `IDD5B` on top of that background:

```text
E_ref = VDD × (IDD5B − IDD3N16) × tRFC
```

As a result, a trace of back-to-back `REFA` commands, one every `tRFC` like
the IDD5B measurement loop, draws exactly `IDD5B`. With `IDD5B` or `nRFC` at 0,
a `REFA` still holds the banks busy for `tRFC` but adds no refresh energy.

### HBM3: split rails

`HBM3_runner` supports a separate I/O rail. When the data-pattern model is enabled,
it knows what share `f_DQ` of the read energy is spent on the external DQ pins,
and it charges reads at

```text
V_RD = f_DQ × VDDQ + (1 − f_DQ) × VDD
```

Otherwise `V_RD = VDD`. Writes stay on `VDD`. The HBM4 case study uses this
to model JESD270-4's separate VDDC/VDDQ rails.

## Read and write currents

The data-pattern model is on by default: each run computes an effective
`IDD4R`/`IDD4W` from the toggle-rate knobs before applying the equations above.
A power file can turn it off with `"datapattern": {"enabled": false}`, which
keeps `IDD4R` and `IDD4W` fixed. The HBM2 case-study configurations do this,
because they model each data pattern with its own measured currents. See
[The data-pattern energy model](data-pattern-model.md).

## Per-bank variation

An optional variation file multiplies each bank's `IDD4R`/`IDD4W` by a
bank-group factor and a bank factor, with separate factors for reads and
writes. This models the structural variation in read and write current across
bank groups and banks observed in the HBM2 characterization.

`config/HBM2_1200MTs/variation.json` holds the measured HBM2 factors. They are
fitted to the bank-group-pair and bank-offset measurement loops of the paper
(Section 5.2), counting only the loop current above the standby share of the
idle pseudo-channels, and normalized to average 1 over the 16 banks of a
pseudo-channel. 

