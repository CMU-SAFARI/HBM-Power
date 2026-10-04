# Runners

Ayna builds two command-line runners into `build/bin/`. They take the same
arguments and print output in the same format.

| Runner | Standard | Refresh energy | I/O rail (`VDDQ`) |
|---|---|---|---|
| `HBM2_runner` | HBM2 (`standards/hbm2`) | yes, `(IDD5B − IDD3N16) × tRFC` per `REFA` | no (single `VDD`) |
| `HBM3_runner` | HBM3 (`standards/hbm3`), also used for HBM3E and HBM4 | yes, `(IDD5B − IDD3N16) × tRFC` per `REFA` | yes, applied to the DQ share of read energy |

## Synopsis

```text
HBM2_runner <organization.json> <timing.json> <power.json> <trace.csv> [variation.json]
            [--dq-rate=R] [--tsv-rate=R] [--bg-rate=R]
HBM3_runner <organization.json> <timing.json> <power.json> <trace.csv> [variation.json]
            [--dq-rate=R] [--tsv-rate=R] [--bg-rate=R]
```

## Arguments

| Argument | Description |
|---|---|
| `organization.json` | Device organization. See [Configuration files](config-files.md#organization-file). |
| `timing.json` | Clock period and timing parameters in cycles. See [Configuration files](config-files.md#timing-file). |
| `power.json` | Voltages, IDD currents, optional `datapattern` block. See [Configuration files](config-files.md#power-file). |
| `trace.csv` | DRAM command trace. See [Trace format](trace-format.md). |
| `variation.json` | Optional per-bank-group and per-bank scaling of read/write current. See [Configuration files](config-files.md#variation-file). |

## Options

| Option | Range | Effect |
|---|---|---|
| `--dq-rate=R` | 0 to 1 | DQ-pin toggle rate |
| `--tsv-rate=R` | 0 to 1 | TSV / 2-bit on-die bus toggle rate |
| `--bg-rate=R` | 0 to 1 | Bank-group bus burst-to-burst toggle rate |

The options must use the `--name=value` form; `--dq-rate 0.5` with a space is
not recognized. Each option overrides its knob; passing any of them also turns
the data-pattern model on if the power file disables it. See [Sweep data-pattern activity](../how-to/sweep-data-pattern-activity.md).

## Output

All output goes to standard output:

```text
Data-pattern model: dq_rate=0.5, tsv_rate=0.5, bg_rate=0.5 (relative form)
HBM3 Configuration:
  Channels: 16, PseudoChannels/Ch: 2
  BankGroups: 4, Banks/BG: 4, Banks/PCh: 16
  tCK=625 ps, tRCD=31, tRAS=45, tRP=26, tRL=20, tWL=10
Loaded 60013 commands from trace file
Trace spans 1 active pseudo-channels
Channel 0 PCh 0 : 60013 cmds, energy = 65531409.003 pJ, power = 1200.855 mW
  Bank  0 : <per-bank energy breakdown>
  ...
Total energy across all pseudo-channels: 65531409.003 pJ
Elapsed cycles: 87313, duration: 54570.625 ns
Average power: 1200.855 mW
```

| Line | Meaning |
|---|---|
| `Data-pattern model: ...` | Printed only when a knob option is passed |
| `Channel c PCh p : ...` | Energy and average power of one simulated pseudo-channel |
| `Bank b : ...` | Per-bank energy by component (activate, precharge, background, read, write, refresh) |
| `Total energy ...` | Sum over all simulated pseudo-channels, in pJ |
| `Elapsed cycles ...` | Timestamp of the last command of the longest pseudo-channel |
| `Average power` | Total energy divided by the elapsed time, in mW |

The case-study scripts parse the `Average power: <value> mW` line.

## Exit status

| Status | Cause |
|---|---|
| 0 | Success |
| 1 | Wrong number of arguments, a missing input file, a configuration that is not valid JSON, or a malformed trace line |

A variation file that cannot be read produces a warning and is ignored. The
run continues.
