# Changes From DRAMPower

Ayna is a fork of [DRAMPower](https://github.com/tukl-msd/DRAMPower) v5.4.1
(library version 5.1). This page lists what the fork adds or changes. The
upstream library, its DDR4/DDR5/LPDDR4/LPDDR5 standards, and its `cli` tool are
otherwise unchanged.

## Added

| Component | Files |
|---|---|
| HBM2 standard | `src/DRAMPower/DRAMPower/standards/hbm2/`, `memspec/MemSpecHBM2.{h,cpp}` |
| HBM3 standard (also used for HBM3E and HBM4) | `src/DRAMPower/DRAMPower/standards/hbm3/`, `memspec/MemSpecHBM3.{h,cpp}` |
| Data-pattern read/write energy model | `src/DRAMPower/DRAMPower/util/datapattern_model.h` |
| HBM trace parsers | `src/cli/lib/DRAMPower/cli/hbm{2,3}_trace_parser.{hpp,cpp}` |
| Runners | `src/cli/main/HBM2_runner.cpp`, `src/cli/main/HBM3_runner.cpp` |

### Structure of the HBM2 and HBM3 standards

- Each pseudo-channel is modeled as one rank of `bankgroups × banks` banks.
  The runners create one instance per pseudo-channel that appears in the
  trace.
- Core energy follows DRAMPower's DDR-style decomposition: activate,
  precharge, active and precharged background, read, and write. See
  [How Ayna computes power](../explanation/how-ayna-computes-power.md).
- `IDD3N1` (one bank open) is the read/write background. Both standards use
  `IDD3N16` (all banks open) to charge each additional open bank.
- Interface (I/O termination) energy is not modeled separately. It is included
  in the measured `IDD4R`/`IDD4W` and in the data-pattern model.

### Features in both standards

- **All-bank refresh energy:** `(IDD5B − IDD3N16) × tRFC` per `REFA`, divided
  across banks. `IDD3N16` is the background the engine already charges while
  all banks refresh. See
  [How Ayna computes power](../explanation/how-ayna-computes-power.md#refresh).
- **Data-pattern model:** replaces `IDD4R`/`IDD4W` with an effective current
  computed from three toggle-rate knobs. See
  [The data-pattern energy model](../explanation/data-pattern-model.md).
- **Per-bank variation:** optional bank-group and bank scaling factors on the
  read and write currents (separate factors for writes are optional).

### HBM3-only features

- **Split rails:** a `VDDQ` I/O rail drives the DQ-pin share of read energy.
  The rest stays on `VDD`.

## Changed

| File | Change |
|---|---|
| `src/DRAMPower/DRAMPower/data/stats.h` | `activeTime()` made `const` |
| `src/DRAMPower/CMakeLists.txt`, `src/cli/**/CMakeLists.txt` | Build the new sources and the two runners |
| `CMakeLists.txt`, `CMakePresets.json` | Remove the unit-test and benchmark targets, whose sources are not part of this fork, and provide `release`/`debug` presets |
