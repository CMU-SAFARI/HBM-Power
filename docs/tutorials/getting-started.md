# Getting Started

This tutorial takes you from a fresh clone to your first power estimates with
the device configurations in `config/`. You will build the two Ayna runners,
run a trace you write by hand, generate a longer read trace, see how the data
pattern changes read power, turn per-pseudo-channel results into stack power,
and compare HBM2, HBM3E and HBM4.

It takes about ten minutes, most of which is the first build.

## Prerequisites

- Linux or macOS with a C++17 compiler (GCC 9+ or Clang 10+)
- CMake 3.22 or later
- Python 3 (standard library only) for the trace generator
- Network access during the first configure: CMake downloads nlohmann_json,
  DRAMUtils, CLI11 and spdlog

All commands below run from the repository root.

## 1. Build the runners

```bash
cmake --preset release
cmake --build --preset release -j
```

The `release` preset configures a Release build in `build/` and builds two
executables:

```text
build/bin/HBM2_runner     # HBM2
build/bin/HBM3_runner     # HBM3, HBM3E and HBM4
```

If you prefer not to use presets, see [Build the engine](../how-to/build-the-engine.md).

## 2. Look at a device configuration

Each directory at the top of `config/` describes one device:

```text
config/HBM2_1200MTs/    HBM2,  1200 MT/s
config/HBM3E_6400MTs/   HBM3E, 6400 MT/s
config/HBM4_8000MTs/    HBM4,  8000 MT/s
```

and holds three files:

| File | Contents |
|---|---|
| `organization.json` | Bank groups and banks per pseudo-channel, transfers per clock |
| `timing.json` | Clock period and timing parameters in clock cycles |
| `power.json` | Supply voltages, IDD currents, and the data-pattern settings |

Two conventions matter for everything below:

- **Per pseudo-channel.** The currents describe one pseudo-channel. A runner
  reports the power of the pseudo-channels a trace drives, and device power is
  the sum over all pseudo-channels (step 6).
- **Uniform-random data by default.** `IDD4R` and `IDD4W` are the currents with
  static data, and the data-pattern knobs in `power.json` default to 0.5
  (uniform-random data). You can override them on the command line (step 5).

## 3. Run a trace you write by hand

A trace is a CSV file with one DRAM command per line, timestamped in clock
cycles. Save this HBM2 trace as `first.csv`. It opens a row, reads it four
times, closes it, and marks the end of the window at cycle 100:

```text
timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id
0,ACT,0,0,0,0,10,
9,RD,0,0,0,0,10,0
13,RD,0,0,0,0,10,1
17,RD,0,0,0,0,10,2
21,RD,0,0,0,0,10,3
29,PRE,0,0,0,0,0,
100,NOP,0,0,0,0,0,
```

Run it with the HBM2 configuration:

```bash
C=config/HBM2_1200MTs
./build/bin/HBM2_runner $C/organization.json $C/timing.json $C/power.json first.csv
```

The runner prints the configuration, a per-bank energy breakdown, and a
summary:

```text
Channel 0 PCh 0 : 7 cmds, energy = 8268.462 pJ, power = 49.610 mW
Total energy across all pseudo-channels: 8268.462 pJ
Elapsed cycles: 100, duration: 166.670 ns
Average power: 49.610 mW
```

The seven commands cost 8.3 nJ over 100 cycles (166.7 ns at tCK = 1666.7 ps),
an average of 49.6 mW for this pseudo-channel. Most of the window is idle, so
standby current dominates. The timestamps respect the HBM2 timing in
`timing.json` (`nRCD` = 9 from `ACT` to `RD`, `nRAS` = 20 from `ACT` to
`PRE`). The engine does not check timing, so whoever writes the trace must.
[Write a command trace](../how-to/write-a-command-trace.md) covers the format
in detail.

## 4. Generate a read trace

Writing long traces by hand is impractical. `examples/make_read_trace.py`
generates a read-streaming trace for one pseudo-channel of any device
configuration. It reads the timing and bank organization from the
configuration, opens a row in every bank group, interleaves four reads per row
across bank groups, and refreshes every `tREFI` with `--refresh`:

```bash
python3 examples/make_read_trace.py config/HBM2_1200MTs --refresh -o read.csv
```

```text
read trace: 8192 reads over 17999 cycles (29998.9 ns), data bus 91% busy
```

Run it:

```bash
./build/bin/HBM2_runner $C/organization.json $C/timing.json $C/power.json read.csv | tail -1
```

```text
Average power: 422.056 mW
```

With the data bus busy 91% of the time, one HBM2 pseudo-channel draws about
422 mW.

## 5. Change the data pattern

Read and write energy depend on how often bits toggle on the DQ pins, the TSVs
and the bank-group bus. The configuration assumes uniform-random data (all
three knobs at 0.5). Set the knobs to 0 for static data, such as all zeros:

```bash
./build/bin/HBM2_runner $C/organization.json $C/timing.json $C/power.json read.csv \
    --dq-rate=0 --tsv-rate=0 --bg-rate=0 | tail -1
```

```text
Average power: 293.304 mW
```

On this read-heavy trace, random data costs 44% more power than static data.
[Sweep data-pattern activity](../how-to/sweep-data-pattern-activity.md) shows
how to explore the knobs further, and
[The data-pattern energy model](../explanation/data-pattern-model.md) explains
them.

## 6. From one pseudo-channel to a stack

An HBM2 stack has 16 pseudo-channels (8 channels × 2). Pseudo-channels that
carry no traffic still draw standby current and refresh, so model them with an
idle trace of the same length:

```bash
python3 examples/make_read_trace.py config/HBM2_1200MTs --refresh --idle -o idle.csv
./build/bin/HBM2_runner $C/organization.json $C/timing.json $C/power.json idle.csv | tail -1
```

```text
idle trace: 17999 cycles (29998.9 ns)
Average power: 19.456 mW
```

Stack power is the sum over all 16 pseudo-channels:

| Traffic | Stack power |
|---|---|
| All 16 pseudo-channels streaming reads | 16 × 422.1 mW = 6.75 W |
| 4 streaming, 12 idle | 4 × 422.1 + 12 × 19.5 mW = 1.92 W |

A trace can also drive several pseudo-channels directly (one
`(channel_id, pseudochannel_id)` pair each). The runner then reports their sum.
For symmetric traffic, simulating one pseudo-channel of each kind and
multiplying is faster.

## 7. Compare HBM2, HBM3E and HBM4

Generate the same kind of trace for each device and run it with the matching
runner:

```bash
for d in HBM2_1200MTs HBM3E_6400MTs HBM4_8000MTs; do
  R=HBM3_runner; [ $d = HBM2_1200MTs ] && R=HBM2_runner
  C=config/$d
  python3 examples/make_read_trace.py $C --refresh -o $d.csv
  ./build/bin/$R $C/organization.json $C/timing.json $C/power.json $d.csv \
      | grep -E "Total energy|Average power"
done
```

```text
read trace: 8192 reads over 17999 cycles (29998.9 ns), data bus 91% busy
Total energy across all pseudo-channels: 12661218.281 pJ
Average power: 422.056 mW
read trace: 8192 reads over 17697 cycles (11060.6 ns), data bus 93% busy
Total energy across all pseudo-channels: 13391274.193 pJ
Average power: 1210.716 mW
read trace: 8192 reads over 18031 cycles (9015.5 ns), data bus 91% busy
Total energy across all pseudo-channels: 13481428.505 pJ
Average power: 1495.361 mW
```

Every read moves 256 bits on all three devices (HBM2: 64 DQ × BL4; HBM3E and
HBM4: 32 DQ × BL8), so each trace reads 8192 × 256 bits. Dividing the total
energy by that gives the energy per bit:

| Device | Power per pseudo-channel | Read bandwidth per pseudo-channel | Energy per bit |
|---|---|---|---|
| HBM2, 1200 MT/s | 422 mW | 70 Gbit/s | 6.04 pJ/bit |
| HBM3E, 6400 MT/s | 1211 mW | 190 Gbit/s | 6.39 pJ/bit |
| HBM4, 8000 MT/s | 1495 mW | 233 Gbit/s | 6.43 pJ/bit |

All three traces keep the data bus about 91–93% busy, so the comparison is
like for like. Per pseudo-channel, HBM4 delivers 23% more read bandwidth than
HBM3E at nearly the same energy per bit (0.7% higher). The HBM4 currents are
forward predictions extrapolated from HBM3E;
[HBM4 power prediction](../explanation/hbm4-power-prediction.md) discusses the
assumptions.

## What you learned

- A run takes an organization, a timing configuration, a power configuration
  and a trace, and reports energy and average power.
- The device configurations in `config/` describe one pseudo-channel and assume
  uniform-random data unless you set the data-pattern knobs.
- Device power is the sum over all pseudo-channels, with unused ones run on an
  idle trace.

## Next steps

- [Write your own command trace](../how-to/write-a-command-trace.md)
- [Model a new HBM device or speed bin](../how-to/model-a-new-device.md)
- [Reproduce the paper's figures](reproducing-the-paper.md), which uses the
  configurations and traces of the case studies
- [Learn how Ayna computes power](../explanation/how-ayna-computes-power.md)
