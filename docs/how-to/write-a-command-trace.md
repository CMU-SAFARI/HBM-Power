# Write a Command Trace

A trace is a CSV file with one DRAM command per line. The full column
definition is in [Trace format](../reference/trace-format.md). This guide shows
how to write a trace that the runners accept and that produces a meaningful
power number.

## 1. Start with the header

```text
timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id
```

The header is optional, but if present it must begin with `timestamp`.

## 2. Issue commands in clock cycles

Timestamps are in DRAM clock cycles (`tCK_ps` in the timing configuration), not
nanoseconds. Commands must be in non-decreasing timestamp order within each
pseudo-channel. This example opens a row, reads four bursts, and closes it, on
HBM2 at 1.2 Gbps:

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

- `RD` comes `nRCD` (9) cycles after `ACT`, and `PRE` comes at least `nRAS` (20)
  cycles after it. The engine does not check timing. A trace that violates the
  timing parameters still runs, but its power is not physical.
- Consecutive reads are 4 cycles apart, the spacing the provided HBM2 traces use
  with this timing configuration (see the read-timing note in
  [Case studies](../reference/case-studies.md#hbm2_model_comparison)).
- The trailing `NOP` sets the end of the trace window. The runner averages
  energy over the time up to the last command, so idle time after the last
  real command counts only if a command marks its end.

## 3. Add refresh

Both runners charge refresh energy for every `REFA` when the power file has an
`IDD5B` and the timing file an `nRFC`. Insert a `REFA` every `tREFI` cycles and
leave a gap of at least `nRFC` cycles after it before the next command. On
HBM3E at 6.4 Gbps (`nRFC` = 560):

```text
0,REFA,0,0,0,0,0,0
560,ACT,0,0,0,0,10,
```

On HBM2 at 1.2 Gbps, `nRFC` is 210 cycles and `tREFI` 2340 cycles.

## 4. Spread traffic over pseudo-channels (optional)

Each `(channel_id, pseudochannel_id)` pair is simulated as an independent
pseudo-channel. The runner adds up their energies and divides by the longest
pseudo-channel's duration. With the device configurations in `config/`, the
power of a device is the sum over all of its pseudo-channels. For symmetric
traffic it is faster to simulate one active pseudo-channel and one idle
pseudo-channel (a trace of `NOP`s spanning the same time), and multiply each
by how many pseudo-channels are in that state.

## 5. Run it

```bash
C=config/HBM2_1200MTs
./build/bin/HBM2_runner $C/organization.json $C/timing.json $C/power.json my_trace.csv
```

If a line is malformed, the runner reports `HBM2 trace line N: expected at
least 7 columns` and exits with status 1.

## Generating traces programmatically

`examples/make_read_trace.py` generates a read-streaming trace, or an idle
trace of the same length, for one pseudo-channel of any device configuration,
taking the timing from its `timing.json`:

```bash
python3 examples/make_read_trace.py config/HBM3E_6400MTs --refresh -o read.csv
python3 examples/make_read_trace.py config/HBM3E_6400MTs --refresh --idle -o idle.csv
```

`--activations` and `--reads-per-act` set the trace length and the reads per
activated row. The script is a short starting point for your own generators.
`case_studies/hbm4_case_study/tracegen.py` is another example: it rebuilds a
random-read trace and a NOP+refresh idle trace for any clock period, so that
`tREFI` and `tRFC` stay constant in nanoseconds across speed bins. To get
traces from a memory-controller simulator instead, convert its command log to
the column layout above. The provided HBM2 microbenchmark traces come from Ramulator.
