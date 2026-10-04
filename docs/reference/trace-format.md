# Trace Format

A trace is a comma-separated file with one DRAM command per line. Both runners
use the same format.

## Columns

```text
timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id[,data]
```

| # | Column | Type | Required | Description |
|---|---|---|---|---|
| 1 | `timestamp` | integer | yes | Issue time in DRAM clock cycles (`tCK_ps`) |
| 2 | `command` | string | yes | Command mnemonic, see below |
| 3 | `channel_id` | integer | yes | Channel index |
| 4 | `pseudochannel_id` | integer | yes | Pseudo-channel index within the channel |
| 5 | `bankgroup_id` | integer | yes | Bank group within the pseudo-channel |
| 6 | `bank_id` | integer | yes | Bank within the bank group |
| 7 | `row_id` | integer | yes | Row address. Ignored by commands that do not use it, but must be present. |
| 8 | `column_id` | integer | no | Column address. May be empty. |
| 9 | `data` | hex string | no | Burst payload for `RD`/`WR`/`RDA`/`WRA`. The HBM energy model does not use it; use the data-pattern knobs instead. |

- A header line is optional. It is recognized when its first field is
  `timestamp`.
- Leading and trailing spaces and tabs are trimmed.
- The flat bank index is `bankgroup_id × banks_per_bankgroup + bank_id`, with
  `banks_per_bankgroup` taken from the organization file.

## Commands

| Command | Meaning | HBM2 | HBM3 |
|---|---|---|---|
| `ACT` | Activate a row | energy | energy |
| `PRE` | Precharge a bank | energy | energy |
| `PREA` | Precharge all banks | energy | energy |
| `RD`, `WR` | Read / write burst | energy | energy |
| `RDA`, `WRA` | Read / write with auto-precharge | energy | energy |
| `REFA` | All-bank refresh | energy | energy |
| `NOP` | No operation; marks time | accepted | accepted |

## Ordering and timing

- Commands are grouped by `(channel_id, pseudochannel_id)`. Each group is
  simulated as an independent pseudo-channel, in file order.
- Within a pseudo-channel, timestamps must not decrease.
- The engine does not check DRAM timing constraints. The trace generator is
  responsible for legal spacing.
- A pseudo-channel's duration ends at its last command, so end a trace with a
  `NOP` if trailing idle time should count.

## Example

HBM3E at 6.4 Gbps (`nRFC` = 560, `nRCD` = 31, `nRAS` = 45):

```text
timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id
0,REFA,0,0,0,0,0,0
560,ACT,0,0,1,2,4711,
591,RD,0,0,1,2,4711,0
595,RD,0,0,1,2,4711,8
636,PRE,0,0,1,2,0,
1248,NOP,0,0,0,0,0,0
```

## Provided traces

| Directory | Traces |
|---|---|
| `case_studies/hbm2_model_comparison/traces/` | Six microbenchmarks (`act_hammer`, `interleaved`, `streaming_all_banks`, `streaming_reads`, `wr_rd_turnaround`, `ws_bg0_bg2`), and the Llama3.1-8B decode batch-size sweep (`bs{1..128}_ctx1024`). The same traces serve both data patterns. |
| `case_studies/hbm3e_validation/traces/` | `hbm3_random_read_4rpa` (random reads, 4 per activate), the same trace with refresh (`_ref`), and a NOP + refresh idle trace |
| `case_studies/hbm4_case_study/traces/` | Source traces for `tracegen.py`, which writes per-speed-bin traces to `traces/generated/` |
