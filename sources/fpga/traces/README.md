# Trace inputs for the BRAM trace runner (Fig 14 / Table 3)

The 14 DRAM command traces that `run_trace_sweep.sh` replays on the FPGA with
`HBMTraceRunnerBRAM` (8-column runner format:
`timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id`).

- **`micro/` (6)** — Ramulator command traces of the memory-access
  microbenchmarks (`act_hammer`, `interleaved`, `streaming_reads_4cyc`,
  `streaming_all_banks_4cyc`, `wr_rd_turnaround_4cyc`, `ws_bg0_bg2`), replayed
  verbatim (they are already in runner format).
- **`llm/` (8)** — LLaMa3.1-8B decode, batch 1→128 at context 1024, pseudo-
  channel 0 only. Produced from the raw traces in `raw/` by
  `convert_llama8Bshort.py` (see below).
- **`raw/llama8Bshort/` (8)** — the raw 13-column Ramulator outputs
  (`clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr,
  pim_cmd, operand, req_type`) the `llm/` set was generated from.

## Reproducing `llm/` from `raw/`

```bash
python3 convert_llama8Bshort.py raw/llama8Bshort /tmp/llama_runner
diff -r /tmp/llama_runner_pc0 llm      # byte-identical
```

The converter drops the `rank` + trailing columns, strips Ramulator's
column-padding whitespace (which would otherwise make every command an
unmatched NOP), appends closing PREs so each trace loops cleanly, and emits a
both-PC set plus one set per pseudo-channel; the AE uses the **pc0** set
(chip 0 / `--channels low`). This reproduction has been verified byte-identical
against the shipped `llm/` files.