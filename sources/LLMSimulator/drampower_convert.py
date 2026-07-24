#!/usr/bin/env python3
"""Convert a raw LLMSimulator HBM2 channel trace into the DRAMPower 8-column
format used by the HBM-Power figure14 reference traces.

Raw input  (cmd_hbm2_40gb.log.ch0): 13 cols, comma-space:
    clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr, pim_cmd, operand, req_type
Output (DRAMPower / figure14 format): 8 cols, plain comma, CRLF:
    timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id

Transforms:
  * keep only pseudochannel 0 (the per-pseudochannel unit the power model uses),
  * drop rank / addr / pim_cmd / operand / req_type,
  * blank the column field on non-column commands (ACT/PRE),
  * append a trailing all-bank precharge flush: 16 PREs (bankgroup 0..3 x bank
    0..3, row 0) at clks maxclk+4, +8, ... closing every bank at end of trace,
  * CRLF line endings.
"""
import sys

def convert(raw_path, out_path):
    HEADER = "timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id"
    rows, maxclk = [], 0
    open_row = {}                                            # (bg,bank) -> last-ACT row (pc0)
    with open(raw_path) as fi:
        for line in fi:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 13:
                continue
            clk, cmd, ch, pch, _rank, bg, bank, row, col = p[:9]
            if pch != "0":
                continue
            colf = col if cmd in ("RD", "WR") else ""       # column blank on ACT/PRE
            rows.append((clk, cmd, ch, pch, bg, bank, row, colf))
            if cmd == "ACT":
                open_row[(bg, bank)] = row                   # track the open row per bank
            c = int(clk)
            if c > maxclk:
                maxclk = c
    # trailing all-bank precharge flush: close every bank at its open row, clks
    # pc0_maxclk+4, +8, ... (step 4 = pc0 half of the 2-pseudochannel flush).
    idx = 0
    for bg in range(4):
        for bank in range(4):
            idx += 1
            r = open_row.get((str(bg), str(bank)), "0")
            rows.append((str(maxclk + 4 * idx), "PRE", "0", "0", str(bg), str(bank), r, ""))
    with open(out_path, "w", newline="") as fo:      # newline="" -> we control CRLF
        fo.write(HEADER + "\r\n")
        for r in rows:
            fo.write(",".join(r) + "\r\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: drampower_convert.py <raw cmd_hbm2_40gb.log.ch0> <out.csv>")
    convert(sys.argv[1], sys.argv[2])
