"""Rate-aware trace generation shared by sweep_hbm4_power.py and plot_hbm3e_vs_hbm4.py.

The provided traces are in DRAM clock cycles, so a fixed trace refreshes more often per unit time at a
higher data rate. To keep refresh at tREFI = 3.9 us and the tRFC stall at 350 ns for every speed bin,
both traces are rebuilt per rate:
  * read trace  -- the random-read trace plus a REFA at every k*nREFI (in the output timeline) and an
                   nRFC stall after each (all later commands shift by nRFC); this mirrors
                   the HBM3E validation trace hbm3_random_read_4rpa_ref.csv, which it reproduces exactly at 625 ps.
  * idle trace  -- a REFA every nREFI with NOP fillers in between, 16 tREFI long.
"""
import csv, math, os

TREFI_NS, TRFC_NS = 3900.0, 350.0

def n_cyc(ns, tck_ns):
    return max(1, math.ceil(ns / tck_ns))

HDR = ["timestamp", "command", "channel_id", "pseudochannel_id", "bankgroup_id", "bank_id", "row_id", "column_id"]

SRC_BANKS_PER_BG = 4                   # the source traces address 4 bank groups x 4 banks

def write_read_trace(src, dst, nREFI, nRFC, banks_per_bg=SRC_BANKS_PER_BG):
    """banks_per_bg remaps each bank's flat index (bg x 4 + bank) onto the target organization,
    e.g. 8 for HBM4's 2 bank groups x 8 banks, so the trace touches the same 16 banks."""
    rows = list(csv.DictReader(open(src)))
    out, k = [], 0                     # k = refreshes inserted so far
    n_rd = 0
    for r in rows:
        t = int(r["timestamp"])
        while t + k * nRFC >= (k + 1) * nREFI:
            out.append([(k + 1) * nREFI, "REFA", 0, 0, 0, 0, 0, 0]); k += 1
        flat = int(r["bankgroup_id"]) * SRC_BANKS_PER_BG + int(r["bank_id"])
        out.append([t + k * nRFC, r["command"], r["channel_id"], r["pseudochannel_id"],
                    flat // banks_per_bg, flat % banks_per_bg, r["row_id"], r["column_id"]])
        n_rd += r["command"] == "RD"
    with open(dst, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(out)
    return dict(n_rd=n_rd, n_ref=k, end_cycle=out[-1][0])

def write_idle_trace(dst, nREFI, n_refi=16, fill=5):
    out = []
    for i in range(n_refi):
        out.append([i * nREFI, "REFA", 0, 0, 0, 0, 0, 0])
        for j in range(1, fill):
            out.append([i * nREFI + j * nREFI // fill, "NOP", 0, 0, 0, 0, 0, 0])
    out.append([n_refi * nREFI, "END", 0, 0, 0, 0, 0, 0])
    with open(dst, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(out)
    return dict(end_cycle=n_refi * nREFI)
