#!/usr/bin/env python3
"""Generate a read-stream command trace for one pseudo-channel of an Ayna device configuration.

The trace opens one row in one bank of every bank group, reads each row `--reads-per-act` times
with the reads interleaved across bank groups, precharges the banks, and moves on to the next bank
of each bank group, overlapping each group's activations with the previous group's reads so the
data bus stays busy. Timing comes from the device's timing.json (nRCD, nRAS, nRP, nRC, nCCD_S,
nCCD_L, nRTP, nBL, and nRFC/nREFI for refresh) and the bank organization from organization.json. It is a simple illustrative schedule, not a full timing checker:
constraints the engine does not use (tRRD, tFAW) are not enforced.

    python3 examples/make_read_trace.py config/HBM2_1200MTs > read.csv
    python3 examples/make_read_trace.py config/HBM2_1200MTs --idle > idle.csv
    python3 examples/make_read_trace.py config/HBM3E_6400MTs --refresh -o read_ref.csv

--idle writes a trace of the same duration that issues no reads (only NOPs, and REFA if --refresh),
for the pseudo-channels that are not in use. A summary (reads, duration, data-bus utilization) goes to stderr.
"""
import argparse, json, os, sys

HEADER = "timestamp,command,channel_id,pseudochannel_id,bankgroup_id,bank_id,row_id,column_id"


def load(device_dir):
    with open(os.path.join(device_dir, "timing.json")) as f:
        t = json.load(f)["timing"]
    with open(os.path.join(device_dir, "organization.json")) as f:
        o = json.load(f)
    return t, o


def schedule(t, o, activations, reads_per_act, refresh):
    """Return the list of (timestamp, command, bankgroup, bank, row, column) and the end time.

    Banks are opened in groups of one bank per bank group. The reads of a group are interleaved
    across its bank groups, nCCD_S apart, so consecutive reads always change bank group and the
    data bus is busy every cycle that nCCD_S and the burst length allow.
    """
    n_bg, n_b = o["bankgroups_per_pseudochannel"], o["banks_per_bankgroup"]
    t_burst = t["nBL"] // o["dataRate"]
    gap = max(t["nCCD_S"], t_burst)                     # cycles between reads to different bank groups
    if n_bg * gap < t["nCCD_L"]:
        gap = -(-t["nCCD_L"] // n_bg)                   # keep same-bank-group reads nCCD_L apart
    rd_to_pre = max(t_burst, t.get("nRTP", 0))
    n_rfc, n_refi = t.get("nRFC", 0), t.get("nREFI", 0)
    if refresh and not (n_rfc and n_refi):
        sys.exit("--refresh needs nRFC and nREFI in timing.json")

    cmds = []
    bank_free = {}                   # (bankgroup, bank) -> earliest next ACT (PRE + nRP, ACT + nRC)
    all_closed = 0                   # when every bank opened so far is precharged again
    next_ref = n_refi if refresh else None
    t0 = 0                           # ACT time of the current group's first bank
    for g in range(-(-activations // n_bg)):
        if refresh and t0 >= next_ref:
            t_ref = max(t0, all_closed)
            cmds.append((t_ref, "REFA", 0, 0, 0, 0))
            t0 = t_ref + n_rfc
            next_ref += n_refi
        bank, row = g % n_b, g // n_b
        t0 = max([t0] + [bank_free.get((bg, bank), 0) - bg * gap for bg in range(n_bg)])
        for bg in range(n_bg):
            t_act = t0 + bg * gap
            cmds.append((t_act, "ACT", bg, bank, row, ""))
            reads = [t_act + t["nRCD"] + k * n_bg * gap for k in range(reads_per_act)]
            for k, t_rd in enumerate(reads):
                cmds.append((t_rd, "RD", bg, bank, row, k * o["dataRate"]))
            t_pre = max(t_act + t["nRAS"], reads[-1] + rd_to_pre)
            cmds.append((t_pre, "PRE", bg, bank, 0, ""))
            bank_free[(bg, bank)] = max(t_pre + t["nRP"], t_act + t["nRC"])
            all_closed = max(all_closed, t_pre + t["nRP"])
        t0 += n_bg * reads_per_act * gap                # next group's first read follows this one's last
    end = max(all_closed, max(c[0] for c in cmds)) + 1
    cmds.sort(key=lambda c: c[0])
    return cmds, end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device", help="device configuration directory, e.g. config/HBM2_1200MTs")
    ap.add_argument("--activations", type=int, default=2048, help="number of row activations, rounded up to a multiple of the bank groups (default 2048)")
    ap.add_argument("--reads-per-act", type=int, default=4, help="reads per activation (default 4)")
    ap.add_argument("--refresh", action="store_true", help="issue REFA every nREFI with an nRFC stall")
    ap.add_argument("--idle", action="store_true", help="same duration, no reads (NOPs, and REFA with --refresh)")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    a = ap.parse_args()

    t, o = load(a.device)
    cmds, end = schedule(t, o, a.activations, a.reads_per_act, a.refresh)
    reads = sum(1 for c in cmds if c[1] == "RD")
    if a.idle:
        n_refi = t.get("nREFI", 0)
        cmds = [(k * n_refi, "REFA", 0, 0, 0, 0) for k in range(1, end // n_refi + 1)] if a.refresh else []
        cmds = [(0, "NOP", 0, 0, 0, "")] + cmds
    cmds.append((end, "NOP", 0, 0, 0, ""))

    out = open(a.output, "w") if a.output else sys.stdout
    out.write(HEADER + "\n")
    for ts, cmd, bg, bk, row, col in cmds:
        out.write(f"{ts},{cmd},0,0,{bg},{bk},{row},{col}\n")
    if a.output:
        out.close()
    ns = end * t["tCK_ps"] / 1000
    if a.idle:
        print(f"idle trace: {end} cycles ({ns:.1f} ns)", file=sys.stderr)
    else:
        busy = reads * (t["nBL"] // o["dataRate"]) / end
        print(f"read trace: {reads} reads over {end} cycles ({ns:.1f} ns), data bus {busy:.0%} busy",
              file=sys.stderr)


if __name__ == "__main__":
    main()
