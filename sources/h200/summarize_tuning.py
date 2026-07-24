#!/usr/bin/env python3
"""Join per-condition bandwidth (stdout) + steady-state power (CSV) into one table.

    python3 summarize_tuning.py <OUTDIR>

Reads <OUTDIR>/manifest.csv and, for each condition, parses:
  * <label>.stdout.txt -> best-of GB/s for Read/Copy/Scale/Add/Triad and the
    last sustained "avg GB/s" of the stress phase.
  * <label>.csv        -> steady-state mem/board power (mean over the load
    plateau = rows whose mem_avg_w >= 0.9 * max(mem_avg_w)).

Writes <OUTDIR>/tuning_summary.csv and prints a readable table.
"""
import csv
import os
import re
import sys

KERNELS = ("Read", "Write", "Copy", "Scale", "Add", "Triad")
# "  Read       3.456 ms     4500.1 GB/s"
_BW_RE = re.compile(r"^\s*(Read|Write|Copy|Scale|Add|Triad)\s+[\d.]+\s*ms\s+([\d.]+)\s*GB/s", re.I)
# "  t= 12.0s   launches=... avg 4498.7 GB/s"
_AVG_RE = re.compile(r"avg\s+([\d.]+)\s*GB/s", re.I)


def parse_stdout(path):
    bw = {k: "" for k in KERNELS}
    sustained = ""
    if not os.path.exists(path):
        return bw, sustained
    with open(path) as f:
        for line in f:
            m = _BW_RE.match(line)
            if m:
                bw[m.group(1).capitalize()] = m.group(2)
            m = _AVG_RE.search(line)
            if m:
                sustained = m.group(1)   # keep the last one (steady)
    return bw, sustained


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def steady_power(path):
    """Median mem/board over the load plateau. Returns dict or None if no data.

    Selects the clearly-loaded region (mem >= idle + 0.5*(max-idle)) and takes the
    MEDIAN, which is robust to the startup-overshoot spike some write/copy runs show
    (a plain top-decile mean got fooled by that spike -> tiny n_steady, wrong value).
    """
    if not os.path.exists(path):
        return None
    mem, b_avg = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                m = float(row["mem_avg_w"])
            except (KeyError, ValueError):
                continue
            mem.append(m)
            try:
                b_avg.append(float(row["board_avg_w"]))
            except (KeyError, ValueError):
                b_avg.append(float("nan"))
    if not mem:
        return None
    lo, hi = min(mem), max(mem)
    idle = sum(mem[:3]) / min(3, len(mem))      # pre-load baseline (logger starts idle)
    thr = lo + 0.5 * (hi - lo)                  # clearly in the load region
    idx = [i for i, m in enumerate(mem) if m >= thr]
    sel_mem = [mem[i] for i in idx]
    sel_board = [b_avg[i] for i in idx if b_avg[i] == b_avg[i]]  # drop NaN
    return {
        "mem_w_steady": _median(sel_mem),
        "board_w_steady": _median(sel_board) if sel_board else float("nan"),
        "mem_idle": idle,
        "mem_min": lo,
        "mem_max": hi,
        "n_steady": len(idx),
        "n_total": len(mem),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: summarize_tuning.py <OUTDIR>")
    outdir = sys.argv[1]
    manifest = os.path.join(outdir, "manifest.csv")
    if not os.path.exists(manifest):
        sys.exit("no manifest.csv in " + outdir)

    cols = ["label", "kernel", "fill", "altRun", "constHex",
            "bw_read_GBs", "bw_write_GBs", "bw_triad_GBs", "bw_sustained_GBs",
            "mem_w_steady", "board_w_steady", "mem_idle", "mem_min", "mem_max",
            "n_steady", "n_total"]
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            label = r["label"]
            bw, sustained = parse_stdout(os.path.join(outdir, label + ".stdout.txt"))
            pw = steady_power(os.path.join(outdir, label + ".csv")) or {}
            rows.append({
                "label": label,
                "kernel": r["kernel"],
                "fill": r["fill"],
                "altRun": r["altRun"],
                "constHex": r.get("constHex", ""),
                "bw_read_GBs": bw.get("Read", ""),
                "bw_write_GBs": bw.get("Write", ""),
                "bw_triad_GBs": bw.get("Triad", ""),
                "bw_sustained_GBs": sustained,
                "mem_w_steady": _fmt(pw.get("mem_w_steady")),
                "board_w_steady": _fmt(pw.get("board_w_steady")),
                "mem_idle": _fmt(pw.get("mem_idle")),
                "mem_min": _fmt(pw.get("mem_min")),
                "mem_max": _fmt(pw.get("mem_max")),
                "n_steady": pw.get("n_steady", ""),
                "n_total": pw.get("n_total", ""),
            })

    out_csv = os.path.join(outdir, "tuning_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # readable table
    hdr = ["label", "kernel", "fill", "altRun", "constHex",
           "mem_W", "board_W", "bw_sust_GBs", "n"]
    print("  ".join(h.ljust(14) for h in hdr))
    for r in rows:
        line = [
            r["label"], r["kernel"], r["fill"], str(r["altRun"]), r["constHex"],
            r["mem_w_steady"], r["board_w_steady"], r["bw_sustained_GBs"],
            str(r["n_steady"]),
        ]
        print("  ".join(c.ljust(14) for c in line))
    print("\nwrote " + out_csv)


def _fmt(v):
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return ""
    return "{:.2f}".format(v) if isinstance(v, float) else str(v)


if __name__ == "__main__":
    main()
