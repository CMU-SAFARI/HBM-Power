#!/usr/bin/env python3
"""Sample H200 power rails via `nvidia-smi -q -d POWER` into a CSV.

Reproduces the column layout of the existing tuning CSVs so the output drops
straight into the same analysis:

    timestamp,elapsed_s,gpu_index,bus_id,board_avg_w,board_inst_w,mem_avg_w

  board_avg_w / board_inst_w  <- "GPU Power Readings" block
  mem_avg_w                   <- "GPU Memory Power Readings" block  (the rail we tune to)

Runs until --duration seconds elapse, or (if --duration 0, the default) until it
receives SIGINT/SIGTERM. Every row is flushed immediately so a kill never loses data.

Standalone use:
    python3 log_power.py --gpu 0 --interval 1.0 --out run.csv          # until Ctrl-C
    python3 log_power.py --gpu 0 --interval 1.0 --duration 120 --out run.csv
"""
import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# `-q` section headers (indented in the raw output; matched after .strip()).
SECTIONS = {
    "GPU Power Readings": "board",
    "GPU Memory Power Readings": "mem",
    "Module Power Readings": "module",
}

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def _num(v):
    """'80.37 W' / '80.37' -> '80.37'; 'N/A' or junk -> '' (blank cell)."""
    v = v.replace("W", "").strip()
    try:
        return "{:.2f}".format(float(v))
    except ValueError:
        return ""


def gpu_identity(gpu_id):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,gpu_bus_id",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    idx, _, bus = out.partition(",")
    return idx.strip(), bus.strip()


def sample(gpu_id):
    """One reading: (board_avg_w, board_inst_w, mem_avg_w) as formatted strings."""
    out = subprocess.run(
        ["nvidia-smi", "-q", "-d", "POWER", "-i", str(gpu_id)],
        capture_output=True, text=True, check=True,
    ).stdout
    section = None
    board_avg = board_inst = mem_avg = ""
    for raw in out.splitlines():
        line = raw.strip()
        if line in SECTIONS:          # section header (no value)
            section = SECTIONS[line]
            continue
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if section == "board":
            if key == "Average Power Draw":
                board_avg = _num(val)
            elif key == "Instantaneous Power Draw":
                board_inst = _num(val)
        elif section == "mem":
            if key == "Average Power Draw":
                mem_avg = _num(val)
    return board_avg, board_inst, mem_avg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default 0)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to log; 0 = until SIGINT/SIGTERM (default)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    idx, bus = gpu_identity(args.gpu)
    sys.stderr.write("[log_power] gpu {} ({}) -> {} every {}s\n".format(
        idx, bus, args.out, args.interval))
    sys.stderr.flush()

    with open(args.out, "w", buffering=1) as f:  # line-buffered
        f.write("timestamp,elapsed_s,gpu_index,bus_id,board_avg_w,board_inst_w,mem_avg_w\n")
        t0 = time.monotonic()
        next_t = t0
        while not _stop:
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            elapsed = time.monotonic() - t0
            try:
                board_avg, board_inst, mem_avg = sample(args.gpu)
            except subprocess.CalledProcessError as e:
                sys.stderr.write("[log_power] nvidia-smi failed: {}\n".format(e))
                board_avg = board_inst = mem_avg = ""
            f.write("{},{:.3f},{},{},{},{},{}\n".format(
                ts, elapsed, idx, bus, board_avg, board_inst, mem_avg))
            if args.duration and elapsed >= args.duration:
                break
            next_t += args.interval
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                next_t = time.monotonic()  # fell behind; resync
    sys.stderr.write("[log_power] stopped\n")


if __name__ == "__main__":
    main()
