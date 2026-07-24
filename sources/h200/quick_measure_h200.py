#!/usr/bin/env python3
"""Quick H200 (HBM3E) power measurement: ONLY random-data read power + idle baseline.

A fast (~30-40 s) subset of run_tuning_sweep.sh, building on the same pieces:
  * hbm_bw_3.cu        -- the CUDA stress kernel (built with nvcc if needed)
  * log_power.py       -- the 1 Hz nvidia-smi mem-power logger
  * summarize_tuning.py-- reused steady_power()/parse_stdout() parsers

Steps:
  1. idle baseline       -- log mem power for a few seconds with the GPU idle
  2. random-data read    -- run `hbm_bw ... read rand` under power logging
  3. report              -- idle mem W, random-read mem W, active (read - idle),
                            achieved read GB/s; written to <outdir>/quick_summary.csv

Usage:
    ./quick_measure_h200.py                       # defaults: ~25 s read + 6 s idle
    ./quick_measure_h200.py --stress-sec 30 --gib 16
"""
import argparse
import csv
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from summarize_tuning import steady_power, parse_stdout   # reuse the proven parsers


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=SCRIPT_DIR, **kw)


def idle_mem_w(csv_path):
    """Mean mem power over a pure-idle log (skip the first sample as warm-up)."""
    vals = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["mem_avg_w"]))
            except (ValueError, KeyError):
                pass
    vals = vals[1:] or vals
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="hbm_bw_3.cu")
    ap.add_argument("--bin", default="./hbm_bw")
    ap.add_argument("--arch", default="sm_90", help="H200 = Hopper = sm_90")
    ap.add_argument("--gib", type=float, default=8.0, help="GiB per buffer (x3); must be >> L2")
    ap.add_argument("--iters", type=int, default=20, help="STREAM best-of iters (bandwidth header)")
    ap.add_argument("--stress-sec", type=int, default=25, help="sustained random-read seconds")
    ap.add_argument("--idle-sec", type=float, default=6.0, help="idle baseline seconds")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5, help="power sample interval (s)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    for tool in ("nvcc", "nvidia-smi"):
        if subprocess.run(["bash", "-lc", f"command -v {tool}"], capture_output=True).returncode:
            sys.exit(f"ERROR: {tool} not found on PATH")

    outdir = args.outdir or os.path.join(SCRIPT_DIR, time.strftime("quick_%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    binpath = os.path.join(SCRIPT_DIR, args.bin) if not os.path.isabs(args.bin) else args.bin

    # 1. build if needed
    if not (os.path.exists(binpath) and os.access(binpath, os.X_OK)):
        print(f">> building {args.src} (arch={args.arch})")
        if sh(["nvcc", "-O3", "-arch", args.arch, "-o", binpath, args.src]).returncode:
            sys.exit("ERROR: nvcc build failed")

    # 2. idle baseline (bounded by --duration; GPU left idle)
    idle_csv = os.path.join(outdir, "idle.csv")
    print(f">> idle baseline ({args.idle_sec:g}s)")
    sh(["python3", "log_power.py", "--gpu", str(args.gpu), "--interval", str(args.interval),
        "--duration", str(args.idle_sec), "--out", idle_csv])

    # 3. random-data read under power logging (background logger + kill, like the sweep)
    read_csv = os.path.join(outdir, "read_rand.csv")
    read_log = os.path.join(outdir, "read_rand.stdout.txt")
    print(f">> random read ({args.stress_sec}s stress, {args.gib:g} GiB/buf)")
    logger = subprocess.Popen(
        ["python3", "log_power.py", "--gpu", str(args.gpu), "--interval", str(args.interval),
         "--out", read_csv], cwd=SCRIPT_DIR)
    try:
        time.sleep(2)  # a couple of idle samples before load
        with open(read_log, "w") as lf:
            sh([binpath, str(args.gib), str(args.iters), str(args.stress_sec), "read", "rand"],
               stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(2)  # capture the power drop after load
    finally:
        logger.terminate()
        try:
            logger.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.kill()

    # 4. parse + report
    idle = idle_mem_w(idle_csv)
    pw = steady_power(read_csv) or {}
    read_w = pw.get("mem_w_steady", float("nan"))
    bw, sustained = parse_stdout(read_log)
    read_gbs = bw.get("Read", sustained or "")
    active = read_w - idle

    summary = os.path.join(outdir, "quick_summary.csv")
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mem_power_W", "read_GBs"])
        w.writerow(["idle_baseline", f"{idle:.2f}", ""])
        w.writerow(["random_read_total", f"{read_w:.2f}", read_gbs])
        w.writerow(["random_read_active(read-idle)", f"{active:.2f}", ""])

    print("\n================ H200 quick measurement ================")
    print(f"  idle (baseline) mem power : {idle:7.2f} W")
    print(f"  random-read mem power     : {read_w:7.2f} W   (read {read_gbs} GB/s)")
    print(f"  active (random - idle)    : {active:7.2f} W")
    print(f"  n_steady samples          : {pw.get('n_steady', '?')}")
    print(f"\n  artifacts: {outdir}/  (idle.csv, read_rand.csv, read_rand.stdout.txt, quick_summary.csv)")


if __name__ == "__main__":
    main()
