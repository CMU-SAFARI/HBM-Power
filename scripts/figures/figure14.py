#!/usr/bin/env python3
"""Figure 14: calibrated HBM2 power-model prediction error (all-0s) across workloads.

Reproduces figures/model_comparison_error_allzeros.pdf (revision/figureA).

Uses BOTH:
  * the DRAMPower power model (sources/drampower -> HBM2_runner), and
  * empirical per-chip ground truth (data/ground_truth_allzeros.csv).

For each workload trace the HBM2_runner is run once with the "Ours" all-0s IDD config;
the predicted average power is compared to each measured chip's VDD power as
error% = (predicted - measured_chip) / measured_chip * 100.
"""
import os
import re, csv, subprocess, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
SRC       = REPO_ROOT / "sources" / "figure14"
ENGINE    = REPO_ROOT / "sources" / "drampower"
BUILD     = ENGINE / "build"
RUNNER    = BUILD / "bin" / "HBM2_runner"
FIG_DIR   = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

CFG    = SRC / "configs"
ORG    = str(CFG / "HBM2_organization.json")
TIMING = str(CFG / "HBM2_1.2Gbps_timing_BL4.json")
IDD_ZEROS = str(CFG / "IDD_ours_allzeros.json")

YLIM = (-28, 28)

# Left panel: (trace stem, display label, all-0s test_name)
MICRO = [
    ("act_hammer",          "Hammer",                "act_hammer"),
    ("interleaved",         "Interleaved",           "interleaved"),
    ("streaming_all_banks", "Streaming\nAll Banks",  "streaming_all_banks"),
    ("streaming_reads",     "Streaming\nOne Bank",   "streaming_reads"),
    ("wr_rd_turnaround",    "Read/Write",            "wr_rd_turnaround"),
    ("ws_bg0_bg2",          "Read BG0-BG2",          "ws_bg0_bg2"),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def ensure_runner():
    """Build the shared DRAMPower HBM2_runner once if it is not present."""
    if RUNNER.exists():
        return
    print(f"[build] {RUNNER.name} not found; building DRAMPower engine (one-time)...", file=sys.stderr)
    subprocess.run(["cmake", "-S", str(ENGINE), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                    "-DDRAMPOWER_BUILD_TESTS=OFF", "-DDRAMPOWER_BUILD_BENCHMARKS=OFF",
                    "-DDRAMPOWER_BUILD_CLI=ON"], check=True)
    subprocess.run(["cmake", "--build", str(BUILD), "--target", "HBM2_runner", "-j"], check=True)
    if not RUNNER.exists():
        sys.exit(f"build did not produce {RUNNER}")


def run_power(idd_cfg, trace):
    out = subprocess.run([str(RUNNER), ORG, TIMING, idd_cfg, trace], capture_output=True, text=True)
    m = re.search(r"Average power:\s+([-\d.]+)\s+mW", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"no power for {trace}:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def load_gt(path):
    gt = defaultdict(list)
    for row in csv.DictReader(open(path)):
        p = float(row["power_vdd_avg"])
        if p > 0:
            gt[row["test_name"]].append(p)
    return gt


def collect(workloads, gt_zeros, xkey):
    """workloads: list of (stem, xvalue, tn_zeros) -> rows dict for seaborn (all-0s only)."""
    rows = {xkey: [], "pct": []}
    for stem, xval, tn_z in workloads:
        trace = str(SRC / "traces" / f"{stem}.csv")
        pred = run_power(IDD_ZEROS, trace)
        meas = gt_zeros[tn_z]
        for x in meas:
            rows[xkey].append(xval)
            rows["pct"].append((pred - x) / x * 100.0)
        errs = [abs(pred - x) / x * 100 for x in meas]
        mape = sum(errs) / len(errs) if errs else float("nan")
        print(f"{str(xval).replace(chr(10),' '):20} All-0s pred={pred:7.1f} mW  "
              f"n={len(meas):3}  MAPE={mape:4.1f}%")
    return rows


def main():
    ensure_runner()
    gt_zeros = load_gt(DATA_DIR / "ground_truth_allzeros.csv")

    print("== microbenchmarks (left) ==")
    micro_wl = [(stem, label, tn_z) for stem, label, tn_z in MICRO]
    rows_micro = collect(micro_wl, gt_zeros, "workload")
    micro_order = [label for _, label, _ in MICRO]

    print("== LLaMa batch sweep (right) ==")
    batch_wl = [(f"bs{bs}_ctx1024", bs,
                 f"llama8Bshort_runner_pc0_bs{bs}_ctx1024_data_zeros") for bs in BATCH_SIZES]
    rows_batch = collect(batch_wl, gt_zeros, "batch")

    # ---- plot: two subplots sharing y; left larger than right. Single all-0s box per category. ----
    color = sns.color_palette("husl", 2)[0]
    box = dict(color=color, width=0.55, linewidth=1.1, fliersize=2.5, showmeans=True,
               meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black",
                              markeredgewidth=1.0, markersize=5, zorder=3))

    fig = plt.figure(figsize=(12, 3.6), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1.45, 1.0], wspace=0.06, figure=fig)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1], sharey=axL)

    sns.boxplot(data=rows_micro, x="workload", y="pct", order=micro_order, ax=axL, **box)
    sns.boxplot(data=rows_batch, x="batch", y="pct", order=BATCH_SIZES, ax=axR, **box)

    for ax in (axL, axR):
        ax.axhline(0.0, color="k", ls="--", lw=1, alpha=0.8)
        ax.grid(axis="y", ls="--", alpha=0.9)
        ax.set_ylim(*YLIM)
    axL.set_ylim(*YLIM)

    axL.set_title("Microbenchmarks", fontsize=17)
    plt.setp(axL.get_xticklabels(), rotation=25, ha="center")
    axL.tick_params(axis="x", labelsize=14)
    axL.tick_params(axis="y", labelsize=14)
    axL.set_ylabel("Prediction Error (%)", fontsize=16)
    axL.set_xlabel("")

    axR.set_title("LLaMa3.1-8B Decode", fontsize=17)
    axR.tick_params(axis="x", labelsize=14)
    axR.set_xlabel("Decode Batch Size", fontsize=16)
    axR.set_ylabel("")
    plt.setp(axR.get_yticklabels(), visible=False)

    for ax in (axL, axR):
        if ax.get_legend():
            ax.get_legend().remove()

    out = FIG_DIR / f"{Path(__file__).stem}.pdf"
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
