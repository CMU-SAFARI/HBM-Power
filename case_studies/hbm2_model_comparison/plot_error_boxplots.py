#!/usr/bin/env python3
"""Companion to Fig. 20 (not in the paper): prediction error of every evaluated power model, both data patterns.

Same layout as the single-model figure: microbenchmarks (left) and the LLaMa3.1-8B decode
batch-size sweep (right) on the x-axis. For every workload there is one box per
(model, data pattern): the distribution over tested HBM2 chips of the signed error
(predicted - measured) / measured. White dot = mean. Ground truth: all-0s 35 chips per
microbenchmark and 18 for LLaMa; random 18 chips. Inputs: configs/, traces/, data/.
"""
import os, re, csv, subprocess, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
TA = HERE; CFG = os.path.join(HERE, "configs")
RUNNER = os.path.join(REPO, "build", "bin", "HBM2_runner")
ORG = os.path.join(CFG, "HBM2_organization.json"); TIM = os.path.join(CFG, "HBM2_1.2Gbps_timing_BL4.json")
OUT = os.path.join(HERE, "model_comparison_error_models")

MICRO = [("act_hammer", "Hammer", "act_hammer", "act_hammer_rand"),
         ("interleaved", "Interleaved", "interleaved", "interleaved_rand"),
         ("streaming_all_banks", "Streaming\nAll Banks", "streaming_all_banks", "streaming_all_banks_4cyc_rand"),
         ("streaming_reads", "Streaming\nOne BG", "streaming_reads", "streaming_reads_4cyc_rand"),
         ("wr_rd_turnaround", "Read/Write", "wr_rd_turnaround", "wr_rd_turnaround_4cyc_rand"),
         ("ws_bg0_bg2", "Read BG0-BG2", "ws_bg0_bg2", "ws_bg0_bg2_rand")]
BATCH = [1, 2, 4, 8, 16, 32, 64, 128]
LLAMA = [(f"bs{b}_ctx1024", str(b), f"llama8Bshort_runner_pc0_bs{b}_ctx1024_data_zeros",
          f"llama8Bshort_runner_pc0_bs{b}_ctx1024_data_rand1") for b in BATCH]
MODELS = [("DRAMSim3", "IDD_dramsim3.json", "IDD_dramsim3.json"),
          ("FGDRAM HBM2", "IDD_oconnor_notoggle.json", "IDD_oconnor_50toggle.json"),
          ("Ayna", "IDD_ours_allzeros.json", "IDD_ours_random.json")]
PATTERNS = ["all-0s", "random"]


def run_power(cfg, stem):
    out = subprocess.run([RUNNER, ORG, TIM, os.path.join(CFG, cfg), os.path.join(TA, "traces", f"{stem}.csv")],
                         capture_output=True, text=True)
    m = re.search(r"Average power:\s+([-\d.]+)\s+mW", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"no power for {cfg} / {stem}:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def load_gt(path):
    gt = defaultdict(list)
    for row in csv.DictReader(open(path)):
        p = float(row["power_vdd_avg"])
        if p > 0:
            gt[row["test_name"]].append(p)
    # LLaMA rows: the artifact's LLaMA ground truth was taken with both HBM2 stacks active;
    # replace them with the single-chip
    # re-measurement (17 chips, zeros and random), keeping the microbenchmark rows as they are.
    llama_new = os.path.join(HERE, "data", "llama_single_chip",
                             "llama_ground_truth_random.csv" if "random" in os.path.basename(path) else "llama_ground_truth_allzeros.csv")
    if os.path.exists(llama_new):
        for k in [k for k in gt if k.startswith("llama")]:
            del gt[k]
        for row in csv.DictReader(open(llama_new)):
            p = float(row["power_vdd_avg"])
            if p > 0:
                gt[row["test_name"]].append(p)
    return gt


def main():
    if not os.path.exists(RUNNER):
        sys.exit(f"HBM2_runner not found at {RUNNER}; build it first (see README.md)")
    gt = {"all-0s": load_gt(os.path.join(TA, "data", "ground_truth_allzeros.csv")),
          "random": load_gt(os.path.join(TA, "data", "ground_truth_random.csv"))}
    rows = []
    for label, cfg0, cfg1 in MODELS:
        for pat, cfg, col in (("all-0s", cfg0, 2), ("random", cfg1, 3)):
            for panel, wl in (("Microbenchmarks", MICRO), ("LLaMa3.1-8B Decode", LLAMA)):
                for stem, xlabel, *tn in wl:
                    pred = run_power(cfg, stem)
                    for x in gt[pat][tn[col - 2]]:
                        rows.append(dict(panel=panel, workload=xlabel, series=f"{label}, {pat}",
                                         model=label, pattern=pat, pct=(pred - x) / x * 100.0))
    df = pd.DataFrame(rows)
    summ = (df.assign(ape=df.pct.abs()).groupby(["panel", "model", "pattern", "workload"])
              .agg(mean_err=("pct", "mean"), mape=("ape", "mean")))
    print(summ.groupby(["panel", "model", "pattern"]).mape.agg(["min", "max", "mean"]).round(1).to_string())

    # box order within a workload: the three all-0s boxes, then the three random boxes;
    # Ayna last in each triple so it sits next to the group boundary. Ayna in red shades,
    # the baselines in blue (DRAMSim3) and green (FGDRAM); light = all-0s, dark = random.
    order = [f"{m}, {p}" for p in PATTERNS for m in ["DRAMSim3", "FGDRAM HBM2", "Ayna"]]
    hue = {"DRAMSim3": "#1f77b4", "FGDRAM HBM2": "#2ca02c", "Ayna": "#d62728"}
    colors = {f"{m}, all-0s": sns.set_hls_values(hue[m], l=0.72) for m in hue}
    colors.update({f"{m}, random": sns.set_hls_values(hue[m], l=0.38) for m in hue})

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), sharey=True, constrained_layout=True,
                             gridspec_kw={"width_ratios": [6, 8]})
    for ax, panel, wl in zip(axes, ["Microbenchmarks", "LLaMa3.1-8B Decode"], [MICRO, LLAMA]):
        sub = df[df.panel == panel]
        sns.boxplot(data=sub, x="workload", y="pct", hue="series", order=[w[1] for w in wl], hue_order=order,
                    palette=colors, ax=ax, width=0.85, linewidth=0.9, fliersize=1.8, showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4})
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        for i in range(len(wl) - 1):
            ax.axvline(i + 0.5, color="gray", linewidth=0.6, alpha=0.6)
        ax.set_title(panel, fontsize=17)
        ax.set_xlabel("Decode Batch Size" if panel.startswith("LLaMa") else "", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.6); ax.set_axisbelow(True)
        ax.legend_.remove()
    axes[0].set_ylabel("Prediction Error (%)", fontsize=16)
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="center")
    handles = [Patch(facecolor=colors[k], edgecolor="black", label=k) for k in order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=6, fontsize=12, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
