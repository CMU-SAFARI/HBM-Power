#!/usr/bin/env python3
"""Fig. 20: measured HBM2 power vs. every evaluated power model.

Two panels as in the single-model figure: the six microbenchmarks (left) and the Llama3.1-8B
decode batch-size sweep (right). Per workload, two boxes give the measured power across
tested chips: all-0s (35 chips per microbenchmark, 18 for Llama) and random (18 chips). The
rightmost group of each panel ("Average") boxes, per data pattern, the distribution across chips of
each chip's power averaged over that panel's workloads; its markers are the mean prediction over
those workloads.
Markers are the model predictions: shape = model (triangle DRAMSim3, square FGDRAM, star
Ayna), face color = data pattern (matching the boxes). Inputs: configs/, traces/ and the
measured ground truth in data/.
"""
import os, re, csv, subprocess, sys
from collections import defaultdict
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np, pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
TA = HERE; CFG = os.path.join(HERE, "configs")
RUNNER = os.path.join(REPO, "build", "bin", "HBM2_runner")
ORG = os.path.join(CFG, "HBM2_organization.json"); TIM = os.path.join(CFG, "HBM2_1.2Gbps_timing_BL4.json")
OUT = os.path.join(HERE, "model_comparison_absolute")

MICRO = [("act_hammer", "Hammer", "act_hammer", "act_hammer_rand"),
         ("interleaved", "Interleaved", "interleaved", "interleaved_rand"),
         ("streaming_all_banks", "Streaming\nAll Banks", "streaming_all_banks", "streaming_all_banks_4cyc_rand"),
         ("streaming_reads", "Streaming\nOne BG", "streaming_reads", "streaming_reads_4cyc_rand"),
         ("wr_rd_turnaround", "Read/Write", "wr_rd_turnaround", "wr_rd_turnaround_4cyc_rand"),
         ("ws_bg0_bg2", "Read BG0-BG2", "ws_bg0_bg2", "ws_bg0_bg2_rand")]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
LLAMA = [(f"bs{b}_ctx1024", str(b), f"llama8Bshort_runner_pc0_bs{b}_ctx1024_data_zeros",
          f"llama8Bshort_runner_pc0_bs{b}_ctx1024_data_rand1") for b in BATCHES]
AVG = "Average"    # extra category per panel: distribution of the per-chip averages over its workloads
MODELS = [("DRAMSim3", "IDD_dramsim3.json", "IDD_dramsim3.json", "^"),
          ("FGDRAM HBM2", "IDD_oconnor_notoggle.json", "IDD_oconnor_50toggle.json", "s"),
          ("Ayna", "IDD_ours_allzeros.json", "IDD_ours_random.json", "*")]
_h = [sns.color_palette("husl", 2)[0], matplotlib.colors.to_rgb("cornflowerblue")]   # Fig. 5 palette: husl pink (all-0s), light blue (random)
PAT = {"all-0s": _h[0], "random": _h[1]}                                              # box fill and marker face
PAT_EDGE = {"all-0s": sns.set_hls_values(_h[0], l=0.3), "random": sns.set_hls_values(_h[1], l=0.3)}   # box line


def run_power(cfg, stem):
    out = subprocess.run([RUNNER, ORG, TIM, os.path.join(CFG, cfg), os.path.join(TA, "traces", f"{stem}.csv")],
                         capture_output=True, text=True)
    m = re.search(r"Average power:\s+([-\d.]+)\s+mW", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"no power for {cfg}/{stem}:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))


def load_gt(path):
    """test_name -> {chip_id: power}; keeping the chip id lets the Average group average per chip."""
    gt = defaultdict(dict)
    for row in csv.DictReader(open(path)):
        p = float(row["power_vdd_avg"])
        if p > 0:
            gt[row["test_name"]][row["chip_id"]] = p
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
                gt[row["test_name"]][row["chip_id"]] = p
    return gt


def per_chip(gt_pat, members, col):
    """One value per chip: its power averaged over `members` (identity for a single-workload group).
    Restricted to the chips measured on every member workload."""
    chips = set.intersection(*[set(gt_pat[m[col]]) for m in members])
    return [float(np.mean([gt_pat[m[col]][c] for m in members])) for c in sorted(chips)]


def main():
    if not os.path.exists(RUNNER):
        sys.exit(f"HBM2_runner not found at {RUNNER}; build it first (see README.md)")
    gt = {"all-0s": load_gt(os.path.join(TA, "data", "ground_truth_allzeros.csv")),
          "random": load_gt(os.path.join(TA, "data", "ground_truth_random.csv"))}
    panels = [("Microbenchmarks", [(w[1], [w]) for w in MICRO] + [(AVG, MICRO)]),
              ("Llama3.1-8B Decode", [(w[1], [w]) for w in LLAMA] + [(AVG, LLAMA)])]
    cats = [(title, label, members) for title, cs in panels for label, members in cs]   # key by panel: "Average" repeats
    meas, pred = {}, {}
    for title, label, members in cats:
        for pat, col in (("all-0s", 2), ("random", 3)):
            meas[(title, label, pat)] = per_chip(gt[pat], members, col)
            for mname, cfg0, cfg1, _ in MODELS:
                cfg = cfg0 if pat == "all-0s" else cfg1
                pred[(title, label, pat, mname)] = np.mean([run_power(cfg, m[0]) for m in members])
    # summary
    print(f"{'workload':18s}{'pattern':8s}{'measured':>9s}{'DRAMSim3':>10s}{'FGDRAM':>9s}{'Ayna':>8s}   (mW; n chips)")
    for title, label, _ in cats:
        for pat in PAT:
            m = np.array(meas[(title, label, pat)])
            print(f"{label.replace(chr(10),' '):18s}{pat:8s}{m.mean():9.0f}" +
                  "".join(f"{pred[(title, label, pat, mn)]:>{w}.0f}" for (mn, *_), w in zip(MODELS, (10, 9, 8))) + f"   n={len(m)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 3.2), sharey=True, constrained_layout=True,
                             gridspec_kw={"width_ratios": [8, 9]})
    W = 0.36
    for ax, (title, pcats) in zip(axes, panels):
        for i, (label, _) in enumerate(pcats):
            for j, pat in enumerate(PAT):
                x = i + (j - 0.5) * (W + 0.06)
                ax.boxplot([meas[(title, label, pat)]], positions=[x], widths=W, patch_artist=True, showfliers=True,
                           showmeans=False, flierprops=dict(marker=".", markersize=3, color=PAT_EDGE[pat]),
                           medianprops=dict(color=PAT_EDGE[pat], linewidth=1.2),
                           boxprops=dict(facecolor=PAT[pat], edgecolor=PAT_EDGE[pat], linewidth=1.0),
                           whiskerprops=dict(color=PAT_EDGE[pat]), capprops=dict(color=PAT_EDGE[pat]))
                offs = np.linspace(-W / 3, W / 3, len(MODELS))
                for (mname, _, _, mk), dx in zip(MODELS, offs):
                    ax.plot(x + dx, pred[(title, label, pat, mname)], marker=mk, markersize=13 if mk == "*" else 9,
                            linestyle="none", markerfacecolor=PAT[pat], markeredgecolor="black",
                            markeredgewidth=1.2, zorder=5)
            if i < len(pcats) - 1:
                sep = pcats[i + 1][0] == AVG
                ax.axvline(i + 0.5, color="gray", linewidth=1.4 if sep else 0.6, alpha=0.9 if sep else 0.6)
        ax.set_xticks(range(len(pcats))); ax.set_xticklabels([c[0] for c in pcats], fontsize=13)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("Decode Batch Size" if title.startswith("Llama") else "", fontsize=16)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.6); ax.set_axisbelow(True); ax.set_ylim(bottom=0)
        ax.set_xlim(-0.6, len(pcats) - 0.4)
    axes[0].set_ylabel("HBM2 Power (mW)", fontsize=16)
    plt.setp(axes[0].get_xticklabels(), rotation=18, ha="center")
    handles = [Patch(facecolor=PAT[p], edgecolor=PAT_EDGE[p], label=f"Measured, {p}") for p in PAT]
    handles += [Line2D([], [], marker=mk, linestyle="none", markersize=13 if mk == "*" else 9, markerfacecolor="white",
                       markeredgecolor="black", markeredgewidth=1.2, label=f"{mn} prediction") for mn, _, _, mk in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.17))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
