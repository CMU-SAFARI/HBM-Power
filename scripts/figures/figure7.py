#!/usr/bin/env python3
"""Figure 7: Distribution of measured IDD4R and IDD4W current across stacks.

Reproduces figures/IDD4_combined_histogram.pdf (results/section5-1.ipynb).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
FIG_DIR = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

# --- Load and prepare aggregated IDD data (section5-1 cell 0) ---
df = pd.read_csv(DATA_DIR / "all_idd_measurements.csv")
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")

# duplicate df_no_hbm entries for odd chip_ids (no_hbm has only even chip_ids)
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    duplicated_chip_data = chip_data.copy()
    duplicated_chip_data["chip_id"] = chip_id + 1
    duplicated_rows.append(duplicated_chip_data)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

# keep only chips present in baseline, then deduct baseline per chip
df = df[df["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df["idd"] = df.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

# take top-40 samples per chip_id/test_loop and average idd to remove outliers
top40 = (
    df.sort_values("sample_id", ascending=False)
    .groupby(["chip_id", "test_loop"], group_keys=False)
    .head(40)
)
avg_idd = top40.groupby(["chip_id", "test_loop"])["idd"].mean().rename("idd")
std_dev_idd = top40.groupby(["chip_id", "test_loop"])["idd"].std().rename("idd_std_dev")
df = (
    top40.sort_values("sample_id", ascending=False)
    .groupby(["chip_id", "test_loop"])
    .first()
    .drop(columns=["idd"])
    .join(avg_idd)
    .join(std_dev_idd)
    .reset_index()
)

# --- Plot (section5-1 cell 16) ---
EVEN_COLOR, ODD_COLOR = sns.color_palette("husl", 2)


def parity_hist(ax, loop, xlabel, nbins=10, legend=False):
    sub = df[df["test_loop"] == loop]
    vals = sub["idd"]
    even = sub[sub["chip_id"].astype(int) % 2 == 0]["idd"]
    odd = sub[sub["chip_id"].astype(int) % 2 == 1]["idd"]
    bins = np.linspace(vals.min(), vals.max(), nbins + 1)
    ax.hist([even, odd], bins=bins, stacked=True,
            color=[EVEN_COLOR, ODD_COLOR], edgecolor="black", linewidth=1.2,
            label=["Even-numbered stacks", "Odd-numbered stacks"])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    if legend:
        ax.legend(fontsize=11)


loops = ["IDD4R", "IDD4W"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for i, loop in enumerate(loops):
    parity_hist(axes[i], loop, f"{loop} (mA)", legend=(i == 0))
axes[0].set_ylabel("Number of HBM2 stacks", fontsize=14)
plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf", bbox_inches="tight")

# --- Numbers reported in the paper (Figure 7, Sec. 5.1) ---
# Paper: IDD4R mean/std = 2881.7 / 103.3 mA; IDD4W mean/std = 1982.5 / 126.6 mA;
#        IDD4R is 45.2% higher than IDD4W. Also: avg IDD4R is 9.6x/11.4x/10.7x/8.3x/45.2%/3.1x
#        "higher than" avg IDD2N/IDD3N1/IDD3N16/IDD0/IDD4W/IDD5B, where "Nx higher" == (ratio-1).
# NOTE: across-stack std reproduces the notebook's barplot cell, which appends a
# synthetic "Average" point (= the mean) to the per-stack values before std.
def paper_all_stats(vals):
    with_avg = pd.concat([vals, pd.Series([vals.mean()])], ignore_index=True)
    return with_avg.mean(), with_avg.std()


r = df[df["test_loop"] == "IDD4R"]["idd"]
w = df[df["test_loop"] == "IDD4W"]["idd"]
rm, rs = paper_all_stats(r)
wm, ws = paper_all_stats(w)
print("\n--- Numbers reported in the paper (Figure 7) ---")
print(f"IDD4R all stacks: mean={rm:.1f} mA, std={rs:.1f} mA (n={len(r)})")
print(f"IDD4W all stacks: mean={wm:.1f} mA, std={ws:.1f} mA (n={len(w)})")
print(f"IDD4R is {(rm - wm) / wm * 100:.1f}% higher than IDD4W")
means = df.groupby("test_loop")["idd"].mean()
print("Average IDD4R relative to other loops ('Nx higher' = mean_IDD4R/mean_loop - 1):")
for loop in ["IDD2", "IDD3N1", "IDD3N16", "IDD0", "IDD4W", "IDD5B"]:
    factor = means["IDD4R"] / means[loop] - 1
    print(f"  vs {loop:<7}: {factor:.1f}x higher ({factor * 100:.1f}%)")
