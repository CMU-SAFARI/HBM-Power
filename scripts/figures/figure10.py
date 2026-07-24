#!/usr/bin/env python3
"""Figure 10: Current drawn by bank group pairs normalized to the BG1-BG3 pair.

Reproduces figures/bank_group_normalized.pdf (results/section5-2.ipynb).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
FIG_DIR = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

# --- Load and prepare bank-group data (section5-2 cell 1) ---
df_bg = pd.read_csv(DATA_DIR / "bank_group_measurements.csv")

# deduct per-FPGA (2x chips) no-HBM baseline IDD
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")

# df_no_hbm has only even chip_ids; duplicate for odd chip_ids
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    dup = chip_data.copy()
    dup["chip_id"] = chip_id + 1
    duplicated_rows.append(dup)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()

# subtract baseline and filter to chips with baseline data
mask = df_bg["chip_id"].isin(avg_idd_no_hbm.index)
df_bg = df_bg[mask].copy()
df_bg["idd"] = df_bg.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)


def preprocess(df, group_cols):
    """Take top 40 samples per group, compute mean and std of idd."""
    top = (
        df.sort_values("sample_id", ascending=False)
        .groupby(group_cols, group_keys=False)
        .head(40)
    )
    avg = (
        top.groupby(group_cols)["idd"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "idd", "std": "idd_std"})
        .reset_index()
    )
    return avg


bg_agg = preprocess(df_bg, ["chip_id", "bankgroup_0", "bankgroup_1"])

# --- Plot: IDD normalized to BG1-BG3 (section5-2 cell 6) ---
bg_ref = bg_agg.copy()
bg_ref["bg_pair"] = bg_ref.apply(lambda r: f"BG{int(r['bankgroup_0'])}-BG{int(r['bankgroup_1'])}", axis=1)
# per-chip IDD at BG1-BG3
bg13_idd = bg_ref[bg_ref["bg_pair"] == "BG1-BG3"].set_index("chip_id")["idd"]
bg_ref = bg_ref[bg_ref["chip_id"].isin(bg13_idd.index)].copy()
bg_ref["idd_normalized"] = bg_ref.apply(lambda r: r["idd"] / bg13_idd[r["chip_id"]], axis=1)

bg_pairs_sorted = sorted(bg_ref["bg_pair"].unique())

plt.figure(figsize=(8, 3))
ax = sns.boxplot(
    data=bg_ref, x="bg_pair", y="idd_normalized",
    order=bg_pairs_sorted,
    palette=sns.color_palette("husl", len(bg_pairs_sorted)),
    linewidth=1.5, fliersize=4,
    showmeans=True,
    showfliers=False,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
)
ax.set_xlabel("Active Bank Group Pair", fontsize=14)
ax.set_ylabel("IDD Normalized\nto BG1-BG3", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.axhline(1, color="gray", linestyle="-", linewidth=0.8)
plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf")

for pair in bg_pairs_sorted:
    s = bg_ref[bg_ref["bg_pair"] == pair]["idd_normalized"]
    print(f"{pair}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")

# --- Numbers reported in the paper (Figure 10, Sec. 5.2) ---
# Paper: accessing bank groups 2 and 3 (BG2-BG3) consumes 8.1% higher current than
#        accessing bank groups 1 and 3 (BG1-BG3), on average across all tested stacks.
print("\n--- Numbers reported in the paper (Figure 10) ---")
print("Average current per bank-group pair relative to BG1-BG3:")
for pair in bg_pairs_sorted:
    m = bg_ref[bg_ref["bg_pair"] == pair]["idd_normalized"].mean()
    print(f"  {pair}: {(m - 1) * 100:+.1f}% vs BG1-BG3")
