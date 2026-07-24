#!/usr/bin/env python3
"""Figure 13: Current vs data transmitted by one HBM2 DQ signal over a burst.

Reproduces figures/idd_vs_beat_pattern.pdf (results/section5-3.ipynb).
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

# --- Load and prepare beat-pattern data (section5-3 cell 4) ---
df_bp = pd.read_csv(DATA_DIR / "beat_pattern_combined_measurements.csv")

# convert beat_pattern to zero-padded binary strings for display
df_bp["beat_pattern_str"] = df_bp["beat_pattern"].astype(str).str.zfill(4)

# filter out beat patterns longer than 4 bits
df_bp = df_bp[df_bp["beat_pattern_str"].str.len() <= 4].copy()

# cleaning: top 40 samples per group, average idd
group_cols = ["chip_id", "test_loop", "beat_pattern", "beat_pattern_str", "col1_inverted"]
top40_bp = (
    df_bp.sort_values("sample_id", ascending=False)
    .groupby(group_cols, group_keys=False)
    .head(40)
)
avg_idd_bp = top40_bp.groupby(group_cols)["idd"].mean().rename("idd")
std_dev_idd_bp = top40_bp.groupby(group_cols)["idd"].std().rename("idd_std_dev")
df_bp = (
    top40_bp.sort_values("sample_id", ascending=False)
    .groupby(group_cols)
    .first()
    .drop(columns=["idd"])
    .join(avg_idd_bp)
    .join(std_dev_idd_bp)
    .reset_index()
)

# deduct no-HBM power
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    duplicated_chip_data = chip_data.copy()
    duplicated_chip_data["chip_id"] = chip_id + 1
    duplicated_rows.append(duplicated_chip_data)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

df_bp = df_bp[df_bp["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df_bp["idd"] = df_bp.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

# remove anomalously low values
df_bp = df_bp[df_bp["idd"] >= 200].reset_index(drop=True)

# count number of toggles (1s) in the beat pattern
df_bp["num_toggles"] = df_bp["beat_pattern_str"].apply(lambda x: sum(int(c) for c in x))

# --- Plot (section5-3 cell 5) ---
order = sorted(df_bp["beat_pattern_str"].unique(), key=lambda x: int(x))

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for ax, inv_val, title in zip(axes, [0, 1], ["No BG Bus Toggle", "BG Bus Toggle"]):
    subset = df_bp[df_bp["col1_inverted"] == inv_val]
    sns.boxplot(
        data=subset, x="beat_pattern_str", y="idd",
        order=order,
        palette=sns.color_palette("husl", 1),
        linewidth=1.5, fliersize=4,
        showmeans=True,
        showfliers=False,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
        ax=ax,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Beat Pattern", fontsize=14)
    ax.set_ylabel("IDD (mA)" if inv_val == 0 else "", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf")

# reported statistics
for inv_val, title in zip([0, 1], ["No BG Bus Toggle", "BG Bus Toggle"]):
    subset = df_bp[df_bp["col1_inverted"] == inv_val]
    max_idd = subset.groupby("beat_pattern_str")["idd"].mean().max()
    min_idd = subset.groupby("beat_pattern_str")["idd"].mean().min()
    print(f"{title}: Max average IDD = {max_idd:.2f} mA, Min average IDD = {min_idd:.2f} mA")

for inv_val, title in zip([0, 1], ["No BG Bus Toggle", "BG Bus Toggle"]):
    subset = df_bp[df_bp["col1_inverted"] == inv_val]
    avg_idd = subset["idd"].mean()
    print(f"{title}: Average IDD across all beat patterns = {avg_idd:.2f} mA")

bg_toggle_avg = df_bp[df_bp["col1_inverted"] == 1]["idd"].mean()
no_bg_toggle_avg = df_bp[df_bp["col1_inverted"] == 0]["idd"].mean()
percentage_increase_bg = ((bg_toggle_avg - no_bg_toggle_avg) / no_bg_toggle_avg) * 100

# --- Numbers reported in the paper (Figure 13, Sec. 5.3) ---
# Paper: highest/lowest avg current = 2873.6 / 1717.9 mA (no BG toggle) and
#        3556.4 / 2646.9 mA (BG toggle); BG-toggle avg is 29.4% higher than no-toggle;
#        the highest-current patterns are "0110" and "1001".
print("\n--- Numbers reported in the paper (Figure 13) ---")
for inv_val, title in zip([0, 1], ["No BG Bus Toggle", "BG Bus Toggle"]):
    means = (df_bp[df_bp["col1_inverted"] == inv_val]
             .groupby("beat_pattern_str")["idd"].mean().sort_values(ascending=False))
    print(f"{title}: highest avg IDD = {means.iloc[0]:.1f} mA, lowest avg IDD = {means.iloc[-1]:.1f} mA")
    top2 = ", ".join(f"{pat} ({val:.1f} mA)" for pat, val in means.head(2).items())
    print(f"{title}: top-2 highest-current beat patterns: {top2}")
print(f"BG-bus-toggle avg current is {percentage_increase_bg:.1f}% higher than no-BG-bus-toggle")
