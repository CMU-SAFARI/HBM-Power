#!/usr/bin/env python3
"""Figure 9: HBM2 current vs temperature linear-fit slopes across IDD loops.

Reproduces figures/temperature_slope_boxplot.pdf (results/section5-1.ipynb).
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

# --- Load raw per-measurement data, baseline-subtracted (section5-1 cell 9) ---
df_raw = pd.read_csv(DATA_DIR / "all_idd_measurements.csv")
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")

# duplicate no_hbm entries for odd chip_ids (no_hbm has only even chip_ids)
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    dup = chip_data.copy()
    dup["chip_id"] = chip_id + 1
    duplicated_rows.append(dup)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

# keep only chips present in baseline, then subtract baseline per chip
df_raw = df_raw[df_raw["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df_raw["idd"] = df_raw.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

# --- Per-chip temperature slopes + boxplot (section5-1 cell 11) ---
rep_idds = ["IDD2", "IDD3N1", "IDD3N16", "IDD0", "IDD4R", "IDD4W", "IDD5B"]
chip_ids_sorted = sorted(df_raw["chip_id"].unique())

# keep only chip_id/test_loop groups whose temperature slope is non-negative
valid_groups = []
for (cid, loop), g in df_raw[df_raw["test_loop"].isin(rep_idds)].groupby(["chip_id", "test_loop"]):
    temp_counts = g["temperature"].value_counts()
    g = g[g["temperature"].isin(temp_counts[temp_counts >= 10].index)]
    if g["temperature"].nunique() > 1:
        slope = np.polyfit(g["temperature"], g["idd"], 1)[0]
        if slope >= 0:
            valid_groups.append((cid, loop))

valid_groups = pd.DataFrame(valid_groups, columns=["chip_id", "test_loop"])
df_raw = df_raw.merge(valid_groups, on=["chip_id", "test_loop"], how="inner")

slope_records = []
for cid in chip_ids_sorted:
    chip_data = df_raw[df_raw["chip_id"] == cid]
    for idd in rep_idds:
        subset = chip_data[chip_data["test_loop"] == idd]
        # filter temperature bins with < 10 samples
        temp_counts = subset["temperature"].value_counts()
        valid_temps = temp_counts[temp_counts >= 10].index
        subset = subset[subset["temperature"].isin(valid_temps)]
        if len(subset["temperature"].unique()) > 1:
            z = np.polyfit(subset["temperature"], subset["idd"], 1)
            slope_records.append({"chip_id": cid, "IDD": idd, "slope_mA_per_C": z[0]})

df_slopes = pd.DataFrame(slope_records)

# add "Average" category: per-chip mean slope across all IDD loops
avg_per_chip = df_slopes.groupby("chip_id")["slope_mA_per_C"].mean().reset_index()
avg_per_chip["IDD"] = "Average"
df_slopes = pd.concat([df_slopes, avg_per_chip], ignore_index=True)

rep_idds_with_avg = rep_idds + ["Average"]

# --- Print summary statistics ---
print("Temperature sensitivity (slope mA/°C) per IDD across chips:\n")
for idd in rep_idds_with_avg:
    s = df_slopes[df_slopes["IDD"] == idd]["slope_mA_per_C"]
    print(f"{idd}:  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}  std={s.std():.2f}  (n={len(s)} chips)")

# --- Boxplot ---
plt.figure(figsize=(9, 3))
ax = sns.boxplot(
    data=df_slopes, x="IDD", y="slope_mA_per_C",
    order=rep_idds_with_avg, palette=sns.color_palette("husl", len(rep_idds_with_avg)),
    showfliers=False,
    linewidth=1.5, fliersize=4, showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
)

# bold the Average x-tick label
for tick in ax.get_xticklabels():
    if tick.get_text() == "Average":
        tick.set_weight("bold")

ax.set_ylabel("Power vs. Temperature\nSlope (mA/°C)", fontsize=14)
ax.set_xlabel("")
ax.tick_params(axis="both", labelsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf")

# --- Numbers reported in the paper (Figure 9, Sec. 5.1.2) ---
# Paper: "on average across all measurement loops and tested HBM2 stacks, current
#         increases by 6.7 mA every 1 C."
per_group = df_slopes[df_slopes["IDD"] != "Average"]["slope_mA_per_C"]
avg_box = df_slopes[df_slopes["IDD"] == "Average"]["slope_mA_per_C"]
print("\n--- Numbers reported in the paper (Figure 9) ---")
print(f"Mean slope across all (loop, stack) pairs: {per_group.mean():.2f} mA/C")
print(f"Mean of per-stack average slopes (Average box): {avg_box.mean():.2f} mA/C")
