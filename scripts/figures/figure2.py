#!/usr/bin/env python3
"""Figure 2: Idle HBM2 VDDC current and temperature across tested HBM2 stacks.

Reproduces figures/idle_idd_barplot.pdf (results/section4-3.ipynb).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
FIG_DIR = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

# --- Load and prepare data (section4-3 cell 0) ---
df = pd.read_csv(DATA_DIR / "all_idd_measurements.csv")
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")

# duplicate df_no_hbm entries for each chip_id (no_hbm has only even chip_ids)
# e.g., chip_id 0 -> chip_id 1, chip_id 2 -> chip_id 3, etc.
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    duplicated_chip_data = chip_data.copy()
    duplicated_chip_data["chip_id"] = chip_id + 1
    duplicated_rows.append(duplicated_chip_data)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

# remove chip_ids that do not exist in df_no_hbm from df, then deduct baseline
df = df[df["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df["idd"] = df.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

df = df[df["test_loop"] == "IDD2"]

# for each chip_id keep only rows near its most frequent temperature (+-1)
most_freq_temp = df.groupby("chip_id")["temperature"].agg(lambda x: x.value_counts().idxmax())
df = df[df.apply(lambda row: abs(row["temperature"] - most_freq_temp[row["chip_id"]]) <= 1, axis=1)].reset_index(drop=True)
df["normalized_ipp"] = df.groupby("chip_id")["ipp"].transform(lambda x: x - x.min())
df["normalized_power_vpp_vddq"] = df["normalized_ipp"] * 2.5

# --- Plot (section4-3 cell 2) ---
fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True)

# sort chip_order by average temperature
chip_order = (
    df.groupby("chip_id")["temperature"].mean().sort_values().index.tolist()
)
temperature_data = (
    df.groupby("chip_id", as_index=False)["temperature"]
    .mean()
    .set_index("chip_id")
    .reindex(chip_order)
    .reset_index()
)
temperature_positions = range(len(chip_order))

sns.barplot(
    data=df, x="chip_id", y="idd", order=chip_order, errorbar="sd",
    ax=axes, palette=sns.color_palette("husl", 1),
    linewidth=2, edgecolor="black",
    err_kws={"color": "black"},
    capsize=0.2,
)

temperature_color = "blue"

# average temperature per chip_id on secondary y axis
ax2 = axes.twinx()
ax2.plot(
    temperature_positions,
    temperature_data["temperature"],
    color=temperature_color, marker="o", linewidth=0,
    label="Average Temperature (°C)",
)
ax2.set_ylabel("Average Temperature (°C)", fontsize=14)
ax2.tick_params(axis="y", labelsize=12)
ax2.set_xlim(axes.get_xlim())
ax2.set_ylim(0, 100)

# combined legend for bars and line
bar_patch = mpatches.Patch(
    color=sns.color_palette("husl", 1)[0],
    edgecolor="black",
    linewidth=2,
    label="Idle IDD (mA)",
)
temp_line = Line2D([0], [0], color=temperature_color, marker="o", linewidth=0, markersize=6,
                   label="Average Temperature (°C)")
axes.legend(handles=[bar_patch, temp_line], loc="upper left", fontsize=12)

axes.set_ylabel("Idle IDD (mA)", fontsize=14)
axes.set_xlabel("")
axes.tick_params(axis="y", labelsize=12)
axes.set_ylim(0, 500)
axes.set_xlabel("Tested HBM2 Stack", fontsize=14)
axes.tick_params(axis="x", labelsize=12, rotation=90)
axes.grid(axis="y", linestyle="--", alpha=0.7)

# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf", bbox_inches="tight")

# --- Numbers reported in the paper (Figure 2, Sec. 4.3) ---
# Paper: min/max mean idle current = 217.5 / 364.8 mA; idle temperature range 48-64 C;
#        avg per-stack std of idle current across measurements = 9.6 mA.
print("\n--- Numbers reported in the paper (Figure 2) ---")
avg_idd_by_chip = df.groupby("chip_id")["idd"].mean()
print(f"Minimum average idle IDD: {avg_idd_by_chip.min():.1f} mA")
print(f"Maximum average idle IDD: {avg_idd_by_chip.max():.1f} mA")
avg_temp_by_chip = df.groupby("chip_id")["temperature"].mean()
print(f"Minimum average idle temperature: {avg_temp_by_chip.min():.1f} C")
print(f"Maximum average idle temperature: {avg_temp_by_chip.max():.1f} C")
avg_std_dev = df.groupby("chip_id")["idd"].std().mean()
print(f"Average per-stack std of idle IDD across measurements: {avg_std_dev:.1f} mA")
