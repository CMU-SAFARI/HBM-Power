#!/usr/bin/env python3
"""Figure 3: Distribution of mean idle HBM2 VDDC current and temperature.

Reproduces figures/idle_idd_histogram.pdf (results/section4-3.ipynb).
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

# --- Load and prepare data (section4-3 cell 0) ---
df = pd.read_csv(DATA_DIR / "all_idd_measurements.csv")
df_no_hbm = pd.read_csv(DATA_DIR / "no_hbm_idd2_measurements.csv")

# duplicate df_no_hbm entries for each chip_id (no_hbm has only even chip_ids)
duplicated_rows = []
for chip_id in df_no_hbm["chip_id"].unique():
    chip_data = df_no_hbm[df_no_hbm["chip_id"] == chip_id]
    duplicated_chip_data = chip_data.copy()
    duplicated_chip_data["chip_id"] = chip_id + 1
    duplicated_rows.append(duplicated_chip_data)
df_no_hbm = pd.concat([df_no_hbm] + duplicated_rows, ignore_index=True)

# remove chip_ids not in df_no_hbm, then deduct baseline
df = df[df["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df["idd"] = df.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

df = df[df["test_loop"] == "IDD2"]

most_freq_temp = df.groupby("chip_id")["temperature"].agg(lambda x: x.value_counts().idxmax())
df = df[df.apply(lambda row: abs(row["temperature"] - most_freq_temp[row["chip_id"]]) <= 1, axis=1)].reset_index(drop=True)

# --- Plot (section4-3 cell 4) ---
# per-stack means (one value per tested HBM2 stack)
avg_idd_by_chip = df.groupby("chip_id")["idd"].mean()
avg_temp_by_chip = df.groupby("chip_id")["temperature"].mean()

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
husl = sns.color_palette("husl", 2)

sns.histplot(avg_idd_by_chip, bins=12, ax=axes[0], color=husl[0],
             edgecolor="black", linewidth=1.5)
axes[0].set_xlabel("Idle IDD (mA)", fontsize=14)
axes[0].set_ylabel("Number of HBM2 stacks", fontsize=14)
axes[0].tick_params(axis="both", labelsize=12)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

sns.histplot(avg_temp_by_chip, bins=12, ax=axes[1], color="blue",
             edgecolor="black", linewidth=1.5)
axes[1].set_xlabel("Temperature (°C)", fontsize=14)
axes[1].set_ylabel("Number of HBM2 stacks", fontsize=14)
axes[1].tick_params(axis="both", labelsize=12)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf", bbox_inches="tight")

# --- Numbers reported in the paper (Figure 3, Sec. 4.3) ---
# Paper: min/max mean idle current = 217.5 / 364.8 mA; idle temperature range 48-64 C;
#        idle-current std across measurements = 9.6 mA on average across stacks.
print("\n--- Numbers reported in the paper (Figure 3) ---")
print(f"Idle IDD across stacks (per-stack mean): min={avg_idd_by_chip.min():.1f} mA, "
      f"max={avg_idd_by_chip.max():.1f} mA, mean={avg_idd_by_chip.mean():.1f} mA")
print(f"Idle temperature across stacks (per-stack mean): min={avg_temp_by_chip.min():.1f} C, "
      f"max={avg_temp_by_chip.max():.1f} C")
print(f"Avg. per-stack std of idle IDD across measurements: "
      f"{df.groupby('chip_id')['idd'].std().mean():.1f} mA")
