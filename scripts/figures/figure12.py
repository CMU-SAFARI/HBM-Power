#!/usr/bin/env python3
"""Figure 12: Current vs number of toggled DQ signals across beats of a burst.

Reproduces figures/idd_vs_num_bitflips.pdf (results/section5-3.ipynb).
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

# --- Load and prepare bitflip data (section5-3 cell 2) ---
df_bf = pd.read_csv(DATA_DIR / "bitflip_measurements.csv")

# cleaning: top 40 samples per group, average idd
top40_bf = (
    df_bf.sort_values("sample_id", ascending=False)
    .groupby(["chip_id", "test_loop", "num_bitflips"], group_keys=False)
    .head(40)
)
avg_idd_bf = top40_bf.groupby(["chip_id", "test_loop", "num_bitflips"])["idd"].mean().rename("idd")
std_dev_idd_bf = top40_bf.groupby(["chip_id", "test_loop", "num_bitflips"])["idd"].std().rename("idd_std_dev")
df_bf = (
    top40_bf.sort_values("sample_id", ascending=False)
    .groupby(["chip_id", "test_loop", "num_bitflips"])
    .first()
    .drop(columns=["idd"])
    .join(avg_idd_bf)
    .join(std_dev_idd_bf)
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

df_bf = df_bf[df_bf["chip_id"].isin(df_no_hbm["chip_id"].unique())].reset_index(drop=True)
avg_idd_no_hbm = df_no_hbm.groupby("chip_id")["idd"].mean()
df_bf["idd"] = df_bf.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)

# keep only test_loop = "IDD4R_full", remove anomalously low values
df_bf = df_bf[df_bf["test_loop"] == "IDD4R_full"].reset_index(drop=True)
df_bf = df_bf[df_bf["idd"] >= 200].reset_index(drop=True)

# --- Plot (section5-3 cell 3) ---
plt.figure(figsize=(8, 3))
ax = sns.boxplot(
    data=df_bf, x="num_bitflips", y="idd",
    palette=sns.color_palette("husl", 1),
    linewidth=1.5, fliersize=4,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
)

# trend line across means
means_bf = df_bf.groupby("num_bitflips")["idd"].mean()
x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
x_positions = np.array(list(range(len(x_labels))))
x_actual = np.array([int(label) for label in x_labels])
y_values = np.array([means_bf[int(label)] for label in x_labels])

z = np.polyfit(x_positions, y_values, 1)
p = np.poly1d(z)

actual_step = np.mean(np.diff(x_actual))
slope_per_bf = z[0] / actual_step
label_text = f"Trend ({slope_per_bf:.1f} mA/bitflip)"

ax.plot(x_positions, p(x_positions), color=sns.color_palette("husl", 2)[1], linestyle="--", linewidth=2, label=label_text, zorder=5)

# fit statistics
y_pred = p(x_positions)
ss_res = np.sum((y_values - y_pred) ** 2)
ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
r_squared = 1 - ss_res / ss_tot
rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
mae = np.mean(np.abs(y_values - y_pred))
print(f"Linear fit: slope = {slope_per_bf:.3f} mA/bitflip, intercept = {z[1]:.3f} mA")
print(f"R² = {r_squared:.6f}, RMSE = {rmse:.3f} mA, MAE = {mae:.3f} mA")
print(f"SS_res = {ss_res:.3f}, SS_tot = {ss_tot:.3f}")

ax.set_xlabel("Number of toggled DQ signals", fontsize=14)
ax.set_ylabel("IDD (mA)", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.legend(fontsize=12)
plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf")

# --- Numbers reported in the paper (Figure 12, Sec. 5.3) ---
# Paper: average current increases by 32.0% as toggled DQ signals go from 2 to 64;
#        linear-fit mean-square error (RMSE) is only 2.0 mA.
print("\n--- Numbers reported in the paper (Figure 12) ---")
print(f"Linear-fit RMSE: {rmse:.1f} mA")
mean_at_2_bf = df_bf[df_bf["num_bitflips"] == 2]["idd"].mean()
print(f"Mean IDD at 2 toggled DQ signals: {mean_at_2_bf:.1f} mA")
mean_at_64_bf = df_bf[df_bf["num_bitflips"] == 64]["idd"].mean()
print(f"Mean IDD at 64 toggled DQ signals: {mean_at_64_bf:.1f} mA")
percentage_increase_bf = ((mean_at_64_bf - mean_at_2_bf) / mean_at_2_bf) * 100
print(f"Current increase, 2 -> 64 toggled DQ signals: {percentage_increase_bf:.1f}%")
