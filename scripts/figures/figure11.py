#!/usr/bin/env python3
"""Figure 11: Power variation across different banks (bank offsets).

Reproduces figures/bank_offset_normalized.pdf (results/section5-2.ipynb).
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

# --- Load and prepare bank-offset data (section5-2 cell 1) ---
df_bo = pd.read_csv(DATA_DIR / "bank_offset_measurements.csv")

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
mask = df_bo["chip_id"].isin(avg_idd_no_hbm.index)
df_bo = df_bo[mask].copy()
df_bo["idd"] = df_bo.apply(lambda row: row["idd"] - avg_idd_no_hbm[row["chip_id"]], axis=1)


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


bo_agg = preprocess(df_bo, ["chip_id", "bank_offset"])

# --- Plot: IDD normalized to bank offset 3 (section5-2 cell 8) ---
bo_ref = bo_agg.copy()
# per-chip IDD at bank_offset 3
bo3_idd = bo_ref[bo_ref["bank_offset"] == 3].set_index("chip_id")["idd"]
bo_ref = bo_ref[bo_ref["chip_id"].isin(bo3_idd.index)].copy()
bo_ref["idd_normalized"] = bo_ref.apply(lambda r: r["idd"] / bo3_idd[r["chip_id"]], axis=1)

bank_offsets_sorted = sorted(bo_ref["bank_offset"].unique())

plt.figure(figsize=(8, 3))
ax = sns.boxplot(
    data=bo_ref, x="bank_offset", y="idd_normalized",
    order=bank_offsets_sorted,
    palette=sns.color_palette("husl", len(bank_offsets_sorted)),
    linewidth=1.5, fliersize=4,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 8},
)
ax.set_xlabel("Bank Offset", fontsize=14)
ax.set_ylabel("IDD Normalized\nto Bank Offset 3", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
ax.axhline(1, color="gray", linestyle="-", linewidth=0.8)
plt.tight_layout()
# save as figureN.pdf in figures/ for visual comparison with the paper
plt.savefig(FIG_DIR / f"{Path(__file__).stem}.pdf")

for bo in bank_offsets_sorted:
    s = bo_ref[bo_ref["bank_offset"] == bo]["idd_normalized"]
    print(f"Bank offset {bo}: mean={s.mean():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")

# --- Numbers reported in the paper (Figure 11, Sec. 5.2) ---
# Paper: accessing bank offsets 0 and 1 consumes 5.3% and 5.3% higher current than
#        bank offset 3; bank offset 2 consumes only 0.4% higher, on average across stacks.
print("\n--- Numbers reported in the paper (Figure 11) ---")
print("Average current per bank offset relative to bank offset 3:")
for bo in bank_offsets_sorted:
    m = bo_ref[bo_ref["bank_offset"] == bo]["idd_normalized"].mean()
    print(f"  bank offset {bo}: {(m - 1) * 100:+.1f}% vs offset 3")
