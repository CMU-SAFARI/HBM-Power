#!/usr/bin/env python3
"""Figure 15: per-pattern parity, FGDRAM single-toggle vs our toggle+coupling model.

Reproduces figures/model_vs_fgdram_single_toggle.pdf (revision/figureB).

Engine-free: this is a pure ordinary-least-squares fit on the empirical beat-pattern
measurement (data/beat_pattern_perpattern.csv) -- the DRAMPower model is not invoked.
Per beat pattern the data-movement energy target is
    pJ/bit = (mean_idd_mA - IDD3N1) / K,   IDD3N1 = 1190.9 mA,  K = 512 mA per pJ/bit.
Two models are fit over the 32 patterns:
  Baseline "DQ + Core Toggle"                 -> R2 ~ 0.41
  Improved "DQ + TSV + BG Toggle & Coupling"  -> R2 ~ 0.97
"""
import os
import csv, itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
FIG_DIR   = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

DATA = DATA_DIR / "beat_pattern_perpattern.csv"
WIRE_ORDER = "best"          # "best" -> data-inferred (1,3,0,2); "linear" -> physical (0,1,2,3)

# ---- calibration constants (HBM2 @ 1.2 Gbps) ----
IDD3N1 = 1190.9                                   # read background incl. off-power (mA)
VDD, tCK, BLDR, BITS, NCYC = 1.2, 1666.7e-12, 4/2, 256, 8
K = (NCYC * BITS * 1.0 * 1e-12) / (VDD * BLDR * tCK) * 1e3   # 512 mA per pJ/bit
IO_FG = 0.80 / 0.5                                # FGDRAM I/O = 1.60 pJ/bit per unit DQ toggle


def _seq(P, ci):
    inv = "".join("1" if c == "0" else "0" for c in P)
    return P + inv if ci == "1" else P


def _tog(s, N):                                   # cyclic mean N-bit-word toggle rate
    w = [s[i:i+N] for i in range(0, len(s), N)]; nb = len(w)
    return 0.0 if nb < 2 else sum(sum(a != b for a, b in zip(w[i], w[(i+1) % nb])) / N
                                  for i in range(nb)) / nb


def _cpl(seq, gran, order):                       # squared (Miller) adjacent-wire coupling
    w = [seq[i:i+gran] for i in range(0, len(seq), gran)]; m = len(w)
    if m < 2:
        return 0.0
    pr = [(order[j], order[j+1]) for j in range(len(order) - 1)]
    return sum((((w[t][a] == "1") - (w[t-1][a] == "1")) - ((w[t][b] == "1") - (w[t-1][b] == "1")))**2
               for t in range(m) for a, b in pr) / (m * len(pr))


def r2(y, p):
    return 1 - ((y - p)**2).sum() / ((y - y.mean())**2).sum()


def main():
    D = []
    for r in csv.DictReader(open(DATA)):
        s = _seq(r["beat_pattern"], r["col1_inverted"])
        D.append(dict(s=s, bf=int(r["col1_inverted"]),
                      lbl=r["beat_pattern"] + ("+flip" if r["col1_inverted"] == "1" else ""),
                      pj=(float(r["mean_idd_mA"]) - IDD3N1) / K,
                      dq=_tog(s, 1), t2=_tog(s, 2), t4=_tog(s, 4)))
    y = np.array([d["pj"] for d in D]); n = len(D)

    # Baseline: pre-GSA floor + fixed I/O + ONE post-GSA toggle (T_4bit)
    FLOOR0 = float(np.mean([d["pj"] for d in D if d["dq"] == 0 and d["t4"] == 0]))
    base = FLOOR0 + IO_FG * np.array([d["dq"] for d in D])
    cA, _, _, _ = np.linalg.lstsq(np.c_[[d["t4"] for d in D]], y - base, rcond=None)
    predA = base + np.c_[[d["t4"] for d in D]] @ cA
    r2A = r2(y, predA)

    # Improved: floor + DQ + 2-bit + busflip + BG/TSV squared coupling (best of 24 wire orders)
    Xb = np.c_[np.ones(n), [d["dq"] for d in D], [d["t2"] for d in D], [d["bf"] for d in D]]
    C2 = np.array([_cpl(d["s"], 2, [0, 1]) for d in D])
    def _Xc(o): return np.c_[Xb, [_cpl(d["s"], 4, o) for d in D], C2]
    def _r2(A):
        c, _, _, _ = np.linalg.lstsq(A, y, rcond=None); return r2(y, A @ c)
    assert WIRE_ORDER in ("best", "linear")
    best = (list(max(itertools.permutations([0, 1, 2, 3]), key=lambda o: _r2(_Xc(list(o)))))
            if WIRE_ORDER == "best" else [1, 0, 3, 2])
    Xc = _Xc(best)
    cB, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None); predC = Xc @ cB
    r2C = r2(y, predC)

    # express on the IDD4R (mA) axis: IDD4R = IDD3N1 + K*pJ/bit (affine; R^2 unchanged)
    y_mA, predA_mA, predC_mA = IDD3N1 + K * y, IDD3N1 + K * predA, IDD3N1 + K * predC
    idx = {d["lbl"]: i for i, d in enumerate(D)}
    OURS, FG = "#1f6fe0", "#e8710a"
    fig, ax = plt.subplots(figsize=(8, 3))
    lo, hi = y_mA.min() - K * 0.3, y_mA.max() + K * 0.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
    ax.scatter(y_mA, predA_mA, s=62, marker="o", facecolors=FG, edgecolors="black",
               linewidths=1.0, zorder=2, label="Baseline")
    ax.scatter(y_mA, predC_mA, s=62, marker="x", facecolors=OURS, edgecolors="black",
               linewidths=2.0, zorder=3, label="Improved")

    def mark(lbl, txt, dx, dy, ha):
        i = idx[lbl]
        ax.plot([y_mA[i], y_mA[i]], [predA_mA[i], predC_mA[i]], color="#999", lw=1, ls="-", zorder=1)
        ax.annotate(txt, xy=(y_mA[i], (predA_mA[i] + predC_mA[i]) / 2),
                    xytext=(y_mA[i] + dx, (predA_mA[i] + predC_mA[i]) / 2 + dy),
                    fontsize=9.5, ha=ha, va="center", color="#222",
                    arrowprops=dict(arrowstyle="->", color="#222", lw=1.1))
    mark("0101+flip", "0101 (BG Bus Toggle)", -K * 0.05, K * 0.55, "right")
    mark("1001", "1001 (No BG Bus Toggle)", K * 0.10, -K * 0.55, "left")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured IDD4R (mA)", fontsize=12)
    ax.set_ylabel("Predicted IDD4R (mA)", fontsize=12)
    ax.legend(fontsize=9.5, loc="lower right"); ax.grid(ls=":", alpha=0.5)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    out = FIG_DIR / f"{Path(__file__).stem}.pdf"
    fig.savefig(out)

    print(f"Baseline (DQ+Core Toggle): R2={r2A:.3f}   "
          f"Improved (DQ+TSV+BG Toggle & Coupling): R2={r2C:.3f}   "
          f"(wire order {tuple(best)} [{WIRE_ORDER}])")
    for l in ("0101+flip", "1001"):
        i = idx[l]
        print(f"  {l:9s} measured={y_mA[i]:.0f} mA  baseline={predA_mA[i]:.0f} mA  improved={predC_mA[i]:.0f} mA")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
