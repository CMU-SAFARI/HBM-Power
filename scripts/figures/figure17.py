#!/usr/bin/env python3
"""Figure 17: HBM3 vs HBM4 read energy/bit across the JEDEC speed bins.

Reproduces figures/hbm3_vs_hbm4_pjbit.pdf (revision/figureE, with revision/figureD folded in).

Forward prediction with the DRAMPower data-pattern model (sources/drampower -> HBM3_runner);
there is no HBM4 silicon, so no empirical data is used. The canonical HBM4 curve that figureE
originally read from figureD's CSV is recomputed here inline with the same HBM4 builder
(hbm4_total_W), so this script is self-contained.

Left panel: read energy/bit (pJ/bit) vs data rate for HBM3 (1.10 V single rail) and three HBM4
split-rail voltage points, each capped at its JEDEC-legal data-rate ceiling (HBM3 <= 6.4 Gbps/pin,
HBM4 <= 8.0). Right panel: each HBM4 voltage point at 8 GT/s, energy/bit normalized to HBM3 at its
best legal point (6.4 GT/s). Both standards run on the same canonical 512-PC device.
"""
import os, re, json, math, tempfile, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC       = REPO_ROOT / "sources" / "figure17"
ENGINE    = REPO_ROOT / "sources" / "drampower"
BUILD     = ENGINE / "build"
RUNNER    = BUILD / "bin" / "HBM3_runner"
FIG_DIR   = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

ORG  = str(SRC / "configs" / "shared_512PC_organization.json")
SYN  = str(SRC / "traces" / "hbm3_random_read_4rpa.csv")
BASE = str(SRC / "traces" / "hbm3_baseline_nop_ref.csv")

# --- canonical organization (matches figureD) ------------------------------
N_PC      = 64 * 8       # 512 PC for both standards (HBM3 forced to match HBM4)
DQ_PER_PC = 32           # PC-mode DQ width
DQ_DEVICE = DQ_PER_PC * N_PC          # = 16384 DQ across the device (same for both standards)
RATES     = [4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0]   # Gbps/pin (full HBM4 JEDEC speed-bin span)
HBM3_VDDS = [1.1]        # HBM3 nominal rail only (1.05 V is below the HBM3 spec).
HBM3_MAX_RATE = 6.4      # legal data-rate ceilings, enforced on the energy/bit plot
HBM4_MAX_RATE = 8.0


def pjbit(total_W, bw_TBs):
    """Energy per transferred bit at peak bandwidth (pJ/bit)."""
    return total_W / (bw_TBs * 8.0) if bw_TBs else 0.0


# --- rails (HBM3E baseline @ 6.4 GT/s; identical to figureD) ----------------
RATE_BASE = 6.4
IDD3N1    = 34.6
IDD4R_B, IDD4W_B = 630.4, 505.1
ARRAY = dict(IDD0=44.6, IDD2N=38.3, IDD3N1=IDD3N1, IDD3N16=36.7)


def io_current(base, rate):
    """Dynamic (above-IDD3N1) part scales linearly with data rate; array part fixed."""
    return IDD3N1 + (base - IDD3N1) * (rate / RATE_BASE)


def cyc(ns, tck_ns, lo=1):
    return max(lo, math.ceil(ns / tck_ns))


# --- HBM3 timings (ramulator2 HBM3_6400 bin, cycles @ tCK=625 ps) -----------
HBM3_625 = dict(nRCD=31, nRCDWR=15, nRL=20, nWL=10, nCCD_S=2, nCCD_L=4,
                nWTR_S=7, nWTR_L=10, nWR=33, nRTP=9)


def hbm3_timing_json(rate):
    tck = 4.0 / rate                 # ns; dataRate=4 transfers/CK
    t = {"tCK_ps": round(tck * 1000, 3), "nBL": 8,
         "nRC": cyc(72 * 0.625, tck), "nRAS": cyc(45 * 0.625, tck), "nRP": cyc(26 * 0.625, tck)}
    for k, c in HBM3_625.items():
        t[k] = cyc(c * 0.625, tck, lo=2 if k.startswith("nCCD") else 1)
    t["nREFI"] = cyc(3900, tck); t["nRFCpb"] = cyc(280, tck)
    return {"timing": t}


# Data-pattern model (identical for both standards; relative form, coupling off, all knobs 0.5).
DP = {"enabled": True, "floor_pJbit": 2.759, "coef_T_DQ": 1.192,
      "coef_T_2bit": 1.331, "coef_busflip": 0.72, "use_coupling": False,
      "dq_rate": 0.5, "tsv_rate": 0.5, "bg_rate": 0.5,
      "ref_dq_rate": 0.0, "ref_tsv_rate": 0.0, "ref_bg_rate": 0.0,
      "apply_to_writes": True}


def idd_block(rate):
    return {**ARRAY, "IDD4R": round(io_current(IDD4R_B, rate), 1),
            "IDD4W": round(io_current(IDD4W_B, rate), 1), "unit": "mA"}


def hbm3_power_json(rate, vdd):
    """Single voltage rail: VDDQ omitted -> engine defaults VDDQ = VDD (no split)."""
    return {"voltage": {"VDD": vdd, "unit": "V"}, "IDD": idd_block(rate), "datapattern": DP}


# --- HBM4 timings (JESD270-4 absolute ns; identical to figureD's timing_json) -----------------
HBM4_625 = dict(nRCD=31, nRCDWR=15, nRL=20, nWL=10, nCCD_S=2, nCCD_L=4,
                nWTR_S=7, nWTR_L=10, nWR=33, nRTP=9)


def hbm4_timing_json(rate):
    tck = 4.0 / rate
    t = {"tCK_ps": round(tck * 1000, 3), "nBL": 8,
         "nRC": cyc(48, tck), "nRAS": cyc(33, tck), "nRP": cyc(15, tck)}
    for k, c in HBM4_625.items():
        t[k] = cyc(c * 0.625, tck, lo=2 if k.startswith("nCCD") else 1)
    t["nREFI"] = cyc(3900, tck); t["nRFCpb"] = cyc(280, tck)
    return {"timing": t}


def hbm4_power_json(rate, vddc, vddq):
    """Split rails: VDDC core + VDDQ on the DQ read I/O only."""
    return {"voltage": {"VDD": vddc, "VDDQ": vddq, "unit": "V"},
            "IDD": idd_block(rate), "datapattern": DP}


RE_POWER = re.compile(r"Average power:\s+([\d.]+)\s+mW")


def ensure_runner():
    if RUNNER.exists():
        return
    print(f"[build] {RUNNER.name} not found; building DRAMPower engine (one-time)...", file=sys.stderr)
    subprocess.run(["cmake", "-S", str(ENGINE), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
                    "-DDRAMPOWER_BUILD_TESTS=OFF", "-DDRAMPOWER_BUILD_BENCHMARKS=OFF",
                    "-DDRAMPOWER_BUILD_CLI=ON"], check=True)
    subprocess.run(["cmake", "--build", str(BUILD), "--target", "HBM3_runner", "-j"], check=True)
    if not RUNNER.exists():
        sys.exit(f"build did not produce {RUNNER}")


def run(timing, power, trace, dq, tsv, bg):
    def tmp(obj):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        return p
    tj, pj = tmp(timing), tmp(power)
    cmd = [str(RUNNER), ORG, tj, pj, trace, f"--dq-rate={dq}", f"--tsv-rate={tsv}", f"--bg-rate={bg}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tj); os.remove(pj)
    m = RE_POWER.search(out.stdout + out.stderr)
    if not m:
        raise RuntimeError("parse fail:\n" + out.stdout + out.stderr)
    return float(m.group(1))   # per-PC mW


def hbm3_total_W(rate, vdd):
    tj, pj = hbm3_timing_json(rate), hbm3_power_json(rate, vdd)
    act = run(tj, pj, SYN,  0.5, 0.5, 0.5) * N_PC / 1000.0
    idl = run(tj, pj, BASE, 0.0, 0.0, 0.0) * N_PC / 1000.0
    return act + idl


def hbm4_total_W(rate, vddc, vddq):
    tj, pj = hbm4_timing_json(rate), hbm4_power_json(rate, vddc, vddq)
    act = run(tj, pj, SYN,  0.5, 0.5, 0.5) * N_PC / 1000.0
    idl = run(tj, pj, BASE, 0.0, 0.0, 0.0) * N_PC / 1000.0
    return act + idl


# HBM4 low-rail variants (canonical VDDC 1.05 / VDDQ 0.9 is computed inline below).
HBM4_VARIANTS = [
    dict(vddc=1.05, vddq=0.7, col="hbm4_c105q07_W", mk="v-", color="#16a085",
         label="HBM4 1.05 VDDC 0.7 VDDQ"),
    dict(vddc=1.00, vddq=0.7, col="hbm4_c100q07_W", mk="D-", color="#7d3c98",
         label="HBM4 1.00 VDDC 0.7 VDDQ"),
]


def main():
    ensure_runner()

    rows = []
    for rate in RATES:
        bw_TBs = DQ_DEVICE * rate / 8 / 1000.0     # peak device BW (TB/s); same for both standards
        # Canonical HBM4 (VDDC 1.05 / VDDQ 0.9): recomputed inline (folds in figureD).
        hbm4_W = hbm4_total_W(rate, 1.05, 0.9)
        hbm4_pJ = pjbit(hbm4_W, bw_TBs)
        rec = {"rate_Gbps": rate, "bw_peak_TBs": round(bw_TBs, 2),
               "hbm4_split_W": round(hbm4_W, 1), "hbm4_split_pJbit": round(hbm4_pJ, 3)}
        for var in HBM4_VARIANTS:
            tot = hbm4_total_W(rate, var["vddc"], var["vddq"])
            rec[var["col"]] = round(tot, 1)
            rec[var["col"].replace("_W", "_pJbit")] = round(pjbit(tot, bw_TBs), 3)
        for v in HBM3_VDDS:
            tag = f"{v:g}".replace(".", "")
            tot = hbm3_total_W(rate, v)
            rec[f"hbm3_{tag}_W"]     = round(tot, 1)
            rec[f"hbm3_{tag}_pJbit"] = round(pjbit(tot, bw_TBs), 3)
        rows.append(rec)
        print(f"{rate:>4} GT/s | HBM4 {rec['hbm4_split_W']:>6.1f} W (1.05/0.9) "
              f"{rec['hbm4_c105q07_W']:>6.1f} (1.05/0.7) {rec['hbm4_c100q07_W']:>6.1f} (1.0/0.7) | "
              f"HBM3 {rec['hbm3_11_W']:>6.1f} (1.1V)")

    R = [r["rate_Gbps"] for r in rows]
    at = {r["rate_Gbps"]: r for r in rows}

    def plot_curves(ax, plot_series, ylabel, title, legend_loc, constrain, legend_bbox=None):
        for label, mk, col, key, lw, rmax in plot_series:
            pts = [(r["rate_Gbps"], r[key]) for r in rows if not (constrain and r["rate_Gbps"] > rmax)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, mk, color=col, ms=5, lw=lw, label=label)
        ax.set_xlabel("Data Rate (Gbps/pin)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        if legend_bbox is None:
            ax.legend(fontsize=7.5, loc=legend_loc, ncol=1)
        else:
            ax.legend(fontsize=7.5, loc=legend_loc, ncol=1, bbox_to_anchor=legend_bbox, borderaxespad=0.0)
        ax.grid(axis="both", ls=":", alpha=0.5); ax.set_xticks(R)

    # (label, marker, color, key-prefix, linewidth, legal-rate ceiling)
    baseline = [
        ("HBM3 1.10 VDDCQ", "o-", "#c0392b", "hbm3_11",   1.6, HBM3_MAX_RATE),
        ("HBM4 1.05 VDDC 0.9 VDDQ", "^-", "#1f6fe0", "hbm4_split", 1.8, HBM4_MAX_RATE),
    ] + [(v["label"], v["mk"], v["color"], v["col"][:-2], 1.8, HBM4_MAX_RATE) for v in HBM4_VARIANTS]
    pjbit_series = [(lbl, mk, col, f"{k}_pJbit", lw, rmax) for lbl, mk, col, k, lw, rmax in baseline]

    # Energy/bit: left = curves (legal ranges); right = HBM4 @ 8 GT/s normalized to HBM3 @ 6.4 GT/s.
    figE, (axL, axR) = plt.subplots(
        1, 2, figsize=(8, 3),
        gridspec_kw={"width_ratios": [2.5, 1.5], "wspace": 0.3},
    )
    plot_curves(axL, pjbit_series, "Read Energy (pJ/bit)", "", "center left", constrain=True,
                legend_bbox=(0.515, 0.8))

    ref = at[HBM3_MAX_RATE]["hbm3_11_pJbit"]      # HBM3 @ 6.4 GT/s (its best legal efficiency point)
    bars = [("VDDC 1.05\nVDDQ 0.9", "#1f6fe0", "hbm4_split_pJbit"),
            ("VDDC 1.05\nVDDQ 0.7", "#16a085", "hbm4_c105q07_pJbit"),
            ("VDDC 1.00\nVDDQ 0.7", "#7d3c98", "hbm4_c100q07_pJbit")]
    xs = list(range(len(bars)))
    norm = [at[HBM4_MAX_RATE][k] / ref for _, _, k in bars]   # pJ/bit @ 8 GT/s, normalized to HBM3 @ 6.4
    axR.bar(xs, norm, width=0.6, color=[c for _, c, _ in bars], linewidth=1.5, edgecolor="black")
    axR.axhline(1.0, ls="--", color="#444", lw=1.2)
    axR.text(len(bars) - 0.6, 1.002, "HBM3 @ 6.4 Gbps", ha="right", va="bottom", fontsize=9, color="#444")
    axR.set_xticks(xs); axR.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=8.5)
    axR.set_ylabel("HBM4 pJ/bit @ 8 Gbps\nnormalized to HBM3 @ 6.4 Gbps")
    axR.set_ylim(0.8, 1.04); axR.grid(axis="y", ls=":", alpha=0.5)

    figE.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88)
    out = FIG_DIR / f"{Path(__file__).stem}.pdf"
    figE.savefig(out, bbox_inches="tight")
    plt.close(figE)

    # report the energy/bit improvement bars (paper: 6.7% / 8.8% / 12.8%)
    for (lbl, _, k), h in zip(bars, norm):
        print(f"  {lbl.replace(chr(10),' '):22} HBM4@8GT/s = {h:.3f} x HBM3@6.4GT/s  ({(1 - h) * 100:.1f}% better)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
