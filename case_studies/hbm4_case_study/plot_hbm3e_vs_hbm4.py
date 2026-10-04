#!/usr/bin/env python3
"""HBM3 vs HBM4 across the JEDEC speed bins -- device power (left) and energy/bit (right).

Four curves on one canonical 512-PC device (8 stacks x 64 PC for HBM4; HBM3 forced to the same
512 PC so the comparison isolates the per-PC standard/voltage differences, not the bus width):

  * HBM4  -- split rails, VDDC core + VDDQ on the DQ read I/O only, at three voltage points:
             (1.05/0.9) the canonical curve taken AS-IS from sweep_hbm4_power.py (hbm4_power_sweep.csv);
             (1.05/0.7) lowest spec VDDQ; and (1.00/0.7) also a lower core rail. The two low-rail
             variants are computed here with the same HBM4 builder as sweep_hbm4_power.py (an assertion checks
             the in-script 1.05/0.9 result matches sweep_hbm4_power.py's CSV).
  * HBM3  -- a hypothetical SINGLE-voltage-rail design (no VDDC/VDDQ split) at HBM3's nominal 1.1 V.
             (1.05 V is below the HBM3 spec -- VDD 1.1 V, VDD_min ~= 1.067 V at -3% -- so it is not a
             legal operating point and is not shown.)

Legal data-rate ranges: HBM3 (JESD238) tops out at 6.4 Gbps/pin; HBM4 (JESD270-4) spans 4.8-8.0. The
energy/bit plot enforces these ceilings (a fair in-spec efficiency comparison -- HBM3 stops at 6.4);
the power plot keeps the full sweep for all curves, showing HBM3 as a hypothetical scaled part.

Everything except timing and voltage is held identical to the HBM4 run (same 512 PC, same HBM3E-
calibrated rails, same linear data-rate scaling of the dynamic read/write current, same data-pattern
knobs at 0.5, same traces). HBM3 uses its own JESD timings (ramulator2 HBM3_6400 bin, scaled to each
data rate at constant absolute ns); HBM4 uses the JESD270-4 absolute timings inside sweep_hbm4_power.py. So the
HBM3<->HBM4 gap here is the combined effect of (a) the timing differences and (b) HBM4's split rail /
lower core voltage -- and the HBM3 1.1 V vs 1.05 V pair isolates the rail-voltage component.

Engine: reuses the HBM3 core engine + data-pattern model (HBM3_runner), same as sweep_hbm4_power.py. Build once
from the repo root:  cmake --build build --target HBM3_runner -j   (see docs/how-to/build-the-engine.md).
Run sweep_hbm4_power.py first (this script reads its CSV for the HBM4 curve).
"""
import os, re, csv, json, math, tempfile, subprocess, sys
import tracegen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(REPO, "build", "bin", "HBM3_runner")
ORG    = os.path.join(HERE, "configs", "HBM4_organization.json")             # HBM4: 2 BG x 8 banks per PC
ORG_H3 = os.path.join(HERE, "configs", "HBM3_16Gb_8hi_organization.json")    # HBM3E: 4 BG x 4 banks per PC
SYN    = os.path.join(HERE, "traces", "hbm3_random_read_4rpa.csv")
BASE   = os.path.join(HERE, "traces", "hbm3_baseline_nop_ref.csv")
HBM4CSV = os.path.join(HERE, "hbm4_power_sweep.csv")   # HBM4 curve from sweep_hbm4_power.py, taken as-is
OUTCSV   = os.path.join(HERE, "hbm3_vs_hbm4_power.csv")
OUTFIG_P = os.path.join(HERE, "hbm3_vs_hbm4_power")    # device power vs data rate
OUTFIG_E = os.path.join(HERE, "hbm3_vs_hbm4_pjbit")    # energy/bit vs data rate

# --- canonical organization (matches sweep_hbm4_power.py) ------------------------------
N_PC      = 64 * 8       # 512 PC for both standards (HBM3 forced to match HBM4)
DQ_PER_PC = 32           # PC-mode DQ width
DQ_DEVICE = DQ_PER_PC * N_PC          # = 16384 DQ across the device (same for both standards)
RATES     = [4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0]   # Gbps/pin (full HBM4 JEDEC speed-bin span)
HBM3_VDDS = [1.1]        # HBM3 nominal rail only. 1.05 V is below the HBM3 spec (VDD 1.1 V, and at
                         # the JEDEC -3% rail tolerance VDD_min ~= 1.067 V), so it is not a legal HBM3
                         # operating point and is dropped.
# Legal per-standard data-rate ceilings, enforced on the energy/bit plot (a fair in-spec efficiency
# comparison). HBM3 (JESD238) tops out at 6.4 Gbps/pin; HBM4 (JESD270-4) spans the full sweep. The
# power plot keeps the full sweep for all curves (HBM3 shown as a hypothetical scaled part).
HBM3_MAX_RATE = 6.4
HBM4_MAX_RATE = 8.0

def pjbit(total_W, bw_TBs):
    """Energy per transferred bit at peak bandwidth (pJ/bit)."""
    return total_W / (bw_TBs * 8.0) if bw_TBs else 0.0   # W / (TB/s * 8 b/B) = pJ/bit

# --- rails (HBM3E baseline @ 6.4 GT/s) --------------------------------------
# Taken from the HBM3E validation config (a copy in configs/), derived from the HBM2
# measurements (per-PC, all-0s, off-power excluded).
RATE_BASE = 6.4
HBM3E_CFG = os.path.join(HERE, "configs", "HBM3_6400_power_datapattern.json")
_H3 = json.load(open(HBM3E_CFG))["IDD"]
IDD3N1    = _H3["IDD3N1"]
IDD4R_B, IDD4W_B = _H3["IDD4R"], _H3["IDD4W"]          # HBM3E all-0s I/O rails at 6.4 GT/s
ARRAY = dict(IDD0=_H3["IDD0"], IDD2N=_H3["IDD2N"], IDD3N1=IDD3N1, IDD3N16=_H3["IDD3N16"], IDD5B=_H3["IDD5B"])
# Fixed per-PC deltas / increments (mA) from the config's `extrapolation` block; the differences of the rails themselves are the fallback.
_X = json.load(open(HBM3E_CFG)).get("extrapolation", {})
D31   = _X.get("IDD3N1_minus_IDD2N_mA",  ARRAY["IDD3N1"]  - ARRAY["IDD2N"])    # IDD3N1  - IDD2N
D316  = _X.get("IDD3N16_minus_IDD3N1_mA", ARRAY["IDD3N16"] - ARRAY["IDD3N1"])   # IDD3N16 - IDD3N1
ACT_INC = _X.get("act_increment_mA", ARRAY["IDD0"] - ARRAY["IDD2N"])            # IDD0 - IDD2N at 6.4 Gbps
REF_INC = _X.get("ref_increment_mA", ARRAY["IDD5B"] - ARRAY["IDD3N1"])          # IDD5B - IDD3N1 at 6.4 Gbps
ACT_SLOPE = _X.get("act_increment_rel_slope_per_MTs", 0.0)                       # relative, per MT/s about 6400
REF_SLOPE = _X.get("ref_increment_rel_slope_per_MTs", 0.0)                       # relative, per MT/s about 6400
def array_at(rate):
    """Standby: IDD2N scales with data rate as measured on HBM2 (Sec. 5.1.2); IDD3N1 and IDD3N16 sit a
    fixed delta above it (the measured IDD3N1 - IDD2N gap is data-rate independent). Activate and
    refresh increments: fixed in mA at 6.4 Gbps, each following its fixed-pacing sweep's trend
    elsewhere. Reproduces the HBM3E config exactly at 6.4 Gbps."""
    s = rate / RATE_BASE
    idd2n = ARRAY["IDD2N"] * s
    idd3n1 = idd2n + D31
    idd0 = idd2n + ACT_INC * (1.0 + ACT_SLOPE * (rate - RATE_BASE) * 1000.0)
    idd5b = idd3n1 + REF_INC * (1.0 + REF_SLOPE * (rate - RATE_BASE) * 1000.0)
    return {k: round(v, 1) for k, v in dict(IDD0=idd0, IDD2N=idd2n, IDD3N1=idd3n1,
                                              IDD3N16=idd3n1 + D316, IDD5B=idd5b).items()}

def io_current(base, rate):
    """Read/write rail scales linearly with data rate (background and dynamic parts alike)."""
    return base * (rate / RATE_BASE)

# --- HBM3 timings (ramulator2 HBM3_6400 bin, cycles @ tCK=625 ps) -----------
# Scaled to each speed bin at constant absolute ns, mirroring sweep_hbm4_power.py's method. nRC/nRAS/nRP are
# HBM3-specific (45/28.125/16.25 ns); the rest carry the HBM3_6400 cycle counts at constant ns.
HBM3_625 = dict(nRCD=31, nRCDWR=15, nRL=20, nWL=10, nCCD_S=2, nCCD_L=4,
                nWTR_S=7, nWTR_L=10, nWR=33, nRTP=9)
def cyc(ns, tck_ns, lo=1):
    return max(lo, math.ceil(ns / tck_ns))

def hbm3_timing_json(rate):
    tck = 4.0 / rate                 # ns; dataRate=4 transfers/CK
    t = {"tCK_ps": round(tck * 1000, 3), "nBL": 8,
         "nRC": cyc(72 * 0.625, tck), "nRAS": cyc(45 * 0.625, tck), "nRP": cyc(26 * 0.625, tck)}
    for k, c in HBM3_625.items():
        t[k] = cyc(c * 0.625, tck, lo=2 if k.startswith("nCCD") else 1)
    t["nREFI"] = cyc(3900, tck); t["nRFCpb"] = cyc(280, tck); t["nRFC"] = cyc(350, tck)   # all-bank tRFC 350 ns (assumption)
    return {"timing": t}

# Data-pattern model (identical for both standards; relative/K-free, coupling off, all knobs 0.5).
DP = {"enabled": True, "floor_pJbit": 2.759, "coef_T_DQ": 1.192,
      "coef_T_2bit": 1.331, "coef_busflip": 0.72, "use_coupling": False,
      "dq_rate": 0.5, "tsv_rate": 0.5, "bg_rate": 0.5,
      "ref_dq_rate": 0.0, "ref_tsv_rate": 0.0, "ref_bg_rate": 0.0,
      "apply_to_writes": True}

def idd_block(rate):
    return {**array_at(rate), "IDD4R": round(io_current(IDD4R_B, rate), 1),
            "IDD4W": round(io_current(IDD4W_B, rate), 1), "unit": "mA"}

def hbm3_power_json(rate, vdd):
    """Single voltage rail: VDDQ omitted -> the engine defaults VDDQ = VDD, so the DQ read I/O is
    on the same rail as everything else (no split)."""
    return {"voltage": {"VDD": vdd, "unit": "V"}, "IDD": idd_block(rate), "datapattern": DP}

# --- HBM4 timings (JESD270-4 absolute ns; identical to sweep_hbm4_power.py's timing_json) -----------------
HBM4_625 = dict(nRCD=31, nRCDWR=15, nRL=20, nWL=10, nCCD_S=2, nCCD_L=4,
                nWTR_S=7, nWTR_L=10, nWR=33, nRTP=9)
def hbm4_timing_json(rate):
    tck = 4.0 / rate
    t = {"tCK_ps": round(tck * 1000, 3), "nBL": 8,
         "nRC": cyc(48, tck), "nRAS": cyc(33, tck), "nRP": cyc(15, tck)}
    for k, c in HBM4_625.items():
        t[k] = cyc(c * 0.625, tck, lo=2 if k.startswith("nCCD") else 1)
    t["nCCD_S"], t["nCCD_L"] = 2, 4  # HBM4 column-to-column delays in clock cycles
    t["nREFI"] = cyc(3900, tck); t["nRFCpb"] = cyc(280, tck); t["nRFC"] = cyc(350, tck)   # all-bank tRFC 350 ns (assumption)
    return {"timing": t}

def hbm4_power_json(rate, vddc, vddq):
    """Split rails: VDDC core + VDDQ on the DQ read I/O only (engine applies VDDQ to the DQ
    component of IDD4R; floor/TSV/BG and everything else stay on VDDC)."""
    return {"voltage": {"VDD": vddc, "VDDQ": vddq, "unit": "V"},
            "IDD": idd_block(rate), "datapattern": DP}

RE_POWER = re.compile(r"Average power:\s+([\d.]+)\s+mW")
def run(org, timing, power, trace, dq, tsv, bg):
    def tmp(obj):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f: json.dump(obj, f)
        return p
    tj, pj = tmp(timing), tmp(power)
    cmd = [RUNNER, org, tj, pj, trace, f"--dq-rate={dq}", f"--tsv-rate={tsv}", f"--bg-rate={bg}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tj); os.remove(pj)
    m = RE_POWER.search(out.stdout + out.stderr)
    if not m: raise RuntimeError("parse fail:\n" + out.stdout + out.stderr)
    return float(m.group(1))   # per-PC mW

TRACE_DIR = os.path.join(HERE, "traces", "generated")
def make_traces(rate, tj, org):
    """Per-rate read trace with REFA every tREFI (3.9 us) and a tRFC (350 ns) stall (see tracegen.py),
    with the banks mapped onto the organization `org`."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    nREFI, nRFC = tj["timing"]["nREFI"], tj["timing"]["nRFC"]
    bpg = json.load(open(org))["banks_per_bankgroup"]
    rd = os.path.join(TRACE_DIR, f"random_read_4rpa_ref_{rate:.1f}Gbps_{bpg}bpg.csv")
    info = tracegen.write_read_trace(SYN, rd, nREFI, nRFC, banks_per_bg=bpg)
    return rd, info

def total_W_and_pjbit(tj, pj, rate, org):
    """Device power from the refresh-bearing read trace (single run, nothing double counted) and
    energy per bit at the bandwidth the trace achieves (refresh stalls included)."""
    rd, info = make_traces(rate, tj, org)
    tot = run(org, tj, pj, rd, 0.5, 0.5, 0.5) * N_PC / 1000.0   # W
    bits = info["n_rd"] * 256 * N_PC
    dur_s = info["end_cycle"] * tj["timing"]["tCK_ps"] * 1e-12
    return tot, tot * dur_s / bits * 1e12

def hbm3_total_W(rate, vdd):
    return total_W_and_pjbit(hbm3_timing_json(rate), hbm3_power_json(rate, vdd), rate, ORG_H3)

def hbm4_total_W(rate, vddc, vddq):
    return total_W_and_pjbit(hbm4_timing_json(rate), hbm4_power_json(rate, vddc, vddq), rate, ORG)

# HBM4 low-rail variants added to the POWER plot only (canonical VDDC 1.05 / VDDQ 0.9 comes from
# sweep_hbm4_power.py as-is). VDDQ 0.7 is the lowest I/O rail in the HBM4 spec; the second variant also drops
# the core to VDDC 1.00.
HBM4_VARIANTS = [
    dict(vddc=1.05, vddq=0.7, col="hbm4_c105q07_W", mk="v-", color="#16a085",
         label="HBM4 1.05 VDDC 0.7 VDDQ"),
    dict(vddc=1.00, vddq=0.7, col="hbm4_c100q07_W", mk="D-", color="#7d3c98",
         label="HBM4 1.00 VDDC 0.7 VDDQ"),
]

def h200_measured_pjbit():
    """Measured HBM3E read energy per bit on the three H200 systems (random-read microbenchmark,
    6.4 Gbps, ~94 % of peak bandwidth): total memory power / read throughput, per unit. Uses the
    same measurements (copied into data/) and loaders as hbm3e_validation/plot_model_vs_h200.py.
    Returns (mean, min, max)."""
    def emp_from_tuning(path):
        for r in csv.DictReader(open(path)):
            if r["label"] == "read_rand":
                return dict(total=float(r["mem_w_steady"]), read_GBs=float(r["bw_read_GBs"]))
    def emp_from_quick(path):
        d = {r["metric"]: r for r in csv.DictReader(open(path))}
        return dict(total=float(d["random_read_total"]["mem_power_W"]),
                    read_GBs=float(d["random_read_total"]["read_GBs"]))
    D = os.path.join(HERE, "data")
    samples = [emp_from_tuning(os.path.join(D, "emp_runs_tuning_summary.csv")),
               emp_from_quick(os.path.join(D, "emp_quick_145512.csv")),
               emp_from_quick(os.path.join(D, "emp_quick_145735.csv"))]
    pj = [s["total"] / (s["read_GBs"] * 8) * 1e3 for s in samples]     # W / (Gbit/s) = nJ/bit -> pJ/bit
    return sum(pj) / len(pj), min(pj), max(pj)


def load_hbm4():
    """HBM4 device power + energy/bit, taken as-is from sweep_hbm4_power.py (rate -> (total_W, pJbit))."""
    assert os.path.exists(HBM4CSV), f"run sweep_hbm4_power.py first to produce {HBM4CSV}"
    with open(HBM4CSV) as f:
        rows = list(csv.DictReader(f))
    return {float(r["rate_Gbps"]): (float(r["total_W"]), float(r["pJbit"])) for r in rows}

def main():
    assert os.path.exists(RUNNER), "build the runner: cmake --build build --target HBM3_runner"
    hbm4 = load_hbm4()

    rows = []
    for rate in RATES:
        bw_TBs = DQ_DEVICE * rate / 8 / 1000.0     # peak device BW (TB/s); same for both standards
        hbm4_W, hbm4_pJ = hbm4[rate]
        # Sanity: recomputing the canonical HBM4 (VDDC 1.05 / VDDQ 0.9) here must match sweep_hbm4_power.py's
        # CSV, confirming the in-script HBM4 builder is identical to sweep_hbm4_power.py's (so the VDDQ 0.7
        # variants below use the same methodology as the curve taken as-is).
        chk, _ = hbm4_total_W(rate, 1.05, 0.9)
        assert abs(chk - hbm4_W) <= 0.2, f"HBM4 recompute {chk:.2f} W != sweep_hbm4_power.py {hbm4_W} W @ {rate}"
        rec = {"rate_Gbps": rate, "bw_peak_TBs": round(bw_TBs, 2),
               "hbm4_split_W": round(hbm4_W, 1), "hbm4_split_pJbit": round(hbm4_pJ, 3)}
        for var in HBM4_VARIANTS:
            tot, pjb = hbm4_total_W(rate, var["vddc"], var["vddq"])
            rec[var["col"]] = round(tot, 1)
            rec[var["col"].replace("_W", "_pJbit")] = round(pjb, 3)
        for v in HBM3_VDDS:
            tag = f"{v:g}".replace(".", "")
            tot, pjb = hbm3_total_W(rate, v)
            rec[f"hbm3_{tag}_W"]     = round(tot, 1)
            rec[f"hbm3_{tag}_pJbit"] = round(pjb, 3)
        rows.append(rec)
        print(f"{rate:>4} GT/s | HBM4 {rec['hbm4_split_W']:>6.1f} W (1.05/0.9) "
              f"{rec['hbm4_c105q07_W']:>6.1f} (1.05/0.7) {rec['hbm4_c100q07_W']:>6.1f} (1.0/0.7) | "
              f"HBM3 {rec['hbm3_11_W']:>6.1f} (1.1V)")

    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.relpath(OUTCSV, REPO))

    # --- figures: device power (standalone); energy/bit (curves + normalized bars) ---
    R = [r["rate_Gbps"] for r in rows]
    # SUB = "HBM3 vs HBM4 (canonical 512-PC device, random reads)"
    SUB = ""
    at = {r["rate_Gbps"]: r for r in rows}

    def plot_curves(ax, plot_series, ylabel, title, legend_loc, constrain, legend_bbox=None, ncol=1):
        for label, mk, col, key, lw, rmax in plot_series:
            pts = [(r["rate_Gbps"], r[key]) for r in rows if not (constrain and r["rate_Gbps"] > rmax)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, mk, color=col, ms=5, lw=lw, label=label)
        ax.set_xlabel("Data Rate (Gbps/pin)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
        if legend_bbox is None:
            ax.legend(fontsize=7.5, loc=legend_loc, ncol=1)
        else:
            ax.legend(fontsize=7.5, loc=legend_loc, ncol=ncol, bbox_to_anchor=legend_bbox, borderaxespad=0.0, frameon=False, columnspacing=1.0)
        ax.grid(axis="both", ls=":", alpha=0.5); ax.set_xticks(R)

    def save(fig, out):
        fig.savefig(out + ".png", dpi=170, bbox_inches="tight")
        fig.savefig(out + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print("wrote", os.path.relpath(out + ".png", REPO), "and .pdf")

    # (label, marker, color, key-prefix, linewidth, legal-rate ceiling). The ceiling is enforced only
    # on the energy/bit plot (constrain=True); HBM3 stops at its 6.4 Gbps spec max there.
    baseline = [
        ("HBM3E 1.10 VDD", "o-", "#c0392b", "hbm3_11",   1.6, HBM3_MAX_RATE),
        ("HBM4 1.05 VDDC 0.9 VDDQ", "^-", "#1f6fe0", "hbm4_split",1.8, HBM4_MAX_RATE),
    ] + [(v["label"], v["mk"], v["color"], v["col"][:-2], 1.8, HBM4_MAX_RATE) for v in HBM4_VARIANTS]
    power_series = [(lbl, mk, col, f"{k}_W",     lw, rmax) for lbl, mk, col, k, lw, rmax in baseline]
    pjbit_series = [(lbl, mk, col, f"{k}_pJbit", lw, rmax) for lbl, mk, col, k, lw, rmax in baseline]

    # Power: standalone, full sweep (HBM3 extrapolated).
    figP, axP = plt.subplots(figsize=(8, 3))
    plot_curves(axP, power_series, "Device power (W)", f"Device power vs data rate\n{SUB}", "upper center", constrain=False)
    figP.tight_layout(); save(figP, OUTFIG_P)

    # Energy/bit: left = curves (legal ranges); right = HBM4 @ 8 GT/s normalized to HBM3 @ 6.4 GT/s.
    # right subplot smaller than left
    figE, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        gridspec_kw={"width_ratios": [2.5, 1.5], "wspace": 0.3},
    )
    # Measured HBM3E reference: H200 random-read energy per bit at 6.4 Gbps (mean of three units,
    # dashed) with the min-max range across units (band). Drawn first so the legend includes it.
    h200_mean, h200_lo, h200_hi = h200_measured_pjbit()
    axL.axhspan(h200_lo, h200_hi, color="#444", alpha=0.18, lw=0)
    axL.axhline(h200_mean, ls="--", color="#444", lw=1.2, label="HBM3E measured (H200, 6.4 Gbps)")
    print(f"H200 measured HBM3E read energy: {h200_mean:.3f} pJ/bit (range {h200_lo:.3f}-{h200_hi:.3f}); "
          f"model HBM3E 1.10 V @ 6.4 Gbps: {at[HBM3_MAX_RATE]['hbm3_11_pJbit']:.3f} pJ/bit")
    plot_curves(axL, pjbit_series, "Read Energy (pJ/bit)", "", "lower center", constrain=True, legend_bbox=(0.5, 1.0), ncol=2)

    ref = at[HBM3_MAX_RATE]["hbm3_11_pJbit"]      # HBM3 @ 6.4 GT/s (its best legal efficiency point)
    bars = [("VDDC 1.05\nVDDQ 0.9", "#1f6fe0", "hbm4_split_pJbit"),
            ("VDDC 1.05\nVDDQ 0.7", "#16a085", "hbm4_c105q07_pJbit"),
            ("VDDC 1.00\nVDDQ 0.7", "#7d3c98", "hbm4_c100q07_pJbit")]
    xs = list(range(len(bars)))
    norm = [at[HBM4_MAX_RATE][k] / ref for _, _, k in bars]   # pJ/bit @ 8 GT/s, normalized to HBM3 @ 6.4
    axR.bar(xs, norm, width=0.6, color=[c for _, c, _ in bars], linewidth=1.5, edgecolor="black")
    axR.axhline(1.0, ls="--", color="#444", lw=1.2, label="HBM3E @ 6.4 Gbps")
    axR.legend(fontsize=8, loc="upper right", frameon=False)
    # for x, h in zip(xs, norm):
    #     axR.annotate(f"{h:.3f}\n(−{(1 - h) * 100:.1f}%)", (x, h), textcoords="offset points",
    #                  xytext=(0, 3), ha="center", va="bottom", fontsize=8.5)
    axR.set_xticks(xs); axR.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=8.5)
    axR.set_ylabel("HBM4 pJ/bit @ 8 Gbps\nnormalized to HBM3E @ 6.4 Gbps")
    # axR.set_title("HBM4 energy/bit improvement\n(8 GT/s vs HBM3 6.4 GT/s)", fontsize=10)
    axR.set_ylim(0.9, 1.05); axR.grid(axis="y", ls=":", alpha=0.5)

    figE.suptitle(SUB, fontsize=11)
    figE.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88)
    save(figE, OUTFIG_E)

if __name__ == "__main__":
    main()
