#!/usr/bin/env python3
"""predict HBM4 random-read power across the JEDEC speed bins using our data-pattern
model -> hbm4_power_vs_datarate.{png,pdf} + hbm4_power_sweep.csv.

Method (mirrors the validated HBM3E pipeline reproduced in hbm3e_validation/):
  per-PC active = HBM3 engine on the synthetic random-read trace at toggle 0.5 (random data,
                  all three buses) with the data-pattern read/write model;
  per-PC idle   = engine standby floor (NOP+REF baseline trace, toggle 0);
  device power  = N_PC * (active + idle).

HBM4 vs HBM3 (JEDEC JESD270-4):
  * 32 channels/stack x 2 PC = 64 PC/stack (2048-bit interface, 2x HBM3's 1024). Canonical
    device = 8 stacks => N_PC = 512.
  * 9 speed bins: 4.8 .. 8.0 Gbps/pin (tCK = 4/rate, since dataRate=4 transfers/CK).
  * No published IDD currents (Table 107 is an empty template). We carry over the HBM3E-
    calibrated rails: IDD2N and the read/write currents scale linearly with data rate; IDD3N1 /
    IDD3N16 sit fixed deltas above IDD2N; the refresh increment is fixed and the activate
    increment follows the fixed-pacing IDD0 trend (see array_at and the HBM3E config's
    `extrapolation` block).
  * Split rails (JESD270-4 p.176): core VDDC typ 1.05 V powers all core/array energy; I/O VDDQ
    typ 0.9 V powers only the external DQ pins, so it scales the DQ component of the read current
    (IDD4R) only. The engine isolates that component from the data-pattern model (the on-die
    floor/TSV/BG terms stay on VDDC); writes stay on VDDC for now.

Engine: HBM4 reuses the HBM3 core engine + data-pattern model (HBM3_runner). Build once from the
repo root:  cmake --build build --target HBM3_runner -j   (see docs/how-to/build-the-engine.md).
See docs/explanation/hbm4-power-prediction.md for the spec facts and assumptions.
"""
import os, re, csv, json, math, tempfile, subprocess
import tracegen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(REPO, "build", "bin", "HBM3_runner")   # HBM4 reuses the HBM3 core engine
ORG    = os.path.join(HERE, "configs", "HBM4_organization.json")
SYN    = os.path.join(HERE, "traces", "hbm3_random_read_4rpa.csv")   # 1 PC, 4 reads/activate
BASE   = os.path.join(HERE, "traces", "hbm3_baseline_nop_ref.csv")   # NOP+REF idle floor
OUTCSV = os.path.join(HERE, "hbm4_power_sweep.csv")
OUTFIG = os.path.join(HERE, "hbm4_power_vs_datarate")

# --- canonical organization -------------------------------------------------
N_PC      = 64 * 8       # 64 PC/stack x 8 stacks
DQ_PER_PC = 32           # PC-mode DQ width
DQ_DEVICE = DQ_PER_PC * N_PC          # = 16384 DQ across the device
RATES     = [4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0]   # Gbps/pin (JEDEC speed bins)
# Split rails (JESD270-4 Table, p.176): core VDDC and I/O VDDQ are separate supplies.
# VDDC powers all core/array/on-die energy; VDDQ powers only the external DQ pins, so it
# applies to the DQ component of the read I/O current (IDD4R) only -- the engine isolates
# that component from the data-pattern model (floor/TSV/BG stay on VDDC).
VDDC = 1.05              # core supply, VDDC typ (1.018 / 1.05 / 1.124 V)
VDDQ = 0.9               # I/O supply,  VDDQ typ (0.873 / 0.9 / 0.963 V)

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

# --- timings ----------------------------------------------------------------
# HBM4 absolute timings (JESD270-4 Table 101): tRC=48, tRAS=33, tRP=15 ns.
# Remaining cycle counts carried from the HBM3E 6.4-GT/s bin at constant absolute ns.
HBM3E_625 = dict(nRCD=31, nRCDWR=15, nRL=20, nWL=10, nCCD_S=2, nCCD_L=4,
                 nWTR_S=7, nWTR_L=10, nWR=33, nRTP=9)
def cyc(ns, tck_ns, lo=1):
    return max(lo, math.ceil(ns / tck_ns))

def timing_json(rate):
    tck = 4.0 / rate                 # ns; dataRate=4 transfers/CK
    t = {"tCK_ps": round(tck * 1000, 3), "nBL": 8,
         "nRC": cyc(48, tck), "nRAS": cyc(33, tck), "nRP": cyc(15, tck)}
    for k, c in HBM3E_625.items():
        t[k] = cyc(c * 0.625, tck, lo=2 if k.startswith("nCCD") else 1)
    t["nCCD_S"], t["nCCD_L"] = 2, 4  # HBM4 column-to-column delays in clock cycles
    t["nREFI"] = cyc(3900, tck); t["nRFCpb"] = cyc(280, tck); t["nRFC"] = cyc(350, tck)   # all-bank tRFC 350 ns (assumption)
    return {"timing": t}

def power_json(rate):
    return {
        "voltage": {"VDD": VDDC, "VDDQ": VDDQ, "unit": "V"},
        "IDD": {**array_at(rate),
                "IDD4R": round(io_current(IDD4R_B, rate), 1),
                "IDD4W": round(io_current(IDD4W_B, rate), 1), "unit": "mA"},
        "datapattern": {
            "enabled": True, "floor_pJbit": 2.759, "coef_T_DQ": 1.192,
            "coef_T_2bit": 1.331, "coef_busflip": 0.72, "use_coupling": False,
            "dq_rate": 0.5, "tsv_rate": 0.5, "bg_rate": 0.5,
            "ref_dq_rate": 0.0, "ref_tsv_rate": 0.0, "ref_bg_rate": 0.0,
            "apply_to_writes": True},
    }

RE_POWER = re.compile(r"Average power:\s+([\d.]+)\s+mW")
def run(timing, power, trace, dq, tsv, bg):
    def tmp(obj):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f: json.dump(obj, f)
        return p
    tj, pj = tmp(timing), tmp(power)
    cmd = [RUNNER, ORG, tj, pj, trace, f"--dq-rate={dq}", f"--tsv-rate={tsv}", f"--bg-rate={bg}"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tj); os.remove(pj)
    m = RE_POWER.search(out.stdout + out.stderr)
    if not m: raise RuntimeError("parse fail:\n" + out.stdout + out.stderr)
    return float(m.group(1))   # per-PC mW

TRACE_DIR = os.path.join(HERE, "traces", "generated")
def make_traces(rate, tj):
    """Per-rate traces: REFA every tREFI (3.9 us) with a tRFC (350 ns) stall, in cycles of this bin."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    nREFI, nRFC = tj["timing"]["nREFI"], tj["timing"]["nRFC"]
    rd = os.path.join(TRACE_DIR, f"random_read_4rpa_ref_{rate:.1f}Gbps.csv")
    idle = os.path.join(TRACE_DIR, f"baseline_nop_ref_{rate:.1f}Gbps.csv")
    info = tracegen.write_read_trace(SYN, rd, nREFI, nRFC, banks_per_bg=json.load(open(ORG))["banks_per_bankgroup"])
    tracegen.write_idle_trace(idle, nREFI)
    return rd, idle, info

def main():
    assert os.path.exists(RUNNER), "build the runner: cmake --build build --target HBM3_runner"
    rows = []
    for rate in RATES:
        tj = timing_json(rate)
        pj = power_json(rate)
        bw_dev_TBs = DQ_DEVICE * rate / 8 / 1000.0     # peak device BW (TB/s)
        tck_ns = tj["timing"]["tCK_ps"] / 1000.0
        rd_trace, idle_trace, info = make_traces(rate, tj)
        # total = the refresh-bearing read trace (standby, refresh and reads in one run, nothing
        # double counted); idle = the NOP+REFA trace. Same decomposition as hbm3e_validation.
        tot = run(tj, pj, rd_trace, 0.5, 0.5, 0.5) * N_PC / 1000.0   # W
        idl = run(tj, pj, idle_trace, 0.0, 0.0, 0.0) * N_PC / 1000.0  # W
        act = tot - idl
        # energy per transferred bit at the bandwidth the trace actually achieves (reads x 256 bit
        # over the trace duration, refresh stalls included)
        bits = info["n_rd"] * 256 * N_PC
        dur_s = info["end_cycle"] * tck_ns * 1e-9
        bw_ach_TBs = bits / 8 / dur_s / 1e12
        pj_bit = tot * dur_s / bits * 1e12
        rec = {"rate_Gbps": rate, "tCK_ps": tj["timing"]["tCK_ps"],
               "IDD4R_mA": pj["IDD"]["IDD4R"], "bw_peak_TBs": round(bw_dev_TBs, 2),
               "bw_ach_TBs": round(bw_ach_TBs, 2),
               "active_W": round(act, 1), "idle_W": round(idl, 1),
               "total_W": round(tot, 1), "pJbit": round(pj_bit, 3)}
        rows.append(rec)
        print(f"{rate:>4} GT/s | tCK {rec['tCK_ps']:>6} ps | BWpk {rec['bw_peak_TBs']:>5.1f} TB/s | "
              f"total {rec['total_W']:>6.1f} W ({rec['pJbit']:.2f} pJ/bit) "
              f"[VDDC={VDDC} V / VDDQ={VDDQ} V]")

    with open(OUTCSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", os.path.relpath(OUTCSV, REPO))

    # --- figure: device power vs data rate (stacked active+idle, split rails) ---
    R = [r["rate_Gbps"] for r in rows]
    act = [r["active_W"] for r in rows]; idl = [r["idle_W"] for r in rows]
    tot = [r["total_W"] for r in rows]
    bw = [r["bw_peak_TBs"] for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.28
    ax.bar(R, act, width=w, color="#1f6fe0", label="Active (random data)")
    ax.bar(R, idl, width=w, bottom=act, color="#f6c9a0", label="Idle (standby+refresh)")
    for x, y in zip(R, tot):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5)
    ax.set_xlabel("Data rate (Gbps/pin)"); ax.set_ylabel("HBM4 device power (W)")
    ax.set_title(f"HBM4 random-read power vs data rate\n"
                 f"(8 stacks, 64 PC/stack, VDDC={VDDC} V / VDDQ={VDDQ} V)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_xticks(R)

    eff = [r["pJbit"] for r in rows]
    axb = ax2.twinx()
    ax2.plot(R, bw, "s-", color="#0a8f3c", label="Peak bandwidth")
    axb.plot(R, eff, "^-", color="#c0392b", label="Energy/bit")
    ax2.set_xlabel("Data rate (Gbps/pin)")
    ax2.set_ylabel("Peak device bandwidth (TB/s)", color="#0a8f3c")
    axb.set_ylabel("Energy efficiency (pJ/bit)", color="#c0392b")
    ax2.set_title("HBM4 bandwidth & energy efficiency", fontsize=10)
    ax2.grid(axis="y", ls=":", alpha=0.5); ax2.set_xticks(R)
    lines = ax2.get_lines() + axb.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTFIG + ".png", dpi=170, bbox_inches="tight")
    plt.savefig(OUTFIG + ".pdf", bbox_inches="tight")
    print("wrote", os.path.relpath(OUTFIG + ".png", REPO), "and .pdf")

if __name__ == "__main__":
    main()
