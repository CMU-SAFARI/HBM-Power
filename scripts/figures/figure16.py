#!/usr/bin/env python3
"""Figure 16: HBM3E (H200) data-pattern power model vs. empirical random-read power.

Reproduces figures/model_vs_empirical_hbm3e.pdf (revision/figureC).

Uses BOTH:
  * the DRAMPower data-pattern model (sources/drampower -> HBM3_runner), and
  * empirical H200 power measurements (data/HBM3E_measurements.csv, 3 units).

Three model bars (device total power at VDD 1.1 / 1.15 / 1.2 V) beside one measured bar
(active + idle, min/max whisker over 3 H200 units). Per pseudo-channel, HBM3_runner is run
on a random-read trace with all toggle knobs at 0.5 (active) and on a NOP baseline trace at
0.0 (idle); both scaled by N_PC = 192 and summed.
"""
import os, re, csv, json, subprocess, tempfile, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = Path(os.environ.get("DATA_DIR", str(REPO_ROOT / "data")))
SRC       = REPO_ROOT / "sources" / "figure16"
ENGINE    = REPO_ROOT / "sources" / "drampower"
BUILD     = ENGINE / "build"
RUNNER    = BUILD / "bin" / "HBM3_runner"
FIG_DIR   = Path(os.environ.get("FIG_DIR", str(REPO_ROOT / "figures")))
FIG_DIR.mkdir(exist_ok=True)

CFG = SRC / "configs"
ORG = str(CFG / "HBM3_16Gb_8hi_organization.json")
TIM = str(CFG / "HBM3_6400Mbps_timing.json")
PWR = str(CFG / "HBM3_6400_power_datapattern.json")
SYN = str(SRC / "traces" / "hbm3_random_read_4rpa.csv")   # random read, 4 reads/activate
BASE = str(SRC / "traces" / "hbm3_baseline_nop_ref.csv")  # NOP idle baseline
N_PC = 192          # pseudo-channels (HBM3E on H200, 2-SID)
VDDS = [1.1, 1.15, 1.2]

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


def run_power(trace, dq, tsv, bg, vdd):
    """Per-pseudo-channel average power (mW) for the given toggle knobs and VDD override."""
    d = json.load(open(PWR)); d["voltage"]["VDD"] = vdd
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    try:
        out = subprocess.run([str(RUNNER), ORG, TIM, tmp, trace,
                              f"--dq-rate={dq}", f"--tsv-rate={tsv}", f"--bg-rate={bg}"],
                             capture_output=True, text=True)
    finally:
        os.remove(tmp)
    m = RE_POWER.search(out.stdout + out.stderr)
    if not m:
        raise RuntimeError("parse fail:\n" + out.stdout + out.stderr)
    return float(m.group(1))


# ---- empirical (3 H200 samples), one tidy row per unit: total/idle power + read BW ----
def load_measurements(path):
    samples = []
    for r in csv.DictReader(open(path)):
        tot, idle = float(r["total_power_W"]), float(r["idle_power_W"])
        samples.append(dict(total=tot, idle=idle, active=tot - idle, read_GBs=float(r["read_GBs"])))
    return samples


def agg(s):
    out = {"n": len(s), "read_GBs": sum(x["read_GBs"] for x in s) / len(s)}
    for k in ("total", "idle", "active"):
        v = [x[k] for x in s]
        out[k], out[k + "_lo"], out[k + "_hi"] = sum(v) / len(v), min(v), max(v)
    return out


def main():
    ensure_runner()

    model = []
    for v in VDDS:
        act = N_PC * run_power(SYN,  0.5, 0.5, 0.5, v) / 1000.0
        idl = N_PC * run_power(BASE, 0.0, 0.0, 0.0, v) / 1000.0
        model.append(dict(vdd=v, active=act, idle=idl, total=act + idl))
        print(f"HBM3E model {v:>4} V: active {act:6.1f} + idle {idl:4.1f} = {act + idl:6.1f} W")

    emp = agg(load_measurements(DATA_DIR / "HBM3E_measurements.csv"))
    print(f"HBM3E empirical (H200, n={emp['n']}): active {emp['active']:.1f} "
          f"[{emp['active_lo']:.1f}-{emp['active_hi']:.1f}] | idle {emp['idle']:.1f} | "
          f"total {emp['total']:.1f} [{emp['total_lo']:.1f}-{emp['total_hi']:.1f}]")

    # ---- plot ----
    ACT, EACT, EIDLE = "#1f6fe0", "#e8710a", "#f6c9a0"
    fig, ax = plt.subplots(figsize=(8, 2.3))
    for i, row in enumerate(model):
        ax.bar(i, row["total"], color=ACT,
               label="HBM3E Power Model (Random Data)" if i == 0 else None)
    x = 3
    ax.bar(x, emp["active"], color=EACT, label="Measured Active Power (Random Data)")
    ax.bar(x, emp["idle"], bottom=emp["active"], color=EIDLE, label="Measured Idle Power")
    ax.errorbar(x, emp["total"], yerr=[[emp["total"] - emp["total_lo"]], [emp["total_hi"] - emp["total"]]],
                fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"Model\nVDD={model[0]['vdd']:g} V", f"Model\nVDD={model[1]['vdd']:g} V",
                        f"Model\nVDD={model[2]['vdd']:g} V", f"HBM3E\nMeasurement\n(n={emp['n']})"],
                       fontsize=10)
    ax.set_ylabel("DRAM Power (W)", fontsize=11); ax.set_ylim(180, 280)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    ax.yaxis.get_major_ticks()[0].label1.set_color("red")
    out = FIG_DIR / f"{Path(__file__).stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
