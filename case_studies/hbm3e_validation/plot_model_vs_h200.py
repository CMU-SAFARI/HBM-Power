#!/usr/bin/env python3
"""Fig. 22: HBM3E (H200) data-pattern power model vs. empirical random-read power
-> model_vs_empirical_hbm3e_{read,idle_refresh}.{png,pdf}, the two subfigures of Fig. 22.

Per panel, three model bars (device total power at VDD 1.067 / 1.100 / 1.177 V) beside one
measured bar (min/max whisker over 3 H200 units): the read panel plots the random-read trace,
the idle+refresh panel the NOP + REFA trace. Each panel is a separate file with its own
y-axis.

Model: the data-pattern read/write energy model integrated into DRAMPower (HBM3, relative
form). Per pseudo-channel, HBM3_runner is run on a random-read trace with all three toggle
knobs at 0.5 (active) and on a NOP baseline trace with knobs at 0.0 (idle); both scaled by
N_PC = 192 pseudo-channels (HBM3E on H200, 2-SID) and summed.

Build the engine once from the repository root:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDRAMPOWER_BUILD_CLI=ON
    cmake --build build --target HBM3_runner -j
"""
import os, re, csv, json, subprocess, tempfile, sys
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(REPO, "build", "bin", "HBM3_runner")
CFG    = os.path.join(HERE, "configs")
ORG    = os.path.join(CFG, "HBM3_16Gb_8hi_organization.json")
TIM    = os.path.join(CFG, "HBM3_6400Mbps_timing.json")
PWR    = os.path.join(CFG, "HBM3_6400_power_datapattern.json")
SYN    = os.path.join(HERE, "traces", "hbm3_random_read_4rpa_ref.csv")   # random-read trace with REFA every tREFI and a tRFC stall per refresh (4.50 TB/s, measured 4.52)
BASE   = os.path.join(HERE, "traces", "hbm3_baseline_nop_ref.csv")   # NOP idle baseline
N_PC   = 192          # pseudo-channels (HBM3E on H200, 2-SID)
VDDS   = [1.067, 1.10, 1.177]   # JEDEC HBM3 VDD: min / nominal / max   # HBM3 nominal VDD is 1.1 V; the H200's operating point is not public

RE_POWER = re.compile(r"Average power:\s+([\d.]+)\s+mW")

def run_power(trace, dq, tsv, bg, vdd):
    """Per-pseudo-channel average power (mW) for the given toggle knobs and VDD override."""
    d = json.load(open(PWR)); d["voltage"]["VDD"] = vdd
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    try:
        out = subprocess.run([RUNNER, ORG, TIM, tmp, trace,
                              f"--dq-rate={dq}", f"--tsv-rate={tsv}", f"--bg-rate={bg}"],
                             capture_output=True, text=True)
    finally:
        os.remove(tmp)
    m = RE_POWER.search(out.stdout + out.stderr)
    if not m:
        raise RuntimeError("parse fail:\n" + out.stdout + out.stderr)
    return float(m.group(1))

# ---- empirical (3 H200 samples): one tuning-sweep summary + two quick-measure summaries ----
def emp_from_tuning(path):
    for r in csv.DictReader(open(path)):
        if r["label"] == "read_rand":
            tot, idle = float(r["mem_w_steady"]), float(r["mem_idle"])
            return dict(total=tot, idle=idle, active=tot - idle, read_GBs=float(r["bw_read_GBs"]))

def emp_from_quick(path):
    d = {r["metric"]: r for r in csv.DictReader(open(path))}
    tot, idle = float(d["random_read_total"]["mem_power_W"]), float(d["idle_baseline"]["mem_power_W"])
    return dict(total=tot, idle=idle, active=tot - idle, read_GBs=float(d["random_read_total"]["read_GBs"]))

def agg(s):
    out = {"n": len(s), "read_GBs": sum(x["read_GBs"] for x in s) / len(s)}
    for k in ("total", "idle", "active"):
        v = [x[k] for x in s]
        out[k], out[k + "_lo"], out[k + "_hi"] = sum(v) / len(v), min(v), max(v)
    return out

def main():
    if not os.path.exists(RUNNER):
        sys.exit(f"HBM3_runner not found at {RUNNER}\nBuild it first (see this script's header).")

    model = []
    for v in VDDS:
        # total = the read trace (it now carries refresh); idle = NOP+REFA trace; active = total - idle,
        # the same decomposition applied to the H200 measurement (benchmark power minus idle power).
        tot = N_PC * run_power(SYN,  0.5, 0.5, 0.5, v) / 1000.0
        idl = N_PC * run_power(BASE, 0.0, 0.0, 0.0, v) / 1000.0
        model.append(dict(vdd=v, active=tot - idl, idle=idl, total=tot))
        print(f"HBM3E model {v:>5} V: total {tot:6.1f} W = idle {idl:4.1f} + active {tot - idl:6.1f} W")

    D = os.path.join(HERE, "data")
    emp = agg([emp_from_tuning(os.path.join(D, "emp_runs_tuning_summary.csv")),
               emp_from_quick(os.path.join(D, "emp_quick_145512.csv")),
               emp_from_quick(os.path.join(D, "emp_quick_145735.csv"))])
    print(f"HBM3E empirical (H200, n={emp['n']}): active {emp['active']:.1f} "
          f"[{emp['active_lo']:.1f}-{emp['active_hi']:.1f}] | idle {emp['idle']:.1f} | "
          f"total {emp['total']:.1f} [{emp['total_lo']:.1f}-{emp['total_hi']:.1f}]")

    # ---- plot: two standalone panels (LaTeX subfigures), one file each. Left in the paper is
    #      the read trace, right is the idle+refresh trace (NOP + REFA); each panel carries its
    #      own y-axis, so nothing shares a scale and no twin axis is needed. Bars: Ayna at each
    #      VDD + the H200 measurement (hatched, min-max whisker over the 3 systems).
    import seaborn as sns
    import numpy as np
    vdd_colors = sns.color_palette("flare", len(model))
    c_meas = sns.color_palette("husl", 2)[1]
    PANELS = [("read", "total", ("total_lo", "total_hi"), "Read\nPower (W)", 300, [0, 100, 200, 300]),
              ("idle_refresh", "idle", ("idle_lo", "idle_hi"), "Idle+Refresh\nPower (W)", 50, [0, 20, 40])]
    # bars are named on the x axis, rotated 90 degrees: two lines each ("Ayna" over its VDD), which
    # costs two line heights of slot width instead of the label's length, so nothing collides.
    xt = [f"Ayna\n{row['vdd']:.3f} V" for row in model] + ["H200"]
    for stem, key, err, ylab, ymax, yticks in PANELS:
        fig, ax = plt.subplots(figsize=(1.63, 0.95))
        for i, row in enumerate(model):
            ax.bar(i, row[key], width=0.8, color=vdd_colors[i], edgecolor="black", linewidth=0.6)
        xm = len(model)
        ax.bar(xm, emp[key], width=0.8, color=c_meas, edgecolor="black", linewidth=0.6, hatch="//")
        ax.errorbar(xm, emp[key], yerr=[[emp[key] - emp[err[0]]], [emp[err[1]] - emp[key]]],
                    fmt="none", ecolor="black", capsize=2.5, lw=0.9, zorder=5)
        ax.set_ylim(0, ymax); ax.set_yticks(yticks)
        ax.set_ylabel(ylab, fontsize=6.5, labelpad=1.5)
        ax.set_xlim(-0.6, xm + 0.6)
        ax.set_xticks(range(xm + 1))
        ax.set_xticklabels(xt, fontsize=6, rotation=90, ha="center", va="top")
        ax.tick_params(axis="y", labelsize=6, pad=1.5); ax.tick_params(axis="x", length=0, pad=1.5)
        ax.grid(axis="y", ls="--", alpha=0.5); ax.set_axisbelow(True)
        fig.tight_layout(pad=0.15)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(HERE, f"model_vs_empirical_hbm3e_{stem}.{ext}"), dpi=170, bbox_inches="tight")
        plt.close(fig)
    print("\nwrote " + ", ".join(f"{HERE}/model_vs_empirical_hbm3e_{stem}.{{png,pdf}}" for stem, *_ in PANELS))

if __name__ == "__main__":
    main()
