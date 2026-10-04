#!/usr/bin/env python3
"""Per-workload MAPE summary across data regimes (the MAPE numbers quoted with Fig. 20)
-> table_mape_summary.tex.

For each (model, data pattern), compute the per-workload MAPE
(mean over per-chip measurements), then report the min / max / mean of those
per-workload MAPEs across the 14 workloads (6 microbenchmarks + the 8-point
LLaMa3.1-8B decode batch-size sweep). Three models x two regimes:

  model        all-0s config             random config            (regime difference)
  DRAMSim3     IDD_dramsim3              IDD_dramsim3             unchanged (no toggle term)
  FGDRAM HBM2  IDD_oconnor_notoggle      IDD_oconnor_50toggle     0% -> 50% toggle
  Ayna         IDD_ours_allzeros         IDD_ours_random          empirical all-0s -> random IDD4R

The model prediction is one engine run per (config, workload); the MAPE spread
comes from the per-chip measured ground truth. Engine = HBM2_runner (build from
the repository root; see docs/how-to/build-the-engine.md).
"""
import os, re, csv, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(REPO, "build", "bin", "HBM2_runner")
CFG    = os.path.join(HERE, "configs")
ORG    = os.path.join(CFG, "HBM2_organization.json")
TIMING = os.path.join(CFG, "HBM2_1.2Gbps_timing_BL4.json")   # read-timing fix: nBL=4

# workload -> (trace stem, all-0s ground-truth test_name, random ground-truth test_name)
WORKLOADS = [
    ("act_hammer",          "act_hammer",          "act_hammer_rand"),
    ("interleaved",         "interleaved",         "interleaved_rand"),
    ("streaming_all_banks", "streaming_all_banks", "streaming_all_banks_4cyc_rand"),
    ("streaming_reads",     "streaming_reads",     "streaming_reads_4cyc_rand"),
    ("wr_rd_turnaround",    "wr_rd_turnaround",    "wr_rd_turnaround_4cyc_rand"),
    ("ws_bg0_bg2",          "ws_bg0_bg2",          "ws_bg0_bg2_rand"),
]
# LLaMa3.1-8B decode batch-size sweep (replaces the single generic 'llm' workload)
WORKLOADS += [
    (f"bs{bs}_ctx1024",
     f"llama8Bshort_runner_pc0_bs{bs}_ctx1024_data_zeros",
     f"llama8Bshort_runner_pc0_bs{bs}_ctx1024_data_rand1")
    for bs in (1, 2, 4, 8, 16, 32, 64, 128)
]

# model -> (display name, all-0s IDD config, random IDD config)
MODELS = [
    ("DRAMSim3",    "IDD_dramsim3.json",         "IDD_dramsim3.json"),
    ("FGDRAM HBM2", "IDD_oconnor_notoggle.json", "IDD_oconnor_50toggle.json"),
    ("Ayna",        "IDD_ours_allzeros.json",    "IDD_ours_random.json"),
]

def run_power(idd_cfg, trace):
    out = subprocess.run([RUNNER, ORG, TIMING, os.path.join(CFG, idd_cfg), trace],
                         capture_output=True, text=True)
    m = re.search(r"Average power:\s+([-\d.]+)\s+mW", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"no power for {trace}:\n{out.stdout}\n{out.stderr}")
    return float(m.group(1))

def load_gt(path):
    gt = defaultdict(list)
    for row in csv.DictReader(open(path)):
        p = float(row["power_vdd_avg"])
        if p > 0:
            gt[row["test_name"]].append(p)
    # LLaMA rows: the artifact's LLaMA ground truth was taken with both HBM2 stacks active;
    # replace them with the single-chip
    # re-measurement (17 chips, zeros and random), keeping the microbenchmark rows as they are.
    llama_new = os.path.join(HERE, "data", "llama_single_chip",
                             "llama_ground_truth_random.csv" if "random" in os.path.basename(path) else "llama_ground_truth_allzeros.csv")
    if os.path.exists(llama_new):
        for k in [k for k in gt if k.startswith("llama")]:
            del gt[k]
        for row in csv.DictReader(open(llama_new)):
            p = float(row["power_vdd_avg"])
            if p > 0:
                gt[row["test_name"]].append(p)
    return gt

def per_workload_mape(idd_cfg, tn_index):
    """min, max, mean of per-workload MAPE across all workloads."""
    gt = GT_ZEROS if tn_index == 1 else GT_RANDOM
    mapes = []
    for row in WORKLOADS:
        stem, tn = row[0], row[tn_index]
        pred = run_power(idd_cfg, os.path.join(HERE, "traces", f"{stem}.csv"))
        meas = gt[tn]
        mapes.append(sum(abs(pred - x) / x * 100 for x in meas) / len(meas))
    return min(mapes), max(mapes), sum(mapes) / len(mapes)

def main():
    global GT_ZEROS, GT_RANDOM
    if not os.path.exists(RUNNER):
        sys.exit(f"HBM2_runner not found at {RUNNER}\nBuild it first (see README.md).")
    GT_ZEROS  = load_gt(os.path.join(HERE, "data", "ground_truth_allzeros.csv"))
    GT_RANDOM = load_gt(os.path.join(HERE, "data", "ground_truth_random.csv"))

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Per-workload MAPE summary across data regimes.}",
         r"\label{tab:mape_summary}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llccc}", r"\hline",
         r"Model & Data Pattern & Min MAPE (\%) & Max MAPE (\%) & Avg MAPE (\%) \\", r"\hline"]
    for name, idd_zeros, idd_random in MODELS:
        zmin, zmax, zavg = per_workload_mape(idd_zeros, 1)
        rmin, rmax, ravg = per_workload_mape(idd_random, 2)
        L.append(f"\\multirow{{2}}{{*}}{{{name}}} & all-0s & {zmin:.1f} & {zmax:.1f} & {zavg:.1f} \\\\")
        L.append(f" & random & {rmin:.1f} & {rmax:.1f} & {ravg:.1f} \\\\")
        L.append(r"\hline")
        print(f"{name:12} all-0s  min={zmin:4.1f} max={zmax:4.1f} avg={zavg:4.1f}   "
              f"random min={rmin:4.1f} max={rmax:4.1f} avg={ravg:4.1f}")
    L += [r"\end{tabular}%", r"}", r"\end{table}"]
    tex = "\n".join(L)

    out = os.path.join(HERE, "table_mape_summary.tex")
    open(out, "w").write(tex + "\n")
    print("\n" + tex)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
