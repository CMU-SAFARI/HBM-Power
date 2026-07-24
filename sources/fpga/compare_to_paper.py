#!/usr/bin/env python3
"""Compare a freshly measured single-chip result against the paper's committed data.

fpga7/chip0 == chip_id 0 in the released fleet CSVs, so we compare the fresh chip
directly against the paper's own chip_id-0 values, and also show where it lands in
the 36-chip fleet distribution.

Usage:  compare_to_paper.py <kind>
  kind in: idd0 idd2 idd3n1 idd3n16 idd4r idd4w idd5b
           no_hbm bank_group bank_offset bitflip beat trace_allzeros trace_random

Env: DATA_DIR (paper, default artifact/data), NEW_DIR (fresh, default artifact/new_data)
Prints a readout and a final 'NOTIFY: ...' one-liner for the push notification.
"""
import csv, os, sys, collections, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.abspath(os.path.join(HERE, "..", ".."))   # HBM-Power/
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ARTIFACT, "data"))
NEW_DIR  = os.environ.get("NEW_DIR",  os.path.join(ARTIFACT, "data/new/fpga"))
CHIP = os.environ.get("CHIP_ID", "0")

# kind -> (csv basename, metric col, test_loop filter or None, human label, unit)
IDD = "all_idd_measurements.csv"
SPEC = {
    "idd0":    (IDD, "idd", "IDD0",    "IDD0 (activate/precharge)", "mA"),
    "idd2":    (IDD, "idd", "IDD2",    "IDD2 (idle standby)",       "mA"),
    "idd3n1":  (IDD, "idd", "IDD3N1",  "IDD3N1 (1 bank active)",    "mA"),
    "idd3n16": (IDD, "idd", "IDD3N16", "IDD3N16 (16 banks active)", "mA"),
    "idd4r":   (IDD, "idd", "IDD4R",   "IDD4R (read)",              "mA"),
    "idd4w":   (IDD, "idd", "IDD4W",   "IDD4W (write)",             "mA"),
    "idd5b":   (IDD, "idd", "IDD5B",   "IDD5B (refresh)",           "mA"),
    "no_hbm":  ("no_hbm_idd2_measurements.csv", "idd", None, "no-HBM baseline", "mA"),
    "bank_group":  ("bank_group_measurements.csv",  "idd", None, "Fig10 bank-group invert", "mA"),
    "bank_offset": ("bank_offset_measurements.csv", "idd", None, "Fig11 bank-offset",       "mA"),
    "bitflip":     ("bitflip_measurements.csv",     "idd", None, "Fig12 DQ bitflip",        "mA"),
    "beat":        ("beat_pattern_combined_measurements.csv", "idd", None, "Fig13 beat-pattern", "mA"),
    "trace_allzeros": ("ground_truth_allzeros.csv", "power_vdd_avg", None, "Fig14 trace (all-0s)",  "mW"),
    "trace_random":   ("ground_truth_random.csv",   "power_vdd_avg", None, "Fig14 trace (random)", "mW"),
}


def load(path, metric, loop):
    """Return {chip_id: [values]} and overall list for chip0."""
    per = collections.defaultdict(list)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            if loop and r.get("test_loop") != loop:
                continue
            try:
                per[r["chip_id"]].append(float(r[metric]))
            except (KeyError, ValueError):
                continue
    return per


def stat(per):
    means = [statistics.mean(v) for v in per.values() if v]
    return (min(means), sum(means)/len(means), max(means)) if means else (None, None, None)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SPEC:
        sys.exit("usage: compare_to_paper.py <%s>" % "|".join(SPEC))
    kind = sys.argv[1]
    base, metric, loop, label, unit = SPEC[kind]
    paper = load(os.path.join(DATA_DIR, base), metric, loop)
    fresh = load(os.path.join(NEW_DIR,  base), metric, loop)

    print("=" * 64)
    print(f"  {label}   [{base}{(' / '+loop) if loop else ''}]")
    print("=" * 64)
    if not fresh or CHIP not in fresh or not fresh[CHIP]:
        print(f"  FRESH: no chip{CHIP} data in {NEW_DIR}/{base} yet.")
        print(f"NOTIFY: {label}: FRESH MISSING (no chip{CHIP} rows)")
        return
    fv = fresh[CHIP]
    fmean = statistics.mean(fv)
    print(f"  FRESH chip{CHIP}: mean={fmean:.1f} {unit}  (n={len(fv)} samples, "
          f"min={min(fv):.0f} max={max(fv):.0f})")

    if not paper or CHIP not in paper or not paper[CHIP]:
        print(f"  PAPER: no chip{CHIP} reference in {DATA_DIR}/{base}.")
        print(f"NOTIFY: {label}: fresh={fmean:.1f}{unit} (no paper chip{CHIP} ref)")
        return
    pv = paper[CHIP]
    pmean = statistics.mean(pv)
    dev = 100.0 * (fmean - pmean) / pmean if pmean else float("nan")
    fmin, favg, fmax = stat(paper)          # fleet distribution across all chips
    inrange = (fmin is not None and fmin <= fmean <= fmax)
    print(f"  PAPER chip{CHIP}: mean={pmean:.1f} {unit}  (n={len(pv)} samples)")
    print(f"  DEVIATION vs paper chip{CHIP}: {dev:+.1f}%")
    print(f"  Fleet (36 chips): min={fmin:.1f} mean={favg:.1f} max={fmax:.1f} {unit}"
          f"   -> fresh is {'WITHIN' if inrange else 'OUTSIDE'} fleet range")
    flag = "OK" if abs(dev) <= 10 else ("WARN" if abs(dev) <= 20 else "LARGE")
    print(f"NOTIFY: {label}: fresh={fmean:.1f}{unit} vs paper-chip0={pmean:.1f}{unit} "
          f"({dev:+.1f}%, {flag}); fleet {'in' if inrange else 'OUT of'} range")


if __name__ == "__main__":
    main()
