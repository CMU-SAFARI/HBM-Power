#!/usr/bin/env python3
"""Selective single-chip standardizer -> new_data/, one piece at a time.

Mirrors standardize_single_chip.py's single-chip patching but runs ONLY the
requested output, so we can standardize progressively after each measurement
without touching subdirs that haven't been measured yet.

Usage: standardize_one.py <what>
  what in: all_idd no_hbm bank_group bank_offset bitflip beat
"""
import os, sys, csv, importlib, collections

ART = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRIV_STD_DIR = os.environ.get(
    "PRIV_STD_DIR",
    os.path.join(ART, "sources", "fpga", "DRAMBender", "sources",
                 "apps", "Power_structural_variation", "standardize"))
RESULTS_DIR = os.path.join(ART, "results")
OUTPUT_DIR = os.path.join(ART, "new_data")
FPGA_NUM, CHIP = 7, 0
IN_SCOPE_IDD = ("IDD0", "IDD2", "IDD3N1", "IDD3N16", "IDD4R", "IDD4W", "IDD5B")

EXTRACTORS = {
    "no_hbm": "extract_no_hbm", "bank_group": "extract_bank_group",
    "bank_offset": "extract_bank_offset", "bitflip": "extract_bitflip",
    "beat": "extract_beat_pattern_combined",
}


def derive_beat_perpattern():
    combined = os.path.join(OUTPUT_DIR, "beat_pattern_combined_measurements.csv")
    out_path = os.path.join(OUTPUT_DIR, "beat_pattern_perpattern.csv")
    if not os.path.isfile(combined):
        print("  WARNING: combined missing; skip perpattern"); return
    agg = collections.defaultdict(lambda: [0.0, 0])
    with open(combined) as f:
        for row in csv.DictReader(f):
            if row.get("test_loop") != "max_power_loop":
                continue
            bp = str(row["beat_pattern"]).zfill(4)
            if len(bp) > 4:
                continue
            key = (bp, row["col1_inverted"])
            agg[key][0] += float(row["idd"]); agg[key][1] += 1
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beat_pattern", "col1_inverted", "mean_idd_mA", "n_chips"])
        for (bp, inv), (total, n) in sorted(agg.items()):
            w.writerow([bp, inv, (total / n) if n else 0.0, n])
    print("Written:", out_path)


def main():
    what = sys.argv[1] if len(sys.argv) == 2 else ""
    if what not in ("all_idd",) + tuple(EXTRACTORS):
        sys.exit("usage: standardize_one.py <all_idd|%s>" % "|".join(EXTRACTORS))

    sys.path.insert(0, PRIV_STD_DIR)
    import generate_standardized_csvs as g
    g.FPGA_NUMBERS = [FPGA_NUM]
    g.CHIPS_PER_FPGA = [CHIP]
    g.RESULTS_DIR = RESULTS_DIR
    g.OUTPUT_DIR = OUTPUT_DIR
    g.IDD_TESTS = {k: v for k, v in g.IDD_TESTS.items() if k in IN_SCOPE_IDD}

    _orig = g.resolve_filepath
    def _alias(f, c, s, fn):
        p = _orig(f, c, s, fn)
        if p is None and "idd4r_full" in fn:
            p = _orig(f, c, s, fn.replace("idd4r_full", "idd4r_16bank"))
        return p
    g.resolve_filepath = _alias

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = g.build_chip_mapping()

    if what == "all_idd":
        g.generate_chip_mapping_csv(mapping, os.path.join(OUTPUT_DIR, "chip_mapping.csv"))
        g.generate_all_tests_csv(mapping, os.path.join(OUTPUT_DIR, "all_idd_measurements.csv"))
    else:
        mod = importlib.import_module(EXTRACTORS[what])
        mod.OUTPUT_DIR = OUTPUT_DIR
        mod.main()
        if what == "beat":
            derive_beat_perpattern()


if __name__ == "__main__":
    main()
