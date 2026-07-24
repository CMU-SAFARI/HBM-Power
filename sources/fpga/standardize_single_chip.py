#!/usr/bin/env python3
"""
Single-chip standardizer for the HBM2-power FPGA artifact evaluation.

The released standardizer (the vendored DRAMBender/.../standardize/) walks the full 36-chip
fleet. For the AE we have ONE chip of ONE FPGA. Rather than fork and re-verify
the extractor regexes, this driver *reuses* the fleet extractors unchanged but
overrides their configuration to a single FPGA / single chip and repoints the
input/output directories at this artifact's tree:

  input :  <artifact>/results/safari-fpga<N>/<subdir>_fixed_reset_full_ipp/   (copy_results.sh)
  output:  <artifact>/new_data/*.csv                                         (the sanitized CSVs)

It produces exactly the in-scope standardized CSVs:
  all_idd_measurements.csv             (Figs 2-9)
  no_hbm_idd2_measurements.csv         (baseline, all figs)
  bank_group_measurements.csv          (Fig 10)
  bank_offset_measurements.csv         (Fig 11)
  bitflip_measurements.csv             (Fig 12)
  beat_pattern_combined_measurements.csv (Fig 13)
  beat_pattern_perpattern.csv          (Fig 15, derived here -- no extractor emits it)
  chip_mapping.csv                     (reference)

Config via environment (all optional):
  PRIV_STD_DIR  path to the fleet standardize/ dir
                (default: <artifact>/sources/fpga/DRAMBender/sources/apps/
                 Power_structural_variation/standardize -- the vendored copy)
  RESULTS_DIR   raw per-run CSV root       (default: <artifact>/results)
  OUTPUT_DIR    sanitized CSV output       (default: <artifact>/data/new/fpga)
  FPGA_NUM      FPGA number label          (default: 7)
  CHIP          chip within the FPGA, 0/1  (default: 0)

Exit codes: 0 success; 1 configuration/plumbing error.
"""

import csv
import importlib
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # HBM-Power/
PRIV_STD_DIR = os.environ.get(
    "PRIV_STD_DIR",
    os.path.join(ARTIFACT_ROOT, "sources", "fpga", "DRAMBender", "sources", "apps",
                 "Power_structural_variation", "standardize"),
)
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ARTIFACT_ROOT, "results"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ARTIFACT_ROOT, "new_data"))
FPGA_NUM = int(os.environ.get("FPGA_NUM", "7"))
CHIP = int(os.environ.get("CHIP", "0"))

# In-scope IDD loops only (drops IDD3/IDD7/actpre to avoid noisy "Missing" warnings)
IN_SCOPE_IDD = ("IDD0", "IDD2", "IDD3N1", "IDD3N16", "IDD4R", "IDD4W", "IDD5B")


def fail(msg):
    sys.stderr.write("standardize: ERROR: " + msg + "\n")
    sys.exit(1)


def _alias_bitflip_idd4r_label(mod):
    """extract_bitflip only matches IDD4R-full files named `idd4r_full_32b`, but the
    current binary labels them `idd4r_16bank_32b` (the --use-full ambiguity, build
    spec 3.4), so a fresh bitflip run standardizes to 0 IDD4R rows (Fig 12). Widen the
    extractor's regex to accept either label. The `(?:...)` is non-capturing, so the
    group indices extract_bitflip.main relies on are unchanged; name-level only, no
    measurement semantics change."""
    import re
    mod._IDD4R_FULL_RE = re.compile(
        mod._IDD4R_FULL_RE.pattern.replace(
            "hbm_idd4r_full_32b_", "hbm_idd4r_(?:full|16bank)_32b_"),
        mod._IDD4R_FULL_RE.flags,
    )


def main():
    if not os.path.isdir(PRIV_STD_DIR):
        fail("fleet standardize dir not found: %s\n"
             "       Set PRIV_STD_DIR to the Power_structural_variation/standardize dir."
             % PRIV_STD_DIR)
    if not os.path.isdir(RESULTS_DIR):
        fail("results dir not found: %s\n"
             "       Run copy_results.sh first so the raw CSVs are present."
             % RESULTS_DIR)

    # Import the fleet standardizer and reconfigure it for a single chip. We
    # patch BEFORE importing any extractor so that each extractor's
    # `from generate_standardized_csvs import ...` binds the patched values.
    sys.path.insert(0, PRIV_STD_DIR)
    import generate_standardized_csvs as g

    g.FPGA_NUMBERS = [FPGA_NUM]
    g.CHIPS_PER_FPGA = [CHIP]
    g.RESULTS_DIR = RESULTS_DIR
    g.OUTPUT_DIR = OUTPUT_DIR
    g.IDD_TESTS = {k: v for k, v in g.IDD_TESTS.items() if k in IN_SCOPE_IDD}

    # Robustness for the IDD4R "--use-full" filename ambiguity (build spec 3.4):
    # the released key is `idd4r_full`, but run_idd4r_8ch_test.sh passes
    # --use-full, which the current binary labels `idd4r_16bank`. Accept either
    # by retrying resolution with the alternate label. This is a name-level
    # fallback only -- it changes no measurement semantics.
    _orig_resolve = g.resolve_filepath

    def _resolve_with_idd4r_alias(fpga, chip_in_fpga, subdir, filename):
        path = _orig_resolve(fpga, chip_in_fpga, subdir, filename)
        if path is None and "idd4r_full" in filename:
            path = _orig_resolve(fpga, chip_in_fpga, subdir,
                                 filename.replace("idd4r_full", "idd4r_16bank"))
        return path

    g.resolve_filepath = _resolve_with_idd4r_alias

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = g.build_chip_mapping()

    print("=== single-chip standardize (fpga%d, chip%d) ===" % (FPGA_NUM, CHIP))
    print("  results: %s" % RESULTS_DIR)
    print("  output : %s" % OUTPUT_DIR)
    print()

    # --- Figs 2-9: all IDD loops ---
    g.generate_chip_mapping_csv(mapping, os.path.join(OUTPUT_DIR, "chip_mapping.csv"))
    g.generate_all_tests_csv(mapping, os.path.join(OUTPUT_DIR, "all_idd_measurements.csv"))

    # --- Extractors (each writes one standardized CSV). They inherit the
    #     single-chip config from the patched `g`; we only repoint OUTPUT_DIR. ---
    for modname in ("extract_no_hbm", "extract_bank_group", "extract_bank_offset",
                    "extract_bitflip", "extract_beat_pattern_combined"):
        mod = importlib.import_module(modname)
        mod.OUTPUT_DIR = OUTPUT_DIR
        if modname == "extract_bitflip":
            _alias_bitflip_idd4r_label(mod)
        mod.main()

    # --- Fig 15: derive beat_pattern_perpattern.csv (no extractor emits it) ---
    derive_beat_pattern_perpattern()

    print("\nDone. Standardized CSVs written to %s" % OUTPUT_DIR)


def derive_beat_pattern_perpattern():
    """Fig 15 input: per-(pattern, col1_inverted) mean of the RAW IDD4R current
    over the 4-bit (max_power_loop) rows of beat_pattern_combined_measurements.csv,
    WITHOUT no-HBM subtraction. `n_chips` is the averaged sample count for this
    chip (mislabeled in the released file; see AE_CSV_FORMATS.md §7)."""
    combined = os.path.join(OUTPUT_DIR, "beat_pattern_combined_measurements.csv")
    out_path = os.path.join(OUTPUT_DIR, "beat_pattern_perpattern.csv")
    if not os.path.isfile(combined):
        print("  WARNING: %s missing; skipping beat_pattern_perpattern.csv" % combined)
        return

    agg = defaultdict(lambda: [0.0, 0])  # (pattern4, inv) -> [sum_idd, n]
    with open(combined) as f:
        for row in csv.DictReader(f):
            if row.get("test_loop") != "max_power_loop":   # 4-bit rows only
                continue
            bp = str(row["beat_pattern"]).zfill(4)
            if len(bp) > 4:                                 # skip any 6-bit rows
                continue
            key = (bp, row["col1_inverted"])
            agg[key][0] += float(row["idd"])
            agg[key][1] += 1

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beat_pattern", "col1_inverted", "mean_idd_mA", "n_chips"])
        for (bp, inv), (total, n) in sorted(agg.items()):
            w.writerow([bp, inv, (total / n) if n else 0.0, n])
    print("Written: %s" % out_path)


if __name__ == "__main__":
    main()
