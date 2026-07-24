#!/usr/bin/env python3
"""
Extract standardized CSV for frequency variation from
frequency_variation_fixed_reset_full_ipp.

The measurements use all 16 channels, so both chip 0 and chip 1 share
the same measurement file.  Each file produces rows for both chips.

Extra column: frequency (MHz)

Output: standardized_csvs/frequency_measurements.csv
"""

import csv
import os
import re

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR,
    VPP_VOLTAGE, build_chip_mapping, get_chip_id, read_measurement_csv,
)

SUBDIR = "frequency_variation_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# Matches all three test types and extracts the frequency:
#   hbm_idd4r_full_multi_...freq300MHz.csv
#   hbm_idd4w_multi_...freq450MHz.csv
#   hbm_max_power_loop_multi_...freq600MHz.csv
_RE = re.compile(
    r"^hbm_(idd4r_full|idd4w|max_power_loop)_multi_"
    r"(?:[0-9a-f]{2}_){4}"                        # data pattern words
    r"(?:\d{1,2}_)*\d{1,2}_"                      # channel list (0-15)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"(?:inv[01]_)?"
    r"dur\d+s_"
    r"freq(\d+)MHz\.csv$"                          # frequency (group 2)
)

TEST_LABEL = {
    "idd4r_full":     "IDD4R",
    "idd4w":          "IDD4W",
    "max_power_loop": "max_power",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "frequency_measurements.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "frequency", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            scan_dir = os.path.join(RESULTS_DIR, f"safari-fpga{fpga}", SUBDIR)
            if not os.path.isdir(scan_dir):
                continue
            for fname in sorted(os.listdir(scan_dir)):
                m = _RE.match(fname)
                if not m:
                    continue
                test_key = m.group(1)
                freq = int(m.group(2))
                test_loop = TEST_LABEL[test_key]
                filepath = os.path.join(scan_dir, fname)
                # All 16 channels → emit rows for both chip 0 and chip 1
                for chip_in_fpga in [0, 1]:
                    chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                    for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                        writer.writerow([chip_id, test_loop, sample_id,
                                         freq, temperature, idd, ipp])
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
