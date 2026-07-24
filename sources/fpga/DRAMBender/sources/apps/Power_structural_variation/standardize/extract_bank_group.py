#!/usr/bin/env python3
"""
Extract standardized CSV for bank-group variation from
max_power_bg_invert_fixed_reset_full_ipp.

Extra columns: bankgroup_0, bankgroup_1

Output: standardized_csvs/bank_group_measurements.csv
"""

import csv
import os
import re

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, RESULTS_DIR_CHIP1, MIN_MEASUREMENTS,
    VPP_VOLTAGE, build_chip_mapping, get_chip_id, chip_from_channels,
    count_csv_rows, read_measurement_csv,
)

SUBDIR = "max_power_bg_invert_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# hbm_max_power_loop_32b_{8 words}_{channels}_pc{P}_bg{B0}_{B1}_banks..._rows..._cols...[_inv{0|1}]_dur{N}s.csv
_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"(?:[0-9a-f]{8}_){8}"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 1)
    r"pc\d+_bg(\d+)_(\d+)_"           # bank groups (groups 2, 3)
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv1_"
    r"dur\d+s\.csv$"
)


def iter_files(fpga):
    """Yield (filepath, match, chip_in_fpga) with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            m = _RE.match(fname)
            if not m:
                continue
            chip_in_fpga = chip_from_channels(m.group(1))
            if chip_in_fpga is None:
                continue
            if is_fallback and chip_in_fpga != 1:
                continue
            filepath = os.path.join(scan_dir, fname)
            if not is_fallback and chip_in_fpga == 1:
                if count_csv_rows(filepath) < MIN_MEASUREMENTS:
                    fb = os.path.join(RESULTS_DIR_CHIP1, f"safari-fpga{fpga}", SUBDIR, fname)
                    if os.path.isfile(fb):
                        filepath = fb
            seen.add(fname)
            yield filepath, m, chip_in_fpga


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "bank_group_measurements.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "bankgroup_0", "bankgroup_1",
                         "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for filepath, m, chip_in_fpga in iter_files(fpga):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                bg0 = int(m.group(2))
                bg1 = int(m.group(3))
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "max_power_bg_invert", sample_id,
                                     bg0, bg1, temperature, idd, ipp])
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
