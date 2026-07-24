#!/usr/bin/env python3
"""
Extract standardized CSV for pseudo-channel variation from
pc_variation_test_fixed_reset_full_ipp.

Each file targets 8 channels (chip 0: 0-7, chip 1: 8-15).

Extra columns: pseudo_channel (0 or 1), test_type (idd4r_full, idd5, max_power_loop)

Output: standardized_csvs/pseudo_channel_measurements.csv
"""

import csv
import os
import re

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, RESULTS_DIR_CHIP1, MIN_MEASUREMENTS,
    VPP_VOLTAGE, build_chip_mapping, get_chip_id, chip_from_channels,
    count_csv_rows, read_measurement_csv,
)

SUBDIR = "pc_variation_test_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# Matches idd4r_full, idd5, and max_power_loop_32b files
_RE = re.compile(
    r"^hbm_(?:idd4r_full|idd5)_"
    r"(?:multi_)?(?:[0-9a-f]{2}_){4}" # multi data pattern (4 bytes, optional multi_ prefix)
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 1)
    r"pc(\d+)_"                        # pseudo channel (group 2)
    r"bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"dur\d+s\.csv$"
)

# Matches max_power_loop_32b files (8 x 32-bit words)
_RE_MAX_POWER = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"(?:[0-9a-f]{8}_){8}"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 1)
    r"pc(\d+)_"                        # pseudo channel (group 2)
    r"bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"(?:inv[01]_)?"
    r"dur\d+s\.csv$"
)

_TEST_TYPE_RE = re.compile(r"^hbm_(idd4r_full|idd5|max_power_loop_32b)_")

_TEST_TYPE_MAP = {
    "idd4r_full": "IDD4R",
    "idd5": "IDD5B",
    "max_power_loop_32b": "MAX_POWER_LOOP",
}


def iter_files(fpga):
    """Yield (filepath, channels, pc, test_type, chip_in_fpga) with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            # Try multi-byte pattern first, then 32-bit word pattern
            m = _RE.match(fname)
            if not m:
                m = _RE_MAX_POWER.match(fname)
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
            pc = int(m.group(2))
            tt = _TEST_TYPE_RE.match(fname)
            test_type = _TEST_TYPE_MAP.get(tt.group(1), tt.group(1)) if tt else "unknown"
            yield filepath, pc, test_type, chip_in_fpga


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "pseudo_channel_measurements.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "pseudo_channel",
                         "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for filepath, pc, test_type, chip_in_fpga in iter_files(fpga):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, test_type,
                                     sample_id, pc,
                                     temperature, idd, ipp])
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
