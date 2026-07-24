#!/usr/bin/env python3
"""
Extract standardized CSV for data-pattern variation (effect of number of 1s)
from data_pattern_variation_fixed_reset_full_ipp.

Includes both IDD4R (custom patterns) and max_power_loop (uniform patterns).

Extra column: num_ones (number of 1-bits in the 32-bit data pattern)

Output: standardized_csvs/data_pattern_ones_measurements.csv
"""

import csv
import os
import re
from collections import defaultdict

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, RESULTS_DIR_CHIP1, MIN_MEASUREMENTS,
    build_chip_mapping, get_chip_id, chip_from_channels,
    count_csv_rows, read_measurement_csv,
)

SUBDIR = "data_pattern_variation_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# IDD4R custom pattern: hbm_idd4r_full_custom_{HEX}_noshift_{channels}_...
_IDD4R_RE = re.compile(
    r"^hbm_idd4r_full_custom_([0-9a-f]{8})_noshift_"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 2)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"dur(\d+)s\.csv$"                 # duration (group 3)
)

# max_power_loop uniform pattern (8 identical 32-bit words)
_MAX_POWER_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 9)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv([01])_dur(\d+)s\.csv$"       # inv (group 10), dur (group 11)
)


def popcount(hex_str):
    """Count number of 1-bits in a hex string."""
    return bin(int(hex_str, 16)).count('1')


def iter_files(fpga):
    """Yield (filepath, match, chip_in_fpga, rtype) with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            # Try IDD4R regex
            m = _IDD4R_RE.match(fname)
            if m:
                chip_in_fpga = chip_from_channels(m.group(2))
                rtype = "idd4r"
            else:
                # Try max_power regex
                m = _MAX_POWER_RE.match(fname)
                if m:
                    # Only include files where all 8 words are identical
                    words = [m.group(i) for i in range(1, 9)]
                    if len(set(words)) != 1:
                        continue
                    chip_in_fpga = chip_from_channels(m.group(9))
                    rtype = "max_power"
                else:
                    continue
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
            yield filepath, m, chip_in_fpga, rtype


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "data_pattern_ones_measurements.csv")

    csv_counts = defaultdict(lambda: defaultdict(int))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "num_ones", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            # Collect best file per (chip_in_fpga, num_ones, rtype),
            # preferring longer duration when duplicates exist
            best = {}
            for filepath, m, chip_in_fpga, rtype in iter_files(fpga):
                if rtype == "idd4r":
                    pattern_hex = m.group(1)
                    dur = int(m.group(3))
                else:
                    pattern_hex = m.group(1)
                    dur = int(m.group(11))

                num_ones = popcount(pattern_hex)
                key = (chip_in_fpga, num_ones, rtype)
                if key in best and best[key][0] >= dur:
                    continue
                best[key] = (dur, filepath, num_ones, rtype)

            for (chip_in_fpga, _, rtype), (_, filepath, num_ones, _) in sorted(best.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                test_loop = "IDD4R" if rtype == "idd4r" else "max_power"
                csv_counts[chip_id][test_loop] += 1
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, test_loop, sample_id,
                                     num_ones, temperature, idd, ipp])

    print(f"Written: {out_path}")
    all_loops = ["IDD4R", "max_power"]
    header = f"  {'chip_id':>8}"
    for loop in all_loops:
        header += f"  {loop:>{max(len(loop), 6)}}"
    print(header)
    print(f"  {'-'*8}" + "".join(f"  {'-'*max(len(loop), 6)}" for loop in all_loops))
    for chip_id in range(40):
        row = f"  {chip_id:>8}"
        for loop in all_loops:
            count = csv_counts[chip_id].get(loop, 0)
            row += f"  {count:>{max(len(loop), 6)}}"
        print(row)


if __name__ == "__main__":
    main()
