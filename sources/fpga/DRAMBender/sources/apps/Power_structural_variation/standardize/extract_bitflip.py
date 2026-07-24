#!/usr/bin/env python3
"""
Extract standardized CSV for DQ bitflip variation from
bitflip_variation_fixed_reset_full_ipp.

Includes max_power_loop files (inv0) and IDD4R full files (32b pattern).

Extra column: num_bitflips (number of bit differences between consecutive
64-bit data blocks on the DQ bus)

Output: standardized_csvs/bitflip_measurements.csv
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

SUBDIR = "bitflip_variation_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 9)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv0_dur(\d+)s\.csv$"            # inv0 only, duration (group 10)
)

_IDD4R_FULL_RE = re.compile(
    r"^hbm_idd4r_full_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 9)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"dur(\d+)s\.csv$"                 # duration (group 10)
)


def popcount(val):
    """Count number of 1-bits in an integer."""
    return bin(val).count('1')


def compute_bitflips(w0, w1, w2, w3):
    """Compute number of bit differences between consecutive 64-bit blocks.

    Block 0 = (w0, w1), Block 1 = (w2, w3).
    """
    return popcount(w0 ^ w2) + popcount(w1 ^ w3)


def iter_files(fpga):
    """Yield (filepath, match, chip_in_fpga) for max_power files with chip-1 fallback."""
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
            # Only include files where the pattern repeats: w[i] == w[i+4]
            words = [int(m.group(i), 16) for i in range(1, 9)]
            if words[0] != words[4] or words[1] != words[5] or \
               words[2] != words[6] or words[3] != words[7]:
                continue
            chip_in_fpga = chip_from_channels(m.group(9))
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


def iter_idd4r_full_files(fpga):
    """Yield (filepath, match, chip_in_fpga) for IDD4R full 32b files with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            m = _IDD4R_FULL_RE.match(fname)
            if not m:
                continue
            # Only include files where the pattern repeats: w[i] == w[i+4]
            words = [int(m.group(i), 16) for i in range(1, 9)]
            if words[0] != words[4] or words[1] != words[5] or \
               words[2] != words[6] or words[3] != words[7]:
                continue
            chip_in_fpga = chip_from_channels(m.group(9))
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
    out_path = os.path.join(OUTPUT_DIR, "bitflip_measurements.csv")

    csv_counts = defaultdict(lambda: defaultdict(int))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "num_bitflips", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            # --- max_power files ---
            best = {}
            for filepath, m, chip_in_fpga in iter_files(fpga):
                words = [int(m.group(i), 16) for i in range(1, 9)]
                num_bitflips = compute_bitflips(words[0], words[1], words[2], words[3])
                dur = int(m.group(10))
                key = (chip_in_fpga, num_bitflips)
                if key in best and best[key][0] >= dur:
                    continue
                best[key] = (dur, filepath, num_bitflips)

            for (chip_in_fpga, _), (_, filepath, num_bitflips) in sorted(best.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                csv_counts[chip_id]["max_power"] += 1
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "max_power", sample_id,
                                     num_bitflips, temperature, idd, ipp])

            # --- IDD4R full files ---
            best_idd4r = {}
            for filepath, m, chip_in_fpga in iter_idd4r_full_files(fpga):
                words = [int(m.group(i), 16) for i in range(1, 9)]
                num_bitflips = compute_bitflips(words[0], words[1], words[2], words[3])
                dur = int(m.group(10))
                key = (chip_in_fpga, num_bitflips)
                if key in best_idd4r and best_idd4r[key][0] >= dur:
                    continue
                best_idd4r[key] = (dur, filepath, num_bitflips)

            for (chip_in_fpga, _), (_, filepath, num_bitflips) in sorted(best_idd4r.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                csv_counts[chip_id]["IDD4R_full"] += 1
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "IDD4R_full", sample_id,
                                     num_bitflips, temperature, idd, ipp])

    print(f"Written: {out_path}")
    all_loops = ["max_power", "IDD4R_full"]
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
