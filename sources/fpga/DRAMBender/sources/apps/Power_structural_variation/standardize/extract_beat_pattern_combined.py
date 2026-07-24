#!/usr/bin/env python3
"""
Extract standardized CSV for beat-pattern variation, combining:
  - beat_pattern_variation_fixed_reset_full_ipp (4-bit patterns, with col1_inverted)
  - test_28_fixed_reset_full_ipp (6-bit patterns from 3col files, col1_inverted=0)

Extra columns:
  beat_pattern  -- 4-char or 6-char binary string (e.g. "0101" or "010101")
  col1_inverted -- 0 or 1 (always 0 for 6-bit patterns)

Output: standardized_csvs/beat_pattern_combined_measurements.csv
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

SUBDIR_4BIT = "beat_pattern_variation_fixed_reset_full_ipp"
SUBDIR_6BIT = "test_28_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# 4-bit beat pattern files (max_power_loop_32b with 4 beat pairs + inv flag)
_4BIT_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_"   # beat pair 0
    r"([0-9a-f]{8})_([0-9a-f]{8})_"   # beat pair 1
    r"([0-9a-f]{8})_([0-9a-f]{8})_"   # beat pair 2
    r"([0-9a-f]{8})_([0-9a-f]{8})_"   # beat pair 3
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 9)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv([01])_dur(\d+)s\.csv$"       # inv flag (group 10), duration (group 11)
)

# 6-bit beat pattern files (max_power_loop_3col with A/B/C hex digits)
_6BIT_RE = re.compile(
    r"^hbm_max_power_loop_3col_3col_"
    r"A([0-9a-f])_B([0-9a-f])_C([0-9a-f])_"  # A (group 1), B (group 2), C (group 3)
    r"((?:\d{1,2}_)*\d{1,2})_"                # channels (group 4)
    r"pc\d+_bg\d+_\d+_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_[0-9a-f]+_"     # 3 col addresses
    r"dur(\d+)s\.csv$"                         # duration (group 5)
)


def beat_pair_bit(b_lo, b_hi):
    """Return '0' if beat pair is (00000000, ffffffff), '1' if flipped."""
    if b_lo == "00000000" and b_hi == "ffffffff":
        return "0"
    if b_lo == "ffffffff" and b_hi == "00000000":
        return "1"
    return None


def abc_to_6bit(a_hex, b_hex, c_hex):
    """Derive 6-bit beat pattern from A, B, C hex digits.

    Pattern A encodes beats (p0,p1,p2,p3) as bit0..bit3.
    Pattern B encodes beats (p4,p5,p0,p1) as bit0..bit3.
    The 6-bit pattern is p0p1p2p3p4p5.
    """
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    bits = []
    for i in range(4):
        bits.append(str((a >> i) & 1))
    bits.append(str((b >> 0) & 1))
    bits.append(str((b >> 1) & 1))
    return "".join(bits)


def iter_files_4bit(fpga):
    """Yield (filepath, match, chip_in_fpga) for 4-bit patterns with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR_4BIT)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            m = _4BIT_RE.match(fname)
            if not m:
                continue
            chip_in_fpga = chip_from_channels(m.group(9))
            if chip_in_fpga is None:
                continue
            if is_fallback and chip_in_fpga != 1:
                continue
            filepath = os.path.join(scan_dir, fname)
            if not is_fallback and chip_in_fpga == 1:
                if count_csv_rows(filepath) < MIN_MEASUREMENTS:
                    fb = os.path.join(RESULTS_DIR_CHIP1, f"safari-fpga{fpga}", SUBDIR_4BIT, fname)
                    if os.path.isfile(fb):
                        filepath = fb
            seen.add(fname)
            yield filepath, m, chip_in_fpga


def iter_files_6bit(fpga):
    """Yield (filepath, match, chip_in_fpga) for 6-bit patterns with chip-1 fallback."""
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", SUBDIR_6BIT)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            m = _6BIT_RE.match(fname)
            if not m:
                continue
            chip_in_fpga = chip_from_channels(m.group(4))
            if chip_in_fpga is None:
                continue
            if is_fallback and chip_in_fpga != 1:
                continue
            filepath = os.path.join(scan_dir, fname)
            if not is_fallback and chip_in_fpga == 1:
                if count_csv_rows(filepath) < MIN_MEASUREMENTS:
                    fb = os.path.join(RESULTS_DIR_CHIP1, f"safari-fpga{fpga}", SUBDIR_6BIT, fname)
                    if os.path.isfile(fb):
                        filepath = fb
            seen.add(fname)
            yield filepath, m, chip_in_fpga


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "beat_pattern_combined_measurements.csv")

    csv_counts = defaultdict(lambda: defaultdict(int))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "beat_pattern", "col1_inverted",
                         "temperature", "idd", "ipp"])

        for fpga in FPGA_NUMBERS:
            # --- 4-bit patterns ---
            # Deduplicate: best file per (chip_in_fpga, beat_pattern, inv)
            best_4bit = {}
            for filepath, m, chip_in_fpga in iter_files_4bit(fpga):
                bits = [
                    beat_pair_bit(m.group(1), m.group(2)),
                    beat_pair_bit(m.group(3), m.group(4)),
                    beat_pair_bit(m.group(5), m.group(6)),
                    beat_pair_bit(m.group(7), m.group(8)),
                ]
                if any(b is None for b in bits):
                    continue
                beat_pattern = "".join(bits)
                col1_inverted = int(m.group(10))
                dur = int(m.group(11))
                key = (chip_in_fpga, beat_pattern, col1_inverted)
                if key in best_4bit and best_4bit[key][0] >= dur:
                    continue
                best_4bit[key] = (dur, filepath)

            for (chip_in_fpga, beat_pattern, col1_inverted), (_, filepath) in sorted(best_4bit.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                csv_counts[chip_id]["max_power_loop"] += 1
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "max_power_loop", sample_id,
                                     beat_pattern, col1_inverted,
                                     temperature, idd, ipp])

            # --- 6-bit patterns ---
            best_6bit = {}
            for filepath, m, chip_in_fpga in iter_files_6bit(fpga):
                beat_pattern = abc_to_6bit(m.group(1), m.group(2), m.group(3))
                dur = int(m.group(5))
                key = (chip_in_fpga, beat_pattern)
                if key in best_6bit and best_6bit[key][0] >= dur:
                    continue
                best_6bit[key] = (dur, filepath)

            for (chip_in_fpga, beat_pattern), (_, filepath) in sorted(best_6bit.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                csv_counts[chip_id]["max_power_loop_3col"] += 1
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "max_power_loop_3col", sample_id,
                                     beat_pattern, 0,
                                     temperature, idd, ipp])

    print(f"Written: {out_path}")
    all_loops = ["max_power_loop", "max_power_loop_3col"]
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
