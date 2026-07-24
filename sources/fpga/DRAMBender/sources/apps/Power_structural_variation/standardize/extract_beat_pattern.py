#!/usr/bin/env python3
"""
Extract standardized CSV for beat-pattern variation from
beat_pattern_variation_fixed_reset_full_ipp.

Extra columns: beat_pattern (4-bit string, e.g. "0101"), col1_inverted (0 or 1)

Output: standardized_csvs/beat_pattern_measurements.csv
"""

import csv
import os
import re

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, RESULTS_DIR_CHIP1, MIN_MEASUREMENTS,
    VPP_VOLTAGE, build_chip_mapping, get_chip_id, chip_from_channels,
    count_csv_rows, read_measurement_csv,
)

SUBDIR = "beat_pattern_variation_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

_RE = re.compile(
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


def beat_pair_bit(b0, b1):
    """Return '0' if pair is (00000000, ffffffff), '1' if inverted."""
    if b0 == "00000000" and b1 == "ffffffff":
        return "0"
    if b0 == "ffffffff" and b1 == "00000000":
        return "1"
    return None


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
    out_path = os.path.join(OUTPUT_DIR, "beat_pattern_measurements.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "beat_pattern", "col1_inverted",
                         "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            # Collect best file per (chip_in_fpga, beat_pattern, inv),
            # preferring longer duration when duplicates exist
            best = {}  # (chip_in_fpga, beat_pattern, inv) -> (duration, filepath)
            for filepath, m, chip_in_fpga in iter_files(fpga):
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
                if key in best and best[key][0] >= dur:
                    continue
                best[key] = (dur, filepath)

            for (chip_in_fpga, beat_pattern, col1_inverted), (_, filepath) in sorted(best.items()):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "beat_pattern_variation", sample_id,
                                     beat_pattern, col1_inverted,
                                     temperature, idd, ipp])
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
