#!/usr/bin/env python3
"""
Extract standardized CSV for channel-combination variation from
max_power_channel_combo_fixed_reset_full_ipp.

Uses the same data source as extract_max_power_by_channel_combo.py.

For chip 1 (channels 8–15), channel 8 is mapped to ch0, channel 9 to ch1, etc.

Extra columns: ch0..ch7 (0 or 1, indicating whether that channel is active)

Output: standardized_csvs/channel_combo_measurements.csv
"""

import csv
import os
import re

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, RESULTS_DIR_CHIP1, MIN_MEASUREMENTS,
    VPP_VOLTAGE, build_chip_mapping, get_chip_id, chip_from_channels,
    count_csv_rows, read_measurement_csv,
)

SUBDIR = "max_power_channel_combo_fixed_reset_full_ipp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"(?:[0-9a-f]{8}_){8}"
    r"((?:\d{1,2}_)*\d{1,2})_"        # channels (group 1)
    r"pc\d+_bg0_2_"
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv1_"
    r"dur90s\.csv$"
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


def channels_to_flags(channel_str, chip_in_fpga):
    """Return list of 8 ints (0/1) for ch0..ch7.

    For chip 1, channel N is mapped to N-8.
    """
    channels = set(int(c) for c in channel_str.split("_"))
    offset = 8 if chip_in_fpga == 1 else 0
    return [int((i + offset) in channels) for i in range(8)]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    out_path = os.path.join(OUTPUT_DIR, "channel_combo_measurements.csv")

    ch_cols = [f"ch{i}" for i in range(8)]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id"] + ch_cols +
                         ["temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for filepath, m, chip_in_fpga in iter_files(fpga):
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                flags = channels_to_flags(m.group(1), chip_in_fpga)
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, "max_power_channel_combo",
                                     sample_id] + flags +
                                    [temperature, idd, ipp])
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
