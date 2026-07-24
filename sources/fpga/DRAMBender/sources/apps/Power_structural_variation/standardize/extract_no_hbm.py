#!/usr/bin/env python3
"""
Extract standardized CSV for the no-HBM-active IDD2 measurement from
temperature_dependence_fixed_reset_full_ipp.

This corresponds to running IDD2 on channel 0 only (no other HBM channels
active).  Only even chip IDs (chip_in_fpga=0) are used.

Output: standardized_csvs/no_hbm_idd2_measurements.csv
"""

import csv
import os

from generate_standardized_csvs import (
    FPGA_NUMBERS, RESULTS_DIR, SUBDIR,
    build_chip_mapping, get_chip_id, read_measurement_csv,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")

# Channel-0-only IDD2 file (90 s measurement, no other HBM active)
FILENAME = "hbm_idd2_0055ffaa_0_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur90s.csv"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mapping = build_chip_mapping()
    output_path = os.path.join(OUTPUT_DIR, "no_hbm_idd2_measurements.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "sample_id", "temperature", "idd", "ipp"])

        for fpga in FPGA_NUMBERS:
            chip_id = get_chip_id(mapping, fpga, chip_in_fpga=0)
            filepath = os.path.join(
                RESULTS_DIR, f"safari-fpga{fpga}", SUBDIR, FILENAME
            )
            if not os.path.isfile(filepath):
                print(f"  WARNING: Missing no-HBM IDD2 data for safari-fpga{fpga}, skipping")
                continue
            for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                writer.writerow([chip_id, sample_id, temperature, idd, ipp])

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
