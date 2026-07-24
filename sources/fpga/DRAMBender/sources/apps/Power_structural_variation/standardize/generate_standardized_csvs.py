#!/usr/bin/env python3
import csv
import os
import re
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RESULTS_DIR_CHIP1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_good_chip1")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "standardized_csvs")
SUBDIR = "temperature_dependence_fixed_reset_full_ipp"
IDD7_SUBDIR = "idd7_experimentation_fixed_reset_full_ipp"
DATA_PATTERN_SUBDIR = "data_pattern_variation_fixed_reset_full_ipp"
BITFLIP_SUBDIR = "bitflip_variation_fixed_reset_full_ipp"
MAX_POWER_SUBDIRS = {
    "max_power_channel_combo": "max_power_channel_combo_fixed_reset_full_ipp",
    "max_power_bank_offset":   "max_power_bank_offset_fixed_reset_full_ipp",
    "max_power_bg_invert":     "max_power_bg_invert_fixed_reset_full_ipp",
}

CHIP0_CHANNELS = "0_1_2_3_4_5_6_7"
CHIP1_CHANNELS = "8_9_10_11_12_13_14_15"
MIN_MEASUREMENTS = 20

# FPGA numbers in order: 7, 10, 42, 56-59, 65-77
FPGA_NUMBERS = [7, 10, 42] + list(range(56, 60)) + list(range(65, 78))
CHIPS_PER_FPGA = [0, 1]

VPP_VOLTAGE = 2.5  # Volts, for converting Power_VPP(mW) to current (mA)

# Mapping from IDD test name to the CSV filename prefix (before the parameters)
# We use the dur3600s variant for idd4r_full
IDD_TESTS = {
    "IDD0":  "hbm_idd0_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur3600s.csv",
    "IDD2":  "hbm_idd2_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur3600s.csv",
    "IDD3":  "hbm_idd3_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur1200s.csv",
    "IDD3N1":  "hbm_idd3n1_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur1800s.csv",
    "IDD3N16": "hbm_idd3n16_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur1800s.csv",
    "IDD4R": "hbm_idd4r_full_multi_00_55_ff_aa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur3600s.csv",
    "IDD4W": "hbm_idd4w_multi_00_55_ff_aa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur3600s.csv",
    "IDD5B": "hbm_idd5_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur3600s.csv",
    "IDD0_3_CYCLE_ACTPRE": "hbm_idd1_max_actpre_0055ffaa_0_1_2_3_4_5_6_7_pc0_bg0_1_banks0123_rows5555_2aaa_cols0a_15_dur1800s.csv",
    "IDD7": "hbm_idd7_actpre_maxpower_32b_00000000_ffffffff_ffffffff_00000000_ffffffff_00000000_00000000_ffffffff_0_1_2_3_4_5_6_7_pc0_bg0_2_banks0123_rows5555_2aaa_cols0a_15_dur1800s.csv",
}

# Tests that live in a different subdir than the default SUBDIR
IDD_TEST_SUBDIRS = {
    "IDD7": IDD7_SUBDIR,
}


def build_chip_mapping():
    """Build and return the chip mapping: list of (fpga_number, chip_in_fpga, chip_id)."""
    mapping = []
    chip_id = 0
    for fpga in FPGA_NUMBERS:
        for chip in CHIPS_PER_FPGA:
            mapping.append((fpga, chip, chip_id))
            chip_id += 1
    return mapping


def get_chip_id(mapping, fpga_number, chip_in_fpga=0):
    """Look up chip_id for a given FPGA and chip number."""
    for fpga, chip, cid in mapping:
        if fpga == fpga_number and chip == chip_in_fpga:
            return cid
    return None


def chip_from_channels(channel_str):
    """Determine chip_in_fpga (0 or 1) from a channel list string like '0_1_2_3'."""
    channels = [int(c) for c in channel_str.split('_')]
    if all(c <= 7 for c in channels):
        return 0
    if all(c >= 8 for c in channels):
        return 1
    return None


def count_csv_rows(filepath):
    """Count the number of data rows in a CSV file."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def make_multi_variant(filename):
    """Generate the multi_ variant: _0055ffaa_ -> _multi_00_55_ff_aa_."""
    return filename.replace('_0055ffaa_', '_multi_00_55_ff_aa_')


_DUR_RE = re.compile(r'_dur(\d+s)\.csv$')
_DURATIONS = ['3600s', '1800s', '1200s']
MIN_ROWS_THRESHOLD = 1800


def _duration_sort_key(path):
    """Sort key: prefer longer durations (3600s=0, 1800s=1, 1200s=2)."""
    m = _DUR_RE.search(path)
    if m:
        try:
            return _DURATIONS.index(m.group(1))
        except ValueError:
            pass
    return len(_DURATIONS)


def _duration_variants(filename):
    """Yield the original filename plus variants with alternate durations."""
    yield filename
    m = _DUR_RE.search(filename)
    if not m:
        return
    for dur in _DURATIONS:
        variant = _DUR_RE.sub(f'_dur{dur}.csv', filename)
        if variant != filename:
            yield variant


def resolve_filepath(fpga, chip_in_fpga, subdir, filename):
    """Resolve file path, preferring multi_ variant, with chip 1 fallback.

    Tries alternate durations (3600s, 1800s, 1200s) when the exact file is missing.
    """
    multi = make_multi_variant(filename)
    base_candidates = [multi, filename] if multi != filename else [filename]
    # expand each base candidate with duration variants
    candidates = []
    for c in base_candidates:
        for v in _duration_variants(c):
            if v not in candidates:
                candidates.append(v)

    def find_in_dir(base_dir):
        existing = []
        for c in candidates:
            path = os.path.join(base_dir, f"safari-fpga{fpga}", subdir, c)
            if os.path.isfile(path):
                existing.append(path)
        if not existing:
            return None
        # Sort by duration preference (3600s > 1800s > 1200s)
        existing.sort(key=_duration_sort_key)
        # Prefer the first file with enough rows
        for path in existing:
            if count_csv_rows(path) >= MIN_ROWS_THRESHOLD:
                return path
        # Fallback: file with the most rows
        return max(existing, key=count_csv_rows)

    if chip_in_fpga == 0:
        return find_in_dir(RESULTS_DIR)

    # chip 1: prefer RESULTS_DIR if enough measurements
    primary = find_in_dir(RESULTS_DIR)
    if primary and count_csv_rows(primary) >= MIN_MEASUREMENTS:
        return primary
    # fallback to RESULTS_DIR_CHIP1
    fallback = find_in_dir(RESULTS_DIR_CHIP1)
    if fallback:
        return fallback
    # last resort: RESULTS_DIR even with few measurements
    return primary


def iter_matching_files(fpga, subdir, regex, channel_group):
    """Iterate over matching files, with chip 1 fallback to results_good_chip1.

    Yields (filepath, match, chip_in_fpga) tuples.
    """
    seen = set()
    for base_dir, is_fallback in [(RESULTS_DIR, False), (RESULTS_DIR_CHIP1, True)]:
        scan_dir = os.path.join(base_dir, f"safari-fpga{fpga}", subdir)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname in seen:
                continue
            m = regex.match(fname)
            if not m:
                continue
            channel_str = m.group(channel_group)
            chip_in_fpga = chip_from_channels(channel_str)
            if chip_in_fpga is None:
                continue
            if is_fallback and chip_in_fpga != 1:
                continue
            filepath = os.path.join(scan_dir, fname)
            if not is_fallback and chip_in_fpga == 1:
                if count_csv_rows(filepath) < MIN_MEASUREMENTS:
                    fallback_path = os.path.join(
                        RESULTS_DIR_CHIP1, f"safari-fpga{fpga}", subdir, fname
                    )
                    if os.path.isfile(fallback_path):
                        filepath = fallback_path
            seen.add(fname)
            yield filepath, m, chip_in_fpga


def read_measurement_csv(filepath):
    """Read a measurement CSV and yield (sample_id, temperature, idd, ipp) tuples."""
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for sample_id, row in enumerate(reader):
            temperature = float(row["Temp1_Ins(Temp)"])
            idd = float(row["Current_Avg(mA)"])
            ipp = float(row["Power_VPP_Avg(mW)"]) / VPP_VOLTAGE
            yield sample_id, temperature, idd, ipp


def generate_chip_mapping_csv(mapping, output_path):
    """CSV 1: chip_mapping.csv."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpga_name", "chip_in_fpga", "chip_id"])
        for fpga, chip, cid in mapping:
            writer.writerow([f"safari-fpga{fpga}", chip, cid])
    print(f"Written: {output_path}")


def generate_idd2_csv(mapping, output_path):
    """CSV 2: idd2 measurements for all FPGAs, both chips."""
    filename_chip0 = IDD_TESTS["IDD2"]
    filename_chip1 = filename_chip0.replace(
        f"_{CHIP0_CHANNELS}_", f"_{CHIP1_CHANNELS}_"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "sample_id", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for chip_in_fpga in CHIPS_PER_FPGA:
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                filename = filename_chip0 if chip_in_fpga == 0 else filename_chip1
                filepath = resolve_filepath(fpga, chip_in_fpga, SUBDIR, filename)
                if filepath is None:
                    print(f"  WARNING: Missing IDD2 data for safari-fpga{fpga} chip {chip_in_fpga}, skipping")
                    continue
                for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                    writer.writerow([chip_id, sample_id, temperature, idd, ipp])
    print(f"Written: {output_path}")


def generate_all_tests_csv(mapping, output_path):
    """CSV 3: all IDD test measurements for all FPGAs, both chips."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for chip_in_fpga in CHIPS_PER_FPGA:
                chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                for test_name, filename_chip0 in IDD_TESTS.items():
                    filename = filename_chip0 if chip_in_fpga == 0 else filename_chip0.replace(
                        f"_{CHIP0_CHANNELS}_", f"_{CHIP1_CHANNELS}_"
                    )
                    test_subdir = IDD_TEST_SUBDIRS.get(test_name, SUBDIR)
                    filepath = resolve_filepath(fpga, chip_in_fpga, test_subdir, filename)
                    if filepath is None:
                        print(f"  WARNING: Missing {test_name} data for safari-fpga{fpga} chip {chip_in_fpga}, skipping")
                        continue
                    for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                        writer.writerow([chip_id, test_name, sample_id, temperature, idd, ipp])
    print(f"Written: {output_path}")


# Regex to parse data-pattern max_power_loop filenames:
# hbm_max_power_loop_32b_{W0}_{...}_{W7}_{channels}_pc{N}_bg{BG0}_{BG1}_banks{XXXX}_rows{R0}_{R1}_cols{C0}_{C1}_inv{0|1}_dur{N}s.csv
_DATA_PAT_MAX_POWER_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"((?:\d{1,2}_)*\d{1,2})_"   # channel list (group 9)
    r"pc\d+_bg(\d+)_(\d+)_"      # bank group pair (groups 10, 11)
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"inv([01])_dur\d+s\.csv$"    # invert flag (group 12)
)

# Regex to parse max_power filenames:
# hbm_max_power_loop_32b_{W0}_{...}_{W7}_{ch0}_{ch1}_..._pc{N}_bg{BG0}_{BG1}_banks{XXXX}_rows{R0}_{R1}_cols{C0}_{C1}_dur{N}s.csv
_MAX_POWER_RE = re.compile(
    r"^hbm_max_power_loop_32b_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_([0-9a-f]{8})_"
    r"((?:\d{1,2}_)*\d{1,2})_"   # channel list (supports channels 0-15)
    r"pc\d+_bg(\d+)_(\d+)_"  # bank group pair
    r"banks\d+_"
    r"rows[0-9a-f]+_[0-9a-f]+_"
    r"cols[0-9a-f]+_[0-9a-f]+_"
    r"dur\d+s\.csv$"
)


def generate_data_pattern_csv(mapping, output_path):
    """CSV 4: data-pattern max_power_loop measurements for all FPGAs, both chips."""
    from collections import defaultdict
    csv_counts = defaultdict(lambda: {"data_pattern": 0, "bitflip": 0})

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "data_pattern", "col1_inverted",
                         "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for subdir, expected_inv in [(DATA_PATTERN_SUBDIR, 0), (BITFLIP_SUBDIR, 1)]:
                label = "data_pattern" if subdir == DATA_PATTERN_SUBDIR else "bitflip"
                for filepath, m, chip_in_fpga in iter_matching_files(
                    fpga, subdir, _DATA_PAT_MAX_POWER_RE, 9
                ):
                    col1_inverted = int(m.group(12))
                    if col1_inverted != expected_inv:
                        continue
                    chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                    csv_counts[chip_id][label] += 1
                    words = m.groups()[:8]
                    data_pattern_hex = "0x" + "".join(words)
                    col1_inverted = int(m.group(12))
                    for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                        writer.writerow([chip_id, "max_power_data_pattern", sample_id,
                                         data_pattern_hex, col1_inverted,
                                         temperature, idd, ipp])
    print(f"Written: {output_path}")
    print(f"  {'chip_id':>8}  {'data_pattern':>13}  {'bitflip':>8}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*8}")
    for chip_id in sorted(csv_counts):
        dp = csv_counts[chip_id]["data_pattern"]
        bf = csv_counts[chip_id]["bitflip"]
        print(f"  {chip_id:>8}  {dp:>13}  {bf:>8}")


def generate_max_power_csv(mapping, output_path):
    """CSV 5: max-power measurements for all FPGAs, both chips."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "test_loop", "sample_id",
                         "data_pattern", "bankgroup_0", "bankgroup_1",
                         "channels", "temperature", "idd", "ipp"])
        for fpga in FPGA_NUMBERS:
            for test_name, subdir in MAX_POWER_SUBDIRS.items():
                for filepath, m, chip_in_fpga in iter_matching_files(
                    fpga, subdir, _MAX_POWER_RE, 9
                ):
                    chip_id = get_chip_id(mapping, fpga, chip_in_fpga=chip_in_fpga)
                    words = m.groups()[:8]
                    data_pattern_hex = "0x" + "".join(words)
                    channels = m.group(9)
                    bg0 = m.group(10)
                    bg1 = m.group(11)
                    for sample_id, temperature, idd, ipp in read_measurement_csv(filepath):
                        writer.writerow([chip_id, test_name, sample_id,
                                         data_pattern_hex, bg0, bg1,
                                         channels, temperature, idd, ipp])
    print(f"Written: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mapping = build_chip_mapping()

    generate_chip_mapping_csv(mapping, os.path.join(OUTPUT_DIR, "chip_mapping.csv"))
    generate_idd2_csv(mapping, os.path.join(OUTPUT_DIR, "idd2_measurements.csv"))
    generate_all_tests_csv(mapping, os.path.join(OUTPUT_DIR, "all_idd_measurements.csv"))
    generate_data_pattern_csv(mapping, os.path.join(OUTPUT_DIR, "data_pattern_measurements.csv"))
    generate_max_power_csv(mapping, os.path.join(OUTPUT_DIR, "max_power_measurements.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
