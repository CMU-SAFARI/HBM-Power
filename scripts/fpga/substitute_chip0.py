#!/usr/bin/env python3
"""Substitute one chip's regenerated FPGA CSV rows into the released data set."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path


DERIVED_BEAT_FILE = "beat_pattern_perpattern.csv"
BEAT_SOURCE_FILE = "beat_pattern_combined_measurements.csv"
NEW_ONLY_REQUIRED_INPUTS = {"chip_mapping.csv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/merged by substituting chip_id rows from data/new/fpga."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--new-dir", type=Path, default=Path("data/new/fpga"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--chip-id", default="0")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_target_chip(value: str, chip_id: str) -> bool:
    return value.strip() == chip_id


def has_substitution_data(new_dir: Path) -> bool:
    for path in new_dir.glob("*.csv"):
        if path.name in NEW_ONLY_REQUIRED_INPUTS:
            continue
        _, rows = read_csv(path)
        if rows:
            return True
    return False


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()


def regenerate_beat_perpattern(output_dir: Path) -> None:
    source = output_dir / BEAT_SOURCE_FILE
    if not source.exists():
        return

    _, rows = read_csv(source)
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("test_loop") != "max_power_loop":
            continue
        beat_pattern = row.get("beat_pattern", "").zfill(4)
        col1_inverted = row.get("col1_inverted", "")
        if len(beat_pattern) > 4:
            continue
        try:
            aggregate[(beat_pattern, col1_inverted)].append(float(row["idd"]))
        except (KeyError, ValueError):
            continue

    fieldnames = ["beat_pattern", "col1_inverted", "mean_idd_mA", "n_chips"]
    derived_rows = []
    for (beat_pattern, col1_inverted), values in sorted(aggregate.items()):
        derived_rows.append(
            {
                "beat_pattern": beat_pattern,
                "col1_inverted": col1_inverted,
                "mean_idd_mA": str(sum(values) / len(values)),
                "n_chips": str(len(values)),
            }
        )
    write_csv(output_dir / DERIVED_BEAT_FILE, fieldnames, derived_rows)


def substitute_file(base_path: Path, new_path: Path, output_path: Path, chip_id: str) -> str:
    base_fields, base_rows = read_csv(base_path)
    new_fields, new_rows = read_csv(new_path)

    if base_fields != new_fields:
        raise ValueError(
            f"schema mismatch for {base_path.name}: data has {base_fields}, new data has {new_fields}"
        )
    if "chip_id" not in base_fields:
        shutil.copy2(base_path, output_path)
        return f"copied {base_path.name} (no chip_id column)"

    unexpected_chips = sorted({row.get("chip_id", "") for row in new_rows if not is_target_chip(row.get("chip_id", ""), chip_id)})
    if unexpected_chips:
        raise ValueError(
            f"{new_path} contains chip_id values other than {chip_id}: {', '.join(unexpected_chips)}"
        )

    kept_rows = [row for row in base_rows if not is_target_chip(row.get("chip_id", ""), chip_id)]
    write_csv(output_path, base_fields, kept_rows + new_rows)
    return (
        f"substituted {base_path.name}: dropped {len(base_rows) - len(kept_rows)} "
        f"old chip {chip_id} rows, added {len(new_rows)} new rows"
    )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    new_dir = args.new_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not data_dir.is_dir():
        print(f"error: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1
    if not new_dir.is_dir() or not has_substitution_data(new_dir):
        print(
            f"error: no FPGA substitution data found in {new_dir}; run the FPGA data step first.",
            file=sys.stderr,
        )
        return 1

    clean_output_dir(output_dir)
    messages: list[str] = []
    base_files = sorted(path for path in data_dir.glob("*.csv") if path.parent == data_dir)

    for base_path in base_files:
        new_path = new_dir / base_path.name
        output_path = output_dir / base_path.name
        if base_path.name == DERIVED_BEAT_FILE:
            continue
        if new_path.exists():
            messages.append(substitute_file(base_path, new_path, output_path, args.chip_id))
        else:
            shutil.copy2(base_path, output_path)
            messages.append(f"copied {base_path.name} (no new FPGA replacement)")

    for new_path in sorted(new_dir.glob("*.csv")):
        if (output_dir / new_path.name).exists() or (data_dir / new_path.name).exists():
            continue
        shutil.copy2(new_path, output_dir / new_path.name)
        messages.append(f"copied {new_path.name} (new FPGA-only file)")

    regenerate_beat_perpattern(output_dir)
    messages.append(f"regenerated {DERIVED_BEAT_FILE} from {BEAT_SOURCE_FILE}")

    for message in messages:
        print(message)
    print(f"merged data written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())