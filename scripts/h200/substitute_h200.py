#!/usr/bin/env python3
"""Merge one fresh H200 Vast result into the HBM3E measurement CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


HBM3E_FILE = "HBM3E_measurements.csv"
DEFAULT_NEW_FILE = Path("data/new/h200/hbm3e_results.csv")
FIELDNAMES = ["sample", "total_power_W", "idle_power_W", "read_GBs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update data/merged/HBM3E_measurements.csv with one fresh H200 result."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged"))
    parser.add_argument("--new-file", type=Path, default=DEFAULT_NEW_FILE)
    parser.add_argument("--sample", default="vast_h200")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"{path} has schema {reader.fieldnames}; expected {FIELDNAMES}")
        return list(reader)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_quick_summary(path: Path, sample: str) -> dict[str, str]:
    with path.open(newline="") as csv_file:
        rows = {row.get("metric", ""): row for row in csv.DictReader(csv_file)}

    try:
        idle_row = rows["idle_baseline"]
        total_row = rows["random_read_total"]
    except KeyError as exc:
        raise ValueError(f"{path} is missing required metric {exc.args[0]!r}") from exc

    merged_row = {
        "sample": sample,
        "total_power_W": total_row.get("mem_power_W", ""),
        "idle_power_W": idle_row.get("mem_power_W", ""),
        "read_GBs": total_row.get("read_GBs", ""),
    }
    for key, value in merged_row.items():
        if value == "":
            raise ValueError(f"{path} did not provide a value for {key}")
    return merged_row


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    new_file = args.new_file.resolve()

    base_path = data_dir / HBM3E_FILE
    output_path = output_dir / HBM3E_FILE

    if not base_path.is_file():
        print(f"error: missing released HBM3E data: {base_path}", file=sys.stderr)
        return 1
    if not new_file.is_file():
        print(
            f"error: missing fresh H200 Vast result: {new_file}\n"
            f"run scripts/h200/vast_run.py first, then rerun scripts/h200/substitute.sh.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = read_rows(base_path)
        new_row = load_quick_summary(new_file, args.sample)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = [row for row in rows if row.get("sample") != args.sample]
    rows.append(new_row)
    write_rows(output_path, rows)

    print(f"merged {new_file} into {output_path}")
    print(f"wrote {len(rows)} HBM3E measurement rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())