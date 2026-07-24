#!/usr/bin/env python3
import csv
import os
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "standardized_csvs", "all_idd_measurements.csv")

counts = Counter()
with open(CSV_PATH, "r") as f:
    for row in csv.DictReader(f):
        counts[(row["chip_id"], row["test_loop"])] += 1

short = {k: v for k, v in counts.items() if v < 25}

if not short:
    print("All (chip_id, test_loop) pairs have >= 25 measurements.")
else:
    print(f"{'chip_id':>8}  {'test_loop':<25}  {'count':>5}")
    print("-" * 44)
    for (chip_id, test_loop), count in sorted(short.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        print(f"{chip_id:>8}  {test_loop:<25}  {count:>5}")
    print(f"\nTotal: {len(short)} pair(s) with < 25 measurements.")
