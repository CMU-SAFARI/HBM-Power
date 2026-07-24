#!/usr/bin/env python3
"""
Summarise BRAM trace-runner power per trace, per chip.

For every results/<fpga>/<trace>_bram.csv produced by run_on_infra_bram.sh,
take the average instantaneous VDD and VPP power over the LAST N rows (steady
state) and tag the measurement with the physical chip id from chip_mapping.csv.

The deployed bitstream is the single-stack chip0 tracer (low channels, stack0),
so each FPGA's run corresponds to its chip_in_fpga == 0 chip.

Output (long format): trace_power_summary.csv with columns
  chip_id, fpga_name, trace, vdd_power_mW, vpp_power_mW, temp1_C, temp2_C
"""

import glob
import os
import pandas as pd

LAST_N        = 10                       # rows to average over (steady state)
CHIP_IN_FPGA  = 0                        # chip0 bitstream -> chip 0 of each FPGA
RESULTS_ROOT  = "results"
CHIP_MAP_CSV  = "../Power_structural_variation/standardized_csvs/chip_mapping.csv"
OUT_CSV       = "trace_power_summary.csv"

VDD_INS = "Power_VDD_Ins(mW)"
VPP_INS = "Power_VPP_Ins(mW)"
TEMP1_INS = "Temp1_Ins(Temp)"
TEMP2_INS = "Temp2_Ins(Temp)"


def fpga_to_chip_id(chip_map):
    """fpga_name -> chip_id for the chip0 (chip_in_fpga == CHIP_IN_FPGA) bitstream."""
    sel = chip_map[chip_map["chip_in_fpga"] == CHIP_IN_FPGA]
    return dict(zip(sel["fpga_name"], sel["chip_id"]))


def trace_name(basename):
    """llm_stack0_pc0_bram.csv -> llm_stack0_pc0"""
    name = basename
    if name.endswith("_bram.csv"):
        name = name[: -len("_bram.csv")]
    elif name.endswith(".csv"):
        name = name[: -len(".csv")]
    return name


def main():
    chip_map = pd.read_csv(CHIP_MAP_CSV)
    chip_id_of = fpga_to_chip_id(chip_map)

    rows = []
    skipped = []
    for fpga_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "safari-fpga*"))):
        fpga_name = os.path.basename(fpga_dir)
        chip_id = chip_id_of.get(fpga_name)
        if chip_id is None:
            skipped.append(f"{fpga_name}: not in chip_mapping.csv")
            continue

        for csv_path in sorted(glob.glob(os.path.join(fpga_dir, "*_bram.csv"))):
            base = os.path.basename(csv_path)
            trace = trace_name(base)

            df = pd.read_csv(csv_path)
            needed = [VDD_INS, VPP_INS, TEMP1_INS, TEMP2_INS]
            if any(c not in df.columns for c in needed):
                skipped.append(f"{fpga_name}/{base}: missing power/temp columns")
                continue
            if len(df) < LAST_N:
                skipped.append(f"{fpga_name}/{base}: only {len(df)} rows (< {LAST_N})")
                continue

            tail = df.tail(LAST_N)
            rows.append({
                "chip_id":      int(chip_id),
                "fpga_name":    fpga_name,
                "trace":        trace,
                "vdd_power_mW": round(tail[VDD_INS].mean(), 2),
                "vpp_power_mW": round(tail[VPP_INS].mean(), 2),
                "temp1_C":      round(tail[TEMP1_INS].mean(), 2),
                "temp2_C":      round(tail[TEMP2_INS].mean(), 2),
            })

    if not rows:
        print("No data found.")
        return

    out = pd.DataFrame(rows).sort_values(["trace", "chip_id"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows ({out['trace'].nunique()} traces, "
          f"{out['chip_id'].nunique()} chips) to {OUT_CSV}")
    if skipped:
        print(f"Skipped {len(skipped)} file(s):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
