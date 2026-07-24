#!/usr/bin/env python3
"""Sweep decode batch sizes and generate LLaMa-3.1-8B HBM2 command traces in the
DRAMPower 8-column format used by figure14.

Always does a CLEAN build of the bundled LLMSimulator (sources/LLMSimulator),
then sweeps the decode batch size at a fixed 1024-token context, reduces each
raw channel-0 trace to pseudochannel 0, and reformats it (see
sources/LLMSimulator/drampower_convert.py). Output CSVs are written to
traces/llm_batch_sweep/bs{bs}_ctx1024.csv.

Simulator config (from sources/LLMSimulator/config.yaml): model llama8Bshort,
gpu_gen A100 -> 40 GB HBM2 (1.2 Gbps / nBL=2 / single rank), decode mode, FP16
(precision_byte=2), single device, output_len=2.

Usage:
    python3 llm_traces.py                        # sweep {1,2,4,8,16,32,64,128}
    BATCH_SIZES="1 32 128" python3 llm_traces.py # custom batch sizes
    CTX=2048 python3 llm_traces.py               # different context length
"""
import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]           # HBM-Power/ (this file: scripts/traces/)
SIM    = ROOT / "sources" / "LLMSimulator"
BUILD  = SIM / "build"
RUN    = BUILD / "run"
CONFIG = SIM / "config.yaml"
OUT    = ROOT / "traces" / "llm_batch_sweep"

CTX         = int(os.environ.get("CTX", "1024"))
OUTPUT_LEN  = int(os.environ.get("OUTPUT_LEN", "2"))   # decode requires output_len > 1
BATCH_SIZES = [int(b) for b in os.environ.get("BATCH_SIZES",
                                              "1 2 4 8 16 32 64 128").split()]

# the trace converter lives in the simulator copy
sys.path.insert(0, str(SIM))
import drampower_convert                                # provides convert(raw, out)


def build():
    """Always build a clean version: wipe build/ and recompile (-j4: a bare -j OOMs)."""
    if BUILD.exists():
        print(f"[build] removing {BUILD.relative_to(ROOT)} for a clean build")
        shutil.rmtree(BUILD)
    print("[build] cmake + make -j4 (fetches ext/ deps, ~minutes)...")
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=BUILD, check=True)
    subprocess.run(["make", "-j4"], cwd=BUILD, check=True)
    if not RUN.exists():
        sys.exit("[build] failed: build/run was not produced")


def templated_config(bs, path):
    """Write a config.yaml variant with this batch size / context / output_len."""
    txt = CONFIG.read_text()
    txt = re.sub(r"(?m)^(\s*max_batch_size:).*", rf"\g<1> {bs}", txt)
    txt = re.sub(r"(?m)^(\s*input_len:).*",      rf"\g<1> {CTX}", txt)
    txt = re.sub(r"(?m)^(\s*output_len:).*",     rf"\g<1> {OUTPUT_LEN}", txt)
    path.write_text(txt)


def main():
    build()
    OUT.mkdir(parents=True, exist_ok=True)
    (BUILD / "log").mkdir(parents=True, exist_ok=True)
    raw = BUILD / "log" / "cmd_hbm2_40gb.log.ch0"        # A100/HBM2 channel-0 trace

    print(f">>> batch sweep: model=llama8Bshort  ctx={CTX}  output_len={OUTPUT_LEN}  "
          f"batches={BATCH_SIZES}")
    for bs in BATCH_SIZES:
        tag = f"bs{bs}_ctx{CTX}"
        cfg = BUILD / f"config_{tag}.yaml"
        templated_config(bs, cfg)
        if raw.exists():
            raw.unlink()
        with open(BUILD / "log" / f"run_{tag}.log", "w") as log:
            subprocess.run([str(RUN), str(cfg)], cwd=BUILD,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if not raw.exists():
            sys.exit(f"[{tag}] no trace at {raw}; see build/log/run_{tag}.log")

        out_csv = OUT / f"{tag}.csv"
        drampower_convert.convert(str(raw), str(out_csv))
        cfg.unlink(missing_ok=True)
        print(f"  [{tag}] -> {out_csv.relative_to(ROOT)}")

    print(f">>> done. traces in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
