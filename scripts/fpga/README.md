# FPGA AE pipeline

One-command single-chip run of the HBM2-power **FPGA** experiments. The deliverable is the
**sanitized CSVs** under `data/new/fpga/` — the per-sample tidy tables for every in-scope experiment
(the IDD/structural measurements plus the trace ground truth). Turning those CSVs into paper
figures is out of scope here (handled separately). The local orchestration scripts live in this
directory, while the measurement scripts they stage onto the FPGA live in
[`../../sources/fpga/`](../../sources/fpga/).

## Running it

Reprogram → measure → standardize into the sanitized CSVs:

```
scripts/fpga/run_fpga_ae.sh   # -> data/new/fpga/*.csv  (the deliverable)
```

If a run fails partway through, resume it instead of starting over:

```
scripts/fpga/run_fpga_ae.sh --resume
```

The driver records completed chunks in `temp/fpga_ae_checkpoint.tsv` by default. On resume, it
skips completed chunks and reprograms only the phase bitstream needed for the first remaining
measurement chunk in that phase. A normal run without `--resume` resets the checkpoint. Override
the path with `CHECKPOINT_FILE=...` or `--checkpoint-file ...`.

Defaults target the artifact-evaluation setup: `aevaluator@safari-fpga7.ethz.ch`, reached
through `aevaluator1@safari-proxy.ethz.ch` with the private key at `$HOME/aevaluator1`. When
running inside the artifact Docker container, `docker-compose.yml` mounts `./aevaluator1` at
`/home/developer/aevaluator1` and sets `FPGA_SSH_KEY` to that path.

Connection env overrides: `FPGA_USER`, `FPGA_HOST`, `FPGA_SSH_KEY`, `FPGA_PROXY_USER`,
`FPGA_PROXY_HOST`, `FPGA_PROXY_JUMP`, `FPGA_PROXY_COMMAND`. Set `FPGA_PROXY_JUMP=""` to
disable the proxy. By default, the helper turns `FPGA_PROXY_JUMP` into an explicit
`ProxyCommand` so the proxy hop uses the same private key and batch-mode settings.

See `run_fpga_ae.sh --help`. The only board-side prerequisite beyond the SoftMC infra is an
artifact checkout on the FPGA host (default `~/HBM-Power`, override with
`REMOTE_ART`): the bitstreams ship in its `sources/fpga/DRAMBender/prebuilt/XCU55/` and the
driver compiles both measurement binaries from its provided sources up-front
(`--skip-build` to skip; each test additionally runs an incremental `BUILD=make` no-op).

## Data flow

```
FPGA  (SoftMC_rdwr, or HBMTraceRunnerBRAM for the trace experiments)
  │   raw per-run CSVs
  ▼
results/<fpga-label>/<test>_fixed_reset_full_ipp/     ← copy_results.sh pulls these back
  │
  ▼
data/new/fpga/*.csv   (the sanitized CSVs, one row per sample)    ← standardize_single_chip.py
data/new/fpga/ground_truth_*.csv                               ← aggregate_trace_ground_truth.py (trace)
```

The sanitized CSVs are the end of this pipeline.

## Who calls what

`run_fpga_ae.sh` is the **only** orchestrator:

```
run_fpga_ae.sh
├─ reprogram_fpga.sh <bitstream>          per phase (no-HBM / chip / trace), only if a selected test needs it
├─ run_step_tmux.sh <run_*.sh> <subdir>   once per measurement (detached tmux, survives SSH drops)
│     └─ copy_results.sh <subdir>         pulls that test's raw CSVs back   (called *inside* run_step_tmux)
├─ aggregate_trace_ground_truth.py        trace phase only → data/new/fpga/ground_truth_*.csv
└─ standardize_single_chip.py             final step → the 8 tidy data/new/fpga/*.csv   (--skip-standardize disables)
```

Checkpoint chunk IDs are readable (`baseline:no_hbm`, `chip:idd0`, `trace:measure`,
`trace:aggregate`, `standardize`) and are appended only after that chunk succeeds. If a chunk
fails before being marked complete, rerunning with `--resume` repeats that chunk rather than
trusting partial output.

`standardize_single_chip.py` runs **once, at the end of `run_fpga_ae.sh`** (or by hand with the
same env vars). `copy_results.sh` is invoked **by `run_step_tmux.sh`**, not by the driver
directly. The measurement scripts (`fpga/run_*.sh`) are staged onto the FPGA and run there — you
don't call them directly; see [`../../sources/fpga/README.md`](../../sources/fpga/README.md) for the per-test map and the
duration knobs.

## Scripts

| Script | Role |
|---|---|
| `run_fpga_ae.sh` | the one-command driver: reprogram → measure each test → standardize |
| `reprogram_fpga.sh` | switch the HBM bitstream, reboot, re-init the SoftMC host (2 phases) |
| `run_step_tmux.sh` | run ONE measurement in a detached `power_experiment` tmux session, poll, copy back |
| `copy_results.sh` | rsync a test's raw CSVs off the FPGA into `results/` |
| `ssh_common.sh` | shared SSH defaults for the FPGA host, proxy host, and key |
| `../../sources/fpga/standardize_single_chip.py` | raw `results/` → the 8 sanitized `data/new/fpga/*.csv` (reuses the fleet extractors, one chip) |
| `../../sources/fpga/aggregate_trace_ground_truth.py` | trace `*_bram.csv` → `data/new/fpga/ground_truth_*.csv` |

## What comes out (and how it differs from `data/`)

A single-chip run emits the **same tidy schema** as the committed `data/` — same filenames, same
columns — so anything downstream consumes it unchanged. It is **not** the same content:

- **One chip, not the fleet.** Output holds a single `chip_id` (e.g. `0`), whereas the released
  `data/` spans all 36 chips; any cross-chip statistic collapses to a single point.
- **AE durations.** The long IDD loops run 1800 s by default vs the paper's 3600 s, so even the
  single chip's row counts differ from the released file's same-chip slice.
- **Line endings.** The extractors emit CRLF; the committed `data/` is LF. Harmless for
  downstream parsing (pandas reads both), but the files are not byte-identical.

> **`OUTPUT_DIR` defaults to `data/new/fpga/`**, so the committed `data/` (the released 36-chip
> reference) is left untouched. Point `OUTPUT_DIR=data` only to deliberately replace it.

## See also

- [`../../sources/fpga/README.md`](../../sources/fpga/README.md) — the measurement scripts, duration knobs, and trace pipeline.
