# H200 (HBM3e) power measurement on rented GPU — `vast_run.py`

`vast_run.py` runs the HBM3e memory-power measurement end-to-end on a real
NVIDIA **H200 SXM** GPU that it rents on demand from [vast.ai](https://vast.ai).
It is fully non-interactive: given an API key, it rents the cheapest qualifying
GPU, builds and runs the CUDA benchmark, fetches the results, and **always
destroys the instance** so billing stops.

You do **not** need a local GPU. Everything GPU-related happens on the rented
instance over SSH.

## What it does

1. Searches vast.ai for the cheapest **verified, on-demand 1× H200** offer
   (reliability ≥ 0.9, CUDA ≥ 12.1, disk ≥ 32 GB).
2. Rents it, on a CUDA `-devel` image (ships `nvcc`).
3. Uploads `../../sources/h200/` and runs `quick_measure_h200.py`: a ~30–40 s
   idle-baseline + random-read measurement.
4. Copies results back and **destroys the instance** (guaranteed, in a `finally`).

## Prerequisites (local machine)

- Python 3 with the `requests` package: `python3 -m pip install --user requests`
- `ssh`, `scp`, `ssh-keygen` (preinstalled on macOS/Linux)

## Supplying the vast.ai API key (artifact evaluators)

The measurement rents a paid GPU, so it needs an API key for a vast.ai account
**with sufficient credit**. **We (the authors) provide this key to the Artifact
Evaluators privately via the HotCRP artifact-submission page** (in the artifact
comments/response to reviewers) — it is intentionally **not** committed to this
repository.

Once you have the key from HotCRP, export it:

```bash
export VAST_API_KEY=<key-from-HotCRP>
```

The script reads `VAST_API_KEY` from the environment (or accepts `--api-key
<key>`). The key is never written to disk or logged.

> The provided key has permission to rent/manage instances. If you use your own
> key instead and get a `401 … requires Two Factor Authentication`, enable 2FA
> on your vast.ai account and generate a fresh key.

## Usage

Run from the repository root:

```bash
# 1. Free, read-only check: confirms the key works and shows the offer that
#    would be rented. Rents nothing, costs nothing.
python3 scripts/h200/vast_run.py --dry-run

# 2. Full end-to-end: rent -> measure -> fetch -> destroy.
python3 scripts/h200/vast_run.py
```

Progress (offer selection, provisioning status, SSH readiness, measurement) is
printed live.

### Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--api-key KEY` | `$VAST_API_KEY` | vast.ai API key |
| `--max-price N` | `5.0` | refuse offers above `N` $/hr |
| `--dry-run` | off | search + select only; rent nothing |
| `--results-dir DIR` | `temp/h200_results` | where verbose results are written |
| `--image IMG` | `nvidia/cuda:12.4.1-devel-ubuntu22.04` | CUDA `-devel` image with `nvcc` |
| `--disk N` | `32` | instance disk in GB |

## Outputs

- **Summary (tracked):** `data/new/h200/hbm3e_results.csv`
- **Verbose (scratch):** `temp/h200_results/out/` — `idle.csv`, `read_rand.csv`,
  `read_rand.stdout.txt`, `quick_summary.csv`

Expected values (H200 SXM, HBM3e):

| metric | mem power (W) | read (GB/s) |
|--------|---------------|-------------|
| idle baseline | ~33 | — |
| random read (total) | ~226 | ~4500 |
| active (read − idle) | ~193 | — |

Exact numbers vary by the specific host/machine that gets rented.

## Cost

A full run bills a few minutes at the on-demand H200 rate (~$3.5–4/hr), i.e.
roughly **$0.30–0.50 per run**. The instance is destroyed automatically when the
run finishes or errors. If a run is interrupted, please contact the authors via HotCRP.

## A note to evaluators

This script has a lot of moving parts — it depends on live vast.ai inventory,
the vast.ai API, on-demand GPU provisioning, SSH connectivity, and a remote
build/measurement — and it has **not been extensively tested** across
environments. Unexpected errors are possible. If you hit one, please reach out
to us via the HotCRP artifact-submission page and we will help you resolve it
on the go.

