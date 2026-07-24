# An Experimental Study of High-Bandwidth Memory (HBM) Power Consumption

This repository contains the data, source code, and scripts needed to reproduce the results in our paper. We provide: (1) the source code of our real HBM2 testing infrastructure based on DRAM Bender, (2) all power-characterization results in the paper, (3) our power model based on DRAMPower and built on our experimental characterization, and (4) all evaluated workloads' memory-access traces. We provide Python scripts to analyze and plot the results.

**Our infrastructure is not designed to accommodate multiple experiments running simultaneously. We kindly ask reviewers to coordinate among themselves and begin evaluating our artifact as early as possible, as only one reviewer can successfully conduct the experiments at a time.**

We expect an end-to-end run of the artifact to take approximately 12 hours. A large fraction of this time is spent on FPGA-based HBM2 experiments (approximately 10 hours), which are designed to run in the background without requiring input from the artifact evaluator.

## Artifact overview

Following Appendix A of the paper, the artifact has four components, each with a single top-level entry point:

| # | Component | Entry point | Output |
|---|-----------|-------------|--------|
| 1 | HBM2 (FPGA) power characterization | `./run_hbm2_characterization.sh` | `data/new/fpga/*.csv` |
| 2 | H200 (HBM3E) power characterization | `./run_h200_characterization.sh` | `data/new/h200/hbm3e_results.csv` |
| 3 | Power-model simulation (DRAMPower) | `./run_power_model.sh` | workload traces + DRAMPower engine |
| 4 | Figure & table generation | `./reproduce_figures_tables.sh` | `figures/*.pdf`, `figures/table3.tex` |

Components 3–4 are a **fast-forward path**: they reproduce every figure and table from the characterization data already committed under `data/`, with no hardware and no remote access. Components 1–2 re-measure that data on real hardware.

## Prerequisites

The artifact runs entirely inside a Docker container **on your own machine** — you need only **Docker** and **Docker Compose**; every other dependency is preinstalled in the image. The measurement components reach two remote endpoints for you, so you need no local FPGA or GPU:

* **Component 1** contacts our FPGA infrastructure over SSH. Place the FPGA SSH key pair we provide via HotCRP (`aevaluator1`, `aevaluator1.pub`) in the repository root before starting the container.
* **Component 2** rents an H200 GPU on demand from [vast.ai](https://vast.ai). Export the API key we provide via HotCRP: `export VAST_API_KEY=<key>`.

The fast-forward path (components 3–4) needs neither key.

We tested the artifact on **Ubuntu 22.04.5 LTS** (Linux `6.8.0-124-generic`) with **Docker 29.1.3** and **Docker Compose v5.1.1**.

### Installing Docker and Docker Compose (Ubuntu 22.04)

Install Docker Engine and the Compose plugin from Docker's official APT repository (the steps below are from Docker's [Install using the apt repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) guide):

```bash
# Add Docker's official GPG key and repository.
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install the engine, CLI, and Compose plugin.
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Run docker without sudo (log out and back in for this to take effect).
sudo usermod -aG docker "$USER"
```

Verify with `docker --version` and `docker compose version`.

> **`docker compose` vs `docker-compose`.** The commands below use `docker compose` (with a space), the modern Compose plugin form installed above. If `docker compose` errors on your system, use the standalone hyphenated command `docker-compose` everywhere in this README instead — e.g. `docker-compose build dev`. Alternatively, install the [Compose plugin](https://docs.docker.com/compose/install/linux/) to get the spaced command.

## Repository structure

```
.
+-- data/                          # Committed characterization data (the fast-forward reference)
+-- figures/                       # Paper figures (regenerated here) and table3.tex
+-- sources/
|   +-- fpga/                      # DRAM Bender programs + HBM2 measurement scripts
|   +-- h200/                      # CUDA benchmark + HBM3E measurement scripts
|   +-- drampower/                 # DRAMPower-based HBM2/HBM3E power model
|   +-- LLMSimulator/              # LLM memory-trace generator
|   +-- figure14,16,17,table3/     # Power-model configs + workload traces
+-- scripts/
|   +-- fpga/                      # FPGA run orchestration     (README inside)
|   +-- h200/                      # vast.ai run orchestration  (README inside)
|   +-- traces/                    # Workload-trace generation
|   +-- figures/ , tables/         # Plotting scripts
+-- run_hbm2_characterization.sh   # Component 1
+-- run_h200_characterization.sh   # Component 2
+-- run_power_model.sh             # Component 3
+-- reproduce_figures_tables.sh    # Component 4
+-- Dockerfile , docker-compose.yml
+-- README.md                      # This file
```

## Getting started

Clone this repository and enter it:

```bash
git clone --branch artifact --single-branch https://github.com/CMU-SAFARI/HBM-Power.git
cd HBM-Power
```

Put the HotCRP-provided FPGA SSH key pair (`aevaluator1`, `aevaluator1.pub`) in the **repository root**, next to `docker-compose.yml`:

```
.
+-- aevaluator1                    # <- FPGA private key (from HotCRP)
+-- aevaluator1.pub                # <- FPGA public key  (from HotCRP)
+-- docker-compose.yml
+-- Dockerfile
+-- README.md
+-- ...                            # (rest of the tree above)
```

`docker compose` bind-mounts these two files into the container at start; both are git- and Docker-build-ignored, so they never leave your machine. Restrict the private key before starting — OpenSSH refuses group/world-readable keys:

```bash
chmod 600 aevaluator1              # required; the read-only mount cannot fix perms later
```

Then build and enter the container:

```bash
# Use your host UID/GID so the container user owns and can read the key.
USER_UID=$(id -u) USER_GID=$(id -g) docker compose build dev
docker compose up -d dev
docker compose exec dev bash        # you land in /workspace
```

If `docker compose` errors, use `docker-compose` everywhere in this README instead — e.g. `docker-compose build dev`. Alternatively, install the Compose plugin to get the spaced command.

Run every command below from `/workspace` inside the container. Stop it with `docker compose down`.

## Reproducing the results

Run everything with `run_all.sh` (below), run the four components one by one, or jump straight to the fast-forward path (3–4).

### Everything at once (`run_all.sh`)

`run_all.sh` runs all four components in dependency order — FPGA characterization → FPGA substitution → H200 characterization → H200 substitution → power model → figures/tables — producing both the released and merged figures/tables. **Export the vast.ai key first:**

```bash
export VAST_API_KEY=<key-from-HotCRP>
./run_all.sh                            # ~10 hours (dominated by the FPGA step)
./run_all.sh --continue                 # resume after a transient failure
```

Each step is checkpointed (`temp/run_all_checkpoint.txt`). If a step fails — e.g. a transient vast.ai error — fix the cause and re-run with `--continue`: completed steps, including the ~10-hour FPGA run, are skipped. The key is only checked while the H200 step is still pending, so a `--continue` that just needs the power-model/figure steps needs no key. On success the script prints `all done`.

The per-component scripts below are the same steps run individually; use them for the hardware-free fast-forward path (3–4) or to run one component at a time.

### 1. HBM2 (FPGA) power characterization

```bash
./run_hbm2_characterization.sh          # add --resume to continue an interrupted run
```

Reprograms our FPGA and runs the single-chip IDD/structural/trace measurements over SSH. **Runtime ~10 hours** — the most time-intensive and involved component. Requires the FPGA SSH keys. See [`scripts/fpga/README.md`](scripts/fpga/README.md).

*Success:* the run finishes with the standardized FPGA CSVs plus `ground_truth_{allzeros,random}.csv` written under `data/new/fpga/`, matching the schema of the committed files in `data/`.

To fold the freshly measured chip into the figures, merge it into a copy of `data/`:

```bash
scripts/fpga/substitute.sh              # data/ + data/new/fpga -> data/merged
```

### 2. H200 (HBM3E) power characterization

```bash
export VAST_API_KEY=<key-from-HotCRP>       # do this first, or the script has nothing to authenticate with
./run_h200_characterization.sh --dry-run    # free: confirms the key and target offer, rents nothing
./run_h200_characterization.sh              # rent -> measure -> fetch -> destroy
```

Rents the cheapest qualifying 1× H200 on vast.ai, runs the memory-power benchmark, writes `data/new/h200/hbm3e_results.csv`, and always destroys the instance. **Runtime a few minutes (~$0.30–0.50).** Requires `VAST_API_KEY`. See [`scripts/h200/README.md`](scripts/h200/README.md).

*Success:* `--dry-run` prints a valid offer, and the full run ends with `data/new/h200/hbm3e_results.csv` written and the instance destroyed (idle ≈33 W, random read ≈226 W; see `scripts/h200/README.md` for expected values).

The API key we provide has **~\$24 of credit loaded**, which should comfortably cover all artifact evaluators. If it runs out, we are happy to top it up — just let us know via HotCRP.

To fold this fresh GPU into the figures, merge it into the released 3-GPU HBM3E data (yielding 4 GPUs total):

```bash
scripts/h200/substitute.sh              # data/new/h200 + data/ -> data/merged/HBM3E_measurements.csv
```

### 3. Power-model simulation (DRAMPower)

```bash
./run_power_model.sh
```

Regenerates the LLaMA workload command traces (`scripts/traces/llm_traces.py`) and builds the DRAMPower `HBM2_runner`. The figure/table step invokes this runner on the workload traces to predict HBM2/HBM3E power. **Runtime a few minutes.** No hardware or remote access.

*Success:* the script prints `done` after building `sources/drampower/build/bin/HBM2_runner` and writing the regenerated traces under `traces/llm_batch_sweep/`.

### 4. Figure & table generation

```bash
./reproduce_figures_tables.sh           # --dry-run to list what would run
./reproduce_figures_tables.sh --skip-h200   # omit the H200-dependent merged figure
```

Runs every plotting script on the committed `data/`, writing `figures/figure{2..17}.pdf` and `figures/table3.tex`. It then additionally reproduces the **merged** figures (`figures/merged/`) for whichever substitutions you have run: the FPGA chip (step 1) and the H200 GPU (step 2). Any missing substitution is reported with a pointer to the script that builds it, and the merged figures are reproduced from whatever is available — nothing errors out. Passing `--skip-h200` deliberately omits the H200-dependent figure (figure16). **Runtime a few minutes.** No hardware or remote access.

*Success:* the script prints `done` after regenerating all released `figures/*.pdf` and `figures/table3.tex`. On the fast-forward path (no substitutions) it notes that the FPGA and H200 merged data are missing and skips `figures/merged/`; that is expected. See *Expected results* below for how to check the figures against the paper.

### Expected results

Compare each regenerated `figures/figureN.pdf` and `figures/table3.tex` against the corresponding figure/table in the paper; they should closely match. We expect that the absolute power consumption values you reproduce for HBM chip/stack id 0 and chip/stack id 1 will likely slightly differ from what is reported in the paper (due to uncontrolled experimental parameters such as HBM stack temperature and current sensor drift). However, the reproduced data will harbor the same trends as described in the paper and will support the conclusions and takeaways in the paper.

Figures 2–13 and 15 come from the HBM2 (FPGA) data; Figures 14, 16, 17 and Table 3 come from the DRAMPower model; Figures 16–17 additionally use the H200 (HBM3E) data.

## Personal-computer scenario

We **strongly recommend the remote setup above** — it is the path we designed and tested. Measuring on your own hardware is possible but will likely incur noticeably more friction: it requires physical boards/GPUs, driver and DRAM Bender setup, and adapting the component 1–2 scripts (which default to our infrastructure) to your machines. **If you intend to use your own hardware, please contact us via HotCRP as early as possible** so we can provide detailed, setup-specific instructions and reduce your burden.

You would additionally need:

* One or more **AMD Alveo U55C** boards with DRAM Bender installed — see [`sources/fpga/DRAMBender/`](sources/fpga/DRAMBender/) for installation instructions.
* One or more **NVIDIA H200** GPUs with CUDA and `nvidia-smi`.

Even then, we recommend running inside the Docker container to avoid dependency issues. Point the component 1 and 2 scripts at your own setup via the environment overrides documented in [`scripts/fpga/README.md`](scripts/fpga/README.md) and [`scripts/h200/README.md`](scripts/h200/README.md). Components 3–4 (model + figures) run locally exactly as in the remote setup.

## Contact

Please reach out via the HotCRP artifact-submission page for questions or credentials.
