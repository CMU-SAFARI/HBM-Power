# Generating HBM Command Traces from LLM Inference

This guide documents how to drive **LLMSimulator** end-to-end to produce a
**cycle-level HBM command trace** (`ACT`/`PRE`/`RD`/`WR`/`REF` …) for a given
LLM running inference on a given GPU. It covers the full path — from the model
config at the top, down to the per-command trace file — plus the supported
models/GPUs, the trace format, and the A100/HBM2 support that was added on top
of the stock simulator.

---

## 1. What this produces, and how (top-down)

```
config.yaml  (model, GPU, workload, use_ramulator: on)
      │
      ▼
LLM inference (eval/test.cpp → Scheduler → Cluster → Model)
      │   builds a per-layer execution graph for the chosen model
      ▼
Per-op DRAM requests  (hardware/*_impl.cpp → issueRamulator())
      │   weight reads, activation r/w, KV-cache reads, output writes
      │   each tensor access is split into 32 B "bundles" (granule)
      ▼
Address generation  (dram/mmap_controller.cpp::getAddrVec)
      │   byte address → 7-level DRAM coordinate
      │   {channel, pseudochannel, rank, bankgroup, bank, row, column}
      ▼
Ramulator 2.0  (DRAMInterface → PIMController frontend → PIM_DRAM_controller)
      │   schedules each request into DDR commands
      ▼
TraceRecorder controller plugin
      │
      ▼
log/cmd_hbm*.log.ch0   ← one line per HBM command
```

Two memory-modeling paths exist, selected by `use_ramulator`:

| `use_ramulator` | behavior |
|---|---|
| `off` (stock default) | analytical model: time = `bytes / bandwidth`, command **counts** estimated in closed form. **No trace.** |
| `on` | cycle-level Ramulator path. Generates real address-resolved requests → **trace.** |

**For trace generation you must set `use_ramulator: on`.**

---

## 2. Prerequisites

- g++ (tested here on 13.3; upstream README used 11.4)
- cmake ≥ 3.14
- A machine with enough RAM for the build. **On this machine, always build with
  `-j4`** — `make -j` (unbounded) spawns ~200 compilers and OOMs the box.

---

## 3. One-time setup

```bash
# 1. Get Ramulator 2.0 (pinned submodule)
git submodule update --init --recursive

# 2. Apply the PIM patch (adds the PIM controller/scheduler and the
#    enriched TraceRecorder that records request semantics alongside commands)
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch
cd ../../..
```

### 3a. g++ ≥ 12/13 fix (required)

Ramulator's headers use `uint64_t` without including `<cstdint>`, which leaked
in transitively on g++ 11 but is rejected by g++ 13. Add `#include <cstdint>`
to these 6 files under `src/dram/ramulator2/src/`:

```
base/utils.h
base/type.h
dram/dram.h
frontend/impl/processor/bhO3/bhcore.h
frontend/impl/processor/bhO3/bhllc.h
frontend/impl/processor/bhO3/bhO3.h
```

(These live in the gitignored submodule, so the fix is local-only and must be
re-applied if you re-init the submodule.)

### 3b. Build

```bash
mkdir build && cd build
cmake ..
make -j4            # 4 cores on this machine
```

This produces `build/run` and copies `config.yaml` + the `dram_config_*.yaml`
files into `build/`.

---

## 4. Configuration (`config.yaml`)

Key knobs for tracing:

```yaml
model:
  model_name: deepseekV3        # see Supported Models below

system:
  gpu_gen: A100                 # A100 | H100 | B100 | B200
  num_device: 8
  processor_type: GPU           # GPU (NOT LOGIC/PIM — those model in-memory compute)
  optimization:
    use_ramulator: on           # REQUIRED for trace generation
    decode_mode: on             # or prefill_mode
    # compressed_kv / use_absorb / use_flash_mla / use_flash_attention …

serving:
  max_batch_size: 32

simulation:
  data: synthesis
  input_len: 1024
  output_len: 10
  precision_byte: 1             # 1 = FP8/INT8
  iter: 5
```

> Keep `processor_type: GPU` for GPU↔HBM traces. The "PIM" naming throughout
> the Ramulator integration (`PIM_DRAM_controller`, `PIMRequest`, …) is inherited
> from the Duplex (MICRO 2024) project this simulator forked from; in GPU mode
> the requests are ordinary HBM reads/writes.

---

## 5. Supported models and GPUs

### Models (`model.model_name`, parsed in `eval/test.cpp`)

`deepseekV3`, `llama4_scout`, `llama4_maverick`, `llama3_405B`, `mixtral`,
`grok1`, `openMoE`, `llama7bMoE`, `llama8B`.

`llama8B` is a custom dense **Llama-3.1-8B** config added in `model_config.h`
(4096 hidden, 32 layers, 32 heads / 8 KV heads GQA, SwiGLU FFN 14336, vocab
128256, no experts) — it fits a single 40 GB A100 with room to spare (~8.6 GB
per device).

Attention variants (MHA/GQA/MQA/MLA) and Mixture-of-Experts are handled per the
model's config.

### GPUs and their memory / trace files

| `gpu_gen` | DRAM model | DRAM config | `memory_scale_factor` (tCK ns) | Trace file |
|-----------|-----------|-------------|-------------------------------|------------|
| H100 | HBM3 (stock) | `dram_config_HBM3_80GB.yaml` | 0.76923 (5.2 Gbps) | `log/cmd_hbm3.log.ch0` |
| B100 / B200 | HBM3E (stock) | `dram_config_HBM3E_192GB.yaml` | 0.5 (8.0 Gbps) | `log/cmd_hbm3E.log.ch0` |
| **A100 (40 GB)** | **HBM2E (added)** | `dram_config_HBM2_40GB.yaml` | 1.6667 (1.2 Gbps / 600 MHz) | `log/cmd_hbm2_40gb.log.ch0` |

The GPU → DRAM-config mapping lives in `src/hardware/device.cpp` (~lines 33–60).

**A100 = 40 GB HBM2 (8 Gb 8-Hi, pseudochannel mode).** `device.cpp` uses
`hbm2_40GB` + `dram_config_HBM2_40GB.yaml` (org `HBM2E_8Gb_2R`), and the A100
`SystemConfig.memory_capacity` is 40 GB. The `hbm2_40GB` geometry mirrors the
real device: `num_channel=16` (8 HBM2 channels × 2 pseudochannels/stack),
`num_rank=2` (SID), `bankgroup=4`×`bank=4` (BA[3:0]), `num_row=16384` (RA[13:0]),
`num_col=32` (CA[5:1]), 256-bit prefetch / 1 KB page; ×5 stacks = 40 GiB.
To switch to the **80 GB HBM2e** part, point both back to `hbm2e_80GB` /
`dram_config_HBM2e_80GB.yaml` (org `HBM2E_A100_80GB`) and set `memory_capacity`
to 80 GB — both sets of artifacts are kept in the tree. Note the 40 GB and 80 GB
single-channel traces differ in size: the 80 GB part has 32 pseudochannels vs
16, so channel-0 in the 40 GB part carries ~2× the commands.

---

## 6. Generate a trace

```bash
cd build
./run > test.log            # or: ./run ../config.yaml
```

The trace is written relative to the run directory, e.g. `build/log/cmd_hbm2e.log.ch0`
for A100. The `.ch0` suffix denotes channel 0 (see Caveats — the model simulates
one representative channel).

To trace a different scenario, edit `config.yaml` (model / gpu_gen / workload)
and re-run.

---

## 7. Trace format

One line per HBM command:

```
clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr, pim_cmd, operand, req_type
```

| Field | Meaning |
|-------|---------|
| `clk` | issue time in **DRAM cycles** (multiply by `memory_scale_factor` for ns) |
| `CMD` | DDR command: `ACT`, `PRE`, `PREA`, `RD`, `WR`, `REFab`, … |
| 7 numeric fields | the physical DRAM address vector |
| `addr` | flat address; `-1` sentinel (this frontend drives Ramulator by address-vector) |
| `pim_cmd` / `operand` / `req_type` | originating request semantics (e.g. `Read` / `DRAM` / `Read`) |

Example (A100 decode):

```
1, ACT, 0, 0, 0, 0, 0, 0, 0, -1, Read, DRAM, Read
9, RD,  0, 0, 0, 0, 0, 0, 0, -1, Read, DRAM, Read
```

Decode is read-dominated (weight + KV-cache reads): a typical run is ~89% `RD`,
with `ACT`/`PRE`/`REF` as the row-management/maintenance commands. Refresh
commands (`REFab`/`PREA`) are controller-generated and have empty
`pim_cmd`/`operand`/`req_type` fields.

The recorder is the `TraceRecorder` controller plugin, configured in each
`dram_config_*.yaml`:

```yaml
Controller:
  plugins:
  - ControllerPlugin:
      impl: TraceRecorder
      path: log/cmd_hbm2e.log
```

---

## 8. A100 / HBM2 support (what was added)

Stock Ramulator only ships HBM2 at 2 Gbps, and its HBM2 model has **6** address
levels (no `rank`), incompatible with the simulator's **7**-level address vector.
So a new 7-level model was written, cloned from the proven HBM3 model:

- **`src/dram/ramulator2/src/dram/impl/HBM2E.cpp`** — registered as `HBM2E`,
  7 levels, A100 org presets, and a timing preset.
  Added to `src/dram/ramulator2/src/dram/CMakeLists.txt`.
- **`dram_config_HBM2e_80GB.yaml`** — `impl: HBM2E`, org `HBM2E_A100_80GB`,
  timing `HBM2E_1.2Gbps`, TraceRecorder → `log/cmd_hbm2e.log`.
- **`src/hardware/memory_config.h`** — `hbm2e_80GB` MemoryConfig (mirrors the
  80 GB geometry).
- **`src/hardware/device.cpp`** — A100 branch: dram config path +
  `memory_scale_factor = 1.6667` + `memory_config = hbm2e_80GB`.
- **`CMakeLists.txt`** — `configure_file` copies the new yaml into `build/`.

### HBM2 timing (`HBM2E_1.2Gbps` preset in `HBM2E.cpp`)

1.2 Gbps / 600 MHz, tCK = 1666.7 ps. The `rate` field is set to `2400` so the
model's `tCK = 1e6/(rate/4)` evaluates to 1666.7 ps (the real data rate is
1.2 Gbps); `rate` only feeds the tCK calc.

Provenance of each field (see comments in `HBM2E.cpp`):

- **From datasheet dump:** `nBL`, `nCL`, `nRCDRD`, `nRCDWR`, `nRP`, `nRAS`,
  `nRC`, `nWR`, `nCWL`, `nCCDS`, `nCCDL`, `nWTRS`, `nWTRL`
- **Estimated (no datasheet value):** `nRTPS`, `nRTPL`, `nRRDS`, `nRRDL`,
  `nRTW`, `nFAW`, `nRFCSB`, `nREFI`, `nRREFD`
- **PIM-only (not issued in GPU mode):** `nCCDAB`, `nCCDSB`
- **From refresh tables (still HBM3 values):** `nRFC`, `nREFISB`
- **`nCL` note:** the dump listed both `nCL=22` and `nRL=7` (contradictory, since
  `RL=AL+CL` needs `AL≥0`). The model's `nCL` *is* the read CAS latency
  (read latency = `nCL+nBL`); `nCL=22` is unphysical for HBM2 (~37 ns), so
  `nCL=7` (the datasheet `nRL`) is used.

### Tuning timing

- **No recompile:** override individual params in `dram_config_HBM2e_80GB.yaml`
  under `timing:` (e.g. `nFAW: 16`).
- **Permanent:** edit the `HBM2E_1.2Gbps` preset in `HBM2E.cpp`, then
  `make -j4`. Refresh `tRFC`/`tREFISB` tables live in `HBM2E.cpp::set_timing_vals()`.

### Disabling refresh

DRAM refresh is injected by the controller's `RefreshManager`. To remove refresh
entirely (no `REF`/refresh-induced `PREA` commands, and rows stay open longer →
far fewer `ACT`/`PRE`), a no-op manager is provided:

- `src/dram/ramulator2/src/dram_controller/impl/refresh/no_refresh.cpp` —
  registered as `NoRefresh` (added to that dir's `CMakeLists.txt`).
- The A100 config selects it: `RefreshManager: { impl: NoRefresh }` in
  `dram_config_HBM2e_80GB.yaml`. Switch back to `AllBank` to re-enable refresh.

The stock H100/B100/B200 configs still use `AllBank`; change their
`RefreshManager` to `NoRefresh` too if you want refresh off there.

---

## 9. Caveats (important for analysis)

1. **Single representative channel.** The Ramulator org is `channel: 1`, and the
   request kernels only emit commands where `addr_vec[0] == 0`
   (`dram/pimkernel/Read.cpp`, `Write.cpp`). The trace is therefore **one
   representative HBM channel** (`.ch0`); full-device behavior is extrapolated by
   `memory_scale_factor`.
2. **Request-shape caching.** `issueRamulator` (`hardware/layer_impl.cpp`) caches
   results by `(layer_type, processor_type, R/W, tensor_size)`. Each *unique*
   request shape is simulated (and traced) once; identical repeats are cache
   hits and are **not re-traced**. The trace is therefore per-unique-shape, not
   every dynamic access.
3. **A100 HBM2 timing is partly estimated** (see §8) — the parameters supplied
   are exact; the rest are reasonable estimates.
4. **Addresses are simulator-internal** (bump-allocated), not CUDA/physical
   addresses.

---

## 10. Extending

- **Full multi-channel trace:** raise the Ramulator `channel` count in the
  DRAM config and remove the `addr_vec[0] == 0` filter in
  `dram/pimkernel/Read.cpp` / `Write.cpp`.
- **Full dynamic trace (every access):** bypass the cache in
  `hardware/layer_impl.cpp::issueRamulator`.
- **New GPU generation:** add a `SystemConfig` (`hardware/hardware_config.h`), a
  `gpu_gen` branch in `device.cpp` (dram config + scale factor + MemoryConfig), a
  `MemoryConfig` in `memory_config.h`, a `dram_config_*.yaml`, and a
  `configure_file` line in the top `CMakeLists.txt`. If it uses a memory the
  stock models don't cover, add a 7-level DRAM model like `HBM2E.cpp`.

---

## 11. File map

| File | Role |
|------|------|
| `config.yaml` | top-level scenario: model, GPU, workload, `use_ramulator` |
| `eval/test.cpp` | entry point; parses config, builds model/cluster, runs iterations |
| `src/hardware/device.cpp` | GPU → DRAM-config / scale-factor / MemoryConfig mapping |
| `src/hardware/*_impl.cpp` | per-op execution; `issueRamulator()` emits DRAM requests |
| `src/dram/mmap_controller.cpp` | byte address → 7-level DRAM coordinate |
| `src/dram/dram_interface.cpp` | drives Ramulator, sends requests, collects stats |
| `src/dram/pimkernel/{Read,Write}.cpp` | tensor → per-bundle DRAM commands |
| `src/dram/ramulator2/src/dram/impl/HBM2E.cpp` | A100 HBM2 device model (added) |
| `dram_config_*.yaml` | Ramulator DRAM + TraceRecorder config per memory type |
| `build/log/cmd_hbm*.log.ch0` | **output: the HBM command trace** |
```
