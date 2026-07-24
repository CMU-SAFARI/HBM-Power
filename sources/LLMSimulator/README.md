# LLMSimulator — HBM2 Command-Trace Generator for DRAM Power Modeling

This is a fork of [LLMSimulator](https://github.com/scale-snu/LLMSimulator) (itself
built on the **Duplex**/MICRO-2024 simulator) repurposed for one job:

> **Produce a cycle-level HBM2 DRAM command trace from LLM inference, for one
> representative HBM2 channel, so that an analytical HBM2 DRAM *power* model can be
> driven and validated against a realistic memory-command workload.**

Given a model (e.g. Llama-3.1-8B), a GPU (e.g. A100), and a workload (prefill or
decode, batch size, sequence lengths, numeric precision), it runs a graph-level
inference simulation, turns every tensor access into address-resolved DRAM
requests, schedules them through a patched [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2),
and records **every issued HBM command** (`ACT`/`PRE`/`RD`/`WR`/…) to a trace file.

The upstream simulator is a general cycle-level LLM performance/energy model; this
fork adds an **A100 / HBM2** memory path and the trace-recording flow, and is meant
to be used as a **trace generator**, not a standalone power model.

---

## What it produces

**1. The command trace** — `build/log/cmd_*.log.ch0`, one line per HBM command for
a single representative channel:

```
clk, CMD, channel, pseudochannel, rank, bankgroup, bank, row, column, addr, pim_cmd, operand, req_type
```

```
1, ACT, 0, 0, 0, 0, 0, 0, 0, -1, Read, DRAM, Read
9, RD,  0, 0, 0, 0, 0, 0, 0, -1, Read, DRAM, Read
```

- `clk` is in **DRAM command cycles** (multiply by `memory_scale_factor` for ns).
- The 7 numeric fields are the physical DRAM coordinate; `addr` is `-1` because the
  frontend drives Ramulator directly by address-vector.
- `pim_cmd`/`operand`/`req_type` carry the originating request's semantics.

**2. Per-command dynamic energy coefficients** — `src/dram/power.h`. In GPU mode each
command costs (per 32 B access): `ACT` = 0.909 nJ, `RD` = `WR` = 0.891 nJ, `PRE` = 0
(precharge is not charged). These let a consumer convert the command trace into
dynamic energy and power.

---

## Pipeline

```
config.yaml  (model, GPU, workload, use_ramulator: on)
   └─▶ inference graph        eval/test.cpp → Scheduler → Cluster → Model
        └─▶ per-op DRAM reqs   hardware/*_impl.cpp :: issueRamulator()   (32 B "bundles")
             └─▶ address map   dram/mmap_controller.cpp :: getAddrVec()  (byte → 7-level coord)
                  └─▶ Ramulator 2.0   PIM frontend → DRAM controller
                       └─▶ TraceRecorder plugin → build/log/cmd_*.log.ch0
```

The full top-down walkthrough — including the A100/HBM2 device model and timing
provenance — lives in [`TRACE_GENERATION.md`](TRACE_GENERATION.md).

---

## Supported models & GPUs

**Models** (`model.model_name`): `deepseekV3`, `llama4_scout`, `llama4_maverick`,
`llama3_405B`, `mixtral`, `grok1`, `openMoE`, `llama7bMoE`, `llama8B`. Attention
variants MHA/GQA/MQA/MLA and Mixture-of-Experts are handled per the model config.
`llama8B` is a custom **dense Llama-3.1-8B** config (4096 hidden, 32 layers,
32 heads / 8 KV heads GQA, SwiGLU FFN 14336, vocab 128256); at FP16 it occupies
~21 GB on a single 40 GB A100 (≈15 GB weights + ≈4 GB KV cache), or ~8.6 GB at FP8.

**GPUs** (`system.gpu_gen`):

| `gpu_gen`   | DRAM model            | trace file                  |
|-------------|-----------------------|-----------------------------|
| **A100**    | 40 GB **HBM2** *(added)* | `log/cmd_hbm2_40gb.log.ch0` |
| H100        | 80 GB HBM3            | `log/cmd_hbm3.log.ch0`      |
| B100 / B200 | 192 GB HBM3E          | `log/cmd_hbm3E.log.ch0`     |

The **A100 / HBM2** path is this fork's main addition: a 7-level HBM2 device model
(`src/dram/ramulator2/src/dram/impl/HBM2E.cpp`) configured as a specific **1.2 Gbps
HBM2** part (600 MHz command clock, `tCK ≈ 1.667 ns`), 8 Gb 8-Hi dies in pseudochannel
mode totaling 40 GiB (16 channels × 2 ranks × 4 bankgroups × 4 banks × 16384 rows ×
32 cols × 32 B, ×5 stacks), with refresh disabled. This is deliberate: `gpu_gen: A100`
is used only as the harness — the memory is modeled as a known 1.2 Gbps HBM2 device,
**not** the A100's stock ~2.4 Gbps HBM2. See `TRACE_GENERATION.md` §8.

---

## Build

Prerequisites: `g++` (tested on 13.3), `cmake ≥ 3.14`.

```bash
git submodule update --init --recursive          # Ramulator 2.0 (pinned)
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch     # PIM controller + enriched TraceRecorder
cd ../../..

mkdir build && cd build
cmake ..
make -j4                                          # see caveats below
```

**Build caveats**

- **Use `-j4` on this machine.** Bare `make -j` spawns ~200 compilers building
  Ramulator and exhausts RAM.
- **g++ ≥ 12/13 fix:** Ramulator's headers use `uint64_t` without `<cstdint>`. Add
  `#include <cstdint>` to `base/utils.h`, `base/type.h`, `dram/dram.h`, and the three
  `frontend/impl/processor/bhO3/bh{core,llc,O3}.h` under `src/dram/ramulator2/src/`.
  (These live in the gitignored submodule, so the fix is local-only.)

The build produces `build/run` and copies `config.yaml` + the `dram_config_*.yaml`
files into `build/`.

---

## Run & configure

```bash
cd build
./run > test.log          # or: ./run ../config.yaml
```

Key `config.yaml` knobs for tracing:

```yaml
model:   { model_name: llama8B }
system:
  gpu_gen: A100                 # A100 | H100 | B100 | B200
  num_device: 1                 # 1 = whole batch on one GPU (no data-parallel split)
  processor_type: GPU           # keep GPU (NOT LOGIC/PIM — those model in-memory compute)
  optimization:
    use_ramulator: on           # REQUIRED for trace generation
    decode_mode: on             # or prefill_mode
simulation:
  input_len: 1024
  output_len: 10
  precision_byte: 2             # 2 = FP16/BF16, 1 = FP8/INT8
  iter: 5
serving: { max_batch_size: 32 }
```

> The "PIM" naming throughout the Ramulator integration (`PIM_DRAM_controller`,
> `PIMRequest`, …) is inherited from Duplex; in GPU mode the requests are ordinary
> HBM reads/writes.

---

## Scope & how to use the trace correctly

The trace models **dynamic, per-channel HBM2 activity**. Read this before drawing
conclusions from it.

1. **One representative channel = 1/40 of the device.** The Ramulator org has
   `channel: 1` and the kernels emit only `addr_vec[0] == 0`. The address mapping
   fine-grain-interleaves traffic across all channels, so channel 0 is an *unbiased*
   sample. Multiply per-channel results by **40** (`num_cube × num_channel / 2`) for
   the full 40 GB A100. All 40 channels are busy simultaneously, so this is valid for
   total active power as well as command counts. The current config sets
   `num_device: 1`, so the whole batch of 32 runs on the one traced GPU — the trace
   already represents a full GPU, with no additional data-parallel (×N) factor.

2. **`clk` → time:** multiply `clk` by `memory_scale_factor` (1.667 ns for A100).

3. **The trace is *per-unique-request-shape*, not a full dynamic stream.**
   `issueRamulator` caches simulated results by `(op, processor, R/W, tensor_size)`,
   so each distinct request shape is traced **once**; identical repeats across the
   32 layers, decode steps, and iterations are cache hits and are **not** re-emitted.
   Consequently:
   - ✅ **Valid for POWER.** Every burst is a bandwidth-saturated streaming access, so
     active power is nearly constant across the trace (**≈0.27 W/channel, ~11 W across
     40 channels**; CoV a few %). The missing repeats do not bias **average or peak
     active power**. Verified invariant across config changes: switching FP8→FP16 and
     batch 4→32/device grew the trace ~101k→451k commands and the span ~320→1420 µs,
     yet the power rate held at ≈0.27 W/channel — precision and batch scale command
     *volume and run time together*, not power.
   - ❌ **Not valid for absolute ENERGY.** Raw command counts are far below the true
     dynamic volume and undercount *non-uniformly* (e.g. weight reads collapse across
     all 32 layers; the three SwiGLU FFN matrices share a byte-size and collapse to
     one). For energy, bypass the cache (`TRACE_GENERATION.md` §10) or reweight each
     shape by its true occurrence count before summing.

4. **Static/background power and refresh are *not* modeled — by design.** Refresh uses
   a `NoRefresh` manager (no `REF`/refresh-induced `PRE` in the trace) and the energy
   model has only dynamic per-command terms. Add refresh/standby analytically if you
   need them.

5. **The current config runs FP16/BF16** (`precision_byte: 2`); set `precision_byte: 1`
   for FP8/INT8. Precision scales command volume and energy, but not active power.

6. **Addresses are simulator-internal** (bump-allocated), not CUDA/physical addresses.

---

## Repository layout

| Path | Role |
|------|------|
| `config.yaml`                         | top-level scenario: model, GPU, workload, `use_ramulator` |
| `eval/test.cpp`                       | entry point; parses config, builds model/cluster, runs iterations |
| `src/model/`, `src/module/`           | LLM graph: layers, attention, FFN/MoE, embedding, lm_head |
| `src/hardware/device.cpp`             | GPU → DRAM-config / scale-factor / MemoryConfig mapping |
| `src/hardware/*_impl.cpp`             | per-op execution; `issueRamulator()` emits DRAM requests |
| `src/dram/mmap_controller.cpp`        | byte address → 7-level DRAM coordinate |
| `src/dram/dram_interface.cpp`         | drives Ramulator, sends requests, collects stats |
| `src/dram/pimkernel/{Read,Write}.cpp` | tensor → per-bundle DRAM commands (single-channel filter) |
| `src/dram/power.h`                    | per-command dynamic energy coefficients |
| `src/dram/ramulator2/.../impl/HBM2E.cpp` | A100 HBM2 device model (added) |
| `dram_config_*.yaml`                  | Ramulator DRAM + TraceRecorder config per memory type |
| `build/log/cmd_*.log.ch0`             | **output: the HBM command trace** |
| `TRACE_GENERATION.md`                 | detailed end-to-end how-to |

---

## Provenance

This is a research fork of [LLMSimulator](https://github.com/scale-snu/LLMSimulator)
(SCALE Lab, SNU), which builds on the simulator from the MICRO-2024 paper
*"Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query
Attention, and Continuous Batching."* Memory modeling uses a patched
[Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) (CMU-SAFARI). The A100/HBM2
device model, trace-recording flow, and the power-modeling scope documented above are
additions in this fork.
