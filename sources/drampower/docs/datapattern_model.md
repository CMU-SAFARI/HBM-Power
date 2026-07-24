# Data-pattern read/write energy model (HBM2 / HBM3)

Makes the per-read (and per-write) **core** energy depend on data activity, instead of a single
fixed `IDD4R`/`IDD4W`. Derived from the HBM2 beat-pattern measurement campaign (§16 model of
record; the fit is reproduced in `figureB/`). Disabled by default — when no `datapattern` block is
present and no CLI knob is passed, energy is byte-identical to before.

## Model

Per-read data-movement energy (pJ/bit) is a linear function of three **aggregate per-bus activity
knobs**, each in `[0,1]` (0 = quiescent / static data, 1 = maximally toggling on that bus):

| Knob | Physical bus | Drives (HBM2) | Drives (HBM3) |
|------|--------------|---------------|---------------|
| `dq_rate`  | DQ pins (I/O)        | `T_DQ`                       | `T_DQ` |
| `tsv_rate` | 2-bit on-die / TSV   | `T_2bit` **+** TSV coupling  | `T_2bit` |
| `bg_rate`  | 4-bit bank-group bus | `busflip` **+** BG coupling  | `busflip` |

```
pJ/bit   = floor + dq_rate·Δ_DQ + tsv_rate·Δ_TSV + bg_rate·Δ_BG
Δ_DQ     = c_dq
Δ_TSV    = c_t2bit  (+ c_tsv_coupling · tsvcpl_full   if use_coupling)
Δ_BG     = c_busflip(+ c_bg_coupling  · bgcpl_full    if use_coupling)
```

The pJ/bit is turned into an effective current (mA) by one of two forms:

- **Absolute** (`K_mA_per_pJbit` > 0): `IDD4R_eff = IDD3N1 + K·pJ/bit`. Reproduces the HBM2
  calibration exactly (`K = 512`). `K` is tied to the bus width / measurement scale, so it does **not**
  transfer between standards — use it only for HBM2.
- **Relative** (`K` omitted / 0): `IDD4R_eff = IDD3N1 + (IDD4R_base − IDD3N1)·S`, with
  `S = pJ/bit(knobs) / pJ/bit(reference)`. Dimensionless, so the **same shape coefficients** work for
  HBM2 and HBM3 — each scales its own dynamic read current. The config `IDD4R` is the current at the
  reference activity (default quiescent ⇒ knobs 0 ⇒ `S = 1` ⇒ unchanged). This reproduces the absolute
  form exactly when the baseline `IDD4R` is the quiescent read current, so it is a strict
  generalization. Writes reuse the same model (`apply_to_writes`).

HBM2 ships with `use_coupling: true` + `K: 512`; **HBM3 reuses the HBM2 shape coefficients but sets
`use_coupling: false`** (cross-coupling not characterized for HBM3 geometry) and uses the relative form.

## Config (`datapattern` block in the power JSON)

```json
"datapattern": {
  "enabled": true,
  "K_mA_per_pJbit": 512.0,      // omit (or 0) for the relative form
  "floor_pJbit": 2.759,
  "coef_T_DQ": 1.192, "coef_T_2bit": 1.331, "coef_busflip": 0.720,
  "use_coupling": true,         // false for HBM3
  "coef_BG_coupling": 0.345, "coef_TSV_coupling": 0.078,
  "bgcpl_full": 4.0, "tsvcpl_full": 4.0,
  "dq_rate": 0.5, "tsv_rate": 0.5, "bg_rate": 0.0,
  "ref_dq_rate": 0.0, "ref_tsv_rate": 0.0, "ref_bg_rate": 0.0,
  "apply_to_writes": true
}
```

Defaults (when keys are omitted) are the HBM2 §16 coefficients; `bgcpl_full`/`tsvcpl_full` default to
`4.0` (the maximum squared-differential coupling = opposite swing on every adjacent wire pair).

Note on the BG knob: `bg_rate` is the **burst-to-burst toggle rate** — `0` = consecutive bursts
identical (constant stream), `1` = consecutive bursts complementary (full inversion, the experiment's
"bus-flip"). Uniform-random data sits at `0.5`, like the other buses.

Example config: `figureC/configs/HBM3_6400_power_datapattern.json` (HBM3, relative form). The HBM2
absolute form (`K`) is the JSON block shown above; `figureD/make_figureD.py` builds HBM4 datapattern
configs inline (relative form, `use_coupling=false`).

## CLI knob overrides

`HBM2_runner` / `HBM3_runner` accept `--dq-rate=R --tsv-rate=R --bg-rate=R` (any subset). Passing any
one **enables** the model (using config or struct-default coefficients) and overrides that knob, so a
toggle-rate sweep needs no JSON edits:

```
build/bin/HBM3_runner org.json timing.json power.json trace.csv --dq-rate=0.7 --tsv-rate=0.7
```

## Implementation

- `src/DRAMPower/DRAMPower/util/datapattern_model.h` — header-only `DataPatternModel` (shape +
  `effective_current_mA`).
- `MemSpecHBM2`/`MemSpecHBM3` carry a `dataPattern` member; parsed in the runners' `buildMemSpec`.
- `core_calculation_HBM2.cpp` / `core_calculation_HBM3.cpp` substitute `IDD4R_eff`/`IDD4W_eff` into
  `E_RD`/`E_WR` only when `dataPattern.enabled`; the per-bank structural-variation factor still applies.
