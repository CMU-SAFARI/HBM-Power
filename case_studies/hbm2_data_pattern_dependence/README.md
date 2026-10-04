# Data-pattern dependence (Section 6.1, Fig. 21): FGDRAM single-toggle vs Ayna

`model_vs_fgdram_single_toggle.pdf` — predicted vs measured per-read IDD4R across the 32
beat patterns, for two data-pattern energy models.

- **Baseline** "DQ + Core Toggle": pre-GSA floor + fixed I/O (1.60·T_DQ) + one post-GSA
  toggle. R² ≈ 0.41 — a single on-die toggle can't order the BG-bus patterns.
- **Ayna** "DQ + TSV + BG Toggle & Coupling": floor + T_DQ + T_2bit + busflip +
  squared (Miller) adjacent-wire coupling on the 4-bit BG bus (best of 24 wire orders) and
  the 2-bit TSV bus. R² ≈ 0.97.

The two labelled patterns make the point: `0101+flip` is already captured by core toggle,
while `1001` (no BG-bus toggle) is missed by the baseline (~1.4 pJ/bit) and recovered by the
coupling term.

## Reproduce

Engine-free — just a least-squares fit on the beat-pattern measurement:

```bash
python3 plot_toggle_models.py        # -> model_vs_fgdram_single_toggle.{png,pdf}
```

Needs Python with `numpy` and `matplotlib`. Prints R² and MAPE for both models, the inferred
BG-bus wire order (1,3,0,2), and the fitted coefficients with and without coupling.

## Inputs

```
data/beat_pattern_perpattern.csv    32 patterns: beat_pattern, col1_inverted, mean_idd_mA, n_chips
configs/IDD_ours_allzeros.json      Ayna all-0s HBM2 config (read background IDD3N1)
```

## Method notes

- **Bus toggle rates.** Each pattern is the 4-beat sequence one DQ pin carries per burst;
  `+flip` (`col1_inverted = 1`) means consecutive bursts alternate between the pattern and its
  inverse, because the read loop alternates two columns holding complementary data. The DQ
  stream is deserialized 1:2 onto two TSV signals (even and odd beats) and 1:2 again onto four
  bank-group (BG) bus signals (BG wire k carries beat k of each burst). A bus's toggle rate is
  the fraction of its own clock cycles in which a wire changes value, averaged over its wires
  (`_tog(s, 1)`, `_tog(s, 2)`, `_tog(s, 4)`). For example, `0101` toggles DQ 100 %, TSV 0 %,
  BG 0 %; `0110+flip` toggles DQ 75 %, TSV 50 %, BG 100 %. The BG toggle rate is 1 exactly for
  the `+flip` patterns and 0 otherwise.
- **Target.** Per pattern, `pJ/bit = (mean_idd_mA − IDD3N1) / K`, with `IDD3N1` the read
  background (raw board current incl. off-power) of the Ayna all-0s config that Fig. 20 uses,
  read at run time from `configs/IDD_ours_allzeros.json` (1014.5 mA; a copy of the
  `hbm2_model_comparison` config),
  and `K = NCYC·BITS / (VDD·(BL/DR)·tCK) = 512 mA per pJ/bit` (the engine's read-dynamic slope).
  Axes are shown back in mA via `IDD4R = IDD3N1 + K·pJ/bit` — an affine rescale, so R², the
  absolute errors and the mA predictions are unchanged whether expressed in pJ/bit or mA. The
  MAPE is not: the script prints it on both axes (pJ/bit: baseline 11.4 %, Ayna 2.3 %, the
  numbers quoted in Section 6.1; mA: 8.2 % and 1.7 %).
- **Fit.** Both models are ordinary least squares over the 32 patterns. The BG-bus wire order
  of the coupling term is the one fitted discrete parameter (best of 24 permutations). Setting
  `WIRE_ORDER = "linear"` in the script uses the fixed order (1,0,3,2) instead.

## Coupling: included vs excluded

The coupling terms model the extra energy when two adjacent wires of a bus switch in opposite
directions (Miller coupling). Per adjacent wire pair and clock cycle, the metric is
`(Δa − Δb)²`, where `Δ` ∈ {−1, 0, +1} is a wire's change: 4 when the two wires swing in opposite
directions, 1 when only one swings, 0 when both stay or swing together (`_cpl` in the script).

Fig. 21 uses the model **with** coupling. The device configurations in `config/`
(`HBM2_1200MTs`, `HBM3E_6400MTs`, `HBM4_8000MTs`) and the engine's built-in defaults use the
same model refit **without** coupling, which needs only the three toggle rates as inputs.

### Fit quality

| Model (least squares on the 32 patterns) | R² | MAPE (pJ/bit) | MAPE (mA) |
|---|---|---|---|
| FGDRAM single toggle (baseline) | 0.41 | 11.4 % | 8.2 % |
| Ayna, no coupling | 0.87 | 4.4 % | 3.2 % |
| Ayna, TSV coupling only | 0.88 | 4.7 % | 3.4 % |
| Ayna, BG coupling only | 0.96 | 2.2 % | 1.6 % |
| Ayna, BG + TSV coupling (Fig. 21) | 0.97 | 2.3 % | 1.7 % |

Almost all of the coupling gain comes from the BG bus. Without coupling, the largest errors are
on the `+flip` patterns whose BG wires all swing the same way (`0000+flip` and `1111+flip`,
over-predicted by about 400 mA) or whose neighbors swing apart (`0111+flip`, under-predicted by
about 235 mA): the BG toggle rate is 1 for all of them, and only the coupling term tells them
apart.

### Coefficients (pJ/bit)

| Coefficient | Engine field | With coupling (Fig. 21) | Without coupling (`config/`) |
|---|---|---|---|
| Floor | `floor_pJbit` | 3.1038 | 3.1798 ± 0.196 |
| DQ toggle | `coef_T_DQ` | 1.1921 | 1.0400 ± 0.248 |
| TSV toggle | `coef_T_2bit` | 1.3315 | 1.4875 ± 0.248 |
| BG bus flip | `coef_busflip` | 0.7198 | 1.4100 ± 0.124 |
| BG coupling | `coef_BG_coupling` | 0.3451 | — |
| TSV coupling | `coef_TSV_coupling` | 0.0780 | — |

Without coupling, the bus-flip coefficient roughly doubles: it absorbs the coupling energy
that every flipped pattern pays on average. DQ and TSV trade off against each other (±0.25
standard errors); the 32 patterns separate them less well than the other terms.

### Sensitivity to the fitted wire order

The BG coupling term depends on which BG wires are adjacent. An order and its reverse are
equivalent, so the 24 permutations give 12 distinct adjacencies:

| BG wire order | R² | MAPE (pJ/bit) |
|---|---|---|
| Best fit (1,3,0,2), used in Fig. 21 | 0.969 | 2.3 % |
| (1,0,3,2), the script's `"linear"` setting | 0.932 | 3.6 % |
| Physical order (0,1,2,3) | 0.877 | 4.8 % |
| No coupling | 0.867 | 4.4 % |

Every order improves on the model without coupling in R² and beats the FGDRAM baseline.

### Effect on the HBM3E and HBM4 case studies

The HBM3E validation (Fig. 22) and the HBM4 case study (Fig. 23) run the engine with the
coupling terms switched off, using the coefficients in
`../hbm3e_validation/configs/HBM3_6400_power_datapattern.json` (floor 2.759, DQ 1.192,
TSV 1.331, bus flip 0.720). Re-running both with the no-coupling fit raises every prediction
slightly and leaves the conclusions unchanged:

| Result | Case study as provided | No-coupling fit |
|---|---|---|
| HBM3E device power at 1.1 V (H200 measured: 248.2 W, range 241.7–255.2) | 242.5 W | 245.5 W |
| HBM3E read energy at 6.4 Gbps (H200 measured: 6.864 pJ/bit) | 6.730 pJ/bit | 6.813 pJ/bit |
| HBM4 (1.05 / 0.9 V) energy per bit, 4.8 → 8.0 Gbps | 6.24 → 6.73 | 6.34 → 6.83 |
| HBM3E (1.1 V) energy per bit, 4.8 → 6.4 Gbps | 6.52 → 6.73 | 6.60 → 6.81 |
| HBM4 savings vs HBM3E at 6.4 Gbps (three voltage points) | 3.71 / 5.25 / 9.51 % | 3.45 / 4.59 / 8.95 % |

Both columns use the current engine; the paper reports 242.3 W and 6.727 pJ/bit for the first
two rows (see [Reproducing the paper](../../docs/tutorials/reproducing-the-paper.md)).

The HBM4 savings shrink slightly because the no-coupling fit gives the DQ pins a smaller
share of the read energy, so less of it moves to HBM4's lower VDDQ rail.

### Using coupling in the engine

The engine still supports the coupling terms. Add them to a power file's `datapattern` block:

```json
"use_coupling": true,
"coef_BG_coupling": 0.345, "coef_TSV_coupling": 0.078,
"bgcpl_full": 4.0, "tsvcpl_full": 4.0
```

together with the with-coupling coefficients above. The engine adds
`coef_BG_coupling × bgcpl_full × bg_rate` (and the TSV analogue), so it assumes the coupling
metric grows linearly with the toggle rate and reaches `bgcpl_full` at full activity. With
`bgcpl_full = 4` this is the worst case, where adjacent wires always swing in opposite
directions. For uniform-random data the expected metric is `2 × toggle rate` (1.0 at a toggle
rate of 0.5), so `bgcpl_full = tsvcpl_full = 2.0` matches random data.
