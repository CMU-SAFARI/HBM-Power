# The Data-Pattern Energy Model

Standard DRAM power models charge every read the same energy (`IDD4R`) and
every write the same energy (`IDD4W`). Our HBM2 measurements show that read
current depends strongly on the data. Across the 32 beat patterns we measured,
the mean read current ranges from 2.54 A to 4.37 A, and how often the DQ pins
toggle does not explain the order of the patterns. Ayna's data-pattern model
captures this with three toggle-rate knobs, one per physical bus that the data
crosses.

## Three buses, three knobs

| Knob | Bus | What it measures |
|---|---|---|
| `dq_rate` | DQ pins (external I/O) | Fraction of DQ wires that toggle between beats |
| `tsv_rate` | 2-bit on-die / TSV bus | Toggling on the through-silicon vias between dies |
| `bg_rate` | 4-bit bank-group bus | Burst-to-burst toggling: 0 means consecutive bursts are identical, 1 means each burst is the inverse of the previous one |

Each knob is between 0 (static data) and 1 (every wire toggles every time).
Uniform-random data gives 0.5 on all three.

## Energy per bit

Read data-movement energy per bit is linear in the knobs:

```text
pJ/bit = floor + c_dq × dq_rate + c_t2bit × tsv_rate + c_busflip × bg_rate
```

The coefficients were fitted by least squares to the measured read current of
the 32 beat patterns. They are the defaults of the engine and of the device
configurations in `config/`:

| Coefficient | Field | Value (pJ/bit) |
|---|---|---|
| `floor` | `floor_pJbit` | 3.1798 |
| `c_dq` | `coef_T_DQ` | 1.0400 |
| `c_t2bit` | `coef_T_2bit` | 1.4875 |
| `c_busflip` | `coef_busflip` | 1.4100 |

This model reaches R² ≈ 0.87 on the 32 patterns, against R² ≈ 0.41 for FGDRAM's
single-toggle model, with an energy-per-bit MAPE of 4.4% vs. 11.4%. The
[`hbm2_data_pattern_dependence`](../reference/case-studies.md#hbm2_data_pattern_dependence)
case study reproduces this fit and the more detailed variant shown in Fig. 21 of
the paper.

## From energy per bit to current

The engine works with currents, so the model converts energy per bit into an
effective `IDD4R` (and, with `apply_to_writes`, `IDD4W`) in one of two ways.

**Relative form** (`K_mA_per_pJbit` = 0, used by every provided configuration that enables the model):

```text
IDD4R_eff = IDD3N1 + (IDD4R − IDD3N1) × pJ/bit(knobs) / pJ/bit(reference)
```

Here `IDD4R` in the configuration is the read current at the reference activity
`ref_*_rate`, which is static data by default. The model only scales the
dynamic part of that current by the ratio of energies. The ratio has no units,
so the same coefficients work for any standard, and each standard scales its
own measured or extrapolated read current.

**Absolute form** (`K_mA_per_pJbit` > 0):

```text
IDD4R_eff = IDD3N1 + K × pJ/bit
```

`K` converts pJ/bit into mA of read current for a given bus width, burst
length, clock, voltage and number of pseudo-channels reading at once. For the
board-level HBM2 currents of the case studies (1.2 Gbps, 8 pseudo-channels
reading) it is 512 mA per pJ/bit. It does not carry over to another
configuration.

## Worked example: HBM3E read with random data

Take the HBM3E device configuration (`config/HBM3E_6400MTs/power.json`). Its
rails are:

```text
IDD3N1 =  71.2 mA      (read background)
IDD4R  = 560.7 mA      (read current with static data, the reference activity)
```

**Reference (static data, all knobs 0):**

```text
pJ/bit(ref) = floor = 3.1798
```

**Uniform-random data (all knobs 0.5, the configuration's default):**

```text
pJ/bit = 3.1798 + 0.5 × 1.0400 + 0.5 × 1.4875 + 0.5 × 1.4100
       = 3.1798 + 0.5200     + 0.7438     + 0.7050
       = 5.1486

S         = 5.1486 / 3.1798 = 1.619
IDD4R_eff = 71.2 + (560.7 − 71.2) × 1.619
          = 71.2 + 489.5 × 1.619
          = 863.8 mA
```

Random data raises the effective read current from 560.7 mA to 863.8 mA. Only
the dynamic part above the 71.2 mA background is scaled, and it grows 1.62×.
Each read burst is charged `VDD × (IDD4R_eff − IDD3N1) × tBurst`, so read
energy per burst grows by the same factor.

Other activity levels, from the same configuration:

| `dq_rate` | `tsv_rate` | `bg_rate` | pJ/bit | S | `IDD4R_eff` (mA) | DQ share `f_DQ` |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 3.180 | 1.000 | 560.7 | 0% |
| 0.5 | 0 | 0 | 3.700 | 1.164 | 640.7 | 14.1% |
| 0 | 0.5 | 0 | 3.924 | 1.234 | 675.2 | 0% |
| 0 | 0 | 0.5 | 3.885 | 1.222 | 669.2 | 0% |
| 0.25 | 0.25 | 0.25 | 4.164 | 1.310 | 712.2 | 6.2% |
| 0.5 | 0.5 | 0.5 | 5.149 | 1.619 | 863.8 | 10.1% |
| 1 | 1 | 1 | 7.117 | 2.238 | 1166.8 | 14.6% |

At equal toggle rates, the TSV bus costs the most per unit of activity, then the
bank-group bus, then the DQ pins.

You can reproduce the 303.1 mA increase with the engine. Run the HBM3E
random-read trace twice and compare the total energies:

```bash
C=config/HBM3E_6400MTs
T=case_studies/hbm3e_validation/traces/hbm3_random_read_4rpa_ref.csv
./build/bin/HBM3_runner $C/organization.json $C/timing.json $C/power.json $T \
    --dq-rate=0 --tsv-rate=0 --bg-rate=0 | grep "Total energy"
./build/bin/HBM3_runner $C/organization.json $C/timing.json $C/power.json $T \
    | grep "Total energy"
```

The difference is 16,668,870 pJ. The trace has 40,000 reads of `tBurst` =
8 / 4 × 625 ps = 1250 ps each at `VDD` = 1.1 V, so the difference corresponds
to 16,668,870 / (1.1 × 1250 × 40,000) = 0.3031 A. That equals 863.8 − 560.7
mA. HBM3E has a single rail, so `VDDQ` equals `VDD` and the split-rail weighting
below has no effect here.

## The DQ share and split rails

With the model enabled, the share of the read energy that goes to the external
pins is known:

```text
f_DQ = c_dq × dq_rate / pJ/bit(knobs)
```

`HBM3_runner` uses it to charge that share of read energy at `VDDQ` instead of
`VDD`. The floor, TSV and bank-group terms are on-die and stay on the core
rail. At uniform-random data, the DQ pins account for about 10% of the read
data-movement energy. See [How Ayna computes power](how-ayna-computes-power.md#hbm3-split-rails).