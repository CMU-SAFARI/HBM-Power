# Sweep Data-Pattern Activity

Read and write energy in Ayna depend on three toggle-rate knobs, each between
0 and 1:

| Knob | Command-line flag | Bus |
|---|---|---|
| DQ | `--dq-rate=R` | External DQ pins |
| TSV | `--tsv-rate=R` | 2-bit on-die / TSV bus |
| BG | `--bg-rate=R` | 4-bit bank-group bus |

0 means static data and 1 means every wire toggles on every transfer.
Uniform-random data is 0.5 on all three. The model behind the knobs is
described in [The data-pattern energy model](../explanation/data-pattern-model.md).

## Override the knobs on the command line

Pass any subset of the flags after the positional arguments:

```bash
C=config/HBM3E_6400MTs
./build/bin/HBM3_runner $C/organization.json $C/timing.json $C/power.json \
    case_studies/hbm3e_validation/traces/hbm3_random_read_4rpa_ref.csv \
    --dq-rate=0.25 --tsv-rate=0.5 --bg-rate=0.5
```

A flag overrides that knob only. A knob you do not pass keeps the value from
the power file's `datapattern` block, or the default of 0.5 (uniform-random
data) if the block does not set it. Passing any flag also turns the model on
if the power file disables it with `"enabled": false`.

The runner confirms the setting on its first line:

```text
Data-pattern model: dq_rate=0.25, tsv_rate=0.5, bg_rate=0.5 (relative form)
```

## Run a sweep

```bash
C=config/HBM3E_6400MTs
H3="$C/organization.json $C/timing.json $C/power.json \
    case_studies/hbm3e_validation/traces/hbm3_random_read_4rpa_ref.csv"

for r in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  p=$(./build/bin/HBM3_runner $H3 --dq-rate=$r --tsv-rate=$r --bg-rate=$r \
        | awk '/Average power/ {print $3}')
  echo "$r,$p"
done
```

The result is per-pseudo-channel power in mW for each toggle rate.

## Set the knobs in the configuration instead

To make a setting the default for a configuration, edit its `datapattern` block:

```json
"datapattern": {
  "enabled": true,
  "dq_rate": 0.15, "tsv_rate": 0.15, "bg_rate": 0.15
}
```

All fields are listed in [Configuration files](../reference/config-files.md#datapattern-block).

## Choosing realistic values

- Uniform-random data is 0.5 on every bus. The case studies use it as an
  upper bound.
- Static data (for example, all zeros) is 0 on every bus. In the device
  configurations, `IDD4R` and `IDD4W` are the currents at this reference point.
