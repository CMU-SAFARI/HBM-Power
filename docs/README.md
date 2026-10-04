# Ayna Documentation

The documentation follows the [Diátaxis](https://diataxis.fr/) structure. Pick
the section that matches what you are trying to do:

| You want to... | Go to |
|---|---|
| Learn Ayna by doing, starting from a fresh clone | [Tutorials](#tutorials) |
| Accomplish one specific task | [How-to guides](#how-to-guides) |
| Look up a fact about the runners, configurations, traces, or case studies | [Reference](#reference) |
| Understand how and why the model works the way it does | [Explanation](#explanation) |

## Tutorials

Step-by-step lessons. Follow them in order.

- [Getting started](tutorials/getting-started.md): build the engine, run a
  trace on a device configuration, change the data pattern, compute stack
  power, and compare HBM2, HBM3E and HBM4.
- [Reproducing the paper](tutorials/reproducing-the-paper.md): regenerate
  Figs. 20 to 23 and the MAPE numbers of the paper, and check them against the
  reference outputs.

## How-to guides

Recipes for one task at a time. They assume Ayna is built.

- [Build the engine](how-to/build-the-engine.md)
- [Write a command trace](how-to/write-a-command-trace.md)
- [Sweep data-pattern activity](how-to/sweep-data-pattern-activity.md)
- [Model a new HBM device or speed bin](how-to/model-a-new-device.md)

## Reference

Facts and formats.

- [Runners](reference/runners.md): `HBM2_runner` and `HBM3_runner` arguments
  and output
- [Configuration files](reference/config-files.md): organization, timing,
  power, `datapattern`, and variation files, and the provided configurations
- [Trace format](reference/trace-format.md)
- [Case studies](reference/case-studies.md): what each directory reproduces,
  its inputs, and its outputs
- [Changes from DRAMPower](reference/changes-from-drampower.md)
- [Upstream DRAMPower README](reference/drampower-upstream.md)

## Explanation

Design discussion and background.

- [How Ayna computes power](explanation/how-ayna-computes-power.md): the
  per-pseudo-channel simulation and the energy equations
- [The data-pattern energy model](explanation/data-pattern-model.md)
- [Where the IDD values come from](explanation/idd-derivation.md): how every
  provided configuration was derived from the HBM2 measurements
- [HBM4 power prediction](explanation/hbm4-power-prediction.md): assumptions
  behind the HBM4 case study
