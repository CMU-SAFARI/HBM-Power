# Build the Engine

Ayna builds two executables, `HBM2_runner` and `HBM3_runner`, into
`build/bin/`. The case-study scripts look for them there.

## Requirements

- A C++17 compiler (GCC 9+ or Clang 10+)
- CMake 3.22 or later
- Network access on the first configure. CMake's FetchContent downloads
  nlohmann_json 3.11.3, DRAMUtils 1.7.0, CLI11 and spdlog 1.9.2.

## With presets

```bash
cmake --preset release
cmake --build --preset release -j
```

For a Debug build with `-Wall -Wextra -Wpedantic`, use the `debug` configure
preset with the same build directory:

```bash
cmake --preset debug
cmake --build build --target HBM2_runner HBM3_runner -j
```

## Without presets

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDRAMPOWER_BUILD_CLI=ON
cmake --build build --target HBM2_runner HBM3_runner -j
```

This is what `run_all.sh` runs.

## Options

| CMake option | Default | Effect |
|---|---|---|
| `DRAMPOWER_BUILD_CLI` | `ON` when top-level | Build the runners and the upstream `cli` tool |
| `OPTIMIZE_FOR_NATIVE` | `ON` | Compile with `-march=native -mtune=native` |
| `CPU_TYPE` | empty | Compile for a specific `-march`; overridden by `OPTIMIZE_FOR_NATIVE` |
| `DRAMPOWER_USE_FETCH_CONTENT` | `ON` when top-level | Download the dependencies listed above |
| `DRAMPOWER_USE_FETCH_CONTENT_*` | follows the above | Turn off one dependency download to use a system package instead |

To build binaries that run on other machines, turn off native tuning:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DOPTIMIZE_FOR_NATIVE=OFF
```

## Offline builds

Install nlohmann_json, CLI11 and spdlog as system packages, provide DRAMUtils
through `CMAKE_PREFIX_PATH`, and configure with
`-DDRAMPOWER_USE_FETCH_CONTENT=OFF`.

## Check the build

```bash
./build/bin/HBM3_runner
```

The runner prints its usage line and exits with status 1. To check the
numbers, run the [getting-started](../tutorials/getting-started.md#3-run-a-trace-you-write-by-hand)
example and compare against its output.
