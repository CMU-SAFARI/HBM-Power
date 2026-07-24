#!/usr/bin/env bash
# Reproduce the HBM-Power figure14 LLaMa batch sweep:
# bs{1,2,4,8,16,32,64,128} @ ctx=1024, converted to the DRAMPower 8-column
# format and verified byte-for-byte against sources/figure14/traces/.
#
# Config: model llama8Bshort (n_layer=1; identical trace to full llama8B under
# the request-shape dedup), decode, FP16 (precision_byte=2), single device,
# committed HBM2 model (1.2 Gbps / nBL=2 / single rank), output_len=2.
# Each raw channel-0 trace is reduced to pseudochannel 0 and reformatted by
# drampower_convert.py (see that file for the exact transform).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$REPO/build"
BASE_CONFIG="$REPO/config.yaml"
CONVERT="$REPO/drampower_convert.py"
REF="${REF:-/home/akakolyris/Documents/safari/HBM-Power/sources/figure14/traces}"
OUT_DIR="$REPO/traces/figure14_batch"
mkdir -p "$OUT_DIR"

read -ra BATCH_SIZES <<< "${BATCH_SIZES:-1 2 4 8 16 32 64 128}"
CTX="${CTX:-1024}"
TRACE_SRC="$BUILD/log/cmd_hbm2_40gb.log.ch0"

echo ">>> Building (make -j4)..."; ( cd "$BUILD" && make -j4 >/dev/null )

pass=0; total=0
for bs in "${BATCH_SIZES[@]}"; do
  total=$((total+1))
  tag="bs${bs}_ctx${CTX}"
  cfg="$BUILD/config_${tag}.yaml"
  sed -e "s/^\(\s*max_batch_size:\).*/\1 ${bs}/" \
      -e "s/^\(\s*input_len:\).*/\1 ${CTX}/" \
      -e "s/^\(\s*output_len:\).*/\1 2/" "$BASE_CONFIG" > "$cfg"

  ( cd "$BUILD" && ./run "$cfg" > "log/run_${tag}.log" 2>&1 )
  [[ -f "$TRACE_SRC" ]] || { echo "!!! [$tag] no trace; see build/log/run_${tag}.log" >&2; exit 1; }
  python3 "$CONVERT" "$TRACE_SRC" "$OUT_DIR/${tag}.csv"
  rm -f "$TRACE_SRC" "$cfg"

  if [[ -f "$REF/${tag}.csv" ]] && cmp -s "$OUT_DIR/${tag}.csv" "$REF/${tag}.csv"; then
    echo "  [$tag]  ✓ byte-identical to reference"
    pass=$((pass+1))
  else
    n=$(diff <(tr -d '\r' <"$OUT_DIR/${tag}.csv") <(tr -d '\r' <"$REF/${tag}.csv" 2>/dev/null) 2>/dev/null | grep -c '^[<>]' || true)
    echo "  [$tag]  ✗ differs from reference (${n} lines after CR-normalize)"
  fi
done
echo ">>> $pass/$total byte-identical.  Output: $OUT_DIR"
