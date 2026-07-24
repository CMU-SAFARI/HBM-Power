#!/usr/bin/env bash
#
# reprogram_fpga.sh -- switch the HBM bitstream on the AE FPGA, then bring the host back.
#
# Programs safari-fpga7 with a named bitstream, waits for the reboot, and re-inits the
# SoftMC host so experiments can run again. Run with --help for options and bitstream
# names.
#
# Two phases:
#   1. program the FPGA (with error detection), then reboot.
#   2. re-init the host -- SoftMC_Host(["hbm00"]) reloads the xdma driver and resets the
#      board. MUST run in a *login* shell (needs PYTHONPATH + SMC_*/discord env); running
#      it as a plain `ssh host 'cmd'` is what made Phase 2 flaky.
#
# Fails gracefully (non-zero) on an unknown bitstream or if the board does not return
# within $REBOOT_WAIT s. No board changes happen if programming fails.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/ssh_common.sh"

# Config (all env-overridable)
BOARD="${BOARD:-XCU55}"
MODULE_LABEL="${MODULE_LABEL:-hbm00}"                 # SoftMC label for the HBM module
PROGRAM_SCRIPT="${PROGRAM_SCRIPT:-/etc/smc_infra/DRAM-Bender/prebuilt/programFPGA.sh}"
SMC_INFRA="${SMC_INFRA:-/etc/smc_infra}"

REBOOT_WAIT="${REBOOT_WAIT:-300}"     # max seconds to wait for the FPGA to come back
PROGRAM_TIMEOUT="${PROGRAM_TIMEOUT:-420}"   # max seconds for the programming step
PHASE2_TIMEOUT="${PHASE2_TIMEOUT:-300}"     # max seconds for host re-init
POLL="${POLL:-10}"                    # seconds between reboot-probe attempts

TARGET="$FPGA_USER@$FPGA_HOST"
fpga_ssh_opts 8 8
SSH=(ssh "${SSH_OPTS[@]}")

# Helpers
log()  { printf '\033[1;34m[reprogram]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[reprogram]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reprogram]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[reprogram]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

usage() {
  cat <<'EOF'
Usage: reprogram_fpga.sh <bitstream-name> [--skip-phase2]

  <bitstream-name>  a .bit in the artifact's prebuilt bitstream dir on the FPGA
                    host (sources/fpga/DRAMBender/prebuilt/XCU55; .bit optional):
                      XCU55_no_hbm                idle / no-HBM offset
                      XCU55_latest_600MHz_chip0   characterization, chip 0
                      XCU55_latest_600MHz_chip1   characterization, chip 1
                      bram_tracer_chip0           trace replay, chip 0
                      bram_tracer_chip1           trace replay, chip 1
  --skip-phase2     program + reboot only; do NOT re-init the host
  -h, --help        this help

  Connection defaults: aevaluator@safari-fpga7.ethz.ch via
                      aevaluator1@safari-proxy.ethz.ch using $HOME/aevaluator1
  Env overrides: FPGA_USER FPGA_HOST FPGA_SSH_KEY FPGA_PROXY_USER
                 FPGA_PROXY_HOST FPGA_PROXY_JUMP FPGA_PROXY_COMMAND
EOF
  exit "${1:-2}"
}

TMPFILES=()
cleanup() { [[ ${#TMPFILES[@]} -gt 0 ]] && rm -f "${TMPFILES[@]}"; }
trap cleanup EXIT
mktmp() { local f; f=$(mktemp); TMPFILES+=("$f"); printf '%s' "$f"; }

# Parse args
SKIP_PHASE2=0
BITSTREAM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)     usage 0 ;;
    --skip-phase2) SKIP_PHASE2=1; shift ;;
    --) shift; break ;;
    -*) die "Unknown option: $1 (see --help)" ;;
    *)  [[ -z "$BITSTREAM" ]] || die "Only one bitstream name expected (got '$BITSTREAM' and '$1')."
        BITSTREAM="$1"; shift ;;
  esac
done
[[ -n "$BITSTREAM" ]] || { err "No bitstream name given."; usage 2; }
name="${BITSTREAM%.bit}"          # accept name with or without the .bit suffix

# 0. Connectivity + resolve remote paths
log "Checking SSH connectivity to $TARGET ..."
REMOTE_HOME=""
for attempt in 1 2 3; do
  REMOTE_HOME=$("${SSH[@]}" -n "$TARGET" 'printf %s "$HOME"' 2>/dev/null) && [[ -n "$REMOTE_HOME" ]] && break
  (( attempt < 3 )) && { warn "SSH probe $attempt/3 failed; retrying in 5s ..."; sleep 5; }
done
[[ -n "$REMOTE_HOME" ]] || die "Cannot SSH to $TARGET after 3 tries (BatchMode). Check your SSH key / hostname / VPN."

# Bitstreams ship with the artifact: prebuilt/XCU55 in the provided DRAM Bender
# tree of the artifact checkout on the FPGA host ($REMOTE_ART, rel. to $HOME).
REMOTE_ART="${REMOTE_ART:-HBM-Power}"
BITSTREAM_DIR="${BITSTREAM_DIR:-$REMOTE_HOME/$REMOTE_ART/sources/fpga/DRAMBender/prebuilt/XCU55}"
BIT_PATH="$BITSTREAM_DIR/$name.bit"

# 1. Validate the bitstream exists on the FPGA (the copy that actually gets programmed)
log "Validating bitstream '$name' on $FPGA_HOST ..."
if ! "${SSH[@]}" -n "$TARGET" "test -f '$BIT_PATH'" 2>/dev/null; then
  err "Bitstream '$name' not found at:"
  err "  $BIT_PATH"
  echo "Available bitstreams in $BITSTREAM_DIR:" >&2
  "${SSH[@]}" -n "$TARGET" \
    "ls -1 '$BITSTREAM_DIR'/*.bit 2>/dev/null | xargs -r -n1 basename | sed 's/\\.bit\$//'" >&2 \
    || err "  (could not list the directory)"
  exit 1
fi
ok "Found $BIT_PATH"

# 2. Phase 1 -- program the FPGA (synchronous, with error detection)
log "Phase 1: programming $BOARD with '$name' (this can take a few minutes) ..."
prog_log=$(mktmp)
# Positional args to the remote login shell: $1=program-script $2=name $3=bitstream-dir.
# programFPGA.sh builds "<arg1>/<name>.bit"; we cd into the flat ~/ae_bitstreams dir
# and pass "." so it resolves to $BITSTREAM_DIR/<name>.bit.
timeout "$PROGRAM_TIMEOUT" "${SSH[@]}" "$TARGET" \
  "bash -l -s -- '$PROGRAM_SCRIPT' '$name' '$BITSTREAM_DIR'" 2>&1 <<'REMOTE' | tee "$prog_log"
set -e
cd "$3"
sudo "$1" . "$2"
REMOTE
prog_rc=${PIPESTATUS[0]}

case "$prog_rc" in
  124) die "Programming timed out after ${PROGRAM_TIMEOUT}s (FPGA left as-is, no reboot).";;
  0)   : ;;
  *)   die "Programming command failed (rc=$prog_rc). FPGA left as-is, no reboot.";;
esac
grep -q "Done programming the board!" "$prog_log" \
  || die "Programming did not complete (no 'Done programming the board!' marker). No reboot."
grep -q "Please assign vivado" "$prog_log" \
  && die "VIVADO_EXEC is not set in the FPGA login shell. No reboot."
grep -qE 'ERROR:' "$prog_log" \
  && die "Vivado reported errors during programming (see output above). No reboot."
ok "Programming reported success."

# 3. Reboot + wait for the FPGA to come back (confirmed via a changed boot id)
OLD_BOOT_ID=$("${SSH[@]}" -n "$TARGET" 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null || true)
log "Rebooting $FPGA_HOST ..."
timeout 30 "${SSH[@]}" "$TARGET" 'sudo /sbin/reboot' </dev/null >/dev/null 2>&1 || true

log "Waiting for $FPGA_HOST to reboot (up to ${REBOOT_WAIT}s) ..."
elapsed=0
new_boot_id=""
while (( elapsed < REBOOT_WAIT )); do
  sleep "$POLL"
  elapsed=$(( elapsed + POLL ))
  new_boot_id=$("${SSH[@]}" -o ConnectTimeout=5 -n "$TARGET" \
                'cat /proc/sys/kernel/random/boot_id' 2>/dev/null || true)
  if [[ -n "$new_boot_id" && "$new_boot_id" != "$OLD_BOOT_ID" ]]; then
    ok "FPGA back up after ~${elapsed}s."
    break
  fi
  printf '  ... still waiting (%ds/%ds)\n' "$elapsed" "$REBOOT_WAIT"
done
if [[ -z "$new_boot_id" || "$new_boot_id" == "$OLD_BOOT_ID" ]]; then
  die "FPGA $FPGA_HOST did not come back within ${REBOOT_WAIT}s."
fi

# Readiness: confirm the Xilinx PCIe device re-enumerated after the reboot.
if "${SSH[@]}" -n "$TARGET" 'lspci -d 10ee: 2>/dev/null | grep -q .' 2>/dev/null; then
  ok "Xilinx PCIe device present after reboot."
else
  warn "Xilinx PCIe device (10ee:) not visible yet -- Phase 2 will surface any problem."
fi

# 4. Phase 2 -- re-init the host (reload xdma driver, reset board, mark ready)
if (( SKIP_PHASE2 )); then
  warn "Skipping Phase 2 (--skip-phase2). The board is programmed but NOT re-initialized;"
  warn "the xdma driver is not loaded, so SoftMC_rdwr will not work until you init it."
else
  log "Phase 2 pre-flight: login-shell env + smc_scripts import ..."
  pre_log=$(mktmp)
  # $1 = smc_infra dir
  "${SSH[@]}" "$TARGET" "bash -l -s -- '$SMC_INFRA'" 2>&1 <<'REMOTE' | tee "$pre_log" >/dev/null
set -e
: "${PYTHONPATH:?PYTHONPATH not set in login shell}"
: "${SMC_DISCORD_CHANNEL_WEBHOOK:?SMC_DISCORD_CHANNEL_WEBHOOK not set in login shell}"
cd "$1"
python3 -c 'import smc_scripts'
REMOTE
  if (( ${PIPESTATUS[0]} != 0 )); then
    cat "$pre_log" >&2
    die "Phase 2 pre-flight failed -- login-shell env is not set up for smc_scripts (see above)."
  fi
  ok "Pre-flight OK (env + import)."

  log "Phase 2: re-initializing host -- SoftMC_Host([\"$MODULE_LABEL\"]) ..."
  p2_log=$(mktmp)
  # $1 = module label, $2 = smc_infra dir
  timeout "$PHASE2_TIMEOUT" "${SSH[@]}" "$TARGET" \
    "bash -l -s -- '$MODULE_LABEL' '$SMC_INFRA'" 2>&1 <<'REMOTE' | tee "$p2_log"
set -e
cd "$2"
python3 - "$1" <<'PY'
import sys
from smc_scripts import SoftMC_Host
infra = SoftMC_Host([sys.argv[1]])
print("PHASE2_OK")
PY
REMOTE
  p2_rc=${PIPESTATUS[0]}

  if (( p2_rc == 124 )); then
    die "Phase 2 timed out after ${PHASE2_TIMEOUT}s."
  fi
  if (( p2_rc != 0 )) || ! grep -q "PHASE2_OK" "$p2_log"; then
    die "Phase 2 (host re-init) failed (rc=$p2_rc). See output above."
  fi
  if grep -qE 'CRITICAL|Traceback|Error loading XDMA|Error resetting' "$p2_log"; then
    die "Phase 2 reported errors (see output above)."
  fi
  ok "Phase 2 succeeded -- host re-initialized and board reset."
fi

# 5. Summary
case "$name" in
  *no_hbm*) ident="no-HBM / idle-offset bitstream" ;;
  *chip0*)  ident="chip 0" ;;
  *chip1*)  ident="chip 1" ;;
  *)        ident="$name" ;;
esac
echo
ok "Done. $FPGA_HOST is now running '$name' ($ident)."
(( SKIP_PHASE2 )) && warn "Remember: Phase 2 was skipped; run the SoftMC_Host init before experiments."
exit 0
