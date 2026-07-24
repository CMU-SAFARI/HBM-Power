#!/usr/bin/env bash
#
# copy_results.sh -- pull raw measurement data for one or more benchmarks off the AE FPGA.
#
# Rsyncs the named results subdir(s) from the FPGA's measurement repo into this artifact's
# results/ tree, ready for the standardizer (never drags the whole results/ folder across).
# Run with --help for options and benchmark names; see sources/fpga/README.md for the
# benchmark -> figure map.
#
# Exit codes: 0 all requested benchmarks copied; 1 one or more failed/missing;
#             2 usage error (no benchmark given / bad flag).

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"       # HBM-Power/
. "$SCRIPT_DIR/ssh_common.sh"

# Config (all env-overridable)
FPGA_LABEL="${FPGA_LABEL:-${FPGA_HOST%%.*}}"           # e.g. safari-fpga7 (matches standardizer)
AUTO_SUFFIX="${AUTO_SUFFIX:-_fixed_reset_full_ipp}"    # appended to a name that lacks it
# Artifact checkout on the FPGA host, relative to the remote $HOME.
REMOTE_ART="${REMOTE_ART:-HBM-Power}"
# --trace mode: where the HBMTraceRunnerBRAM app writes its flat *_bram.csv, and
# the local subdir they land in (aggregate_trace_ground_truth.py globs for these).
TRACE_APP_REL="$REMOTE_ART/sources/fpga/DRAMBender/sources/apps/HBMTraceRunnerBRAM"
TRACE_DEST_SUBDIR="${TRACE_DEST_SUBDIR:-trace_runner}"

DEST_ROOT="${DEST_ROOT:-$REPO_ROOT/results}"

TARGET="$FPGA_USER@$FPGA_HOST"
fpga_ssh_opts 8 8
SSH=(ssh "${SSH_OPTS[@]}")
# shellcheck disable=SC2206
RSYNC_OPTS=(${RSYNC_OPTS:--az -h --stats})

# Helpers
log()  { printf '\033[1;34m[copy]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[copy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[copy]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[copy]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
SSH_WRAPPER="$(fpga_ssh_wrapper)" || die "could not create temporary SSH wrapper for rsync"
cleanup(){ rm -f "$SSH_WRAPPER"; }
trap cleanup EXIT
usage() {
  cat <<'EOF'
Usage: copy_results.sh [options] <benchmark> [<benchmark> ...]
       copy_results.sh --trace [options]          (trace-runner CSVs)

  <benchmark>   a directory under the FPGA's results/ dir. The
                "_fixed_reset_full_ipp" suffix is auto-appended if omitted, so
                these copy the same thing:
                  copy_results.sh temperature_dependence
                  copy_results.sh temperature_dependence_fixed_reset_full_ipp

  In-scope benchmark names (single-chip FPGA AE):
    temperature_dependence   max_power_bg_invert   max_power_bank_offset
    bitflip_variation        beat_pattern_variation   [test_28]

  Options:
    --trace         pull the BRAM trace-runner CSVs (*_bram.csv) for Fig 14 /
                    Table 3 instead of a benchmark subdir (takes no <benchmark>).
    -n, --dry-run   show what rsync would transfer, copy nothing
    --flat          use results/<benchmark>/ instead of the nested
                    results/<fpga-label>/<benchmark>/ (standardizer expects nested)
    --no-suffix     do not auto-append "_fixed_reset_full_ipp"
    -h, --help      this help

  Destination (default): results/<fpga-label>/<benchmark>_fixed_reset_full_ipp/
                         results/<fpga-label>/trace_runner/   (--trace)
  Config (env): FPGA_USER FPGA_HOST FPGA_SSH_KEY FPGA_PROXY_USER FPGA_PROXY_HOST
                FPGA_PROXY_JUMP FPGA_PROXY_COMMAND REMOTE_RESULTS_DIR DEST_ROOT
                FPGA_LABEL AUTO_SUFFIX RSYNC_OPTS REMOTE_TRACE_APP TRACE_DEST_SUBDIR
EOF
  exit "${1:-2}"
}

# Parse args
DRY_RUN=0
FLAT=0
TRACE=0
BENCHMARKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)     usage 0 ;;
    -n|--dry-run)  DRY_RUN=1; shift ;;
    --flat)        FLAT=1; shift ;;
    --trace)       TRACE=1; shift ;;
    --no-suffix)   AUTO_SUFFIX=""; shift ;;
    --)            shift; while [[ $# -gt 0 ]]; do BENCHMARKS+=("$1"); shift; done ;;
    -*)            err "Unknown option: $1 (see --help)"; exit 2 ;;
    *)             BENCHMARKS+=("$1"); shift ;;
  esac
done
if (( TRACE )); then
  [[ ${#BENCHMARKS[@]} -eq 0 ]] || { err "--trace takes no <benchmark> args (got: ${BENCHMARKS[*]})."; usage 2; }
else
  [[ ${#BENCHMARKS[@]} -gt 0 ]] || { err "No benchmark directory name given."; usage 2; }
fi

# reject path-y names so a benchmark can only be a single dir under results/
for b in "${BENCHMARKS[@]}"; do
  case "$b" in
    */*|..|.) err "Invalid benchmark name '$b' (must be a plain directory name)."; exit 2 ;;
  esac
done

(( DRY_RUN )) && RSYNC_OPTS+=(-n -v)

# 0. Connectivity + resolve remote results dir (retry: the first SSH after an idle
#    spell often fails cold, and a transient blip should not abort the whole copy)
log "Checking SSH connectivity to $TARGET ..."
REMOTE_HOME=""
for attempt in 1 2 3; do
  REMOTE_HOME=$("${SSH[@]}" -n "$TARGET" 'printf %s "$HOME"' 2>/dev/null) && [[ -n "$REMOTE_HOME" ]] && break
  (( attempt < 3 )) && { warn "SSH probe $attempt/3 failed; retrying in 5s ..."; sleep 5; }
done
[[ -n "$REMOTE_HOME" ]] || die "Cannot SSH to $TARGET after 3 tries (BatchMode). Check your SSH key / hostname / VPN."

# 0b. --trace mode: pull the flat *_bram.csv from the trace-runner app dir, then exit
if (( TRACE )); then
  REMOTE_TRACE_APP="${REMOTE_TRACE_APP:-$REMOTE_HOME/$TRACE_APP_REL}"
  src="$REMOTE_TRACE_APP/results"
  if (( FLAT )); then dest="$DEST_ROOT/$TRACE_DEST_SUBDIR"; else dest="$DEST_ROOT/$FPGA_LABEL/$TRACE_DEST_SUBDIR"; fi
  log "Remote trace results: $TARGET:$src"
  log "Local destination   : $dest/"
  (( DRY_RUN )) && warn "DRY RUN -- nothing will be written."

  if ! "${SSH[@]}" -n "$TARGET" "test -d '$src'" 2>/dev/null; then
    die "Remote trace results dir not found: $src
       Set REMOTE_TRACE_APP, or run run_trace_sweep.sh on the FPGA first."
  fi
  mkdir -p -- "$dest" || die "Cannot create local dir: $dest"
  # only the power CSVs; the app dir's results/ holds nothing else, but be explicit
  if rsync "${RSYNC_OPTS[@]}" -e "$SSH_WRAPPER" \
        --include='*/' --include='*_bram.csv' --exclude='*' \
        "$TARGET:$src/" "$dest/"; then
    if (( DRY_RUN )); then ok "Would copy *_bram.csv -> $dest/"; else
      n=$(find "$dest" -maxdepth 1 -name '*_bram.csv' | wc -l | tr -d ' ')
      ok "Copied $n *_bram.csv -> $dest/"
      (( n == 0 )) && warn "No *_bram.csv landed -- did the sweep produce output?"
    fi
    exit 0
  fi
  die "rsync failed for the trace results (rc=$?)."
fi

REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-$REMOTE_HOME/$REMOTE_ART/sources/fpga/DRAMBender/sources/apps/Power_structural_variation/results}"
log "Remote results dir: $TARGET:$REMOTE_RESULTS_DIR"
log "Local destination : $DEST_ROOT$( ((FLAT)) || printf '/%s' "$FPGA_LABEL" )/"
(( DRY_RUN )) && warn "DRY RUN -- nothing will be written."

# 1. Copy each requested benchmark
copied=0
failed=0
for b in "${BENCHMARKS[@]}"; do
  # resolve the on-FPGA subdir name (auto-append the suffix unless already present)
  sub="$b"
  if [[ -n "$AUTO_SUFFIX" && "$sub" != *"$AUTO_SUFFIX" ]]; then
    sub="$sub$AUTO_SUFFIX"
    log "'$b' -> '$sub'"
  fi

  src="$REMOTE_RESULTS_DIR/$sub"
  if (( FLAT )); then dest="$DEST_ROOT/$sub"; else dest="$DEST_ROOT/$FPGA_LABEL/$sub"; fi

  echo
  log "=== $sub ==="
  # confirm the source exists remotely, for a clean error instead of a raw rsync failure
  if ! "${SSH[@]}" -n "$TARGET" "test -d '$src'" 2>/dev/null; then
    err "Remote directory not found: $src"
    warn "Available directories under $REMOTE_RESULTS_DIR:"
    "${SSH[@]}" -n "$TARGET" "ls -1 '$REMOTE_RESULTS_DIR' 2>/dev/null" >&2 \
      || err "  (could not list the remote results dir)"
    failed=$(( failed + 1 ))
    continue
  fi

  mkdir -p -- "$dest" || { err "Cannot create local dir: $dest"; failed=$(( failed + 1 )); continue; }

  # trailing slash on src copies its *contents* into dest/
  if rsync "${RSYNC_OPTS[@]}" -e "$SSH_WRAPPER" "$TARGET:$src/" "$dest/"; then
    (( DRY_RUN )) && ok "Would copy -> $dest/" || ok "Copied -> $dest/"
    copied=$(( copied + 1 ))
  else
    err "rsync failed for '$sub' (rc=$?)."
    failed=$(( failed + 1 ))
  fi
done

# 2. Summary
echo
if (( failed == 0 )); then
  ok "Done. $copied/${#BENCHMARKS[@]} benchmark(s) copied into $DEST_ROOT/."
  exit 0
fi
warn "Finished with problems: $copied copied, $failed failed (of ${#BENCHMARKS[@]})."
exit 1
