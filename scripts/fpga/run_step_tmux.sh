#!/usr/bin/env bash
# run_step_tmux.sh <run_script.sh> <copy_subdir> [flags for the run script]
#
# Run ONE FPGA measurement in a DETACHED tmux session ('power_experiment', the upstream
# run_on_infra convention) so an SSH drop can't SIGHUP it mid-run; poll for a done-marker
# (reconnect-tolerant), then copy results back with retry. Called once per test by
# run_fpga_ae.sh. Example:  run_step_tmux.sh run_idd2_test.sh temperature_dependence
#
# Env (all optional; defaults target the Power_structural_variation app):
#   FPGA_USER, FPGA_HOST, FPGA_SSH_KEY, FPGA_PROXY_*   FPGA connection settings
#   FPGA_USER, FPGA_HOST   the FPGA to run on
#   REMOTE_ART  artifact repo checkout on the FPGA host, relative to $HOME
#               (default: HBM-Power)
#   APP_REL   remote app dir (relative to $HOME) to stage into and run in
#   BINARY    measurement binary   (default: ./SoftMC_rdwr)
#   BUILD     shell run before the measurement, e.g. the trace git+make (default: none)
#   COPY_ARGS args passed to copy_results.sh (default: "<copy_subdir>"; traces use "--trace")
#   WARMUP_S MEASURE_S SWEEP_S NOHBM_S TRACE_S   forwarded to the run script when set
#   MAX_SECS  backstop wall-clock cap (default 6000s)   POLL  poll interval (default 30s)
set -uo pipefail

SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"     # scripts/fpga/
ART="$(cd -- "$SRC/../.." && pwd)"                            # HBM-Power/
. "$SRC/ssh_common.sh"
FPGA_DIR="$ART/sources/fpga"; COPY="$SRC/copy_results.sh"
TARGET="$FPGA_USER@$FPGA_HOST"; FPGA_LABEL="${FPGA_HOST%%.*}"
fpga_ssh_opts 10 4
SSH_WRAPPER="$(fpga_ssh_wrapper)" || { echo "FATAL: could not create temporary SSH wrapper for rsync"; exit 1; }
cleanup(){ rm -f "$SSH_WRAPPER" "${tmp:-}"; }
trap cleanup EXIT
SESSION=power_experiment
BINARY="${BINARY:-./SoftMC_rdwr}"; MAX_SECS="${MAX_SECS:-6000}"; POLL="${POLL:-30}"

# retry <n> <sleep> <cmd...> -- run cmd up to n times; a transient SSH/rsync blip
# (common on the first contact after idle) then succeeds on a later attempt
retry() {
  local n="$1" s="$2"; shift 2; local i
  for ((i=1; i<=n; i++)); do
    "$@" && return 0
    [ "$i" -lt "$n" ] && { echo "  retry $i/$n failed; ${s}s ..."; sleep "$s"; }
  done
  return 1
}

[ $# -ge 2 ] || { echo "usage: run_step_tmux.sh <run_script.sh> <copy_subdir> [flags]"; exit 2; }
SCRIPT="$1"; SUBDIR="$2"; shift 2; FLAGS="$*"
COPY_ARGS="${COPY_ARGS:-$SUBDIR}"

echo "=== [tmux-step] $SCRIPT ${FLAGS:+($FLAGS) }-> $SUBDIR @ $(date -u +%H:%M:%S)UTC ==="

REMOTE_HOME=""
for attempt in 1 2 3; do
  REMOTE_HOME="$(ssh "${SSH_OPTS[@]}" "$TARGET" 'printf %s "$HOME"' 2>/dev/null)" && [ -n "$REMOTE_HOME" ] && break
  [ "$attempt" -lt 3 ] && { echo "  ssh probe $attempt/3 failed; retrying in 5s"; sleep 5; }
done
[ -n "$REMOTE_HOME" ] || { echo "FATAL: ssh to $TARGET failed after 3 tries"; exit 1; }
REMOTE_ART="${REMOTE_ART:-HBM-Power}"
APP_DIR="$REMOTE_HOME/${APP_REL:-$REMOTE_ART/sources/fpga/DRAMBender/sources/apps/Power_structural_variation}"
DONEF="$APP_DIR/.ae_done"; LOGF="$APP_DIR/.ae_log"; LAUNCHER="$APP_DIR/.ae_launch.sh"

# stage the run script into the app dir
retry 3 5 ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$APP_DIR'" || { echo "FATAL: mkdir $APP_DIR"; exit 1; }
retry 3 5 rsync -az -e "$SSH_WRAPPER" "$FPGA_DIR/$SCRIPT" "$TARGET:$APP_DIR/" || { echo "FATAL: could not stage $SCRIPT"; exit 1; }

# forward only the duration knobs that are set
envs=""; for v in WARMUP_S MEASURE_S SWEEP_S NOHBM_S TRACE_S; do [ -n "${!v:-}" ] && envs+="$v=${!v} "; done

# optional BUILD step -- computed here so the heredoc interpolates a plain string. Inlining
# ${BUILD:+... { ...; } } in the heredoc is broken: the brace group's first } closes the ${}
# early and leaks a literal } into the launcher (a syntax error that silently no-ops the run).
build_line=""
[ -n "${BUILD:-}" ] && build_line="$BUILD || { echo DONE_build_failed > '$DONEF'; exit 4; }"

# stage a launcher (avoids quote-nesting through tmux); optional BUILD, then measure, then mark done
tmp="$(mktemp)"
cat > "$tmp" <<EOF
#!/usr/bin/env bash
cd '$APP_DIR' || exit 3
rm -f '$DONEF'
$build_line
${envs}bash -l '$APP_DIR/$SCRIPT' $FLAGS '$BINARY' > '$LOGF' 2>&1
echo "DONE_\$?" > '$DONEF'
EOF
scp -q "${SSH_OPTS[@]}" "$tmp" "$TARGET:$LAUNCHER" || { rm -f "$tmp"; echo "FATAL: could not stage the launcher"; exit 1; }
rm -f "$tmp"

ssh "${SSH_OPTS[@]}" "$TARGET" "tmux kill-session -t $SESSION 2>/dev/null; chmod +x '$LAUNCHER'; tmux new-session -d -s $SESSION \"bash -l '$LAUNCHER'\"" \
  || { echo "FATAL: could not start tmux session"; exit 1; }
echo "=== [tmux-step] running detached in tmux '$SESSION'; polling every ${POLL}s (backstop ${MAX_SECS}s) ==="

# poll; a failed poll (network blip) is harmless -- the run is safe in tmux, so retry next tick
# remote probe: print the done-marker if present, else RUNNING if tmux is alive, else GONE
probe="if [ -f '$DONEF' ]; then cat '$DONEF'; elif tmux has-session -t $SESSION 2>/dev/null; then echo RUNNING; else echo GONE; fi"
elapsed=0; done=""
while [ "$elapsed" -lt "$MAX_SECS" ]; do
  sleep "$POLL"; elapsed=$((elapsed+POLL))
  st="$(ssh "${SSH_OPTS[@]}" "$TARGET" "$probe" 2>/dev/null)" \
    || { echo "  [${elapsed}s] poll ssh blip -- run safe in tmux, retrying"; continue; }
  case "$st" in
    DONE_*) done="$st"; echo "=== [tmux-step] finished ($st) after ~${elapsed}s ==="; break ;;
    GONE)   done="GONE"; break ;;
    *)      [ $((elapsed % 300)) -eq 0 ] && echo "  [${elapsed}s] still running" ;;
  esac
done

# A healthy run always leaves a DONE_<rc> marker (the launcher writes it even if the binary
# core-dumps on exit). GONE / DONE_build_failed mean the measurement never produced data, so
# fail loudly rather than copy stale results and report success.
case "$done" in
  GONE)              echo "FATAL: tmux ended with no DONE marker -- the measurement did not run (launcher/build failed). Remote log: $TARGET:$LOGF"; exit 1 ;;
  DONE_build_failed) echo "FATAL: the BUILD step failed -- see $TARGET:$LOGF"; exit 1 ;;
esac
[ -z "$done" ] && echo "=== [tmux-step] WARNING: hit ${MAX_SECS}s backstop; copying whatever exists ==="

# copy back with retry
copied=0
for a in 1 2 3 4 5; do
  FPGA_USER="$FPGA_USER" FPGA_HOST="$FPGA_HOST" "$COPY" $COPY_ARGS && { copied=1; break; }
  echo "  copy attempt $a failed; retry in 8s"; sleep 8
done
[ "$copied" = 1 ] || { echo "FATAL: copy_results.sh failed 5x ($COPY_ARGS)"; exit 1; }
echo "=== [tmux-step] DONE $SCRIPT @ $(date -u +%H:%M:%S)UTC ==="
