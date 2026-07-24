#!/usr/bin/env bash
# run_fpga_ae.sh -- single-chip HBM2-power FPGA artifact driver (chip 0).
#
# Reprograms the bitstream for each phase and runs every measurement via run_step_tmux.sh
# (detached tmux -- survives SSH drops), then standardizes the raw CSVs into data/new/fpga/.
# The deliverable is the sanitized CSVs in data/new/fpga/ (figures are out of scope here). Run --help.
#
#   Phases:  baseline (no-HBM)  ->  chip (IDD + structural)  ->  trace (Fig 14 / Table 3)
#
# Env: FPGA_USER, FPGA_HOST, FPGA_SSH_KEY, FPGA_PROXY_*, PYTHON, OUTPUT_DIR.
set -uo pipefail

SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ART="$(cd -- "$SRC/../.." && pwd)"
. "$SRC/ssh_common.sh"
RUNNER="$SRC/run_step_tmux.sh"
REPROGRAM="$SRC/reprogram_fpga.sh"
FPGA_SOURCES="$ART/sources/fpga"
STANDARDIZE="$FPGA_SOURCES/standardize_single_chip.py"
AGGREGATE="$FPGA_SOURCES/aggregate_trace_ground_truth.py"
PYTHON="${PYTHON:-python3}"
FPGA_LABEL="${FPGA_HOST%%.*}"; FPGA_NUM="${FPGA_LABEL#safari-fpga}"
OUTPUT_DIR="${OUTPUT_DIR:-$ART/data/new/fpga}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$ART/temp/fpga_ae_checkpoint.tsv}"

# bitstreams (chip 0 only, for now)
NO_HBM_BITSTREAM=XCU55_no_hbm
CHIP_BITSTREAM=XCU55_latest_600MHz_chip0
TRACE_BITSTREAM=bram_tracer_chip0
# artifact checkout on the FPGA host (relative to remote $HOME); both apps live
# in its provided DRAM Bender tree and are built there with plain `make`
REMOTE_ART="${REMOTE_ART:-HBM-Power}"
export REMOTE_ART
PSV_APP_REL=$REMOTE_ART/sources/fpga/DRAMBender/sources/apps/Power_structural_variation
TRACE_APP_REL=$REMOTE_ART/sources/fpga/DRAMBender/sources/apps/HBMTraceRunnerBRAM

# --- test registry:  name | run_script | copy_subdir --------------------------
BASELINE_TESTS=(
  "no_hbm|run_no_hbm.sh|temperature_dependence"
)
CHIP_TESTS=(
  "idd4r|run_idd4r_8ch_test.sh|temperature_dependence"       # first: heats the die
  "idd2|run_idd2_test.sh|temperature_dependence"
  "idd0|run_idd0_test.sh|temperature_dependence"
  "idd3n1|run_idd3n1_test.sh|temperature_dependence"
  "idd3n16|run_idd3n16_test.sh|temperature_dependence"
  "idd4w|run_idd4w_test.sh|temperature_dependence"
  "idd5|run_idd5_test.sh|temperature_dependence"
  "bg_invert|run_max_power_bg_invert.sh|max_power_bg_invert"
  "bank_offset|run_max_power_bank_offset.sh|max_power_bank_offset"
  "bitflip|run_dq_bitflip_pattern_sweep.sh|bitflip_variation"
  "beat|run_beat_pattern_sweep_max_power.sh|beat_pattern_variation"
)

usage() {
  cat <<'EOF'
Usage: run_fpga_ae.sh [options]

  --tests "a b c"     run only these named tests (registry keys below, plus "trace");
                      a phase's bitstream is loaded only if one of its tests is used
  --quick             short durations for a plumbing shakeout (NOT paper-accurate)
  --resume            skip chunks already marked complete in the checkpoint file
  --checkpoint-file F  checkpoint file (default: temp/fpga_ae_checkpoint.tsv)
  --skip-build        do not `make` the two measurement binaries up-front
  --skip-reprogram    assume the right bitstream is already loaded
  --skip-trace        skip the trace phase
  --skip-standardize  do not regenerate data/*.csv
  -n, --dry-run       print the steps without touching the FPGA
  -h, --help          this help

  Connection defaults: aevaluator@safari-fpga7.ethz.ch via
                      aevaluator1@safari-proxy.ethz.ch using $HOME/aevaluator1
EOF
  exit "${1:-0}"
}

# --- options -----------------------------------------------------------------
TESTS_FILTER=""; QUICK=0; RESUME=0; DO_BUILD=1; DO_REPROGRAM=1; DO_TRACE=1; DO_STANDARDIZE=1; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --tests)            TESTS_FILTER="$2"; shift 2 ;;
  --quick)            QUICK=1; shift ;;
  --resume)           RESUME=1; shift ;;
  --checkpoint-file)  CHECKPOINT_FILE="$2"; shift 2 ;;
  --skip-build)       DO_BUILD=0; shift ;;
  --skip-reprogram)   DO_REPROGRAM=0; shift ;;
  --skip-trace)       DO_TRACE=0; shift ;;
  --skip-standardize) DO_STANDARDIZE=0; shift ;;
  -n|--dry-run)       DRY=1; shift ;;
  -h|--help)          usage 0 ;;
  *) echo "unknown option: $1 (see --help)"; exit 2 ;;
esac; done
[ "$QUICK" = 1 ] && export WARMUP_S=30 MEASURE_S=60 SWEEP_S=10 NOHBM_S=20 TRACE_S=15

log(){ printf '\033[1;34m[ae]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[ae]\033[0m %s\n' "$*" >&2; exit 1; }
run(){ if [ "$DRY" = 1 ]; then local cmd; printf -v cmd '%q ' "$@"; log "(dry-run) ${cmd% }"; else "$@"; fi; }
wanted(){ [ -z "$TESTS_FILTER" ] && return 0; case " $TESTS_FILTER " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

checkpoint_has(){
  [[ -f "$CHECKPOINT_FILE" ]] \
    && awk -F '\t' -v id="$1" '$1 == id { found=1 } END { exit !found }' "$CHECKPOINT_FILE"
}

chunk_done(){ (( RESUME )) && checkpoint_has "$1"; }

mark_done(){
  local id="$1"
  if [ "$DRY" = 1 ]; then
    log "(dry-run) checkpoint complete: $id"
    return 0
  fi
  mkdir -p -- "$(dirname -- "$CHECKPOINT_FILE")"
  checkpoint_has "$id" || printf '%s\t%s\n' "$id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CHECKPOINT_FILE"
  log "CHECKPOINT complete: $id"
}

init_checkpoint(){
  if [ "$DRY" = 1 ]; then
    if (( RESUME )); then log "(dry-run) resume checkpoint: ${CHECKPOINT_FILE#$ART/}"; else log "(dry-run) reset checkpoint: ${CHECKPOINT_FILE#$ART/}"; fi
    return 0
  fi
  mkdir -p -- "$(dirname -- "$CHECKPOINT_FILE")"
  if (( RESUME )); then
    touch "$CHECKPOINT_FILE"
    log "RESUME checkpoint: ${CHECKPOINT_FILE#$ART/}"
  else
    : > "$CHECKPOINT_FILE"
    log "CHECKPOINT file: ${CHECKPOINT_FILE#$ART/}"
  fi
}

reprogram(){ run env FPGA_USER="$FPGA_USER" FPGA_HOST="$FPGA_HOST" "$REPROGRAM" "$1" || die "reprogram $1 failed"; }

prepare_remote_artifact(){
  local target="$FPGA_USER@$FPGA_HOST" status ssh_wrapper rsync_rc
  if chunk_done prepare; then log "SKIP prepare (checkpoint)"; return 0; fi
  fpga_ssh_opts 10 4

  if [ "$DRY" = 1 ]; then
    log "(dry-run) probe $target:~/$REMOTE_ART and sync $ART/ if missing or empty"
    return 0
  fi

  status="$(ssh "${SSH_OPTS[@]}" "$target" "bash -l -s -- '$REMOTE_ART'" <<'REMOTE'
set -e
dir="$HOME/$1"
psv="$dir/sources/fpga/DRAMBender/sources/apps/Power_structural_variation"
trace="$dir/sources/fpga/DRAMBender/sources/apps/HBMTraceRunnerBRAM"
if [[ ! -d "$dir" ]]; then
  echo missing
elif ! find "$dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo empty
elif [[ -d "$psv" && -d "$trace" ]]; then
  echo ready
else
  echo incomplete
fi
REMOTE
  )" || die "could not inspect ~/$REMOTE_ART on $FPGA_HOST"

  case "$status" in
    ready)
      log "REMOTE checkout exists at ~/$REMOTE_ART"
      ;;
    missing|empty)
      log "SYNC local artifact -> $FPGA_HOST:~/$REMOTE_ART/ ($status)"
      ssh "${SSH_OPTS[@]}" "$target" "mkdir -p '$REMOTE_ART'" \
        || die "could not create ~/$REMOTE_ART on $FPGA_HOST"
      ssh_wrapper="$(fpga_ssh_wrapper)" || die "could not create temporary SSH wrapper for rsync"
      rsync_rc=0
      rsync -az --delete -e "$ssh_wrapper" \
        --exclude='.git/' \
        --exclude='.claude/' \
        --exclude='.pytest_cache/' \
        --exclude='.mypy_cache/' \
        --exclude='.ruff_cache/' \
        --exclude='aevaluator1' \
        --exclude='aevaluator1.pub' \
        --exclude='build/' \
        --exclude='cmake-build-*/' \
        --exclude='temp/' \
        --exclude='results/' \
        --exclude='new_data/' \
        --exclude='data/new/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        "$ART/" "$target:$REMOTE_ART/"
      rsync_rc=$?
      rm -f "$ssh_wrapper"
      (( rsync_rc == 0 )) || die "syncing local artifact to ~/$REMOTE_ART on $FPGA_HOST failed"
      ;;
    incomplete)
      die "~/$REMOTE_ART exists on $FPGA_HOST but is missing the vendored FPGA app dirs; move it aside, empty it, or set REMOTE_ART"
      ;;
    *)
      die "unexpected remote artifact status '$status' for ~/$REMOTE_ART on $FPGA_HOST"
      ;;
  esac
  mark_done prepare
}

# compile both measurement binaries from the vendored sources before anything runs
# (the per-test BUILD=make below is then an incremental no-op safety net)
build_phase(){
  (( DO_BUILD )) || return 0
  if chunk_done build; then log "SKIP build (checkpoint)"; return 0; fi
  log "BUILD SoftMC_rdwr + HBMTraceRunnerBRAM on $FPGA_HOST"
  fpga_ssh_opts 10 4
  run ssh "${SSH_OPTS[@]}" "$FPGA_USER@$FPGA_HOST" \
      "bash -l -c 'set -e; cd \"\$HOME/$PSV_APP_REL\" && make -s && cd \"\$HOME/$TRACE_APP_REL\" && make -s'" \
    || die "building the measurement binaries failed (is the artifact checkout at ~/$REMOTE_ART on $FPGA_HOST?)"
  mark_done build
}

# do_phase <phase> <bitstream> <registry entries...>: reprogram (if any remaining test is wanted), then run them
do_phase(){
  local phase="$1" bs="$2"; shift 2
  local pick=() e name script subdir chunk
  for e in "$@"; do
    IFS='|' read -r name _ _ <<< "$e"
    wanted "$name" || continue
    chunk="$phase:$name"
    if chunk_done "$chunk"; then log "SKIP $name ($phase checkpoint)"; else pick+=("$e"); fi
  done
  [ ${#pick[@]} -eq 0 ] && return 0
  (( DO_REPROGRAM )) && reprogram "$bs"
  for e in "${pick[@]}"; do
    IFS='|' read -r name script subdir <<< "$e"
    log "MEASURE $name  ($script -> $subdir)"
    run env FPGA_USER="$FPGA_USER" FPGA_HOST="$FPGA_HOST" BUILD="${BUILD:-make}" \
        "$RUNNER" "$script" "$subdir" \
      || die "measurement '$name' failed"
    mark_done "$phase:$name"
  done
}

trace_phase(){
  wanted trace || return 0
  if chunk_done trace:measure; then
    log "SKIP trace measurement (checkpoint)"
  else
    (( DO_REPROGRAM )) && reprogram "$TRACE_BITSTREAM"
    log "MEASURE trace  (run_trace_sweep.sh -> trace_runner)"
    run env FPGA_USER="$FPGA_USER" FPGA_HOST="$FPGA_HOST" \
          APP_REL="$TRACE_APP_REL" BINARY=./HBMTraceRunnerBRAM BUILD="${BUILD:-make}" COPY_ARGS="--trace" \
          "$RUNNER" run_trace_sweep.sh trace_runner --channels low \
      || die "trace sweep failed"
    mark_done trace:measure
  fi
  if (( DO_STANDARDIZE )); then
    if chunk_done trace:aggregate; then
      log "SKIP trace aggregation (checkpoint)"
    else
      run mkdir -p "$OUTPUT_DIR" || die "could not create output dir: $OUTPUT_DIR"
      run env RESULTS_DIR="$ART/results" DATA_DIR="$OUTPUT_DIR" CHIP_ID=0 \
          "$PYTHON" "$AGGREGATE" --no-backup || die "trace aggregation failed"
      mark_done trace:aggregate
    fi
  fi
}

standardize(){
  (( DO_STANDARDIZE )) || return 0
  if chunk_done standardize; then log "SKIP standardize (checkpoint)"; return 0; fi
  log "STANDARDIZE (fpga${FPGA_NUM}, chip0) -> ${OUTPUT_DIR#$ART/}/"
  run env RESULTS_DIR="$ART/results" OUTPUT_DIR="$OUTPUT_DIR" FPGA_NUM="$FPGA_NUM" CHIP=0 \
        "$PYTHON" "$STANDARDIZE" || die "standardization failed"
  mark_done standardize
}

# --- run ---------------------------------------------------------------------
msg="single-chip AE on $FPGA_USER@$FPGA_HOST (fpga${FPGA_NUM}, chip0)"
[ -n "$TESTS_FILTER" ] && msg+="  tests=[$TESTS_FILTER]"
(( QUICK )) && msg+="  [quick]"
log "$msg"
init_checkpoint
prepare_remote_artifact
build_phase
do_phase baseline "$NO_HBM_BITSTREAM" "${BASELINE_TESTS[@]}"
do_phase chip     "$CHIP_BITSTREAM"   "${CHIP_TESTS[@]}"
(( DO_TRACE )) && trace_phase
standardize
log "done. sanitized CSVs -> ${OUTPUT_DIR#$ART/}/"
