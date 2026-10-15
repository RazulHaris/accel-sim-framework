#!/bin/bash
# Run real workloads with one frozen build (docs/snake/SNAKE_PLAN.md §9).
#
# usage: run_apps.sh <out dir> <build> <config list> [parallel]
#   <build>   "current": freeze the current build into <out dir>/bin (both
#             trees must be clean, on Snake, and the build _modified_0.0);
#             or a frozen build directory (accel-sim.out + lib/libcudart.so),
#             e.g. /home/razul/snake/bin_ref_v2.
#   <configs> comma-separated run_simulations.py configs.
#   APPS      run_simulations.py -B selectors (rodinia-3.1 and parboil apps
#             go to their trace roots); default: the 7 suite apps (§10).
#   ONLY      extended regex over <app>/<args>/<config>: only matching jobs
#             are kept and run (the others are removed before anything runs),
#             e.g. to run one input of an app with several (S3c).
# Jobs go to <out dir>/runs/<app>/<args>/<config>/sim.out.  A job that exists
# already is never re-run (the script refuses an existing <out dir>/runs).
#   DRY=1     generate the jobs and freeze the build, launch nothing (S5).
#   RESUME=1  for a dry-run <out dir>: run, with its frozen <out dir>/bin,
#             only the jobs that have no sim.out yet; a job with an
#             unfinished sim.out is listed and left for diagnosis.  <build>
#             and <configs> are ignored.

set -euo pipefail
OUTARG=${1:?usage: run_apps.sh <out dir> <build> <config list> [parallel]}
BUILD=${2:?}; CFGS=${3:?}; PAR=${4:-12}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(mkdir -p "$OUTARG" && cd "$OUTARG" && pwd)
APPS=${APPS:-rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1,rodinia-3.1:lud-rodinia-3.1,rodinia-3.1:nw-rodinia-3.1,parboil:parboil-histo,parboil:parboil-mri-q}

cd "$ACCELSIM"
for r in . gpu-simulator/gpgpu-sim; do
  [ "$(git -C $r branch --show-current)" = Snake ] || { echo "ABORT: $r is not on Snake"; exit 1; }
  [ -z "$(git -C $r status --porcelain)" ] || { echo "ABORT: $r is dirty"; exit 1; }
done
set +u; source gpu-simulator/setup_environment.sh release >/dev/null 2>&1; set -u
if [ "${RESUME:-0}" = 1 ]; then
  [ -d "$OUT/runs" ] && [ -x "$OUT/bin/accel-sim.out" ] || { echo "ABORT: $OUT is not a dry-run dir"; exit 1; }
  BIN=$OUT/bin
  tags="$(strings "$BIN/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1) $(strings "$BIN/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
  [ "$(echo "$tags" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: build is not _modified_0.0 twice"; exit 1; }
  echo "$(date '+%F %T') resume with $BIN: $tags" | tee -a "$OUT/BUILDS.txt"
  todo=$(find "$OUT/runs" -name justrun.sh -printf '%h\n' | sort | while read -r j; do
    if [ ! -e "$j/sim.out" ]; then echo "$j"
    elif ! grep -q 'exit detected' "$j/sim.out"; then echo "UNFINISHED (not re-run): ${j#$OUT/runs/}" >&2; fi; done)
  echo "$(echo "$todo" | grep -c . || true) jobs to run"
  [ -n "$todo" ] || exit 0
else
[ ! -e "$OUT/runs" ] || { echo "ABORT: $OUT/runs exists"; exit 1; }
if [ "$BUILD" = current ]; then
  BIN=$OUT/bin; mkdir -p "$BIN/lib"
  cp -a gpu-simulator/bin/release/accel-sim.out "$BIN/"
  cp -a "$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG/." "$BIN/lib/"
else
  BIN=$(cd "$BUILD" && pwd)
fi
tags="$(strings "$BIN/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1) $(strings "$BIN/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
{ echo "$(date '+%F %T') build $BIN: $tags"; echo "configs: $CFGS"; echo "apps: $APPS"; } | tee "$OUT/BUILDS.txt"
[ "$(echo "$tags" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: build is not _modified_0.0 twice"; exit 1; }

ROD=$(echo "$APPS" | tr ',' '\n' | { grep '^rodinia-3.1' || true; } | paste -sd, -)
PB=$(echo "$APPS" | tr ',' '\n' | { grep '^parboil' || true; } | paste -sd, -)
[ -z "$ROD" ] || python3 util/job_launching/run_simulations.py -B "$ROD" -C "$CFGS" \
    -T hw_run/rodinia-3.1/11.0 -N "$(basename "$OUT")_rod" -r "$OUT/runs" -n >/dev/null
[ -z "$PB" ] || python3 util/job_launching/run_simulations.py -B "$PB" -C "$CFGS" \
    -T hw_run/parboil -N "$(basename "$OUT")_pb" -r "$OUT/runs" -n >/dev/null
if [ -n "${ONLY:-}" ]; then
  find "$OUT/runs" -name justrun.sh -printf '%h\n' | sed "s#^$OUT/runs/##" | { grep -Ev "$ONLY" || true; } |
    while read -r j; do rm -rf "${OUT:?}/runs/$j"; done
fi
find "$OUT/runs" -name justrun.sh -printf '%h\n' | sort | sed "s#^$OUT/runs/##" | tee "$OUT/JOBS.txt"
echo "$(wc -l < "$OUT/JOBS.txt") jobs"
if [ "${DRY:-0}" = 1 ]; then echo "DRY=1: nothing launched"; exit 0; fi
todo=$(find "$OUT/runs" -name justrun.sh -printf '%h\n')
fi

run_job() {  # <job dir>
  cd "$1" && LD_LIBRARY_PATH="$BIN/lib" "$BIN/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/runs/}"
}
export -f run_job; export OUT BIN
echo "$todo" | xargs -P "$PAR" -I{} bash -c 'run_job "{}"'
