#!/bin/bash
# Snake final evaluation (docs/snake/SNAKE_PLAN.md §8 S5, §10): the Tier 1 and
# Tier 2 run directories and their configs, on the 9 workloads (7 apps) of
# run_apps.sh, 1B instructions.
#
# usage: s5_runs.sh <tier1|tier2> dry          generate the run dirs (DRY=1)
#        s5_runs.sh <tier1|tier2> list         print the planned runs
#        s5_runs.sh <tier1|tier2> run [par]    run every job of the tier that
#            has no sim.out yet, in one pool of [par] (default 15) jobs, each
#            with its run dir's frozen build; longest workloads first.  A job
#            with an unfinished sim.out is listed and not re-run.
# Run from a clean tree on branch Snake (run_apps.sh checks it).

set -euo pipefail
TIER=${1:?usage: s5_runs.sh <tier1|tier2> <dry|list|run> [parallel]}
MODE=${2:?}; PAR=${3:-15}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
B=QV100-SASS-PAPER_V100-1B_INSN
S=$B-SNAKE-SNAKE_FP1   # Snake, exact footprint: the reference Snake config

# <run dir>|<comma-separated configs>.  Configs that coincide are listed once:
# Snake exact = Tail 10 = lru_popcount at Tail 10 = throttle 50 = every
# Tier 2 default.
tier1() {
  echo "sim_run_baseline|$B,$B-PAPER_LAT"
  echo "sim_run_snake|$S,$B-SNAKE,$B-SNAKE_ISO-SNAKE_FP1,$B-PAPER_LAT-SNAKE-SNAKE_FP1"
  echo "sim_run_snake_dt|$B-SNAKE_DT-SNAKE_FP1"
  echo "sim_run_snake_t|$B-SNAKE_T-SNAKE_FP1"
  echo "sim_run_ssnake|$B-S_SNAKE-SNAKE_FP1"
  echo "sim_run_snake_sweep_tail|$S-SNAKE_TAIL20,$S-SNAKE_TAIL40,$S-SNAKE_TAILINF"
  echo "sim_run_snake_sweep_evict|$S-SNAKE_EVPOP,$S-SNAKE_EVPOP-SNAKE_TAIL20,$S-SNAKE_EVPOP-SNAKE_TAIL40"
  echo "sim_run_snake_sweep_throttle|$S-SNAKE_THR25,$S-SNAKE_THR75,$S-SNAKE_THR100,$S-SNAKE_THR150,$S-SNAKE_THR200,$S-SNAKE_THR300"
}
tier2() {
  local c="" t
  for t in SNAKE_IWD1 SNAKE_IWD2 SNAKE_IWD8 SNAKE_MQR4 SNAKE_STALE200 SNAKE_STALE1000 \
           SNAKE_BWBYTES SNAKE_IWC3 SNAKE_DV0 SNAKE_Q64 SNAKE_ISSUE2; do c="$c,$S-$t"; done
  echo "sim_run_snake_sens|${c#,}"
  echo "sim_run_baseline_1132|QV100-SASS-PAPER_V100_1132-1B_INSN"
  echo "sim_run_snake_1132|QV100-SASS-PAPER_V100_1132-1B_INSN-SNAKE-SNAKE_FP1"
}
case $TIER in tier1|tier2) ;; *) echo "unknown tier $TIER"; exit 1;; esac

cd "$ACCELSIM"
case $MODE in
  list)
    n=0
    while IFS='|' read -r d cfgs; do
      for c in ${cfgs//,/ }; do echo "$d  $c"; n=$((n + 1)); done
    done < <($TIER)
    echo "$n configs x 9 workloads = $((n * 9)) runs" ;;
  dry)
    while IFS='|' read -r d cfgs; do
      echo "######## $d"
      DRY=1 util/snake/run_apps.sh "$d" current "$cfgs" | tail -2
    done < <($TIER) ;;
  run)
    jobs=$(mktemp)
    while IFS='|' read -r d _; do
      [ -x "$d/bin/accel-sim.out" ] || { echo "ABORT: $d has no dry run"; exit 1; }
      tags="$(strings "$d/bin/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1) $(strings "$d/bin/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
      [ "$(echo "$tags" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: $d build is not _modified_0.0 twice" >&2; exit 1; }
      find "$ACCELSIM/$d/runs" -name justrun.sh -printf '%h\n' | sort | while read -r j; do
        if [ ! -e "$j/sim.out" ]; then echo "$ACCELSIM/$d|$j"
        elif ! grep -q 'exit detected' "$j/sim.out"; then echo "UNFINISHED (not re-run): $j" >&2; fi
      done
    done < <($TIER) > "$jobs"
    # Longest workloads first (S5 estimate), so the pool drains evenly.
    order='parboil-histo srad_v1 nw-rodinia parboil-mri-q hotspot-rodinia-3.1/1024 lud-rodinia-3.1/_i backprop lud-rodinia-3.1/_s hotspot-rodinia-3.1/512'
    sorted=$(mktemp)
    for k in $order; do grep -F "/$k" "$jobs" || true; done > "$sorted"
    [ "$(wc -l < "$sorted")" = "$(wc -l < "$jobs")" ] || { echo "ABORT: job order incomplete"; exit 1; }
    echo "$(wc -l < "$sorted") jobs to run, $PAR at a time"
    run_job() {  # <run dir>|<job dir>
      local d=${1%%|*} j=${1#*|}
      cd "$j" && LD_LIBRARY_PATH="$d/bin/lib" "$d/bin/accel-sim.out" \
          -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
      echo "[done $(date '+%T')] rc=$? ${j#$d/runs/} ($(basename "$d"))"
    }
    export -f run_job
    xargs -P "$PAR" -I{} bash -c 'run_job "{}"' < "$sorted"
    rm -f "$jobs" "$sorted" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
