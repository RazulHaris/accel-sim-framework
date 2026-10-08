#!/bin/bash
# Parboil counterpart of run_uncoal_v2.sh: re-runs the artifacts affected by the
# uncoalesced predicate change (`generated > 1` -> `generated > floor_transactions`
# in gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc).
#
# Phase 1: load-address dumps      -> sim_run_parboil_uncoal_dump_v2 (A100-SASS-DUMP_UNCOAL)
# Phase 2: baseline + ideal oracle -> sim_run_parboil_oracle_v2      (A100-SASS, A100-SASS-ORACLE_PREF)
#
# parboil-bfs is skipped: it aborts on an upstream Accel-Sim teardown assert
# (accel-sim.cc:144, `assert(k)` with an empty kernels_info) after ~23 min,
# producing no dump.  Unrelated to coalescing.
#
# parboil-lbm / tpacf / mri-gridding are commented out in define-all-apps.yml
# and have no traces, so they cannot be launched here at all.
#
# Every job writes its own uncoalesced_loads.log in its own job directory; the
# simulator caps each log at 20 GiB (UNCOALESCED_LOAD_LOG_MAX_BYTES).

ACCELSIM=/home/razul/accel-sim-framework
DUMP_DIR=$ACCELSIM/sim_run_parboil_uncoal_dump_v2
ORACLE_DIR=$ACCELSIM/sim_run_parboil_oracle_v2
EXCLUDE=parboil-bfs
export UNCOALESCED_LOAD_LOG_MAX_BYTES=$((20*1024*1024*1024))
MIN_FREE_GB=40

cd "$ACCELSIM"
source gpu-simulator/setup_environment.sh release >/dev/null 2>&1

run_one() {
    d=$(dirname "$1")
    name="${d#$ACCELSIM/}"
    free_gb=$(df -BG --output=avail "$ACCELSIM" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "[SKIP  $(date '+%H:%M:%S')] $name  (only ${free_gb}G free, need ${MIN_FREE_GB}G)"
        return
    fi
    echo "[START $(date '+%H:%M:%S')] $name"
    ( cd "$d" && bash justrun.sh > job.log 2>&1 )
    if grep -q "GPGPU-Sim: \*\*\* exit detected \*\*\*\|GPGPU-Sim Simulator Statistics" "$d/job.log" 2>/dev/null; then
        sz=$(stat -c %s "$d/uncoalesced_loads.log" 2>/dev/null || echo 0)
        echo "[DONE  $(date '+%H:%M:%S')] $name  dump=${sz}B"
    else
        echo "[FAIL  $(date '+%H:%M:%S')] $name  (see $d/job.log)"
    fi
}
export -f run_one; export ACCELSIM MIN_FREE_GB UNCOALESCED_LOAD_LOG_MAX_BYTES

echo "########## PHASE 1: parboil load dumps  $(date) ##########"
echo "jobs: $(find "$DUMP_DIR" -name justrun.sh | grep -vc $EXCLUDE), parallelism: 4  (excluding $EXCLUDE)"
find "$DUMP_DIR" -name justrun.sh | grep -v $EXCLUDE | sort | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "phase 1 dump total: $(du -sh "$DUMP_DIR" | cut -f1)"

echo "########## PHASE 2: parboil baseline + oracle  $(date) ##########"
echo "jobs: $(find "$ORACLE_DIR" -name justrun.sh | grep -vc $EXCLUDE), parallelism: 7  (excluding $EXCLUDE)"
find "$ORACLE_DIR" -name justrun.sh | grep -v $EXCLUDE | sort | xargs -P 7 -I{} bash -c 'run_one "$@"' _ {}

echo "########## ALL FINISHED $(date) ##########"
