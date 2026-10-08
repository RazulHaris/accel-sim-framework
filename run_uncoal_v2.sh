#!/bin/bash
# Re-run the Rodinia-3.1 artifacts invalidated by changing the uncoalesced
# predicate from `generated > 1` to `generated > floor_transactions`
# (gpu-simulator/gpgpu-sim/src/abstract_hardware_model.cc).
#
# Phase 1: load-address dumps      -> sim_run_rodinia_uncoal_dump_v2  (A100-SASS-DUMP_UNCOAL)
# Phase 2: baseline + ideal oracle -> sim_run_oracle_v2               (A100-SASS, A100-SASS-ORACLE_PREF)
#
# Each job writes its own uncoalesced_loads.log in its own job directory; the
# simulator caps every log at 20 GiB (UNCOALESCED_LOAD_LOG_MAX_BYTES).

ACCELSIM=/home/razul/accel-sim-framework
DUMP_DIR=$ACCELSIM/sim_run_rodinia_uncoal_dump_v2
ORACLE_DIR=$ACCELSIM/sim_run_oracle_v2
export UNCOALESCED_LOAD_LOG_MAX_BYTES=$((20*1024*1024*1024))   # 20 GiB per log
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

echo "########## PHASE 1: load dumps  $(date) ##########"
echo "jobs: $(find "$DUMP_DIR" -name justrun.sh | wc -l), parallelism: 6"
find "$DUMP_DIR" -name justrun.sh | sort | xargs -P 6 -I{} bash -c 'run_one "$@"' _ {}
echo "phase 1 dump total: $(du -sh "$DUMP_DIR" | cut -f1)"

echo "########## PHASE 2: baseline + oracle  $(date) ##########"
echo "jobs: $(find "$ORACLE_DIR" -name justrun.sh | wc -l), parallelism: 12"
find "$ORACLE_DIR" -name justrun.sh | sort | xargs -P 12 -I{} bash -c 'run_one "$@"' _ {}

echo "########## ALL FINISHED $(date) ##########"
