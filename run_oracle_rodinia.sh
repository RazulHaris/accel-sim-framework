#!/bin/bash
# Run the Rodinia-3.1 suite twice: baseline (oracle off) and with the oracle
# uncoalesced prefetcher enabled, using the same binary so the IPC comparison
# is apples to apples.
#
#   screen -dmS oracle_rodinia bash /home/razul/accel-sim-framework/run_oracle_rodinia.sh
#   screen -r oracle_rodinia
#
# The run directories and per-job justrun.sh scripts are created beforehand by:
#   python3 util/job_launching/run_simulations.py -B rodinia-3.1 \
#       -C A100-SASS,A100-SASS-ORACLE_PREF -T hw_run/rodinia-3.1/11.0 \
#       -N oracle_pref -r sim_run_oracle -n
#
# This script runs those scripts directly with xargs instead of going through
# procman, which is spawned as a child of the launching shell and dies with it.
#
# The environment is sourced here so every job inherits LD_LIBRARY_PATH
# pointing at the freshly built gcc-11.4.0/cuda-11080 libcudart.so; justrun.sh
# does not set it up itself, and a stale build lives alongside the new one.

ACCELSIM=/home/razul/accel-sim-framework
RUN_DIR="${1:-$ACCELSIM/sim_run_oracle_buf}"
PARALLEL=${2:-10}

cd "$ACCELSIM"
source gpu-simulator/setup_environment.sh release

echo "=== launching $(date) ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "jobs: $(find "$RUN_DIR" -name justrun.sh | wc -l), parallelism: $PARALLEL"

run_one() {
    d=$(dirname "$1")
    name="${d#$RUN_DIR/}"
    echo "[START $(date '+%H:%M:%S')] $name"
    ( cd "$d" && bash justrun.sh > job.log 2>&1 )
    # justrun.sh pipes through tee, so its exit status is tee's; look for the
    # simulator's own completion marker instead
    if grep -q "GPGPU-Sim: \*\*\* exit detected \*\*\*\|GPGPU-Sim Simulator Statistics" "$d/job.log" 2>/dev/null; then
        echo "[DONE  $(date '+%H:%M:%S')] $name"
    else
        echo "[FAIL  $(date '+%H:%M:%S')] $name  (see $d/job.log)"
    fi
}
export -f run_one
export RUN_DIR

find "$RUN_DIR" -name justrun.sh | sort | \
    xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo "=== all jobs finished $(date) ==="
echo "collect with:"
echo "  python3 util/job_launching/get_stats.py -R -B rodinia-3.1 -C A100-SASS,A100-SASS-ORACLE_PREF -r sim_run_oracle > oracle_rodinia_stats.csv"
