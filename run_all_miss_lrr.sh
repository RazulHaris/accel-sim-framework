#!/bin/bash
# Rodinia-3.1 with the oracle uncoalesced prefetcher at a 128KB and 256KB
# unified L1D+shmem pool, bracketing the 192KB oracle-v2 run.
#
# The shared-memory carveout each kernel picks is unchanged from the 192KB run
# (verified over 728 kernel launches), so the only variable is L1D capacity.
#
#   screen -dmS l1d_sweep bash /home/razul/accel-sim-framework/run_l1d_sweep.sh
#   screen -r l1d_sweep
#
# Run dirs were created by:
#   python3 util/job_launching/run_simulations.py -B rodinia-3.1 \
#       -C A100-SASS-RR-ORACLE_ALL_MISS \
#       -T hw_run/rodinia-3.1/11.0 -N all_miss_lrr -r sim_run_all_miss_lrr -n

ACCELSIM=/home/razul/accel-sim-framework
RUN_DIR="${1:-$ACCELSIM/sim_run_all_miss_lrr}"
PARALLEL=${2:-12}

cd "$ACCELSIM"
source gpu-simulator/setup_environment.sh release

echo "=== launching $(date) ==="
echo "jobs: $(find "$RUN_DIR" -name justrun.sh | wc -l), parallelism: $PARALLEL"

run_one() {
    d=$(dirname "$1")
    name="${d#$RUN_DIR/}"
    echo "[START $(date '+%m-%d %H:%M:%S')] $name"
    ( cd "$d" && bash justrun.sh > job.log 2>&1 )
    # justrun.sh pipes through tee, so its exit status is tee's; look for the
    # simulator's own completion marker instead
    if grep -q "GPGPU-Sim: \*\*\* exit detected \*\*\*\|GPGPU-Sim Simulator Statistics" "$d/job.log" 2>/dev/null; then
        echo "[DONE  $(date '+%m-%d %H:%M:%S')] $name"
    else
        echo "[FAIL  $(date '+%m-%d %H:%M:%S')] $name  (see $d/job.log)"
    fi
}
export -f run_one; export RUN_DIR

find "$RUN_DIR" -name justrun.sh | sort | \
    xargs -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo "=== all jobs finished $(date) ==="
