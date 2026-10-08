#!/bin/bash
# Rodinia 2.0-ft sector-mask dump run (A100-SASS-DUMP_SECTOR).
#
# One detached screen session per benchmark, named sec_<benchmark>.  Each job
# writes its own sector_masks.log in its own job directory:
#   cycle,sm_id,warp_id,uid,pc,num_coalesced_transactions,sector_mask,sector,addr,status
# one record per L1D transaction of every global load.
#
#   screen -ls                 list the sessions
#   screen -r sec_backprop     attach to one (ctrl-a d to detach)
#   bash run_sector_dump.sh    (re)launch; skips benchmarks already running
#
# Logs are capped per job.  10 jobs x 5 GiB is the worst case, which fits the
# free space on /home with room to spare; raise only after checking df.

ACCELSIM=/home/razul/accel-sim-framework
RUN_DIR=$ACCELSIM/sim_run_sector_dump
export SECTOR_MASK_LOG_MAX_BYTES=$((5 * 1024 * 1024 * 1024))   # 5 GiB per log
MIN_FREE_GB=30

cd "$ACCELSIM" || exit 1
source gpu-simulator/setup_environment.sh release >/dev/null 2>&1

launched=0
skipped=0

for js in $(find "$RUN_DIR" -name justrun.sh | sort); do
    jobdir=$(dirname "$js")
    # .../sim_run_sector_dump/<benchmark>/<args>/<config>/justrun.sh
    bench=$(basename "$(dirname "$(dirname "$jobdir")")")
    session="sec_${bench%%-rodinia-2.0-ft}"

    if screen -ls | grep -q "[.]${session}[[:space:]]"; then
        echo "[SKIP ] $session already has a running screen"
        skipped=$((skipped + 1))
        continue
    fi

    free_gb=$(df -BG --output=avail "$ACCELSIM" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "[ABORT] only ${free_gb}G free, need ${MIN_FREE_GB}G -- stopping"
        break
    fi

    screen -dmS "$session" bash -c "
        cd '$jobdir' || exit 1
        export SECTOR_MASK_LOG_MAX_BYTES=$SECTOR_MASK_LOG_MAX_BYTES
        echo \"[START \$(date '+%F %H:%M:%S')] $bench\"
        bash justrun.sh > job.log 2>&1
        rc=\$?
        if grep -q 'exit detected' job.log 2>/dev/null; then
            echo \"[DONE  \$(date '+%F %H:%M:%S')] $bench  dump=\$(du -h sector_masks.log 2>/dev/null | cut -f1)\"
        else
            echo \"[FAIL  \$(date '+%F %H:%M:%S')] $bench  rc=\$rc  see job.log\"
        fi
        # keep the window alive so the result stays readable on attach
        exec sleep infinity
    "
    echo "[START] $session  ->  ${jobdir#$ACCELSIM/}"
    launched=$((launched + 1))
done

echo
echo "launched $launched, skipped $skipped"
screen -ls
