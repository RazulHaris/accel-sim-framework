#!/bin/bash
# Rodinia-3.1 sector-mask dump run (A100-SASS-DUMP_SECTOR), RAM-aware.
#
# One screen session runs this scheduler; it starts jobs in parallel but only
# while there is memory for them:
#
#   admit the next job  iff  running < MAX_PAR
#                       and  MemAvailable - RESERVE - headroom >= EST
#
# where headroom = sum over running jobs of max(0, EST - current RSS), i.e. room
# kept back for jobs that have not grown to their estimate yet.  A job that has
# grown past EST is already accounted for by MemAvailable itself.  Jobs run
# longest-first (by their runtime in sim_run_oracle_v2) so the multi-hour ones
# start immediately instead of becoming a tail.
#
#   screen -dmS sector_r31 bash run_sector_dump_rodinia3.sh     # launch
#   screen -r sector_r31                                        # watch (ctrl-a d)
#   cat sim_run_sector_dump_rodinia3/scheduler_status.txt       # live snapshot
#
# Re-running skips jobs whose job.log already shows a completed simulation.
# Per-job peak RSS and wall time go to sim_run_sector_dump_rodinia3/job_memory.csv.

ACCELSIM=/home/razul/accel-sim-framework
RUN_DIR=$ACCELSIM/sim_run_sector_dump_rodinia3
HIST_DIR=$ACCELSIM/sim_run_oracle_v2        # runtimes used for longest-first order

MAX_PAR=${1:-12}          # hard cap on concurrent jobs (16 cores on this box)
EST_GB=${EST_GB:-4}       # memory assumed per job until its real RSS is known
RESERVE_GB=${RESERVE_GB:-8}   # never plan to use the last 8 GB
MIN_FREE_GB=30            # stop launching if /home gets this low
POLL=10

STATUS=$RUN_DIR/scheduler_status.txt
MEMLOG=$RUN_DIR/job_memory.csv

cd "$ACCELSIM" || exit 1
source gpu-simulator/setup_environment.sh release >/dev/null 2>&1

ts() { date '+%m-%d %H:%M:%S'; }
mem_avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
session_rss_kb() { ps -o rss= --sid "$1" 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
completed() { grep -q "exit detected\|GPGPU-Sim Simulator Statistics" "$1/job.log" 2>/dev/null; }

# ---- build the queue, longest historical runtime first ----------------------
declare -a QUEUE=()
while read -r secs d; do QUEUE+=("$d"); done < <(
    for js in $(find "$RUN_DIR" -name justrun.sh); do
        d=$(dirname "$js")
        key=$(echo "${d#$RUN_DIR/}" | cut -d/ -f1,2)
        h=$(ls "$HIST_DIR/$key"/A100-SASS/job.log 2>/dev/null | head -1)
        secs=$(grep -oE "gpgpu_simulation_time = .*\(([0-9]+) sec\)" "$h" 2>/dev/null \
               | tail -1 | grep -oE "\([0-9]+ sec" | tr -dc 0-9)
        echo "${secs:-999999} $d"        # no history: treat as longest
    done | sort -rn)

[ -f "$MEMLOG" ] || echo "job,state,peak_rss_mb,wall_s" > "$MEMLOG"

declare -A PID=() START=() PEAK=()
next=0; launched=0; done_n=0; fail_n=0; skipped=0

echo "=== launching $(date) ==="
echo "jobs: ${#QUEUE[@]}  max parallel: $MAX_PAR  est/job: ${EST_GB}G  reserve: ${RESERVE_GB}G"

while :; do
    # ---- reap finished jobs and refresh RSS of running ones -----------------
    headroom_kb=0
    for d in "${!PID[@]}"; do
        pid=${PID[$d]}
        name=${d#$RUN_DIR/}
        if kill -0 "$pid" 2>/dev/null; then
            rss=$(session_rss_kb "$pid")
            (( rss > ${PEAK[$d]:-0} )) && PEAK[$d]=$rss
            gap=$(( EST_GB * 1048576 - rss ))
            (( gap > 0 )) && headroom_kb=$(( headroom_kb + gap ))
        else
            wait "$pid" 2>/dev/null
            wall=$(( $(date +%s) - START[$d] ))
            peak_mb=$(( ${PEAK[$d]:-0} / 1024 ))
            if completed "$d"; then
                st=DONE; done_n=$((done_n + 1))
                sz=$(du -h "$d/sector_masks.log" 2>/dev/null | cut -f1)
                echo "[DONE  $(ts)] $name  peak=${peak_mb}MB  ${wall}s  dump=${sz:-none}"
            else
                st=FAIL; fail_n=$((fail_n + 1))
                echo "[FAIL  $(ts)] $name  peak=${peak_mb}MB  ${wall}s  (see job.log)"
            fi
            echo "$name,$st,$peak_mb,$wall" >> "$MEMLOG"
            unset "PID[$d]" "START[$d]" "PEAK[$d]"
        fi
    done

    # ---- admit as many queued jobs as memory allows -------------------------
    while (( next < ${#QUEUE[@]} )) && (( ${#PID[@]} < MAX_PAR )); do
        d=${QUEUE[$next]}
        if completed "$d"; then
            echo "[SKIP  $(ts)] ${d#$RUN_DIR/}  (already completed)"
            skipped=$((skipped + 1)); next=$((next + 1)); continue
        fi
        avail_kb=$(mem_avail_kb)
        budget_kb=$(( avail_kb - RESERVE_GB * 1048576 - headroom_kb ))
        (( budget_kb < EST_GB * 1048576 )) && break          # wait for memory
        free_gb=$(df -BG --output=avail "$ACCELSIM" | tail -1 | tr -dc '0-9')
        if (( free_gb < MIN_FREE_GB )); then
            echo "[HOLD  $(ts)] only ${free_gb}G free on disk, not launching more"
            break
        fi
        # setsid: each job gets its own session, so its RSS can be summed
        # across accel-sim.out, tee and the wrapping shells
        ( cd "$d" && exec setsid bash -c 'bash justrun.sh > job.log 2>&1' ) &
        PID[$d]=$!; START[$d]=$(date +%s); PEAK[$d]=0
        launched=$((launched + 1)); next=$((next + 1))
        headroom_kb=$(( headroom_kb + EST_GB * 1048576 ))
        echo "[START $(ts)] ${d#$RUN_DIR/}  (running ${#PID[@]}, MemAvailable $((avail_kb / 1048576))G)"
    done

    # ---- status snapshot ----------------------------------------------------
    {
        echo "updated $(date)"
        echo "queued $(( ${#QUEUE[@]} - next ))  running ${#PID[@]}  done $done_n  failed $fail_n  skipped $skipped"
        echo "MemAvailable $(( $(mem_avail_kb) / 1048576 ))G  reserved-for-growth $(( headroom_kb / 1048576 ))G  disk free $(df -BG --output=avail "$ACCELSIM" | tail -1 | tr -dc '0-9')G"
        echo
        for d in "${!PID[@]}"; do
            printf "  %-70s rss %5dMB  peak %5dMB  %6ss\n" "${d#$RUN_DIR/}" \
                "$(( $(session_rss_kb "${PID[$d]}") / 1024 ))" "$(( ${PEAK[$d]:-0} / 1024 ))" \
                "$(( $(date +%s) - START[$d] ))"
        done
    } > "$STATUS"

    (( next >= ${#QUEUE[@]} && ${#PID[@]} == 0 )) && break
    sleep "$POLL"
done

echo "=== ALL FINISHED $(date) ==="
echo "done $done_n  failed $fail_n  skipped $skipped  (of ${#QUEUE[@]})"
echo "total dump: $(find "$RUN_DIR" -name sector_masks.log -exec du -cb {} + 2>/dev/null | tail -1 | cut -f1) bytes"
exec sleep infinity       # keep the screen open so the summary stays readable
