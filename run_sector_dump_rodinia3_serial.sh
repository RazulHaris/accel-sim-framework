#!/bin/bash
# Rodinia-3.1 sector-mask dump (A100-SASS-DUMP_SECTOR), strictly serial: one
# benchmark at a time, in the order below, never two simulations at once.
#
# Why serial: in the parallel attempt on 07 Oct, lavaMD alone reached 46 GB RSS
# and was still growing.  Next to anything else it does not fit in this 62 GB
# box -- the OOM killer took 28-35 GB accel-sim processes 7 times in Aug-Sep.
#
# Order: shortest first (by runtime in sim_run_oracle_v2), so most dumps land
# early and the three multi-hour jobs (srad_v1, lavaMD, myocyte) run last.
# Expected total: roughly 15-18 h.
#
# A job whose job.log already shows a finished simulation is skipped; that
# covers the 7 that completed in the parallel attempt.  FORCE=1 re-runs them.
# A failed job is reported and the script moves on to the next one.
#
#   screen -dmS sector_r31_serial bash -c 'bash /home/razul/accel-sim-framework/run_sector_dump_rodinia3_serial.sh 2>&1 | tee /home/razul/accel-sim-framework/sim_run_sector_dump_rodinia3/serial.log'
#   screen -r sector_r31_serial          (ctrl-a d to detach)
#
# Each job writes job.log, sector_masks.log and peak_mem.txt (peak RSS in KB and
# wall seconds, from /usr/bin/time) in its own directory.

ACCELSIM=/home/razul/accel-sim-framework
RUN=$ACCELSIM/sim_run_sector_dump_rodinia3
SIM="$RUN/gpgpu-sim-builds/accelsim-commit-ba09ee6_modified_2.0_26-10-05-00-00-53gpgpu-sim_git-commit-7126b169_modified_3.0./accel-sim.out"
MIN_FREE_GB=30
FORCE=${FORCE:-0}

source "$ACCELSIM/gpu-simulator/setup_environment.sh" release >/dev/null 2>&1

ts() { date '+%m-%d %H:%M:%S'; }
finished() { tail -c 50000 "$1/job.log" 2>/dev/null | grep -q "exit detected\|GPGPU-Sim Simulator Statistics"; }

n_done=0; n_fail=0; n_skip=0

# begin <label> <dir>: decides whether the job runs.  Returns 0 (and leaves us
# inside the job directory) if it should, 1 if it is skipped.
begin() {
    LABEL=$1; DIR=$2; RAN=0
    if [ "$FORCE" != 1 ] && finished "$DIR"; then
        echo "[SKIP  $(ts)] $LABEL  (already completed)"
        n_skip=$((n_skip + 1)); return 1
    fi
    free_gb=$(df -BG --output=avail "$ACCELSIM" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "[STOP  $(ts)] only ${free_gb}G free on disk, need ${MIN_FREE_GB}G -- stopping before $LABEL"
        exit 1
    fi
    # strictly one simulation at a time, even against ones started elsewhere
    while pgrep -x accel-sim.out >/dev/null; do
        echo "[WAIT  $(ts)] another accel-sim.out is running; holding $LABEL"
        sleep 60
    done
    cd "$DIR" || { echo "[FAIL  $(ts)] $LABEL  (no such directory)"; n_fail=$((n_fail + 1)); return 1; }
    echo "[START $(ts)] $LABEL  (MemAvailable $(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)G)"
    RAN=1; T0=$(date +%s); return 0
}

# end: reports the job that begin() just let through.
end() {
    [ "$RAN" = 1 ] || return 0
    wall=$(( $(date +%s) - T0 ))
    peak=$(tail -1 peak_mem.txt 2>/dev/null | cut -d, -f1)
    peak_gb=$(awk -v k="${peak:-0}" 'BEGIN {printf "%.1f", k / 1048576}')
    if finished "$DIR"; then
        echo "[DONE  $(ts)] $LABEL  ${wall}s  peak ${peak_gb}G  dump $(du -h sector_masks.log 2>/dev/null | cut -f1)"
        n_done=$((n_done + 1))
    else
        echo "[FAIL  $(ts)] $LABEL  ${wall}s  peak ${peak_gb}G  (see $DIR/job.log)"
        n_fail=$((n_fail + 1))
    fi
    cd "$ACCELSIM"
}

if pgrep -x accel-sim.out >/dev/null; then
    echo "accel-sim.out is already running -- refusing to start a serial run beside it:"
    pgrep -a -x accel-sim.out
    exit 1
fi

echo "=== serial sector-mask dump, Rodinia-3.1, started $(date) ==="

# ---- 1/24  gaussian  matrix4            (~5 s) -------------------------------
begin "gaussian matrix4" "$RUN/gaussian-rodinia-3.1/_f___data_matrix4_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 2/24  nn                           (~9 s) -------------------------------
begin "nn" "$RUN/nn-rodinia-3.1/__data_filelist_4__r_5__lat_30__lng_90/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 3/24  bfs  graph4096               (~31 s) ------------------------------
begin "bfs graph4096" "$RUN/bfs-rodinia-3.1/__data_graph4096_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 4/24  gaussian  s_16               (~38 s) ------------------------------
begin "gaussian s_16" "$RUN/gaussian-rodinia-3.1/_s_16/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 5/24  dwt2d  192x192               (~44 s) ------------------------------
begin "dwt2d 192x192" "$RUN/dwt2d-rodinia-3.1/__data_192_bmp__d_192x192__f__5__l_3/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 6/24  hotspot  512                 (~1.5 min) ---------------------------
begin "hotspot 512" "$RUN/hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 7/24  gaussian  s_64               (~3 min) -----------------------------
begin "gaussian s_64" "$RUN/gaussian-rodinia-3.1/_s_64/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 8/24  lud  s_256                   (~3 min) -----------------------------
begin "lud s_256" "$RUN/lud-rodinia-3.1/_s_256__v/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 9/24  particlefilter_naive         (~4.5 min) ---------------------------
begin "particlefilter_naive" "$RUN/particlefilter_naive-rodinia-3.1/_x_128__y_128__z_10__np_1000/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 10/24 backprop  65536              (~5 min) -----------------------------
begin "backprop 65536" "$RUN/backprop-rodinia-3.1/65536/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 11/24 bfs  graph65536              (~6 min) -----------------------------
begin "bfs graph65536" "$RUN/bfs-rodinia-3.1/__data_graph65536_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 12/24 hotspot  1024                (~7 min) -----------------------------
begin "hotspot 1024" "$RUN/hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 13/24 dwt2d  1024x1024             (~8 min)  [done in parallel attempt] -
begin "dwt2d 1024x1024" "$RUN/dwt2d-rodinia-3.1/__data_rgb_bmp__d_1024x1024__f__5__l_3/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 14/24 pathfinder                   (~10 min) [done in parallel attempt] -
begin "pathfinder" "$RUN/pathfinder-rodinia-3.1/100000_100_20___result_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 15/24 b+tree                       (~13 min) [done in parallel attempt] -
begin "b+tree" "$RUN/b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 16/24 lud  512                     (~16 min) [done in parallel attempt] -
begin "lud 512" "$RUN/lud-rodinia-3.1/_i___data_512_dat/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 17/24 particlefilter_float         (~17 min) [done in parallel attempt] -
begin "particlefilter_float" "$RUN/particlefilter_float-rodinia-3.1/_x_128__y_128__z_10__np_1000/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 18/24 gaussian  matrix208          (~19 min) [done in parallel attempt] -
begin "gaussian matrix208" "$RUN/gaussian-rodinia-3.1/_f___data_matrix208_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 19/24 gaussian  s_256              (~32 min) [done in parallel attempt] -
begin "gaussian s_256" "$RUN/gaussian-rodinia-3.1/_s_256/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 20/24 bfs  graph1MW                (~57 min) ----------------------------
begin "bfs graph1MW" "$RUN/bfs-rodinia-3.1/__data_graph1MW_6_txt/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 21/24 nw  2048                     (~1.2 h) -----------------------------
begin "nw 2048" "$RUN/nw-rodinia-3.1/2048_10/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 22/24 srad_v1                      (~3.4 h) -----------------------------
begin "srad_v1" "$RUN/srad_v1-rodinia-3.1/100_0_5_502_458/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 23/24 lavaMD  boxes1d_10           (~4.3 h, >46 GB RSS) -----------------
begin "lavaMD boxes1d_10" "$RUN/lavaMD-rodinia-3.1/_boxes1d_10/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

# ---- 24/24 myocyte                      (~5.7 h) -----------------------------
begin "myocyte" "$RUN/myocyte-rodinia-3.1/100_1_0/A100-SASS-DUMP_SECTOR" && \
    /usr/bin/time -o peak_mem.txt -f "%M,%e" "$SIM" -config ./gpgpusim.config -trace ./traces/kernelslist.g > job.log 2>&1
end

echo "=== finished $(date) ==="
echo "done $n_done  failed $n_fail  skipped $n_skip  (of 24)"
echo "total dump: $(find "$RUN" -name sector_masks.log -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)"
