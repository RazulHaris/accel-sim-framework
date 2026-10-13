#!/bin/bash
# Run two frozen simulator builds on the off-regression workloads and compare
# their final statistics (used for bin_ref_paperera vs bin_ref_v2, see
# docs/snake/VERSION_DIFF.md, "Pre-756ed33 traces").
#
# Each build is a directory with accel-sim.out and lib/libcudart.so.  Jobs:
# the off-regression set (rodinia-3.1 backprop, hotspot x2, nw, srad_v1 and
# parboil histo), QV100-SASS-PAPER_V100-<cap>_INSN.  Results go to
# sim_run_snake_refcmp/<name>/{a,b}.
#
# usage: compare_bins.sh <name> <build a> <build b> [insn_cap] [parallel]
#   COMPARE_ONLY=1 compare_bins.sh ...   re-prints the table of an existing run

set -euo pipefail
NAME=${1:?usage: compare_bins.sh <name> <build a> <build b> [insn_cap] [parallel]}
A=${2:?}; B=${3:?}
INSN_CAP=${4:-100M}
PAR=${5:-12}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ACCELSIM/sim_run_snake_refcmp/$NAME
CFG=QV100-SASS-PAPER_V100-${INSN_CAP}_INSN
ROD_APPS=${ROD_APPS:-rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:nw-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1}
PB_APPS=${PB_APPS:-parboil:parboil-histo}

cd "$ACCELSIM"
if [ "${COMPARE_ONLY:-0}" != 1 ]; then
for r in . gpu-simulator/gpgpu-sim; do
  [ "$(git -C $r branch --show-current)" = Snake ] || { echo "ABORT: $r is not on Snake"; exit 1; }
  [ -z "$(git -C $r status --porcelain)" ] || { echo "ABORT: $r is dirty"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "ABORT: $OUT exists; pick a new name"; exit 1; }
mkdir -p "$OUT"
for v in a b; do
  d=$A; [ $v = b ] && d=$B
  t="$(strings "$d/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1) $(strings "$d/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
  echo "$v: $d  $t" | tee -a "$OUT/BUILDS.txt"
  [ "$(echo "$t" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: build $v is not _modified_0.0 twice"; exit 1; }
done
set +u; source gpu-simulator/setup_environment.sh release >/dev/null 2>&1; set -u
for v in a b; do
  python3 util/job_launching/run_simulations.py -B $ROD_APPS -C $CFG -T hw_run/rodinia-3.1/11.0 \
      -N snake_refcmp_${NAME}_${v}_rod -r "$OUT/$v" -n >/dev/null
  python3 util/job_launching/run_simulations.py -B $PB_APPS -C $CFG -T hw_run/parboil \
      -N snake_refcmp_${NAME}_${v}_pb -r "$OUT/$v" -n >/dev/null
done
run_job() {  # <job dir> <build dir>
  cd "$1" && LD_LIBRARY_PATH="$2/lib" "$2/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/}"
}
export -f run_job; export OUT
{
  for d in $(find "$OUT/a" -name justrun.sh -printf '%h\n'); do echo "$d $A"; done
  for d in $(find "$OUT/b" -name justrun.sh -printf '%h\n'); do echo "$d $B"; done
} | xargs -P "$PAR" -L 1 bash -c 'run_job "$0" "$1"'
fi

set +e +o pipefail
last() { grep -E "$2" "$1" | tail -1 | awk '{print $NF}'; }
pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if (a == 0) print "n/a"; else printf "%+.2f%%", 100 * (b - a) / a }'; }
echo "| workload | cycles a -> b | L1D read hits a -> b | L1D read misses a -> b | DRAM reads a -> b | re-decoded lines (b) |"
echo "|---|---|---|---|---|---|"
for j in $(cd "$OUT/a" && find . -name sim.out | sort); do
  ra=$OUT/a/$j; rb=$OUT/b/$j
  grep -q 'exit detected' "$ra" && grep -q 'exit detected' "$rb" || { echo "| $j | run did not finish | | | | |"; continue; }
  row="| $(echo "$j" | cut -d/ -f2,3 | cut -c1-60) "
  for re in 'gpu_tot_sim_cycle = ' 'Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] ' \
            'Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\] ' 'total dram reads'; do
    x=$(last "$ra" "$re"); y=$(last "$rb" "$re")
    row+="| $x -> $y ($(pct "$x" "$y")) "
  done
  echo "$row| $(last "$rb" 'pre-756ed33 base\+delta trace lines re-decoded') |"
done
