#!/bin/bash
# Snake off-regression (docs/snake/SNAKE_PLAN.md section 9.1).
#
# Runs the same jobs with
#   ref : the clean-base reference build   ($SNAKE_BIN_REF, default
#         /home/razul/snake/bin_ref: accel-sim.out + lib/libcudart.so)
#   off : the current build, Snake off      (default -snake_enable 0)
#   on  : the current build, -snake_enable 1
# and requires ref == off byte for byte after removing only volatile lines
# (dates, wall-clock, simulation rate, build strings, paths) and the added
# -snake_* lines of gpgpu-sim's option dump (see regress_norm.py).  on vs off
# is checked too: until Snake issues prefetches (S3) it must not change timing,
# so it must match apart from its own block.  Independently of the
# normalization, the final cycles, instructions, L1D read hits/misses and DRAM
# reads of ref/off/on are compared raw.
#
# Jobs: rodinia-3.1 backprop, hotspot (2 inputs), nw, srad_v1 and parboil histo,
# QV100-SASS-PAPER_V100 with an instruction cap.  Override with ROD_APPS/PB_APPS
# (run_simulations.py -B syntax: suite:exe; suite:exe:N is not supported).
#
# usage: off_regression.sh <stage> [insn_cap] [parallel]
#   e.g. off_regression.sh S1
#   COMPARE_ONLY=1 off_regression.sh S1   re-compares an existing run
# Run from a clean tree after building it; the script refuses otherwise.

set -euo pipefail
STAGE=${1:?usage: off_regression.sh <stage> [insn_cap] [parallel]}
INSN_CAP=${2:-100M}
PAR=${3:-12}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
REF=${SNAKE_BIN_REF:-/home/razul/snake/bin_ref}
NORM=$ACCELSIM/util/snake/regress_norm.py   # what is removed and why: see that file
OUT=$ACCELSIM/sim_run_snake_regress/$STAGE
CFG=QV100-SASS-PAPER_V100-${INSN_CAP}_INSN
ROD_APPS=${ROD_APPS:-rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:nw-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1}
PB_APPS=${PB_APPS:-parboil:parboil-histo}

cd "$ACCELSIM"
if [ "${COMPARE_ONLY:-0}" != 1 ]; then
for r in . gpu-simulator/gpgpu-sim; do
  [ "$(git -C $r branch --show-current)" = Snake ] || { echo "ABORT: $r is not on Snake"; exit 1; }
  [ -z "$(git -C $r status --porcelain)" ] || { echo "ABORT: $r is dirty"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "ABORT: $OUT exists; pick a new stage name"; exit 1; }

set +u; source gpu-simulator/setup_environment.sh release >/dev/null 2>&1; set -u
LIBDIR=$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG

# Freeze the current build next to the results.
mkdir -p "$OUT/bin_new/lib"
cp -a gpu-simulator/bin/release/accel-sim.out "$OUT/bin_new/"
cp -a "$LIBDIR/." "$OUT/bin_new/lib/"
{
  echo "stage $STAGE  $(date '+%F %T')"
  echo "accel-sim $(git log -1 --format='%h %s')"
  echo "gpgpu-sim $(git -C gpu-simulator/gpgpu-sim log -1 --format='%h %s')"
  echo "new: $(strings "$OUT/bin_new/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1)" \
       "$(strings "$OUT/bin_new/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
  echo "ref: $(strings "$REF/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1)" \
       "$(strings "$REF/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
} > "$OUT/BUILDS.txt"
cat "$OUT/BUILDS.txt"
[ "$(sed -n 4p "$OUT/BUILDS.txt" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: new build is not _modified_0.0 twice"; exit 1; }

# Job directories (configs, trace links); -n = do not launch.
for v in ref off on; do
  c=$CFG; [ $v = on ] && c=$CFG-SNAKE
  python3 util/job_launching/run_simulations.py -B $ROD_APPS -C $c -T hw_run/rodinia-3.1/11.0 \
      -N snake_regress_${STAGE}_${v}_rod -r "$OUT/$v" -n >/dev/null
  python3 util/job_launching/run_simulations.py -B $PB_APPS -C $c -T hw_run/parboil \
      -N snake_regress_${STAGE}_${v}_pb -r "$OUT/$v" -n >/dev/null
done

run_job() {  # <job dir> <binary dir>
  cd "$1" && LD_LIBRARY_PATH="$2/lib" "$2/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/}"
}
export -f run_job; export OUT
{
  for d in $(find "$OUT/ref" -name justrun.sh -printf '%h\n'); do echo "$d $REF"; done
  for d in $(find "$OUT/off" "$OUT/on" -name justrun.sh -printf '%h\n'); do echo "$d $OUT/bin_new"; done
} | xargs -P "$PAR" -L 1 bash -c 'run_job "$0" "$1"'
fi

# Comparison.  Differences are expected, so no errexit/pipefail from here on.
set +e +o pipefail
norm() { python3 -I "$NORM" "$@" || { echo "ABORT: normalization failed on $1" >&2; exit 2; }; }
keystats() {  # final cumulative values, read from the raw output
  printf '%s ' \
    "cyc=$(grep -o 'gpu_tot_sim_cycle = [0-9]*' "$1" | tail -1 | awk '{print $3}')" \
    "insn=$(grep -o 'gpu_tot_sim_insn = [0-9]*' "$1" | tail -1 | awk '{print $3}')" \
    "l1r_hit=$(grep -E 'Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] ' "$1" | tail -1 | awk '{print $3}')" \
    "l1r_miss=$(grep -E 'Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\] ' "$1" | tail -1 | awk '{print $3}')" \
    "dram_rd=$(grep 'total dram reads' "$1" | tail -1 | awk '{print $NF}')"
}
status=0
for j in $(cd "$OUT/ref" && find . -name sim.out); do
  r=$OUT/ref/$j; o=$OUT/off/$j
  n=$(echo "$OUT/on/$j" | sed "s#/$CFG/#/$CFG-SNAKE/#")
  name=$(echo "$j" | cut -d/ -f2,3)
  grep -q 'exit detected' "$r" && grep -q 'exit detected' "$o" || { echo "FAIL $name: a run did not finish"; status=1; continue; }
  nr=$(norm "$r" | wc -l); no=$(norm "$o" | wc -l)
  [ "$nr" -gt 1000 ] && [ "$no" -gt 1000 ] || { echo "FAIL $name: too few lines after normalization ($nr/$no)"; status=1; continue; }
  if diff -q <(norm "$r") <(norm "$o") >/dev/null; then
    echo "PASS ref==off  $name  ($nr of $(wc -l < "$r") lines compared; excluded: $(grep -c '^-snake_' "$o") -snake_ option lines + volatile lines; mf-dump runs sorted)"
  else
    echo "FAIL ref!=off  $name"; diff <(norm "$r") <(norm "$o") | head -20; status=1
  fi
  if diff -q <(norm "$o") <(norm "$n" --strip-snake-block) >/dev/null; then
    echo "     off==on   $name  (Snake block: $(grep -c '^snake_' "$n") lines)"
  else
    echo "FAIL off!=on   $name"; diff <(norm "$o") <(norm "$n" --strip-snake-block) | head -10; status=1
  fi
  kr=$(keystats "$r"); ko=$(keystats "$o"); kn=$(keystats "$n")
  if [ "$kr" = "$ko" ] && [ "$ko" = "$kn" ]; then
    echo "     raw stats ref==off==on: $kr"
  else
    echo "FAIL raw stats  ref: $kr"; echo "                off: $ko"; echo "                 on: $kn"; status=1
  fi
done
exit $status
