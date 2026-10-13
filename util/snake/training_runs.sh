#!/bin/bash
# Snake training plausibility runs (docs/snake/SNAKE_PLAN.md S2 gate).
#
# Runs rodinia-3.1 hotspot, backprop and srad_v1 with Snake on in each lane
# mode (OQ-17: QV100-SASS-PAPER_V100-<cap>_INSN-SNAKE = lane mode 1, the
# default; ...-SNAKE-SNAKE_LANE0 = lane mode 0) on a frozen copy of the current
# build, then prints training_report.py for every run.
#
# usage: training_runs.sh <stage> [insn_cap] [parallel]
# Run from a clean tree after a clean build; the script refuses otherwise.

set -euo pipefail
STAGE=${1:?usage: training_runs.sh <stage> [insn_cap] [parallel]}
INSN_CAP=${2:-100M}
PAR=${3:-8}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ACCELSIM/sim_run_snake_train/$STAGE
CFG=QV100-SASS-PAPER_V100-${INSN_CAP}_INSN-SNAKE
APPS=${APPS:-rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1}

cd "$ACCELSIM"
for r in . gpu-simulator/gpgpu-sim; do
  [ "$(git -C $r branch --show-current)" = Snake ] || { echo "ABORT: $r is not on Snake"; exit 1; }
  [ -z "$(git -C $r status --porcelain)" ] || { echo "ABORT: $r is dirty"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "ABORT: $OUT exists; pick a new stage name"; exit 1; }

set +u; source gpu-simulator/setup_environment.sh release >/dev/null 2>&1; set -u
mkdir -p "$OUT/bin/lib"
cp -a gpu-simulator/bin/release/accel-sim.out "$OUT/bin/"
cp -a "$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG/." "$OUT/bin/lib/"
tags="$(strings "$OUT/bin/accel-sim.out" | grep -o 'accelsim-commit-[^ "]*' | head -1) $(strings "$OUT/bin/lib/libcudart.so" | grep -o 'gpgpu-sim_git-commit-[^ "]*' | head -1)"
echo "build: $tags" | tee "$OUT/BUILDS.txt"
[ "$(echo "$tags" | grep -o '_modified_0\.0' | wc -l)" = 2 ] || { echo "ABORT: build is not _modified_0.0 twice"; exit 1; }

python3 util/job_launching/run_simulations.py -B $APPS -C "$CFG,$CFG-SNAKE_LANE0" \
    -T hw_run/rodinia-3.1/11.0 -N snake_train_$STAGE -r "$OUT/runs" -n >/dev/null

run_job() {  # <job dir>
  cd "$1" && LD_LIBRARY_PATH="$OUT/bin/lib" "$OUT/bin/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/}"
}
export -f run_job; export OUT
find "$OUT/runs" -name justrun.sh -printf '%h\n' | xargs -P "$PAR" -I{} bash -c 'run_job "{}"'

python3 -I util/snake/training_report.py $(find "$OUT/runs" -name sim.out | sort)
