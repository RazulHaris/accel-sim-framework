#!/bin/bash
# Snake synthetic-trace check (docs/snake/SNAKE_PLAN.md section 9.2).
#
# Writes the synthetic traces (gen_synthetic_traces.py), runs them with
# Snake on and its debug log (QV100-SASS-SYNTH_LRR1-SNAKE-SNAKE_DBG) on a
# frozen copy of the current build, and checks them (check_synthetic.py).
#
# usage: run_synthetic.sh <stage> [parallel]
# Run from a clean tree after a clean build; the script refuses otherwise.

set -euo pipefail
STAGE=${1:?usage: run_synthetic.sh <stage> [parallel]}
PAR=${2:-3}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ACCELSIM/sim_run_snake_synth/$STAGE
CFG=QV100-SASS-SYNTH_LRR1-SNAKE-SNAKE_DBG

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

python3 -I util/snake/gen_synthetic_traces.py "$OUT/traces"
python3 util/job_launching/run_simulations.py -B snake-synth -C $CFG -T "$OUT/traces" \
    -N snake_synth_$STAGE -r "$OUT/runs" -n >/dev/null

run_job() {  # <job dir>
  cd "$1" && LD_LIBRARY_PATH="$OUT/bin/lib" "$OUT/bin/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/}"
}
export -f run_job; export OUT
find "$OUT/runs" -name justrun.sh -printf '%h\n' | xargs -P "$PAR" -I{} bash -c 'run_job "{}"'

python3 -I util/snake/check_synthetic.py "$OUT/runs" "$OUT/traces"
