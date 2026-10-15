#!/bin/bash
# Snake synthetic-trace check (docs/snake/SNAKE_PLAN.md section 9.2).
#
# Writes the synthetic traces (gen_synthetic_traces.py), runs them with
# Snake on and its debug log on a frozen copy of the current build, and
# checks each config (check_synthetic.py).  Configs: $CFGS (comma-separated);
# the default is the S3a set, Snake-DT (no decoupling or throttling) with the
# line footprint and with the exact footprint (-SNAKE_FP1, OQ-9).  The S2
# gates used QV100-SASS-SYNTH_LRR1-SNAKE-SNAKE_DBG and -SNAKE_LANE0.
# $FIG15_CFGS (comma-separated, may be empty) are run on the Fig. 15 trace
# only.  From S3e the default config is the paper-literal one (no chain from
# inter-warp targets, OQ-29), whose Fig. 15 (c) must match the figure
# exactly; the extra default -SNAKE_IWC3 checks the chain extension
# (Fig. 15 (c) plus {3100, 3200, 3300}).
#
# usage: run_synthetic.sh <stage> [parallel]
# Run from a clean tree after a clean build; the script refuses otherwise.

set -euo pipefail
STAGE=${1:?usage: run_synthetic.sh <stage> [parallel]}
PAR=${2:-3}
ACCELSIM=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ACCELSIM/sim_run_snake_synth/$STAGE
CFG=QV100-SASS-SYNTH_LRR1-SNAKE_DT-SNAKE_DBG
CFGS=${CFGS:-$CFG,$CFG-SNAKE_FP1}
FIG15_CFGS=${FIG15_CFGS-$CFG-SNAKE_IWC3}
# $DEP_CFGS (may be empty) run trace (d) only; from S3e the >= 90% coverage
# gate applies with the inter-warp chain on (OQ-29), the default reports it.
DEP_CFGS=${DEP_CFGS-$CFG-SNAKE_FP1-SNAKE_IWC3}
# $RAND_CFGS (may be empty) run trace (b) only; from S4 full Snake (decoupling
# mode 1 and throttling) must be throttled > 50% of its cycles there.
RAND_CFGS=${RAND_CFGS-QV100-SASS-SYNTH_LRR1-SNAKE-SNAKE_DBG}

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
python3 util/job_launching/run_simulations.py -B snake-synth -C "$CFGS" -T "$OUT/traces" \
    -N snake_synth_$STAGE -r "$OUT/runs" -n >/dev/null
[ -z "$FIG15_CFGS" ] || python3 util/job_launching/run_simulations.py \
    -B snake-synth:snake-synth-fig15 -C "$FIG15_CFGS" -T "$OUT/traces" \
    -N snake_synth_${STAGE}_fig15 -r "$OUT/runs" -n >/dev/null
[ -z "$DEP_CFGS" ] || python3 util/job_launching/run_simulations.py \
    -B snake-synth:snake-synth-strided-dep -C "$DEP_CFGS" -T "$OUT/traces" \
    -N snake_synth_${STAGE}_dep -r "$OUT/runs" -n >/dev/null
[ -z "$RAND_CFGS" ] || python3 util/job_launching/run_simulations.py \
    -B snake-synth:snake-synth-random -C "$RAND_CFGS" -T "$OUT/traces" \
    -N snake_synth_${STAGE}_rand -r "$OUT/runs" -n >/dev/null
find "$OUT/runs" -name justrun.sh -printf '%h\n' | sort | sed "s#^$OUT/runs/##"
echo "$(find "$OUT/runs" -name justrun.sh | wc -l) runs"

run_job() {  # <job dir>
  cd "$1" && LD_LIBRARY_PATH="$OUT/bin/lib" "$OUT/bin/accel-sim.out" \
      -config ./gpgpusim.config -trace ./traces/kernelslist.g > sim.out 2>&1
  echo "[done $(date '+%T')] rc=$? ${1#$OUT/}"
}
export -f run_job; export OUT
find "$OUT/runs" -name justrun.sh -printf '%h\n' | xargs -P "$PAR" -I{} bash -c 'run_job "{}"'

status=0
for c in ${CFGS//,/ }; do
  echo "######## $c"
  python3 -I util/snake/check_synthetic.py "$OUT/runs" "$OUT/traces" "$c" || status=1
done
for c in ${FIG15_CFGS//,/ }; do
  echo "######## $c (Fig. 15 only)"
  python3 -I util/snake/check_synthetic.py "$OUT/runs" "$OUT/traces" "$c" snake-synth-fig15 || status=1
done
for c in ${DEP_CFGS//,/ }; do
  echo "######## $c (trace (d) only)"
  python3 -I util/snake/check_synthetic.py "$OUT/runs" "$OUT/traces" "$c" snake-synth-strided-dep || status=1
done
for c in ${RAND_CFGS//,/ }; do
  echo "######## $c (trace (b) only)"
  python3 -I util/snake/check_synthetic.py "$OUT/runs" "$OUT/traces" "$c" snake-synth-random || status=1
done
exit $status
