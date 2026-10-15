# Final evaluation, Tier 1 (2026-10-15/16)

Build: accel-sim `2c3a333`, gpgpu-sim `3f635b67` (clean, `_modified_0.0` twice), frozen in every run dir's `bin/`; launch tooling from `e89c9d0`. QV100-SASS-PAPER_V100 (GTO, 1530 MHz), 1B-instruction cap, 9 workloads, 21 configs (results/S5.md §1). Per-job CSV: `results/final_tier1.csv` (`util/snake/collect_results.py`); tables: `util/snake/compare_paper.py` over the 8 Tier 1 run dirs.

## 0. Pre-flight (results/S5.md conditions)

- **Peak memory:** 1.85 GB for one short run (mri-q, Snake exact, 100M; `sim_run_snake_preflight`). Traces are read one thread block at a time, so the trace size does not add to it.
  - Across all 189 Tier 1 jobs (`time.txt`) the peak is **2.23 GB** (mri-q).
  - 15 jobs in parallel ≤ 33 GB, 54% of the 62 GB RAM, under the 80% limit.
- **Disk:** 160 GB free; the 189 jobs wrote < 1 GB besides traces.
- **`snake_debug_trace`:** off in all 306 Tier 1 + 2 configs.
- **Launch gate:**
  - both trees clean on `Snake` (checked by `s5_runs.sh run`)
  - `accel-sim.out` + `libcudart.so` snapshotted per run dir
  - no rebuild from the dry-run to the last job
- **Energy (AccelWattch):**
  - The base repo has QV100 SASS AccelWattch (`Accelwattch_SASS_SIM`).
  - A paired run of hotspot 512 (baseline, power off vs on, side by side) took 42.5 s vs 66.0 s: **+55%**, above the 30% limit.
  - No POWER configs were added, and Fig. 19 energy stays unavailable (SPEC §8).
  - That needed one power-off reference run beyond the one smoke run: 4 short pre-flight runs in total.

## 1. Canary: baseline and Snake exact × 9 workloads (18 jobs, 23:10–23:39)

- All 18 finished with rc 0; every build tag is `_modified_0.0`; no crash or stall; every invariant OK (`collect_results.py`).
- Wall time 3.93 core-hours vs 5.38 estimated (0.73×); per job 0.51–1.42× the estimate.
- **Results vs the 100M gates:** each run's first 100M instructions reproduce the gate exactly.
  - hotspot 512 (complete in both) is identical.
  - srad_v1 at 101M: coverage_paper 22.8%, coverage_pred 84.8%, as in S4b.
  - backprop's kernel 1: coverage_paper 43.7% (S3b, Snake-DT: 44.2%).
- **Full-run differences come from kernels the 100M gates never reached:**
  - backprop's `bpnn_adjust_weights` (770k demand sectors, ~4% coverage_paper, 267k sectors dropped with the queue full)
  - srad_v1's `srad2` (coverage_pred 85%, coverage_paper 2.2%)
  - Judged not off: Tier 1 continued.

## 2. Tier 1 completion

- **189 / 189 jobs** finished, rc 0, 0 failed checks (`collect_results.py --require-all`).
- **Unlimited Tail (1,024 entries), Tail evictions per workload** (Tail 10 in parentheses):
  - **backprop 107,498 (236,353)**; its `adjust_weights` kernel allocates ~2,370 entries per SM-launch ((222,266 − 32,848) / 80), so 1,024 is not unlimited there. Its kernel 1 evicts 0
  - **0 on the other 8 workloads.** At Tail 10 they evict: hotspot ×2 0, lud 512 26,464, lud 256 4,520, nw 112,712, histo 22,256, mri-q 95, srad_v1 378,132.
  - The 1,024-entry jobs ran at the same speed as the others (backprop 427 s, histo 3,738 s).

## 3. Ours vs the paper (Snake, exact footprint)

Averages: per app first (hotspot and lud have two inputs), then over the 7 apps. The paper's averages are recomputed over the same 7 apps. Coverage "pred" = coverage_pred, "paper-def" = coverage_paper; accuracy = accuracy_paper (SPEC OQ-14).

| app / input | coverage: paper / pred / paper-def | accuracy: paper / ours | IPC gain: paper / ours | demand L1 hit: baseline → Snake (ISO) | energy (norm.) |
|---|---|---|---|---|---|
| backprop / 65536 | 83 / 64.6 / 12.4 | 68 / 0.2 | +1% / -0.2% | 41.7 → 42.3 (42.3) | unavailable |
| hotspot / 1024_2_2___data_temp_102 | 88 / 93.3 / 69.1 | 85 / 27.0 | +13% / +0.2% | 0.7 → 27.9 (37.9) | unavailable |
| hotspot / 512_2_2___data_temp_512_ | 88 / 90.6 / 59.5 | 85 / 26.4 | +13% / +0.1% | 0.2 → 26.6 (32.5) | unavailable |
| lud / _i___data_512_dat | 91 / 82.2 / 28.4 | 82 / 2.7 | +18% / +0.1% | 4.2 → 6.8 (6.9) | unavailable |
| lud / _s_256__v | 91 / 74.9 / 33.4 | 82 / 0.7 | +18% / +0.1% | 0.0 → 0.7 (0.7) | unavailable |
| nw / 2048_10 | 12 / 0.0 / 0.0 | 10 / 0.0 | +5% / +0.0% | 1.5 → 1.5 (1.5) | unavailable |
| histo / _i___data_default_input_ | 97 / 34.4 / 23.6 | 96 / 10.8 | +33% / -1.9% | 59.1 → 69.3 (70.1) | unavailable |
| mri-q / _i___data_small_input_32 | 72 / 88.0 / 9.3 | 60 / 0.0 | +4% / +0.1% | 0.0 → 0.0 (0.0) | unavailable |
| srad_v1 / 100_0_5_502_458 | 87 / 84.8 / 12.5 | 83 / 3.5 | +29% / -0.2% | 25.7 → 32.0 (31.8) | unavailable |
| **Average** (7 apps) | 75.7 / 63.2 / 21.9 | 69.1 / 6.1 | +14.1% / -0.3% | 18.7 → 25.2 (26.4); paper 45 → 79 (84) | unavailable |

- **Snake (line footprint):** coverage_pred 48.1, coverage_paper 12.9, accuracy 3.1, IPC −0.3%.
- **PAPER_LAT** (L1 28 / L2 212 cycles), Snake vs PAPER_LAT baseline: 63.5 / 22.7 / 5.9, IPC −0.2%. Raising the latencies to the paper's does not change the picture.
- **Ablations (IPC, GMEAN):** Snake −0.3%, Snake-T −0.3%, Snake-DT −0.3%, s-Snake −0.1%, Snake-ISO −0.1%. Snake vs Snake-DT ±0.0% (paper +13%), vs Snake-T +0.0% (paper +7%).
- **Fig. 20:** coverage_pred 63.2 → 63.8 → 64.1 → 64.4 for Tail 10 → 20 → 40 → 1,024. Tail 10 costs 1.2 pp vs "unlimited" (paper: 8%).
- **Fig. 22:** popcount alone is 1.9–3.4 pp below lru_popcount in coverage_pred at every size.
- **Fig. 23:** coverage_paper falls from 22.2% (25 cycles) through 21.9 (50) to 17.6 (300); accuracy_paper from 6.1 to 4.3; IPC flat. Paper: 50 cycles gives 75% accuracy at a 2% coverage loss.

**Summary.** Snake predicts much of what the paper predicts (coverage_pred 63% vs the paper's 76% over these apps; hotspot, mri-q and srad_v1 at 85–93%). Few predictions become timely prefetches (accuracy 6% vs 69%), and IPC does not move (−0.3% vs +14%). The causes are those recorded at 100M (SPEC §8):
- prefetches issued late or not at all (miss-queue slots, queue full, one request per cycle)
- hotspot's prefetches used ~900 cycles after issue
- histo's unpredictable accesses
- throttling

## 4. Time: actual vs estimate

| | estimate (results/S5.md) | actual |
|---|---|---|
| core-hours (189 jobs) | 56.5 | **64.9** (1.15×) |
| wall time at 15 parallel | ≈ 3.8 h | **4 h 33 min** (canary 29 min + 171 jobs 4 h 04 min, 23:40–03:44) |
| peak RSS per job | — | 2.23 GB |

Per workload, mean wall time vs estimate:
- backprop 375 s (1.35×), hotspot 512 95 s (0.76×), hotspot 1024 458 s (0.92×)
- lud 512 591 s (1.19×), lud 256 125 s (0.93×)
- **nw 2,888 s (1.79×)**: nw slows most under 15 parallel jobs (1,575 s in the lighter canary)
- histo 3,427 s (1.03×), mri-q 1,123 s (0.86×), srad_v1 2,040 s (1.07×)

Rescaled by 1.15, Tier 2 is ≈ 40 core-hours, ≈ 2.7 h at 15 parallel.

## 5. Full compare_paper.py output

## 1. Ours vs the paper, per app (Snake, exact footprint, vs the baseline)

| app / input | paper coverage | coverage_pred | coverage_paper | paper accuracy | accuracy_paper | paper IPC gain | IPC gain | paper energy (norm.) | energy (norm.) | Snake's own energy (mJ) |
|---|---|---|---|---|---|---|---|---|---|---|
| backprop / 65536 | 83 | 64.6 | 12.4 | 68 | 0.2 | +1% | -0.2% | 0.92 | unavailable | 0.026 |
| hotspot / 1024_2_2___data_temp_102 | 88 | 93.3 | 69.1 | 85 | 27.0 | +13% | +0.2% | 0.86 | unavailable | 0.023 |
| hotspot / 512_2_2___data_temp_512_ | 88 | 90.6 | 59.5 | 85 | 26.4 | +13% | +0.1% | 0.86 | unavailable | 0.007 |
| lud / _i___data_512_dat | 91 | 82.2 | 28.4 | 82 | 2.7 | +18% | +0.1% | 0.82 | unavailable | 0.338 |
| lud / _s_256__v | 91 | 74.9 | 33.4 | 82 | 0.7 | +18% | +0.1% | 0.82 | unavailable | 0.158 |
| nw / 2048_10 | 12 | 0.0 | 0.0 | 10 | 0.0 | +5% | +0.0% | 0.95 | unavailable | 0.790 |
| histo / _i___data_default_input_ | 97 | 34.4 | 23.6 | 96 | 10.8 | +33% | -1.9% | 0.64 | unavailable | 0.531 |
| mri-q / _i___data_small_input_32 | 72 | 88.0 | 9.3 | 60 | 0.0 | +4% | +0.1% | 0.96 | unavailable | 0.082 |
| srad_v1 / 100_0_5_502_458 | 87 | 84.8 | 12.5 | 83 | 3.5 | +29% | -0.2% | 0.71 | unavailable | 0.253 |
| **Average** (7 apps) | 75.7 | 63.2 | 21.9 | 69.1 | 6.1 | +14.1% | -0.3% | 0.83 | unavailable | |
| **Snake, line footprint** (7 apps) |  | 48.1 | 12.9 |  | 3.1 |  | -0.3% |  | unavailable | |
| **PAPER_LAT: Snake vs PAPER_LAT baseline** (7 apps) |  | 63.5 | 22.7 |  | 5.9 |  | -0.2% |  | unavailable | |

## 2. Demand L1 hit rate (Fig. 25, §5.7; paper averages: baseline 45, Snake 79, Isolated-Snake 84)

| app / input | baseline | Snake | Snake-ISO | Snake-DT |
|---|---|---|---|---|
| backprop / 65536 | 41.7 | 42.3 | 42.3 | 42.3 |
| hotspot / 1024_2_2___data_temp_102 | 0.7 | 27.9 | 37.9 | 38.0 |
| hotspot / 512_2_2___data_temp_512_ | 0.2 | 26.6 | 32.5 | 32.3 |
| lud / _i___data_512_dat | 4.2 | 6.8 | 6.9 | 6.8 |
| lud / _s_256__v | 0.0 | 0.7 | 0.7 | 0.7 |
| nw / 2048_10 | 1.5 | 1.5 | 1.5 | 1.5 |
| histo / _i___data_default_input_ | 59.1 | 69.3 | 70.1 | 69.0 |
| mri-q / _i___data_small_input_32 | 0.0 | 0.0 | 0.0 | 0.0 |
| srad_v1 / 100_0_5_502_458 | 25.7 | 32.0 | 31.8 | 32.4 |
| **Average** (apps) | 18.7 | 25.2 | 26.4 | 26.3 |

## 3. Ablations: IPC gain vs the baseline (§5.2: Snake beats Snake-DT by 13% and Snake-T by 7%)

| app / input | Snake | Snake-T | Snake-DT | s-Snake | Snake-ISO |
|---|---|---|---|---|---|
| backprop / 65536 | -0.2% | -0.0% | -0.0% | +0.1% | -0.3% |
| hotspot / 1024_2_2___data_temp_102 | +0.2% | +0.1% | +0.1% | +0.2% | +0.1% |
| hotspot / 512_2_2___data_temp_512_ | +0.1% | +0.1% | +0.0% | -0.1% | +0.3% |
| lud / _i___data_512_dat | +0.1% | +0.1% | +0.0% | +0.1% | +0.1% |
| lud / _s_256__v | +0.1% | +0.1% | +0.1% | +0.1% | +0.1% |
| nw / 2048_10 | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| histo / _i___data_default_input_ | -1.9% | -2.4% | -2.0% | -0.4% | -0.8% |
| mri-q / _i___data_small_input_32 | +0.1% | +0.1% | +0.1% | -0.1% | +0.1% |
| srad_v1 / 100_0_5_502_458 | -0.2% | -0.0% | -0.0% | -0.4% | -0.1% |
| **GMEAN** (apps) | -0.3% | -0.3% | -0.3% | -0.1% | -0.1% |

Snake vs Snake-DT: -0.0%; vs Snake-T: +0.0% (paper +13%, +7%).

## 4a. Fig. 20: Tail entries (paper: Tail 10 costs 8% coverage vs unlimited)

| config | coverage_pred | coverage_paper | accuracy_paper | IPC gain (GMEAN) | demand L1 hit | apps |
|---|---|---|---|---|---|---|
| Tail 10 (default) | 63.2 | 21.9 | 6.1 | -0.3% | 25.2 | 7 |
| Tail 20 | 63.8 | 22.1 | 6.1 | -0.3% | 25.1 | 7 |
| Tail 40 | 64.1 | 22.2 | 6.1 | -0.3% | 25.2 | 7 |
| Tail unlimited (1024) | 64.4 | 22.2 | 6.1 | -0.4% | 25.1 | 7 |

## 4b. Fig. 22: Tail eviction policy x Tail entries

| config | coverage_pred | coverage_paper | accuracy_paper | IPC gain (GMEAN) | demand L1 hit | apps |
|---|---|---|---|---|---|---|
| lru_popcount, 10 | 63.2 | 21.9 | 6.1 | -0.3% | 25.2 | 7 |
| lru_popcount, 20 | 63.8 | 22.1 | 6.1 | -0.3% | 25.1 | 7 |
| lru_popcount, 40 | 64.1 | 22.2 | 6.1 | -0.3% | 25.2 | 7 |
| popcount, 10 | 59.8 | 20.5 | 6.1 | -0.4% | 25.1 | 7 |
| popcount, 20 | 61.4 | 20.9 | 6.1 | -0.2% | 25.2 | 7 |
| popcount, 40 | 62.2 | 21.0 | 6.1 | -0.3% | 25.1 | 7 |

## 4c. Fig. 23: space-throttle halt (paper: 50 cycles gives 75% accuracy at 2% coverage loss)

| config | coverage_pred | coverage_paper | accuracy_paper | IPC gain (GMEAN) | demand L1 hit | apps |
|---|---|---|---|---|---|---|
| 25 cycles | 63.2 | 22.2 | 6.1 | -0.3% | 25.2 | 7 |
| 50 cycles (default) | 63.2 | 21.9 | 6.1 | -0.3% | 25.2 | 7 |
| 75 cycles | 63.2 | 21.5 | 6.0 | -0.3% | 25.1 | 7 |
| 100 cycles | 63.2 | 21.2 | 5.9 | -0.3% | 24.9 | 7 |
| 150 cycles | 63.2 | 20.3 | 5.5 | -0.3% | 24.6 | 7 |
| 200 cycles | 63.2 | 19.4 | 5.1 | -0.2% | 24.1 | 7 |
| 300 cycles | 63.3 | 17.6 | 4.3 | -0.2% | 23.3 | 7 |

