# SK_SC: Snake as SnakeCharmer's baseline (2026-10-26)

SnakeCharmer (Tang et al., MICRO 2026, arXiv 2610.30990v1) builds on Snake and uses it as its baseline "SK". It publishes values for parameters the Snake paper leaves open. `SK_SC` is that configuration in our Snake; it is compared here with our current Snake (`SNAKE-SNAKE_FP1`) and with no prefetching, using SnakeCharmer's metrics.

- **Builds:** accel-sim `5d55de3`, gpgpu-sim `9622012e` (both `_modified_0.0`), frozen in each run dir. The PAPER_V100 no-prefetch runs of the 7 apps are the cached `bin_ref_v2` outputs (`sim_run_snake_refcache`), equal to Snake off by the off-regression.
- **Runs:** 100M instruction cap, `sim_run_snake_skc/{v100p_7apps, qv100_7apps, v100p_sc, qv100_sc, cta}` (87 runs), plus 17 gate runs. Tables: `util/snake/sk_sc_report.py sim_run_snake_skc sim_run_snake_refcache/bin_ref_v2/runs`.
- **SnakeCharmer was read on arXiv** (HTML v1); its PDF is not on disk. Interpretations: SPEC OQ-33…OQ-36.

## 1. Setup

### SK_SC (SPEC OQ-33)
| parameter | SnakeCharmer §III-A | SK_SC | our default |
|---|---|---|---|
| prefetch queue | "a single 128-entry prefetch queue" | `-snake_queue_size 128` | 32 |
| inter-PC (inter-thread) chain | maximum distance 10 | `-snake_max_chain_depth 10`; steps after the first need a promoted entry with the warp's bit (`-snake_chain_deep_promoted 1`, OQ-35) | 3; trained entry |
| intra-warp | maximum distance 8 | `-snake_intra_distance 8`, every iteration 1..8 (OQ-34) | next iteration |
| inter-warp | maximum distance 63; Fig. 8 prefetches for "lagging warps 2, 3, and 4" | resident warps with a larger global warp id, nearest first, up to 63 (`-snake_interwarp_scope 1 -snake_interwarp_degree 63`) | next 4 warps of the CTA |
| training confidence | "a prefetch is generated once the corresponding training confidence reaches 3" | 3 warps for T1, T2 and the inter-warp stride | same |

Everything else is our Snake: exact footprint, decoupling mode 1 (variant 1), throttling, lane mode 1. `SK_SC_CTA` is SK_SC with the inter-warp targets kept in the warp's CTA (up to 63).

### GPU setups
- **PAPER_V100:** the Snake paper's V100 (GTO, 1530 MHz), our main config.
- **Base QV100:** LRR, 1132 MHz. This is SnakeCharmer's V100 (Table I: 1132 MHz, round-robin, 80 SMs, 128 KB L1, 512 L1 MSHRs, 6 MB L2 in 64 slices).
- SnakeCharmer's main GPU is an A100 (1410 MHz, 108 SMs, 192 KB L1). It was not run (§2).

### Workloads
- **Our 7 apps** (9 workloads, as in every stage).
- **SnakeCharmer apps with local traces** (6 workloads, all sm_70 traces in `hw_run`):

| app (SnakeCharmer Table II input) | our trace | input | mis-decoded global memory lines after the parser fix |
|---|---|---|---|
| b+tree (1M keys, 2 commands) | rodinia-3.1 `mil.txt` + `command.txt` | same | 0 |
| bfs (1M nodes, 5,999,970 edges) | rodinia-3.1 `graph1MW_6` | same | **24.8%** (graph65536 24.4%, graph4096 25.0%) |
| particlefilter (128×128×10, 1000 particles) | rodinia-3.1 float and naive | same | 0 / 0 |
| heartwall (48.41 MiB AVI, 20 frames) | rodinia_2.0-ft `test.avi`, 1 frame | **differs** | 0 |
| streamcluster (65,536 points × 256 dims) | rodinia_2.0-ft, 1,024 points × 16 dims | **differs** | 2.3% |

  - The last column is from `util/snake/scan_legacy_traces.py`. bfs's lines are "base+delta wrong", a tracer bug the parser fix cannot repair (VERSION_DIFF "Pre-756ed33 traces"), so **bfs results are not reliable**. streamcluster's are base+stride gaps.
  - heartwall (6.9M instructions), streamcluster (20.1M), particlefilter_naive (26.9M), lud 256 (23.8M) and hotspot 512 (93.2M) finish before the 100M cap; the others stop at it.
- **Not available:** LonestarGPU mst and sssp, Polybench gemm, Tango AlexNet and ResNet. Table II's other four apps (nn, dotf, sto, fr) were not in scope.

## 2. Traces and build checks
- **No A100 (sm_80) traces exist on this machine.**
  - All 47 trace sets in `hw_run` are sm_70.
  - All `sim_run_oracle*` A100 runs (10 dirs) link their traces to `hw_run/rodinia-3.1/11.0`, i.e. V100 traces run on the A100 config.
  - `/home/razul/GPU/GhOST_yvr/GhOST/GhOST_SASS_traces.tar.gz` holds sm_75 (Turing) traces, except particlefilter_float (sm_70). Its sssp is Pannotia's.
- **Runs since 2026-10-22, by build tag:** none paired accel-sim from `scap` with gpgpu-sim `Snake`. The only runs are the 48 outputs of `sim_run_oracle_coalesced`: accel-sim `278cf01` with gpgpu-sim `8699c3a5`, both on `oracle-coalesced-miss` (worktree `/home/razul/oracle-wt`). The main tree's build was still `ac28769` / `7ed7810e` (Snake) from 2026-10-16.

## 3. Gates
- **Off-regression** vs `bin_ref_v2`: PASS on 6 workloads (`sim_run_snake_regress/SK_SC`, reference outputs from S2c).
- **Synthetic:** PASS (11 runs, `sim_run_snake_synth/SK_SC`). Every output is identical to S6 apart from the new lines (`snake_sc_*`, chain-depth histogram 9 and 10).
- **Default Snake unchanged:** the 5 PAPER_V100 Snake runs that S4b also ran (hotspot 512, lud 512, histo, mri-q, srad_v1) are identical to S4b. The only lines excluded are the new ones and `snake_pred_inter_warp_other_cta`, added in S6.
- **Option parsing:** the dump shows every SK_SC value, and `SK_SC_CTA`'s later `-snake_interwarp_scope 0` wins.
- **All 87 runs** finish, keep `pf_woken_warps` at 0, and pass the outcome and funnel invariants. Load first uses equal `pf_useful_timely` except for 0–56 first uses by stores per run.

## 4. Ours vs SnakeCharmer's SK
SnakeCharmer's numbers are A100-only: the AVG over its 14 apps (Fig. 12) and Fig. 4's timeliness, for full runs with a zero-compressed cache. Ours are arithmetic means over workloads (SnakeCharmer does not say how it averages), at 100M. nw issues no prefetch and is left out of accuracy and timeliness. Coverage is given as on-time only / with late uses (OQ-36).

| metric | SnakeCharmer SK (A100) | SK_SC, base QV100: their apps (6) | SK_SC, base QV100: all (15) | SK_SC, PAPER_V100: their apps (6) | SK_SC, PAPER_V100: all (15) |
|---|---|---|---|---|---|
| coverage | 11.5 | 16.4 / 27.2 | 8.6 / 21.4 | 21.7 / 31.1 | 17.0 / 29.1 |
| accuracy | 23.1 | 66.2 | 64.4 | 67.2 | 62.5 |
| on time | 48.3 | 38.8 | 22.8 | 41.7 | 34.1 |
| too early | 28.7 | 4.7 | 2.4 | 4.6 | 2.6 |
| too late | 23.0 | 56.5 | 74.8 | 53.7 | 63.4 |

- **Coverage** is of the same order (on-time only 8.6–21.7% vs 11.5%).
- **Accuracy** is about 3× SnakeCharmer's.
- **Timeliness is the opposite of theirs:** almost nothing is too early and most uses are too late. This is the S3c/S6 finding: inter-warp predictions come ~20 cycles before use.
- **Not tested, possible reasons:**
  - Our decoupled prefetch region (mode 1) protects prefetched data.
  - Our exact footprint fetches only the lanes' sectors.
  - Our "too early" counts only sectors demanded later in the same kernel.
  - Their A100 has a 192 KB L1 and a zero-compressed cache, and its runs go to completion.
  - The app sets differ: 5 of their 14 apps, two with other inputs, bfs partly mis-decoded.

## 5. IPC vs no prefetching (SnakeCharmer reports none for SK)
GMEAN over all 15 workloads:

| | PAPER_V100 | base QV100 |
|---|---|---|
| Snake | +4.30% | +3.71% |
| SK_SC | +1.44% | +2.96% |
| Snake, without particlefilter_naive | +0.92% | +0.30% |
| SK_SC, without particlefilter_naive | −2.59% | −1.01% |

- **particlefilter_naive** gains +65% (Snake) and +79% (SK_SC) on both setups. The whole app runs (26.9M instructions, 9 kernels, the same in all runs) at 5.9% occupancy, so memory latency is exposed. Cycles: 676,687 → 409,433 (Snake) → 378,075 (SK_SC), PAPER_V100. This is the latency-bound case S6 found missing in our 7 apps.
- **particlefilter_float** gains +8.3–8.8% with either.
- **SK_SC loses most on b+tree** (−17.8% PAPER_V100, −12.6% QV100; Snake +4.9% / −1.9%), histo (−6.4% / −8.7%; Snake −1.9% / −3.8%) and hotspot on PAPER_V100 (§6).
  - **b+tree:** inter-warp issue grows from 68k to 461k sectors, accuracy falls to 4.4%, too early reaches 27%. Throttled cycles grow 8× and MISS_QUEUE_FULL 2.4×.
  - **histo** (one CTA per SM, so not the scope): chains run to depth 10 (36k loads), intra-warp issues 50k sectors (11k wrong), wrong inter-thread prefetches grow 2.4×, and queue-full drops grow 5×.
  - **lud 512** (−2.3% / −1.2%): 3.05M inter-warp targets in other CTAs, L1 hit 8.2 → 5.6%.
- **Per-workload tables:** §9.

## 6. hotspot
| setup | config | IPC vs no prefetch (1024 / 512) | accuracy_sc | on time | demand L1 hit (1024) |
|---|---|---|---|---|---|
| PAPER_V100 | Snake | +0.13% / +0.13% | 82.5 / 85.6 | 39.2 / 42.4 | 25.1% |
| PAPER_V100 | SK_SC | **−11.03% / −6.25%** | 35.4 / 35.9 | 10.6 / 17.4 | 3.5% |
| PAPER_V100 | SK_SC_CTA | +0.15% / −0.02% | 79.9 / 84.7 | 41.3 / 45.9 | 26.9% |
| base QV100 | Snake | +0.06% / +0.79% | 98.2 / 96.5 | 1.5 / 1.6 | |
| base QV100 | SK_SC | +0.44% / +1.13% | 54.5 / 53.5 | 2.0 / 2.4 | |
| base QV100 | SK_SC_CTA | +0.26% / +2.42% | 97.7 / 95.0 | 1.6 / 2.1 | |

- **S6 reconciled.** In S6 (hotspot 512, 100M, degree 4), scope 1 gave +0.10% and scope 2 gave −4.73%, with accuracy_paper 0.0% and 0 timely inter-warp sectors. The README's "−4.7% (2D grid)" is scope 2.
- **Why SK_SC drops hotspot.** Scope 1 at degree 63 reaches many CTAs. hotspot 1024 gets 401k inter-warp targets in other CTAs (0 with SK_SC_CTA). On a 2-D grid their addresses are wrong, so on PAPER_V100:
  - demand reservation failures grow from 11.8k to 74.7k (33.9k caused by prefetches)
  - MISS_QUEUE_FULL doubles
  - the demand L1 hit rate falls from 25.1% to 3.5%
- **The drop exceeds 2%, so SK_SC_CTA ran** on hotspot and srad_v1 on both setups (6 runs). It restores hotspot (and on base QV100 gives hotspot 512 +2.42%, the best here). srad_v1: +0.02% (PAPER_V100), −1.33% (QV100), vs SK_SC +0.07% / −0.78%.
- **On base QV100 hotspot loses nothing:** under LRR almost every prefetch is late (on time ≤ 2.4%) in every config.

## 7. Does SK_SC improve timeliness on our 7 apps?
**Partly, and it does not turn into IPC.**

| 7 apps (9 workloads) | on time | too early | too late | coverage_sc on-time | accuracy_sc | IPC (GMEAN) |
|---|---|---|---|---|---|---|
| PAPER_V100 Snake | 21.5 | 1.2 | 77.2 | 7.8 | 85.2 | −0.08% |
| PAPER_V100 SK_SC | 28.3 | 1.0 | 70.7 | 13.8 | 58.9 | −2.95% |
| base QV100 Snake | 8.2 | 0.1 | 91.7 | 2.6 | 86.8 | −0.43% |
| base QV100 SK_SC | 10.8 | 0.6 | 88.6 | 3.3 | 63.0 | −1.12% |

- **The mean on-time share rises.** It comes mostly from backprop (PAPER_V100 2.1 → 92.5%; QV100 0 → 12.8%), srad_v1 (28 → 42%) and histo (47 → 54%). Reaching the warps of later CTAs gives the lead time that S6's scope 2 gave.
- **hotspot and lud get less timely on PAPER_V100**, and accuracy falls on every app except mri-q (100% in both).
- **backprop's on-time prefetches gain nothing:** −0.18% IPC with 92.5% on time. It is DRAM-bound (S6).

## 8. Limits
- V100 only; SnakeCharmer's main results are A100 (no sm_80 traces).
- 100M instructions, not full runs; 5 of SnakeCharmer's 14 apps, two with other inputs; bfs partly mis-decoded.
- SK_SC changes five things at once. Only the inter-warp scope was isolated (SK_SC_CTA), so the causes in §5 for b+tree and histo are observed mixes, not ablations.
- No zero-compressed cache and no AccelWattch.

## 9. Per-workload results
Columns: IPC vs no prefetching; coverage_sc (on-time only, with late); accuracy_sc; on time / too early / too late among used prefetches; prefetch-induced demand evictions as a share of victim sectors. "-" = nothing to divide (no prefetch, or no eviction). † input differs from SnakeCharmer Table II. ‡ trace partly mis-decoded after the parser fix (bfs 24.8%, streamcluster 2.3% of global memory lines).

### PAPER_V100 (GTO, 1530 MHz)
| workload | config | IPC vs no prefetch | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions |
|---|---|---|---|---|---|---|---|---|---|
| backprop | Snake | -0.10% | 0.7 | 31.6 | 100.0 | 2.1 | 0.0 | 97.9 | 0.0 |
| backprop | SK_SC | -0.18% | 80.5 | 87.0 | 97.7 | 92.5 | 0.0 | 7.5 | 0.0 |
| hotspot 1024 | Snake | +0.13% | 18.3 | 44.3 | 82.5 | 39.2 | 4.8 | 55.9 | 29.1 |
| hotspot 1024 | SK_SC | -11.03% | 2.6 | 24.3 | 35.4 | 10.6 | 2.1 | 87.3 | 22.3 |
| hotspot 1024 | SK_SC_CTA | +0.15% | 19.7 | 45.3 | 79.9 | 41.3 | 5.1 | 53.6 | 28.0 |
| hotspot 512 | Snake | +0.13% | 19.8 | 44.8 | 85.6 | 42.4 | 4.4 | 53.2 | 26.3 |
| hotspot 512 | SK_SC | -6.25% | 4.3 | 24.0 | 35.9 | 17.4 | 2.1 | 80.5 | 19.0 |
| hotspot 512 | SK_SC_CTA | -0.02% | 22.1 | 46.1 | 84.7 | 45.9 | 4.3 | 49.8 | 26.2 |
| lud 512 | Snake | +0.25% | 2.5 | 22.4 | 48.0 | 11.2 | 0.1 | 88.7 | 22.3 |
| lud 512 | SK_SC | -2.29% | 1.1 | 12.7 | 13.0 | 8.4 | 0.5 | 91.0 | 21.6 |
| lud 256 | Snake | +0.08% | 0.5 | 25.2 | 79.1 | 2.1 | 0.0 | 97.9 | - |
| lud 256 | SK_SC | -0.03% | 0.3 | 22.2 | 41.4 | 1.2 | 0.0 | 98.8 | - |
| nw | Snake | +0.00% | 0.0 | 0.0 | - | - | - | - | - |
| nw | SK_SC | +0.00% | 0.0 | 0.0 | - | - | - | - | - |
| histo | Snake | -1.85% | 20.5 | 43.1 | 87.8 | 47.2 | 0.6 | 52.2 | 20.1 |
| histo | SK_SC | -6.37% | 23.9 | 42.8 | 71.9 | 53.8 | 3.6 | 42.6 | 22.5 |
| mri-q | Snake | +0.12% | 0.0 | 8.6 | 100.0 | 0.0 | 0.0 | 100.0 | - |
| mri-q | SK_SC | +0.19% | 0.0 | 8.7 | 100.0 | 0.0 | 0.0 | 100.0 | - |
| srad_v1 | Snake | +0.56% | 8.2 | 29.2 | 98.5 | 28.0 | 0.0 | 72.0 | 3.7 |
| srad_v1 | SK_SC | +0.07% | 11.9 | 28.2 | 75.8 | 42.2 | 0.0 | 57.8 | 8.3 |
| srad_v1 | SK_SC_CTA | +0.02% | 10.3 | 29.8 | 95.9 | 34.6 | 0.0 | 65.4 | 7.8 |
| b+tree | Snake | +4.94% | 0.5 | 14.3 | 85.4 | 3.3 | 0.0 | 96.7 | 16.5 |
| b+tree | SK_SC | -17.84% | 0.9 | 4.8 | 4.4 | 14.1 | 27.1 | 58.8 | 10.4 |
| bfs ‡ | Snake | +1.22% | 0.0 | 47.4 | 99.3 | 0.0 | 0.0 | 100.0 | - |
| bfs ‡ | SK_SC | +1.76% | 24.0 | 60.5 | 99.3 | 39.6 | 0.0 | 60.4 | - |
| heartwall † | Snake | -0.17% | 0.0 | 1.6 | 8.6 | 1.9 | 0.0 | 98.1 | - |
| heartwall † | SK_SC | +0.23% | 0.0 | 2.3 | 9.9 | 1.5 | 0.0 | 98.5 | - |
| particlefilter_float | Snake | +8.25% | 24.3 | 29.8 | 90.8 | 81.4 | 0.3 | 18.3 | 19.8 |
| particlefilter_float | SK_SC | +8.47% | 39.5 | 44.1 | 91.6 | 89.2 | 0.4 | 10.4 | 25.5 |
| particlefilter_naive | Snake | +65.27% | 45.1 | 58.4 | 99.3 | 77.3 | 0.0 | 22.7 | - |
| particlefilter_naive | SK_SC | +78.98% | 65.4 | 74.0 | 98.0 | 88.4 | 0.0 | 11.6 | - |
| streamcluster † ‡ | Snake | -0.23% | 0.1 | 1.4 | 100.0 | 7.3 | 0.0 | 92.7 | 0.0 |
| streamcluster † ‡ | SK_SC | -0.17% | 0.2 | 1.1 | 100.0 | 17.6 | 0.0 | 82.4 | 0.0 |

| set | config | n | IPC (GMEAN) | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 apps | Snake | 9 | -0.08% | 7.8 | 27.7 | 85.2 | 21.5 | 1.2 | 77.2 | 16.9 |
| 7 apps | SK_SC | 9 | -2.95% | 13.8 | 27.8 | 58.9 | 28.3 | 1.0 | 70.7 | 15.6 |
| 7 apps | SK_SC_CTA | 3 | +0.05% | 17.4 | 40.4 | 86.8 | 40.6 | 3.1 | 56.3 | 20.7 |
| SnakeCharmer apps | Snake | 6 | +11.22% | 11.7 | 25.5 | 80.6 | 28.5 | 0.0 | 71.4 | 12.1 |
| SnakeCharmer apps | SK_SC | 6 | +8.42% | 21.7 | 31.1 | 67.2 | 41.7 | 4.6 | 53.7 | 12.0 |
| all | Snake | 15 | +4.30% | 9.4 | 26.8 | 83.2 | 24.5 | 0.7 | 74.7 | 15.3 |
| all | SK_SC | 15 | +1.44% | 17.0 | 29.1 | 62.5 | 34.1 | 2.6 | 63.4 | 14.4 |

### Base QV100 (LRR, 1132 MHz; SnakeCharmer's V100)
| workload | config | IPC vs no prefetch | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions |
|---|---|---|---|---|---|---|---|---|---|
| backprop | Snake | -0.25% | 0.0 | 21.5 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| backprop | SK_SC | -0.85% | 3.7 | 28.7 | 99.6 | 12.8 | 0.0 | 87.2 | 0.0 |
| hotspot 1024 | Snake | +0.06% | 0.4 | 27.1 | 98.2 | 1.5 | 0.0 | 98.5 | 29.3 |
| hotspot 1024 | SK_SC | +0.44% | 0.4 | 22.6 | 54.5 | 2.0 | 0.1 | 98.0 | 27.3 |
| hotspot 1024 | SK_SC_CTA | +0.26% | 0.4 | 27.2 | 97.7 | 1.6 | 0.0 | 98.4 | 27.6 |
| hotspot 512 | Snake | +0.79% | 0.4 | 26.0 | 96.5 | 1.6 | 0.0 | 98.4 | 27.8 |
| hotspot 512 | SK_SC | +1.13% | 0.5 | 20.5 | 53.5 | 2.4 | 0.2 | 97.5 | 27.0 |
| hotspot 512 | SK_SC_CTA | +2.42% | 0.5 | 25.8 | 95.0 | 2.1 | 0.0 | 97.9 | 25.6 |
| lud 512 | Snake | +0.02% | 1.5 | 14.8 | 44.0 | 10.1 | 0.1 | 89.8 | 29.9 |
| lud 512 | SK_SC | -1.20% | 0.6 | 7.8 | 10.2 | 7.5 | 0.8 | 91.7 | 29.6 |
| lud 256 | Snake | +0.03% | 0.6 | 22.9 | 72.3 | 2.6 | 0.0 | 97.4 | - |
| lud 256 | SK_SC | -0.01% | 0.4 | 21.0 | 39.3 | 1.8 | 0.0 | 98.2 | - |
| nw | Snake | +0.00% | 0.0 | 0.0 | - | - | - | - | - |
| nw | SK_SC | +0.00% | 0.0 | 0.0 | - | - | - | - | - |
| histo | Snake | -3.78% | 20.4 | 42.2 | 87.3 | 48.0 | 0.5 | 51.5 | 20.8 |
| histo | SK_SC | -8.66% | 23.6 | 42.1 | 70.7 | 53.9 | 3.9 | 42.2 | 22.4 |
| mri-q | Snake | +0.17% | 0.0 | 9.0 | 100.0 | 0.0 | 0.0 | 100.0 | - |
| mri-q | SK_SC | +0.20% | 0.0 | 9.0 | 100.0 | 0.0 | 0.0 | 100.0 | - |
| srad_v1 | Snake | -0.87% | 0.1 | 7.2 | 96.1 | 1.8 | 0.0 | 98.2 | 1.0 |
| srad_v1 | SK_SC | -0.78% | 0.4 | 6.7 | 76.7 | 5.8 | 0.0 | 94.2 | 2.0 |
| srad_v1 | SK_SC_CTA | -1.33% | 0.2 | 6.9 | 88.2 | 3.4 | 0.0 | 96.6 | 1.9 |
| b+tree | Snake | -1.87% | 0.3 | 12.8 | 76.4 | 2.7 | 0.0 | 97.3 | 21.5 |
| b+tree | SK_SC | -12.63% | 1.1 | 4.2 | 4.3 | 19.4 | 27.9 | 52.7 | 10.1 |
| bfs ‡ | Snake | +1.50% | 0.0 | 47.1 | 99.3 | 0.0 | 0.0 | 100.0 | - |
| bfs ‡ | SK_SC | +1.11% | 9.0 | 52.4 | 99.3 | 17.1 | 0.0 | 82.9 | - |
| heartwall † | Snake | +0.22% | 0.0 | 0.4 | 6.5 | 0.0 | 0.0 | 100.0 | - |
| heartwall † | SK_SC | -0.10% | 0.0 | 0.6 | 4.9 | 0.0 | 0.0 | 100.0 | - |
| particlefilter_float | Snake | +8.68% | 19.2 | 24.5 | 90.3 | 78.2 | 0.3 | 21.5 | 15.9 |
| particlefilter_float | SK_SC | +8.83% | 22.4 | 29.5 | 90.4 | 75.8 | 0.1 | 24.0 | 15.6 |
| particlefilter_naive | Snake | +65.60% | 45.2 | 58.4 | 99.3 | 77.3 | 0.0 | 22.7 | - |
| particlefilter_naive | SK_SC | +78.49% | 65.4 | 74.0 | 98.0 | 88.4 | 0.0 | 11.6 | - |
| streamcluster † ‡ | Snake | -0.02% | 0.9 | 2.4 | 100.0 | 36.7 | 0.0 | 63.3 | 0.0 |
| streamcluster † ‡ | SK_SC | -0.01% | 0.8 | 2.4 | 100.0 | 32.2 | 0.0 | 67.8 | 0.0 |

| set | config | n | IPC (GMEAN) | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 apps | Snake | 9 | -0.43% | 2.6 | 19.0 | 86.8 | 8.2 | 0.1 | 91.7 | 18.1 |
| 7 apps | SK_SC | 9 | -1.12% | 3.3 | 17.6 | 63.0 | 10.8 | 0.6 | 88.6 | 18.1 |
| 7 apps | SK_SC_CTA | 3 | +0.44% | 0.4 | 20.0 | 93.6 | 2.3 | 0.0 | 97.7 | 18.4 |
| SnakeCharmer apps | Snake | 6 | +10.25% | 10.9 | 24.3 | 78.6 | 32.5 | 0.1 | 67.5 | 12.5 |
| SnakeCharmer apps | SK_SC | 6 | +9.40% | 16.4 | 27.2 | 66.2 | 38.8 | 4.7 | 56.5 | 8.5 |
| all | Snake | 15 | +3.71% | 5.9 | 21.1 | 83.3 | 18.6 | 0.1 | 81.3 | 16.2 |
| all | SK_SC | 15 | +2.96% | 8.6 | 21.4 | 64.4 | 22.8 | 2.4 | 74.8 | 14.9 |

The SK_SC_CTA rows cover hotspot ×2 and srad_v1 only, so their means are not comparable with the 9-workload ones.

## 10. Reproduce
```bash
P=QV100-SASS-PAPER_V100-100M_INSN; Q=QV100-SASS-100M_INSN
util/snake/run_apps.sh sim_run_snake_skc/v100p_7apps current "$P-SNAKE-SNAKE_FP1,$P-SNAKE-SNAKE_FP1-SK_SC"
util/snake/run_apps.sh sim_run_snake_skc/qv100_7apps current "$Q,$Q-SNAKE-SNAKE_FP1,$Q-SNAKE-SNAKE_FP1-SK_SC"
SC=rodinia-3.1:b+tree-rodinia-3.1,rodinia-3.1:bfs-rodinia-3.1,rodinia-3.1:particlefilter_float-rodinia-3.1,rodinia-3.1:particlefilter_naive-rodinia-3.1,rodinia_2.0-ft:heartwall-rodinia-2.0-ft,rodinia_2.0-ft:streamcluster-rodinia-2.0-ft
ONLY='^(b\+tree|particlefilter|heartwall|streamcluster)|^bfs-rodinia-3.1/__data_graph1MW_6_txt/'
APPS=$SC ONLY=$ONLY util/snake/run_apps.sh sim_run_snake_skc/v100p_sc current "$P,$P-SNAKE-SNAKE_FP1,$P-SNAKE-SNAKE_FP1-SK_SC"
APPS=$SC ONLY=$ONLY util/snake/run_apps.sh sim_run_snake_skc/qv100_sc current "$Q,$Q-SNAKE-SNAKE_FP1,$Q-SNAKE-SNAKE_FP1-SK_SC"
APPS=rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1 \
  util/snake/run_apps.sh sim_run_snake_skc/cta sim_run_snake_skc/v100p_7apps/bin "$P-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA,$Q-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA"
python3 util/snake/sk_sc_report.py $PWD/sim_run_snake_skc $PWD/sim_run_snake_refcache/bin_ref_v2/runs
```
(The four main dirs were generated with `DRY=1` and run with `RESUME=1`; gate logs are in `/home/razul/snake/`: `off_regression_SK_SC.log`, `synthetic_SK_SC.log`, `legacy_scan_SK_SC.md`.)
