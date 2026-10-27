# Fixing Snake at Tier 1 conditions (2026-10-27)

Every run is at Tier 1 conditions: PAPER_V100 (GTO, 1530 MHz), 1B instructions, our 7 apps (9 workloads), against Tier 1's no-prefetch runs (`sim_run_baseline`).

**Success criteria:**
- (a) histo's whole run ≥ 0 vs no prefetching
- (b) no workload more than 1% below no prefetching
- (c) 7-app whole-run GMEAN above Tier 1 Snake (−0.29%)
- (d) the timely share of used prefetches goes up

**Perfect-L1 ceilings (whole run, D1_ceilings.md):** histo +39%, srad_v1 +21%, nw +17%, backprop +42% (DRAM-bound), lud +3–4%, mri-q +2%, hotspot +0.3%.

## Phase 0: why histo's prescan and final kernels lose under SK_R0_D0 (no new runs)
Data:
- the existing 1B outputs: no prefetching (`sim_run_baseline`), Tier 1 Snake (`sim_run_snake`), SK_R0_D0 (`sim_run_snake_skm/v100p_1b`)
- every complete launch of each kernel (9 or 10 launches each)
- script: `util/snake/kernel_diag.py`

Notes on the columns:
- Latency means and p90s come from log2 histogram buckets, so they are approximate.
- The memory-fetch latency counts every read the SMs send, prefetches included.
- The simulator prints no per-PC prediction counts. Only the kernels' load PCs from the traces are listed.

| kernel (share of no-prefetch cycles) | config | IPC gain | prefetches issued / demand sector | issued: timely / late / evicted / unused at kernel end % | accuracy_sc | demand MISS_QUEUE_FULL / access (vs no prefetching) | L1 hit % | prefetch-induced evictions % | demand miss-queue wait mean / p90 | memory-fetch latency mean / p90 | DRAM reads vs no prefetching | issued by source % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `histo_prescan_kernel` (6.5%) | no prefetching | | | | | 0.7 | 50.5 | | | 478 / 1023 | 1.00 | |
| | Tier 1 Snake | −10.83% | 0.71 | 7 / 43 / 0 / 50 | n/a (older build) | 0.0 (×0.00) | 56.7 | n/a | 5.2 / 15 | 710 / 2047 | ×1.88 | inter-thread 66, inter-warp 27, promotion 7 |
| | SK_R0_D0 | **−46.50%** | **1.68** | 4 / 17 / 0 / **79** | **20.5** | 2.5 (×3.46) | 58.8 | n/a (no evictions) | 8.0 / 15 | **2324 / 8191** | **×4.08** | inter-thread 63, inter-warp 18, intra-warp 15, promotion 4 |
| `histo_final_kernel` (10.2%) | no prefetching | | | | | 99.2 | 0.5 | | | 368 / 1023 | 1.00 | |
| | Tier 1 Snake | −2.82% | 0.66 | 27 / 55 / 0 / 18 | n/a | 65.6 (×0.66) | 18.3 | n/a | 9.4 / 15 | 401 / 1023 | ×1.17 | inter-thread 58, inter-warp 34, promotion 8 |
| | SK_R0_D0 | **−23.79%** | **0.90** | 34 / 26 / 2 / **37** | 60.9 | 72.6 (×0.73) | 31.6 | **30.5** | 9.4 / 31 | 569 / 2047 | **×1.50** | inter-thread 39, **intra-warp 35**, inter-warp 18, promotion 7 |
| `histo_intermediates_kernel` (17.1%) | SK_R0_D0 | +42.17% | 0.57 | 17 / 81 / 1 / 0 | 97.5 | 64.4 (×1.38) | 11.3 | 0.1 | 9.3 / 31 | 391 / 1023 | ×0.73 | promotion 44, inter-warp 49 |
| `histo_main_kernel` (66.1%) | SK_R0_D0 | +0.07% | 0.06 | 63 / 24 / 4 / 8 | 88.0 | 0.6 (×1.71) | 73.6 | 3.6 | 4.6 / 15 | 322 / 511 | ×0.93 | inter-thread 62, inter-warp 24, intra-warp 12 |

Load PCs (L1-cached global loads, from the first launch's trace):
- **`histo_prescan_kernel`:** two PCs (`02a0`, `0420`), 4,096 warp loads each.
- **`histo_final_kernel`:** 17 PCs (`0f00` 6,272 warp loads, `01a0` 1,344, fifteen others 576 each).

The full table, with reservation fails and wrong issued sectors per source, is in `/home/razul/snake/phase0_histo.md`.

**Cause.** In both losing kernels SK_R0_D0 floods the memory system with prefetches that are never used, and the extra DRAM traffic raises memory latency for the demand loads.
- **The miss queue is not the bottleneck.**
  - The demand miss-queue wait stays at 8–9 cycles on average, about Tier 1 Snake's.
  - Demand MISS_QUEUE_FULL is a small share of `histo_prescan_kernel`'s accesses (2.5%), and below no prefetching in `histo_final_kernel` (×0.73).
- **`histo_prescan_kernel`:** SK_R0_D0 issues 1.68 prefetched sectors per demand sector, against 0.71 for Tier 1 Snake. 79% are never used before the kernel ends (Tier 1 Snake: 50%).
  - Most come from inter-thread chains, now up to depth 10 (63% of issued), plus intra-warp (15%).
  - DRAM reads rise 4.1×, and the mean memory-fetch latency rises from 478 to 2,324 cycles.
- **`histo_final_kernel`:** the new intra-warp prediction (8 iterations ahead) adds 35% of the issued sectors.
  - 37% of issued sectors are never used. DRAM reads rise 1.5×.
  - 30.5% of the demand data evicted from L1 is evicted by prefetches.
- **So R2 alone, which only separates the injection path, is not expected to fix these two kernels.** A usefulness filter that stops unproductive entries is (Phase 3; the condition for Phase 3 holds).

