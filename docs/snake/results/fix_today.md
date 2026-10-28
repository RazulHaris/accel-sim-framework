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

### Phase 0 addendum: which source floods, and the filter threshold
Used share per prediction source, by the mode a prefetch was issued under. Data: `snake_pf_issued_<mode>` and `snake_hist_lead_issue_<mode>_{timely,late,evicted}`, per kernel from the existing 1B runs. The funnel's per-mode classes follow a record's latest prediction mode instead, so they are not used here.

| kernel | config | source (mode at issue) | issued (share of all issued) | used = timely + late, % of the source's issued | timely % | late % | evicted unused % | unused at kernel end % |
|---|---|---|---|---|---|---|---|---|
| `histo_prescan_kernel` | Tier 1 Snake | inter_thread | 187082 (66%) | **32.2** | 8.7 | 23.5 | 0.0 | 67.8 |
| `histo_prescan_kernel` | Tier 1 Snake | intra_warp | 366 (0%) | **1.4** | 0.0 | 1.4 | 0.0 | 98.6 |
| `histo_prescan_kernel` | Tier 1 Snake | promotion | 18984 (7%) | **64.8** | 3.9 | 61.0 | 0.0 | 35.2 |
| `histo_prescan_kernel` | Tier 1 Snake | inter_warp | 75047 (27%) | **83.7** | 1.4 | 82.3 | 0.0 | 16.3 |
| `histo_final_kernel` | Tier 1 Snake | inter_thread | 355150 (58%) | **71.3** | 36.0 | 35.4 | 0.0 | 28.6 |
| `histo_final_kernel` | Tier 1 Snake | promotion | 49708 (8%) | **99.9** | 12.7 | 87.2 | 0.0 | 0.1 |
| `histo_final_kernel` | Tier 1 Snake | inter_warp | 204672 (34%) | **96.8** | 14.1 | 82.7 | 0.0 | 3.2 |
| `histo_prescan_kernel` | SK_R0_D0 | inter_thread | 417777 (63%) | **7.9** | 2.3 | 5.6 | 0.0 | 92.1 |
| `histo_prescan_kernel` | SK_R0_D0 | intra_warp | 99832 (15%) | **2.2** | 1.8 | 0.5 | 0.0 | 97.8 |
| `histo_prescan_kernel` | SK_R0_D0 | promotion | 27818 (4%) | **35.1** | 1.6 | 33.5 | 0.0 | 64.9 |
| `histo_prescan_kernel` | SK_R0_D0 | inter_warp | 117167 (18%) | **57.9** | 2.9 | 55.0 | 0.0 | 42.1 |
| `histo_final_kernel` | SK_R0_D0 | inter_thread | 327414 (39%) | **54.0** | 34.5 | 19.5 | 0.0 | 46.0 |
| `histo_final_kernel` | SK_R0_D0 | intra_warp | 290801 (35%) | **42.8** | 36.9 | 5.9 | 0.0 | 57.2 |
| `histo_final_kernel` | SK_R0_D0 | promotion | 60629 (7%) | **99.9** | 24.0 | 75.9 | 0.0 | 0.1 |
| `histo_final_kernel` | SK_R0_D0 | inter_warp | 152770 (18%) | **90.4** | 33.5 | 56.9 | 0.0 | 9.6 |

- **`histo_prescan_kernel`:** inter-thread chains (inter-PC) flood it.
  - Tier 1 Snake: 66% of issued, 32% used.
  - SK_R0_D0: 63% of issued, 8% used, with chains up to depth 10.
  - SK_R0_D0's intra-warp prediction adds 15% of issued at 2% used. Its promotion (35% used) and inter-warp (58%) also stay below 60%.
- **`histo_final_kernel`, SK_R0_D0:** intra-warp (distance 1..8: 35% of issued, 43% used) and inter-thread (39%, 54%). Tier 1 Snake's sources there are 71–100% used.
- **Inter-warp target distance:** the existing outputs do not record it.
  - The Phase 3/4 build adds `snake_iw_dist_<d>_issued` / `_used` (warp distance 1, 2–3, 4–7, … 64+), so the new runs report it.
  - With SK settings, inter-warp targets stay in the CTA, so the distances are at most a CTA's warps.
- **Threshold: 60%.**
  - **Why not 25%:** it triggers only on SK_R0_D0's prescan sources. It misses Tier 1 Snake's prescan inter-thread (32%) and both flood sources of the final kernel (43%, 54%).
  - **Above the floods:** 60% is above every flood source of both kernels (2–54%).
  - **Below the useful sources:** across all 9 workloads, the sources of the kernels that gain (hotspot, histo intermediates and main, srad_v1, mri-q, backprop) are 73–100% used (`/home/razul/snake/phase0_sources_all.md`).
  - **Side effect:** it also stops some of lud's sources (17–65% used), whose kernels move IPC by at most ±0.6%.
  - The option default stays 0.25. Token `SNAKE_ACCF` sets 0.60.

## Phase 1: R2, a separate prefetch-queue injection path (SPEC OQ-38)
Implemented as specified in `docs/snake/R2_SPEC.md`: gpgpu-sim `bb20e7aa`, tokens in accel-sim `c09021a`.
- **Options:**
  - `-snake_pq_path 1`: prefetch sectors that pass today's duplicate and resource checks wait in a per-L1D prefetch queue (`-snake_pq_size`, 128 sectors). They never enter the L1D miss queue, which carries demand only (no miss-queue reserve).
  - One request per cycle leaves the L1D in total (`baseline_cache::snake_pq_cycle`).
  - `-snake_pq_arb` 0: demand first. 1: the queues take turns when both hold a request.
  - `-snake_pq_mshr` 0: line and MSHR are taken when the prefetch is sent, after tags and MSHR are checked again; the sector counts as issued only then. 1: taken at enqueue, so a demand to the sector merges and is late.
- **Throttling and bandwidth meter:** throttling still applies, and sends count in the bandwidth meter.
- **Per-kernel stats** (cumulative, so per kernel by differences): `snake_pq_*` (pushed, full drops, sent, drops at send, blocked sends, occupancy, both-queues cycles, arbitration wins) and `snake_dist_pq_wait`.
- **Gates:**
  - Off-regression vs `bin_ref_v2`: PASS on 6 workloads (`sim_run_snake_regress/R2`).
  - Synthetic: PASS, 12 runs, with Fig. 15 also run with `SNAKE_R2` (`sim_run_snake_synth/R2b`). The first attempt (`R2`) launched nothing: a shell variable was lost, so the config names were empty.
  - Smoke tests (hotspot 512, 100M; `pq_mshr` 0 and 1; `pq_arb` 0 and 1): invariants OK, no prefetch enters the miss queue, and demand first wins every contested cycle. Round-robin splits them (20,771 : 18,772).
- **Provenance of the Phase 2 runs:** every job runs `sim_run_snake_fix/p2/bin/accel-sim.out` with `p2/bin/lib/libcudart.so` (checked in `/proc/<pid>/maps` for all running jobs). The library is a real copy, the binary has no RPATH, and queued jobs start through the same `run_job` with `BIN` fixed. So a rebuild cannot change them.

## Phase 3 and 4 code (gpgpu-sim `19e3d746`, tokens accel-sim `4b51495`; all default off)
- **Accuracy filter (`-snake_entry_acc_filter 1`, SPEC OQ-39, our extension):**
  - Per source (target load PC, mode), issued prefetches are counted in windows of 32, each when its outcome is known.
    - **Used:** first use, timely or late.
    - **Unused:** evicted unused, or resident unused 1,000 cycles after its fill.
  - A window under the threshold stops the source for 1,000 cycles, and its queued lines are dropped.
  - **Why outcomes, not issue time:** counting uses as they arrived, in windows of 32 issued, stopped good sources on hotspot 512 (smoke test: 301 stops, issue halved, though its sources are about 90% used). Uses arrive after a burst of issue has closed the window.
- **Single inter-warp target (`-snake_interwarp_single 1`, OQ-40):** only the D-th resident warp ahead, by global warp id among the SM's resident warps.
  - "Exactly D global warp ids ahead" almost never finds a resident warp, because consecutive CTAs go to different SMs (smoke test: 31 targets on hotspot 512).
- **CTA anchoring (`-snake_interwarp_cta_anchor 1`, OQ-40):** a target in another CTA takes the address its CTA's first warp last loaded at this PC, plus its warp index times the stride. It is skipped until that warp has run the PC.
- **Distance counters:** `snake_iw_dist_*`.
- **Build:** a first incremental build left stale objects after the header change (smoke runs aborted, one used 19.8 GB). A clean full build fixed it, and the gates ran on that build.

