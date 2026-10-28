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

## Phase 2: R2 at Tier 1 conditions (27 runs) and Phase 3, first part (D, E: 18 runs)
- **Runs:**
  - `sim_run_snake_fix/p2`: A, B and C, build `c09021a` / `bb20e7aa`.
  - `sim_run_snake_fix/p3`: D and E, build `4b51495` / `19e3d746`.
  - All runs finished with 0 failed checks (outcome and funnel invariants, no warp woken by a prefetch, both build tags `_modified_0.0`).
  - Baseline: Tier 1's no-prefetch runs.
- **The session break.** A session break killed 6 Phase 2 jobs (histo and nw under A, B and C) and the P34 off-regression. The partial outputs are kept as `sim.out.killed`, and the jobs were rerun on the same pinned build (`RESUME=1`).
- **Phase 3/4 build gates:**
  - Off-regression P34b: PASS on 6 workloads.
  - Synthetic P34: PASS, 12 runs, with Fig. 15 also run with `SNAKE_ACCF`.
- **Configs:**
  - A: Tier 1 Snake + R2 (arbitration 0, MSHR at send)
  - B: SK_R0_D0 + R2 (same settings)
  - C: B with round-robin arbitration
  - D: Tier 1 Snake + filter (60%)
  - E: SK_R0_D0 + filter (60%)
- **Script:** `util/snake/fix_report.py`.

### Whole run (primary)
| workload | Tier 1 Snake | SK_R0_D0 | A | B | C | D | E | perfect-L1 ceiling |
|---|---|---|---|---|---|---|---|---|
| backprop | -0.20% | +0.16% | +0.03% | +0.24% | -0.19% | -0.20% | -0.05% | +41.7% |
| hotspot 1024 | +0.16% | +0.14% | +0.11% | +0.08% | +0.14% | +0.19% | +0.20% | +0.1% |
| hotspot 512 | +0.13% | -0.09% | -0.10% | +0.17% | +0.05% | +0.20% | +0.07% | +0.3% |
| lud 512 | +0.12% | -0.11% | -0.07% | -0.65% | -0.55% | +0.14% | -0.15% | +3.6% |
| lud 256 | +0.08% | +0.09% | +0.09% | +0.10% | +0.09% | +0.08% | +0.09% | +3.0% |
| nw | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +16.6% |
| histo | -1.93% | -3.44% | -4.96% | -3.25% | -3.13% | -1.77% | -1.88% | +39.2% |
| mri-q | +0.09% | +0.21% | +0.24% | -0.06% | +0.15% | +0.09% | +0.21% | +2.2% |
| srad_v1 | -0.21% | -0.53% | -1.57% | -0.10% | -0.60% | -1.35% | -0.56% | +20.6% |
| **GMEAN** (7 apps) | **-0.29%** | **-0.52%** | **-0.91%** | **-0.48%** | **-0.56%** | **-0.42%** | **-0.31%** | +16.6% |

### Success criteria
| config | (a) histo ≥ 0 | (b) lowest workload (≥ −1%) | (c) GMEAN > Tier 1 Snake | (d) timely share of used: pooled / mean over workloads | all four |
|---|---|---|---|---|---|
| Tier 1 Snake | -1.93% no | -1.93% (histo) no | -0.29% - | 39.1% / 21.4% - | no |
| SK_R0_D0 | -3.44% no | -3.44% (histo) no | -0.52% no | 38.5% / 24.5% no | no |
| A | -4.96% no | -4.96% (histo) no | -0.91% no | 34.5% / 22.2% no | no |
| B | -3.25% no | -3.25% (histo) no | -0.48% no | 40.2% / 26.1% yes | no |
| C | -3.13% no | -3.13% (histo) no | -0.56% no | 38.1% / 24.2% no | no |
| D | -1.77% no | -1.77% (histo) no | -0.42% no | 38.3% / 19.7% no | no |
| E | -1.88% no | -1.88% (histo) no | -0.31% no | 36.6% / 23.9% no | no |

The timely share is pooled over the 9 workloads (`pf_useful_timely / (timely + late)`; Tier 1's build has no `snake_sc_*`). The mean over workloads is also given.

### Per kernel: histo
Columns: IPC gain / prefetches issued per demand sector / unused share of issued / DRAM reads vs no prefetching / R2 prefetch-queue wait / arbitration wins demand:prefetch.

| histo kernel (share) | config | IPC gain | issued / demand | unused % | DRAM × | pq wait | wins d:p |
|---|---|---|---|---|---|---|---|
| `histo_main_kernel` (66.1%) | Tier 1 Snake | -3.88% | 0.20 | 1 | 1.08 | - | - |
| `histo_main_kernel` (66.1%) | SK_R0_D0 | +0.07% | 0.06 | 8 | 0.93 | - | - |
| `histo_main_kernel` (66.1%) | A | -8.43% | 0.24 | 0 | 1.08 | 31.7 | 223228:0 |
| `histo_main_kernel` (66.1%) | B | -0.28% | 0.07 | 9 | 0.92 | 36.5 | 112355:0 |
| `histo_main_kernel` (66.1%) | C | -0.16% | 0.07 | 9 | 0.92 | 36.0 | 97407:87235 |
| `histo_main_kernel` (66.1%) | D | -3.31% | 0.19 | 1 | 1.07 | - | - |
| `histo_main_kernel` (66.1%) | E | +0.39% | 0.06 | 5 | 1.02 | - | - |
| `histo_intermediates_kernel` (17.1%) | Tier 1 Snake | +11.48% | 0.52 | 0 | 0.91 | - | - |
| `histo_intermediates_kernel` (17.1%) | SK_R0_D0 | +42.17% | 0.57 | 0 | 0.73 | - | - |
| `histo_intermediates_kernel` (17.1%) | A | +11.97% | 0.64 | 0 | 0.92 | 68.1 | 638404:0 |
| `histo_intermediates_kernel` (17.1%) | B | +43.32% | 0.62 | 1 | 0.73 | 71.5 | 714583:0 |
| `histo_intermediates_kernel` (17.1%) | C | +42.23% | 0.66 | 1 | 0.73 | 52.5 | 448533:414322 |
| `histo_intermediates_kernel` (17.1%) | D | +11.21% | 0.52 | 0 | 0.92 | - | - |
| `histo_intermediates_kernel` (17.1%) | E | +31.85% | 0.59 | -0 | 0.72 | - | - |
| `histo_final_kernel` (10.2%) | Tier 1 Snake | -2.82% | 0.66 | 18 | 1.17 | - | - |
| `histo_final_kernel` (10.2%) | SK_R0_D0 | -23.79% | 0.90 | 37 | 1.50 | - | - |
| `histo_final_kernel` (10.2%) | A | -2.97% | 0.82 | 24 | 1.27 | 50.0 | 705846:0 |
| `histo_final_kernel` (10.2%) | B | -20.44% | 1.04 | 44 | 1.55 | 81.2 | 700571:0 |
| `histo_final_kernel` (10.2%) | C | -20.14% | 1.07 | 42 | 1.54 | 61.7 | 362932:331037 |
| `histo_final_kernel` (10.2%) | D | -4.74% | 0.64 | 15 | 1.14 | - | - |
| `histo_final_kernel` (10.2%) | E | -17.59% | 0.74 | 30 | 1.30 | - | - |
| `histo_prescan_kernel` (6.5%) | Tier 1 Snake | -10.83% | 0.71 | 50 | 1.88 | - | - |
| `histo_prescan_kernel` (6.5%) | SK_R0_D0 | -46.50% | 1.68 | 79 | 4.08 | - | - |
| `histo_prescan_kernel` (6.5%) | A | -9.88% | 0.72 | 50 | 1.87 | 9.9 | 14935:0 |
| `histo_prescan_kernel` (6.5%) | B | -47.42% | 1.86 | 84 | 4.45 | 42.0 | 26072:0 |
| `histo_prescan_kernel` (6.5%) | C | -47.04% | 1.86 | 84 | 4.46 | 41.0 | 24270:17171 |
| `histo_prescan_kernel` (6.5%) | D | -10.86% | 0.71 | 50 | 1.87 | - | - |
| `histo_prescan_kernel` (6.5%) | E | -39.48% | 1.52 | 77 | 3.83 | - | - |

### Per kernel: srad_v1
| srad_v1 kernel (share) | config | IPC gain | issued / demand | unused % | DRAM × | pq wait | wins d:p |
|---|---|---|---|---|---|---|---|
| `reduce` (46.2%) | Tier 1 Snake | +0.19% | 0.32 | 0 | 7.20 | - | - |
| `reduce` (46.2%) | SK_R0_D0 | +0.44% | 0.34 | 1 | 15.20 | - | - |
| `reduce` (46.2%) | A | +0.17% | 0.31 | 0 | 15.00 | 15.1 | 379496:0 |
| `reduce` (46.2%) | B | +0.23% | 0.29 | 2 | 37.70 | 48.6 | 389356:0 |
| `reduce` (46.2%) | C | +0.15% | 0.31 | 2 | 42.10 | 49.8 | 65673:41770 |
| `reduce` (46.2%) | D | +0.20% | 0.32 | 0 | 7.10 | - | - |
| `reduce` (46.2%) | E | +0.25% | 0.34 | 1 | 15.60 | - | - |
| `srad` (23.1%) | Tier 1 Snake | -1.20% | 0.19 | 3 | 1.03 | - | - |
| `srad` (23.1%) | SK_R0_D0 | -2.12% | 0.21 | 3 | 1.17 | - | - |
| `srad` (23.1%) | A | -6.68% | 0.19 | 3 | 1.03 | 14.4 | 272244:0 |
| `srad` (23.1%) | B | +0.14% | 0.21 | 3 | 0.93 | 28.4 | 243268:0 |
| `srad` (23.1%) | C | -2.10% | 0.21 | 3 | 0.98 | 20.3 | 115644:105228 |
| `srad` (23.1%) | D | -5.74% | 0.18 | 3 | 1.05 | - | - |
| `srad` (23.1%) | E | -2.14% | 0.21 | 2 | 1.01 | - | - |
| `srad2` (17.3%) | Tier 1 Snake | -0.60% | 0.02 | 6 | 2.14 | - | - |
| `srad2` (17.3%) | SK_R0_D0 | -1.65% | 0.02 | 18 | 3.71 | - | - |
| `srad2` (17.3%) | A | -0.57% | 0.02 | 7 | 2.19 | 46.5 | 313785:0 |
| `srad2` (17.3%) | B | -1.59% | 0.02 | 27 | 5.95 | 82.8 | 338508:0 |
| `srad2` (17.3%) | C | -2.07% | 0.03 | 21 | 3.45 | 38.2 | 57047:47402 |
| `srad2` (17.3%) | D | -0.75% | 0.02 | 6 | 1.95 | - | - |
| `srad2` (17.3%) | E | -1.38% | 0.02 | 18 | 3.67 | - | - |
| `prepare` (12.1%) | Tier 1 Snake | +0.60% | 0.15 | 0 | - | - | - |
| `prepare` (12.1%) | SK_R0_D0 | +0.17% | 0.20 | 0 | - | - | - |
| `prepare` (12.1%) | A | +0.52% | 0.08 | 0 | - | 14.2 | 543462:0 |
| `prepare` (12.1%) | B | +0.20% | 0.08 | 1 | - | 14.4 | 546303:0 |
| `prepare` (12.1%) | C | +1.46% | 0.12 | 1 | - | 25.3 | 118938:23255 |
| `prepare` (12.1%) | D | +0.44% | 0.15 | 0 | - | - | - |
| `prepare` (12.1%) | E | +0.36% | 0.20 | 0 | - | - | - |
| `extract` (1.3%) | Tier 1 Snake | +2.56% | 0.51 | 0 | 1.00 | - | - |
| `extract` (1.3%) | SK_R0_D0 | +3.00% | 0.65 | 0 | 1.00 | - | - |
| `extract` (1.3%) | A | +2.21% | 0.28 | 0 | 1.00 | 20.0 | 22389:0 |
| `extract` (1.3%) | B | +2.32% | 0.28 | 1 | 1.00 | 33.5 | 22344:0 |
| `extract` (1.3%) | C | +1.78% | 0.46 | 0 | 1.00 | 31.0 | 6772:5193 |
| `extract` (1.3%) | D | +2.56% | 0.51 | 0 | 1.00 | - | - |
| `extract` (1.3%) | E | +3.00% | 0.65 | 0 | 1.00 | - | - |

### Reading
- **R2 does not fix histo** (Phase 0 predicted this).
  - B and C leave `histo_prescan_kernel` (−47%) and `histo_final_kernel` (−20%) as they were, and histo's whole run stays at −3.1 to −3.3%.
  - On Tier 1 Snake (A), demand-first injection makes `histo_main_kernel` lose more (−8.4% vs −3.9%), and srad's `srad` kernel too (−6.7% vs −1.2%). A has the worst GMEAN (−0.91%).
  - On SK_R0_D0, R2 helps `srad` (−2.1% → +0.1%), and B raises the pooled timely share above Tier 1 Snake's (40.2% vs 39.1%, criterion (d)). Its GMEAN (−0.48%) stays below Tier 1 Snake's.
- **The 60% filter helps histo, but its 1,000-cycle stop is too short.**
  - E: histo −1.88% (SK_R0_D0 −3.44%), GMEAN −0.31% (best so far; Tier 1 Snake −0.29%). D: histo −1.77%.
  - In `histo_prescan_kernel` the filter stops sources 16.5 times per launch per SM, each for 1,000 cycles, in a 9,700-cycle launch. Between stops the flood resumes until the next window closes (up to 1,000 cycles of aging). Prescan keeps 77% of its prefetches unused (SK_R0_D0: 79%).
  - E also costs `histo_intermediates_kernel` 10 points (+42.2% → +31.9%).
  - D cannot reach histo ≥ 0. Tier 1 Snake's `histo_main_kernel` loss (−3.3% with the filter, 66% of the cycles) comes with only 1% unused prefetches, so a usefulness filter cannot remove it.
- **Inter-warp target distance (E, new counters):** the used share falls with distance.
  - Whole histo run: 94% (distance 1), 89% (2–3), 78% (4–7), 69% (8–15), 42% (16–31).
  - `histo_prescan_kernel`: 96 / 70 / 57 / 61 / 0%.
  - `histo_final_kernel`: 91 / 84 / 66 / 57% (distances 1 to 8–15).
- **None of A–E meets (a) or (b):** histo stays below −1% in every config.

## Phase 3, second part (F, E10: 18 runs, `sim_run_snake_fix/p3b`)
- **Configs:**
  - F: B + filter (SK_R0_D0 + R2 + filter at 60%), as requested.
  - E10: E with a 10,000-cycle stop (`SNAKE_ACCFB10K`, about one `histo_prescan_kernel` launch). A parameter-only variant of OQ-39 aimed at the flood E left.
- **Build:** `4b51495` / `19e3d746`. All 18 runs finished with 0 failed checks.

### Whole run, all configs so far
| workload | Tier 1 Snake | SK_R0_D0 | A | B | C | D | E | F | E10 | perfect-L1 ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| backprop | -0.20% | +0.16% | +0.03% | +0.24% | -0.19% | -0.20% | -0.05% | +0.15% | -0.10% | +41.7% |
| hotspot 1024 | +0.16% | +0.14% | +0.11% | +0.08% | +0.14% | +0.19% | +0.20% | +0.11% | +0.30% | +0.1% |
| hotspot 512 | +0.13% | -0.09% | -0.10% | +0.17% | +0.05% | +0.20% | +0.07% | +0.24% | +0.07% | +0.3% |
| lud 512 | +0.12% | -0.11% | -0.07% | -0.65% | -0.55% | +0.14% | -0.15% | -0.43% | -0.13% | +3.6% |
| lud 256 | +0.08% | +0.09% | +0.09% | +0.10% | +0.09% | +0.08% | +0.09% | +0.10% | +0.09% | +3.0% |
| nw | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +16.6% |
| histo | -1.93% | -3.44% | -4.96% | -3.25% | -3.13% | -1.77% | -1.88% | -1.80% | -1.64% | +39.2% |
| mri-q | +0.09% | +0.21% | +0.24% | -0.06% | +0.15% | +0.09% | +0.21% | -0.06% | +0.21% | +2.2% |
| srad_v1 | -0.21% | -0.53% | -1.57% | -0.10% | -0.60% | -1.35% | -0.56% | -0.14% | -0.46% | +20.6% |
| **GMEAN** (7 apps) | **-0.29%** | **-0.52%** | **-0.91%** | **-0.48%** | **-0.56%** | **-0.42%** | **-0.31%** | **-0.26%** | **-0.26%** | +16.6% |

### Success criteria
| config | (a) histo ≥ 0 | (b) lowest workload (≥ −1%) | (c) GMEAN > Tier 1 Snake | (d) timely share of used: pooled / mean over workloads | all four |
|---|---|---|---|---|---|
| Tier 1 Snake | -1.93% no | -1.93% (histo) no | -0.29% - | 39.1% / 21.4% - | no |
| SK_R0_D0 | -3.44% no | -3.44% (histo) no | -0.52% no | 38.5% / 24.5% no | no |
| A | -4.96% no | -4.96% (histo) no | -0.91% no | 34.5% / 22.2% no | no |
| B | -3.25% no | -3.25% (histo) no | -0.48% no | 40.2% / 26.1% yes | no |
| C | -3.13% no | -3.13% (histo) no | -0.56% no | 38.1% / 24.2% no | no |
| D | -1.77% no | -1.77% (histo) no | -0.42% no | 38.3% / 19.7% no | no |
| E | -1.88% no | -1.88% (histo) no | -0.31% no | 36.6% / 23.9% no | no |
| F | -1.80% no | -1.80% (histo) no | -0.26% yes | 39.2% / 25.4% yes | no |
| E10 | -1.64% no | -1.64% (histo) no | -0.26% yes | 34.8% / 22.8% no | no |

### histo per kernel, F and E10 (columns as above)
| histo kernel (share) | config | IPC gain | issued / demand | unused % | DRAM × | pq wait | wins d:p |
|---|---|---|---|---|---|---|---|
| `histo_main_kernel` (66.1%) | F | +0.05% | 0.07 | 6 | 0.97 | 39.3 | 109524:0 |
| `histo_main_kernel` (66.1%) | E10 | +0.33% | 0.05 | 5 | 1.03 | - | - |
| `histo_intermediates_kernel` (17.1%) | F | +32.11% | 0.62 | 0 | 0.71 | 70.4 | 694090:0 |
| `histo_intermediates_kernel` (17.1%) | E10 | +28.26% | 0.59 | 0 | 0.72 | - | - |
| `histo_final_kernel` (10.2%) | F | -17.42% | 0.94 | 41 | 1.43 | 90.3 | 709501:0 |
| `histo_final_kernel` (10.2%) | E10 | -15.51% | 0.65 | 25 | 1.21 | - | - |
| `histo_prescan_kernel` (6.5%) | F | -38.00% | 1.57 | 78 | 3.95 | 39.7 | 18735:0 |
| `histo_prescan_kernel` (6.5%) | E10 | -37.47% | 1.51 | 77 | 3.81 | - | - |

### Reading
- **GMEAN:** F (−0.264%) and E10 (−0.262%) are the first configs above Tier 1 Snake (−0.288%), meeting criterion (c). F also meets (d) (timely share 39.2% vs 39.1%).
- **histo** stays below −1% (F −1.80%, E10 −1.64%), so (a) and (b) fail for every config.
- **The longer stop does not cure `histo_prescan_kernel`** (E10: −37.5%, 77% unused, 3.8× DRAM reads). The filter stops sources often there, but other sources keep issuing. `histo_final_kernel` improves (−15.5% vs −17.6%).
- **Phase 4 condition holds:** the pooled late share is still about 61%. Phase 4 runs on E10 (best GMEAN so far, and best histo).

## Phase 4: one inter-warp target further ahead, with CTA anchoring (27 runs, `sim_run_snake_fix/p4`)
- **Why it ran:** the late share was still about 61% after Phase 3, above the 50% condition.
- **Configs:** on top of E10 (best GMEAN and best histo after Phase 3), `SNAKE_IWD<D>` for D = 8, 16, 32 (OQ-40).
  - Scope 1 (targets in any resident CTA).
  - Only the D-th resident warp ahead by global warp id.
  - A target in another CTA based on that CTA's first warp at the PC (CTA anchoring).
- **Build:** `4b51495` / `19e3d746`. All 27 runs finished with 0 failed checks.

### Whole run, all configs
| workload | Tier 1 Snake | SK_R0_D0 | A | B | C | D | E | F | E10 | G8 | G16 | G32 | perfect-L1 ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| backprop | -0.20% | +0.16% | +0.03% | +0.24% | -0.19% | -0.20% | -0.05% | +0.15% | -0.10% | +0.17% | -0.22% | +0.24% | +41.7% |
| hotspot 1024 | +0.16% | +0.14% | +0.11% | +0.08% | +0.14% | +0.19% | +0.20% | +0.11% | +0.30% | +0.17% | +0.13% | +0.10% | +0.1% |
| hotspot 512 | +0.13% | -0.09% | -0.10% | +0.17% | +0.05% | +0.20% | +0.07% | +0.24% | +0.07% | +0.17% | +0.33% | +0.09% | +0.3% |
| lud 512 | +0.12% | -0.11% | -0.07% | -0.65% | -0.55% | +0.14% | -0.15% | -0.43% | -0.13% | -0.17% | -0.23% | -0.23% | +3.6% |
| lud 256 | +0.08% | +0.09% | +0.09% | +0.10% | +0.09% | +0.08% | +0.09% | +0.10% | +0.09% | +0.10% | +0.10% | +0.10% | +3.0% |
| nw | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +16.6% |
| histo | -1.93% | -3.44% | -4.96% | -3.25% | -3.13% | -1.77% | -1.88% | -1.80% | -1.64% | -1.96% | -1.88% | -2.16% | +39.2% |
| mri-q | +0.09% | +0.21% | +0.24% | -0.06% | +0.15% | +0.09% | +0.21% | -0.06% | +0.21% | +0.12% | +0.01% | +0.01% | +2.2% |
| srad_v1 | -0.21% | -0.53% | -1.57% | -0.10% | -0.60% | -1.35% | -0.56% | -0.14% | -0.46% | -0.11% | -0.12% | -0.74% | +20.6% |
| **GMEAN** (7 apps) | **-0.29%** | **-0.52%** | **-0.91%** | **-0.48%** | **-0.56%** | **-0.42%** | **-0.31%** | **-0.26%** | **-0.26%** | **-0.24%** | **-0.29%** | **-0.38%** | +16.6% |

### Success criteria
| config | (a) histo ≥ 0 | (b) lowest workload (≥ −1%) | (c) GMEAN > Tier 1 Snake | (d) timely share of used: pooled / mean over workloads | all four |
|---|---|---|---|---|---|
| Tier 1 Snake | -1.93% no | -1.93% (histo) no | -0.29% - | 39.1% / 21.4% - | no |
| SK_R0_D0 | -3.44% no | -3.44% (histo) no | -0.52% no | 38.5% / 24.5% no | no |
| A | -4.96% no | -4.96% (histo) no | -0.91% no | 34.5% / 22.2% no | no |
| B | -3.25% no | -3.25% (histo) no | -0.48% no | 40.2% / 26.1% yes | no |
| C | -3.13% no | -3.13% (histo) no | -0.56% no | 38.1% / 24.2% no | no |
| D | -1.77% no | -1.77% (histo) no | -0.42% no | 38.3% / 19.7% no | no |
| E | -1.88% no | -1.88% (histo) no | -0.31% no | 36.6% / 23.9% no | no |
| F | -1.80% no | -1.80% (histo) no | -0.26% yes | 39.2% / 25.4% yes | no |
| E10 | -1.64% no | -1.64% (histo) no | -0.26% yes | 34.8% / 22.8% no | no |
| G8 | -1.96% no | -1.96% (histo) no | -0.24% yes | 39.8% / 20.4% yes | no |
| G16 | -1.88% no | -1.88% (histo) no | -0.29% no | 34.0% / 14.4% no | no |
| G32 | -2.16% no | -2.16% (histo) no | -0.38% no | 31.1% / 8.5% no | no |

### histo per kernel, E10 and Phase 4
| histo kernel (share) | config | IPC gain | issued / demand | unused % | DRAM × | pq wait | wins d:p |
|---|---|---|---|---|---|---|---|
| `histo_main_kernel` (66.1%) | E10 | +0.33% | 0.05 | 5 | 1.03 | - | - |
| `histo_main_kernel` (66.1%) | G8 | +0.60% | 0.04 | 5 | 0.99 | - | - |
| `histo_main_kernel` (66.1%) | G16 | +0.71% | 0.04 | 6 | 0.99 | - | - |
| `histo_main_kernel` (66.1%) | G32 | +0.03% | 0.03 | 8 | 0.95 | - | - |
| `histo_intermediates_kernel` (17.1%) | E10 | +28.26% | 0.59 | 0 | 0.72 | - | - |
| `histo_intermediates_kernel` (17.1%) | G8 | +23.59% | 0.45 | 0 | 0.74 | - | - |
| `histo_intermediates_kernel` (17.1%) | G16 | +20.68% | 0.38 | 0 | 0.74 | - | - |
| `histo_intermediates_kernel` (17.1%) | G32 | +20.95% | 0.38 | 0 | 0.75 | - | - |
| `histo_final_kernel` (10.2%) | E10 | -15.51% | 0.65 | 25 | 1.21 | - | - |
| `histo_final_kernel` (10.2%) | G8 | -17.64% | 0.55 | 28 | 1.20 | - | - |
| `histo_final_kernel` (10.2%) | G16 | -16.41% | 0.48 | 26 | 1.15 | - | - |
| `histo_final_kernel` (10.2%) | G32 | -15.68% | 0.48 | 26 | 1.15 | - | - |
| `histo_prescan_kernel` (6.5%) | E10 | -37.47% | 1.51 | 77 | 3.81 | - | - |
| `histo_prescan_kernel` (6.5%) | G8 | -35.44% | 1.33 | 81 | 3.61 | - | - |
| `histo_prescan_kernel` (6.5%) | G16 | -34.34% | 1.24 | 85 | 3.55 | - | - |
| `histo_prescan_kernel` (6.5%) | G32 | -34.24% | 1.24 | 85 | 3.55 | - | - |

### Reading
- **G8 has the best GMEAN of all configs:** −0.238% vs Tier 1 Snake's −0.288%. It meets (c), and (d) with a pooled timely share of 39.8% vs 39.1%.
- **G16 and G32 lose.** One target 16 or 32 warps ahead is used less often, and the mean timely share over workloads falls (14.4%, 8.5%).
- **In histo, G8–G32 help `histo_main_kernel` (up to +0.7%) and `histo_prescan_kernel` (−34 to −35% vs −37.5%).** They cost `histo_intermediates_kernel` (+21 to +24% vs +28%), so histo stays at −1.9 to −2.2%.

## FINAL
**No config meets (a) and (b).** histo stays between −1.6% and −5.0% vs no prefetching in every config. So the rule "best whole-run GMEAN subject to (a) and (b)" selects nothing.
- **By GMEAN alone the best is G8:** −0.238%, above Tier 1 Snake (−0.288%). It meets (c) and (d), not (a) or (b).
- **The config closest to (a) and (b) is E10:** histo −1.64%, GMEAN −0.262%.

### Whole run: Tier 1 Snake, SK_R0_D0, the best by GMEAN (G8), E10 and the perfect-L1 ceiling
| workload | Tier 1 Snake | SK_R0_D0 | **G8 (best GMEAN)** | E10 (best histo) | perfect-L1 ceiling |
|---|---|---|---|---|---|
| backprop | -0.20% | +0.16% | **+0.17%** | -0.10% | +41.7% |
| hotspot 1024 | +0.16% | +0.14% | **+0.17%** | +0.30% | +0.1% |
| hotspot 512 | +0.13% | -0.09% | **+0.17%** | +0.07% | +0.3% |
| lud 512 | +0.12% | -0.11% | **-0.17%** | -0.13% | +3.6% |
| lud 256 | +0.08% | +0.09% | **+0.10%** | +0.09% | +3.0% |
| nw | +0.00% | +0.00% | **+0.00%** | +0.00% | +16.6% |
| histo | -1.93% | -3.44% | **-1.96%** | -1.64% | +39.2% |
| mri-q | +0.09% | +0.21% | **+0.12%** | +0.21% | +2.2% |
| srad_v1 | -0.21% | -0.53% | **-0.11%** | -0.46% | +20.6% |
| **GMEAN** (7 apps) | -0.29% | -0.52% | **-0.24%** | -0.26% | +16.6% |

| criterion | Tier 1 Snake | G8 | E10 |
|---|---|---|---|
| (a) histo ≥ 0 | −1.93% | −1.96% no | −1.64% no |
| (b) no workload below −1% | histo | histo no | histo no |
| (c) GMEAN > Tier 1 Snake | −0.288% | −0.238% **yes** | −0.262% **yes** |
| (d) timely share of used, pooled | 39.1% | 39.8% **yes** | 34.8% no |

The headroom is still where D1 put it (perfect-L1 GMEAN +16.6%: backprop +41.7%, histo +39.2%, srad_v1 +20.6%, nw +16.6%). No config gains more than +0.24% on any of those four workloads at the whole-run level (backprop, B).

### Best-performing kernel (diagnostic only)
Each value is the kernel IPC normalized to the same kernel's IPC without prefetching. The best kernel has the largest gain among kernels with at least 1% of the baseline cycles, over complete launches. A kernel's gain does not predict the run's: histo's best kernel gains 24–42% while every histo run loses.

| workload | Tier 1 Snake | SK_R0_D0 | G8 | E10 |
|---|---|---|---|---|
| backprop | `bpnn_layerforward_CUDA` 1.0007 | `bpnn_adjust_weights_cuda` 1.0029 | `bpnn_adjust_weights_cuda` 1.0030 | `bpnn_layerforward_CUDA` 1.0001 |
| hotspot 1024 | `calculate_temp` 1.0016 | `calculate_temp` 1.0014 | `calculate_temp` 1.0017 | `calculate_temp` 1.0030 |
| hotspot 512 | `calculate_temp` 1.0013 | `calculate_temp` 0.9991 | `calculate_temp` 1.0017 | `calculate_temp` 1.0007 |
| lud 512 | `lud_internal` 1.0059 | `lud_perimeter` 1.0005 | `lud_perimeter` 1.0001 | `lud_perimeter` 1.0002 |
| lud 256 | `lud_internal` 1.0043 | `lud_internal` 1.0051 | `lud_internal` 1.0054 | `lud_internal` 1.0051 |
| nw | `needle_cuda_shared_1` 1.0000 | `needle_cuda_shared_1` 1.0000 | `needle_cuda_shared_1` 1.0000 | `needle_cuda_shared_1` 1.0000 |
| histo | `histo_intermediates_kernel` 1.1148 | `histo_intermediates_kernel` 1.4217 | `histo_intermediates_kernel` 1.2359 | `histo_intermediates_kernel` 1.2826 |
| mri-q | `ComputeQ_GPU` 1.0010 | `ComputeQ_GPU` 1.0022 | `ComputeQ_GPU` 1.0012 | `ComputeQ_GPU` 1.0022 |
| srad_v1 | `extract` 1.0256 | `extract` 1.0300 | `extract` 1.0248 | `extract` 1.0300 |

### Options of G8 (best GMEAN)
Config: `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0-SNAKE_ACCF-SNAKE_ACCFB10K-SNAKE_IWD8`. Effective values come from the run's option dump; later tokens override earlier ones.

| option | G8 | Tier 1 Snake | origin | SPEC |
|---|---|---|---|---|
| PAPER_V100, 1B_INSN | GTO, 1530 MHz, 1B | same | paper's V100 setup | §5, OQ-18, OQ-23 |
| `-snake_enable 1` | Head / Tail tables, inter-thread chains, promotion, inter-warp and intra-warp prediction, throttling | same | **paper Snake** | §1–§3 |
| `-snake_footprint 1` | exact footprint (the target load's lanes) | same | paper Snake, our reading | OQ-9 |
| `-snake_tail_entries 10`, `-snake_throttle_enable 1`, `-snake_lane_mode 1`, `-snake_mshr_reserve 8` | as Tier 1 | same | paper Snake (throttling §3.3); lane mode and MSHR reserve our readings | OQ-10, OQ-11, OQ-17 |
| `-snake_queue_size 128` | 128 | 32 | SnakeCharmer's SK setting | OQ-33 |
| `-snake_max_chain_depth 10`, `-snake_chain_deep_promoted 1` | depth 10, deeper steps on promoted entries | 3, trained entries | SnakeCharmer's SK setting | OQ-7, OQ-33, OQ-35 |
| `-snake_intra_distance 8`, `-snake_intra_distance_mode 0` | iterations 1..8 ahead | 1 (next iteration) | SnakeCharmer's SK setting | OQ-34 |
| `-snake_promote_warps 3`, `-snake_intra_confirm_warps 3`, `-snake_interwarp_min_warps 3` | 3 | 3 | paper Snake (SnakeCharmer's confidence 3 agrees) | OQ-28, OQ-35 |
| `-snake_head_two_warps 0` | 64-entry per-warp Head | per-PC rows of two warps | SnakeCharmer's Head table | OQ-1b, OQ-37 |
| `-snake_missq_reserve 0` | no miss-queue slots reserved for demand | 8 | our implementation parameter (SK match Step 1) | OQ-11 |
| `-snake_decouple_mode 0` | memory decoupling off (throttling stays on) | 1 (unified prefetch region) | **departs from paper Snake** (§3.2 decoupling) | OQ-12 |
| `-snake_entry_acc_filter 1`, `-snake_acc_filter_min 0.60`, `-snake_acc_filter_block 10000` (window 32, age 1,000) | per-source accuracy filter | off | **our extension** | OQ-39 |
| `-snake_interwarp_scope 1`, `-snake_interwarp_single 1`, `-snake_interwarp_degree 8`, `-snake_interwarp_cta_anchor 1` | one inter-warp target, the 8th resident warp ahead; other CTAs anchored on their first warp | next 4 warps of the CTA | **our extension** (scope 1 from S6 / OQ-33) | OQ-8, OQ-33, OQ-40 |
| R2 (`-snake_pq_path`) | off | off | our reading of SnakeCharmer §III-A, tested in Phase 2 (A–C, F), not in G8 | OQ-38 |

### What did not work and why (for the next step)
- **R2** (OQ-38) changes the injection order, not the volume. histo's loss is the volume of unused prefetches (Phase 0), so R2 leaves it (B, C) or deepens it with Tier 1 Snake (A: `histo_main_kernel` −8.4%).
- **The accuracy filter** (OQ-39) halves histo's loss with SK settings (−3.44% → −1.64%) but does not stop `histo_prescan_kernel`'s flood. Prescan stays 77–85% unused with 3.5–3.8× DRAM reads in every filtered config, although sources there are stopped often. Other sources keep issuing, and an unused prefetch only counts after 1,000 cycles.
- **Tier 1 Snake's `histo_main_kernel` loss** (−3.3 to −3.9% over 66% of histo's cycles) comes with only 1% unused prefetches. It is not a flood, and a usefulness filter cannot remove it. Every Tier 1-based config (A, D) stays at histo ≤ −1.77%.
- **Untested next candidates:**
  - stop inter-thread chains (and intra-warp prediction) in kernels whose demand is mostly streaming reads read once, as histo's prescan is
  - make the filter stop a source until it shows accuracy again, instead of for a fixed time
  - keep filter state across launches of the same kernel (each prescan launch starts untrained)

