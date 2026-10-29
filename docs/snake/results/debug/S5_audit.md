# Debug plan Stage 5: implementation audit (2026-10-29)

**Status:** done.
- **Code reading:** gpgpu-sim `1558559d`.
- **Data:** the Stage 3 runs (`sim_run_snake_debug/s3`; log on SMs 0, 16, 32, 48, 64; counters from all SMs), from `util/snake/debug/s5_audit.py`.
- **Micro-traces:** the synthetic suite (gate DBG3b), and its logging-on copy (DBG3blc).
- **Bug fix:** one, stats only, applied to the report scripts. No IPC changes, so no 100M or 1B reruns.

## TLDR

- **No simulator bug that changes IPC was found.**
  - Trigger point, retries, chain walk, fill and insert, and late-merge wake-up all pass: no demand ever missed a sector holding an unused prefetch, and no read was left unanswered in a run that finished.
  - The wake-up after a prefetch fill is 4-10 cycles (median).
- **One metrics bug, fixed.** `snake_sc_ld_miss` counts every MISS return, so it already holds the late loads. The reports added `snake_sc_ld_late` a second time, understating SnakeCharmer coverage: hotspot 1024 19.1% instead of 27.3%, histo 19.8% instead of 26.0%.
  - Fixed in `util/snake/sk_sc_report.py` (`metrics()`) and the definitions in `make_slides_data.py`; SPEC OQ-36 updated.
  - Deliverables generated earlier (the slides workbook, `Snake_all_versions_2026-10-28.xlsx`, SK_SC.md, SK_match_qv100.md) carry the old values until regenerated.
- **Findings that are design limits, not bugs:**
  - **Inter-warp targets (item 6):** 30-54% of inter-warp predictions go to warps that have already executed the PC (none to other CTAs). Snake picks the next 4 resident warps of the CTA without checking their progress, and under GTO 30-54% of them are already ahead.
  - **nw (item 11):** Snake never trains. Each of its 255 launches puts 1.16 single-warp CTAs (16 threads) on an SM on average, while a stride needs 3 warps to promote, and the tables are cleared at every launch (OQ-19).
  - **Tail thrash (item 11):** on backprop, 99.3% of Tail allocations evict an entry (hit rate 1.2%, 872 promotions vs 794 demotions); on srad_v1, 95.3%.
  - **histo's 34% coverage (item 11)** comes from `histo_main` (75% of histo's loads; 24% of its demand sectors predicted). A quarter of its loads fit no lane run (data-dependent addresses). The Tail is not the limit there: 378 allocations, 0 evictions; its hit rate over the run is 96.5%.
  - **Lane filter (item 12):** lane mode 1 excludes 0-19% of a workload's loads (25% in `histo_main`). The paper-literal reading (one 32-lane run) would exclude every load of backprop and hotspot (all are two 16-lane runs) and 83-91% of lud's.
  - **Throttle and decoupling (items 4, 5):** throttled 0.3-18% of Snake cycles, mostly by the bandwidth trigger. Confinement evicts demand lines (hotspot 1024: 85 per 1,000 demand sectors), but costs at most 0.2 IPC points against Snake-DT (no decoupling, no throttling).
- **Labelling notes:**
  - DUP_L1 ("already in L1") also covers sectors that are reserved and still in flight: 28-96% of DUP_L1 offers per workload (hotspot 1024 28%, lud 256 96%), split in item 8.
  - `-snake_l1_filter` is registered but never read.
- **Stage 6 candidates (for approval; per G2 the extensions are skipped):** check an inter-warp target's progress before predicting for it (item 6). Rename DUP_L1 or split its counter. Remove or document `-snake_l1_filter`.

## Code reading and verdicts

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | Trigger point | **pass** | `ldst_unit::issue` (`shader.cc:2800`) calls `observe()` when the instruction enters the LD/ST unit from the operand collector: once per instruction, before its first L1D access. That is after the scheduler issued it (operand collection in between), not at L1 accept and not after retries. H5 (trigger too late) is rejected for the trigger point |
| 2 | Reservation-fail retries | **pass** | A load that gets RESERVATION_FAIL stays in the memory stage and retries its access every cycle (`ldst_unit::memory_cycle`, `L1_latency_queue_cycle`). `observe()` is not on that path, so a retry does not re-train or re-trigger. `on_demand_access` skips coverage counting on a failed attempt (`snake_prefetcher.cc`, the `reservation_fail` branch), so a retried load counts once |
| 3 | Prefetch queue | **pass** (works as designed; the main loss point, S3 3a) | Drained once per core cycle after the L1D's own cycle and the demand accesses (`ldst_unit::cycle` → `snake_issue`, `shader.cc:3036`). At most `issue_width` = 1 line per cycle, only when the data port is free, and only its sectors in banks no demand used that cycle; demand always goes first. Reserves: 8 MSHR entries and 8 of the 16 miss-queue slots stay free for demand (`l1_cache::snake_prefetch`). Own-warp lines go before inter-warp lines (OQ-31) |
| 4 | Throttle meter | **pass** | Meter = L1D miss-queue sends per core cycle over 100-cycle windows (`on_mq_cycle`: demand and prefetch requests), against a peak of 1 per cycle; halt at ≥ 0.70, resume at ≤ 0.50 (`tick()`). One read request returns one 32 B sector, so this is the share of the SM's reply bandwidth. The space trigger halts for 50 cycles after an allocation finds no free line |
| 5 | Decoupling caps | **pass** (≤ 0.2 IPC points vs Snake-DT) | Variant 1 (Tier 1): the 50% L1 cap holds until the first promotion only for the first 10 Snake cycles of a launch (`untrained_cap`), and for at most 50 cycles from the start of each throttle episode (`l1_capped`). Demand allocations then evict L1-region lines (`l1_quota_victims`) |
| 6 | Inter-warp targets | **finding**: 30-54% aimed at warps already past the PC | With `-snake_interwarp_scope 0` the targets are the next 4 resident warps above the trigger in the same CTA slot (`predict()`). Nothing checks whether a target has already executed the PC. The data counts how often it had |
| 7 | Chain walk | **pass** | `predict()`: the inter-thread chain stays on the trigger warp `w`; step d adds the Tail entry's stride to the previous step's address and moves to its PC2; depth ≤ 3; step 1 needs a promoted entry (with `w`'s bit, or any with OQ-32), deeper steps a trained one with `w`'s bit. One trigger yields at most 3 chain targets (w, PC2..PC4), 4 inter-warp targets (w+1..w+4 at the same PC) and 1 intra-warp target (w, same PC, next iteration). The synthetic Fig. 15 (c) check matches the figure exactly (gate DBG3b) |
| 8 | In-flight and duplicate filters | **pass**, with a labelling note | `snake_prefetch()` returns DUP_L1 for a sector that is valid **or reserved** (its line is waiting for a fill), and DUP_MSHR only for an MSHR entry without a reserved line. So "in L1" in the counters also holds sectors in flight; the data splits them. A prediction whose demand arrives first is closed at that demand as "queued at demand" (counted, not lost). A prediction made while a demand miss for the sector is already in flight is born late; the data counts these |
| 9 | Fill and insert | **pass** (0 cases) | Prefetch fills use `baseline_cache::fill` like demand fills (sector valid and readable); `on_fill` marks the sector in `m_pf_bits`; a demand to it hits and counts as timely (`on_tag_hit`). The data checks that no demand misses a sector holding an unused prefetch |
| 10 | Late-merge wake-up | **pass**, one modelling cost | Demands merged into a prefetch's MSHR entry become ready at its fill (`mshr_table::mark_ready`). The prefetch's own response pops first and is retired in the writeback stage without taking the writeback slot (`shader.cc`, client 4), so each merged demand gets its data at least one cycle after the fill: no extra round trip, no lost wake-up. Cost: every prefetch fill uses one L1D response pop (at most one per cycle), so demand responses queued behind it wait. 3c measures the merged demands' latency |
| 11 | Head and Tail capacity | **finding** | Tail thrash on backprop (99% of allocations evict) and srad_v1 (95%); nw never promotes (1.16 single-warp CTAs per launch and SM, 3 warps needed, tables cleared per launch); histo's low coverage is `histo_main` (24% predicted, 25% of its loads fit no lane run, Tail not thrashing: 378 allocations, 0 evictions) |
| 12 | Lane and shape filter (OQ-17) | **finding** (mode 1 excludes 0-19% per workload, 25% in `histo_main`; mode 0 would exclude all of backprop and hotspot) | Lane mode 1 (Tier 1) accepts a load whose lanes fit runs of 32, 16, 8 or 4 at equal strides; mode 0 (the paper-literal §3.4 reading) accepts only one run of 32 at one stride. Loads with runs of 16, 8 or 4 would be excluded in mode 0 |

Other findings:
- **Metric bug (stats only, IPC unaffected).** `on_demand_access` counts `sc_ld_miss` for every demand load whose access returns MISS. `rd_miss_base` returns MISS also for a load that merges into an existing MSHR entry (`gpu-cache.cc:2146`). So a load merging into an in-flight prefetch is counted in `sc_ld_late` (first merge per prefetch, `on_mshr_merge`) **and** in `sc_ld_miss`. `sk_sc_report.metrics()` and `make_slides_data.py` compute SnakeCharmer coverage as on / (on + late + miss): late loads are counted twice in the denominator, and coverage is understated. **Fixed** in the report scripts: on / (on + miss) and (on + late) / (on + miss); SPEC OQ-36 updated.
- **Dead option.** `-snake_l1_filter` ("skip lines already in L1", §3.2) is registered but never read. The in-L1 check happens at offer time instead (DUP_L1).


## Data

Runs: `sim_run_snake_debug/s3/runs`, config `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SNAKE_LC16`. Log items: SMs 0, 16, 32, 48 and 64; counter items: all SMs.

### Item 3: prefetch queue (whole run)

| workload | offered lines waited for the port (per issued sector) | drops: MSHR reserve | drops: miss-queue reserve | drops: miss queue full | queue full (sectors) | displaced |
|---|---|---|---|---|---|---|
| backprop 65536 | 4.63 | 2 | 77729 | 561 | 267395 | 18982 |
| hotspot 1024_2_2__ | 0.34 | 0 | 186099 | 652 | 429094 | 27374 |
| hotspot 512_2_2___ | 0.41 | 0 | 38616 | 229 | 144273 | 5408 |
| lud _i___data_ | 0.60 | 0 | 263949 | 8481 | 1262185 | 149445 |
| lud _s_256__v | 0.66 | 0 | 32721 | 2011 | 38072 | 4640 |
| nw 2048_10 | n/a | 0 | 0 | 0 | 0 | 0 |
| histo _i___data_ | 2.13 | 0 | 3717729 | 46519 | 6137146 | 1321236 |
| mri-q _i___data_ | 5.41 | 0 | 7082 | 885 | 78032 | 12660 |
| srad_v1 100_0_5_50 | 5.17 | 222 | 535952 | 12580 | 10402130 | 495774 |

### Item 4: throttling (whole run)

| workload | throttled (% of Snake cycles) | by space | by bandwidth | space events | bandwidth events | meter windows >= 70% (%) |
|---|---|---|---|---|---|---|
| backprop 65536 | 10.3 | 0.1 | 10.1 | 108 | 2276 | 8.7 |
| hotspot 1024_2_2__ | 14.4 | 11.7 | 2.8 | 10441 | 809 | 2.0 |
| hotspot 512_2_2___ | 12.9 | 7.2 | 5.7 | 1656 | 251 | 2.7 |
| lud _i___data_ | 13.0 | 0.7 | 12.4 | 967 | 4622 | 8.5 |
| lud _s_256__v | 3.3 | 0.0 | 3.3 | 0 | 374 | 2.4 |
| nw 2048_10 | 0.4 | 0.0 | 0.4 | 0 | 2449 | 0.4 |
| histo _i___data_ | 13.9 | 2.8 | 11.5 | 20950 | 32489 | 10.6 |
| mri-q _i___data_ | 0.3 | 0.0 | 0.3 | 0 | 246 | 0.3 |
| srad_v1 100_0_5_50 | 18.0 | 0.0 | 18.0 | 20 | 12482 | 15.7 |

### Item 5: decoupling costs to demand data (whole run)

| workload | L1 capped, untrained (% of Snake cycles) | L1 capped, confined while throttled (%) | demand lines evicted for the cap (per 1,000 demand sectors) | demand-fetched sectors evicted by a prefetch allocation (% of victim sectors) | unused prefetches evicted from prefetch-region lines | from L1-region lines |
|---|---|---|---|---|---|---|
| backprop 65536 | 0.0 | 3.0 | 3.53 | 16.4 | 0 | 0 |
| hotspot 1024_2_2__ | 0.0 | 12.4 | 85.27 | 23.5 | 104398 | 129 |
| hotspot 512_2_2___ | 0.1 | 8.3 | 50.26 | 26.3 | 15608 | 8 |
| lud _i___data_ | 0.4 | 3.9 | 8.53 | 21.9 | 29095 | 253 |
| lud _s_256__v | 0.6 | 1.2 | 0.00 | n/a | 0 | 0 |
| nw 2048_10 | 0.2 | 0.2 | 0.00 | n/a | 0 | 0 |
| histo _i___data_ | 0.1 | 6.6 | 2.16 | 20.1 | 267991 | 18 |
| mri-q _i___data_ | 0.0 | 0.1 | 0.00 | n/a | 0 | 0 |
| srad_v1 100_0_5_50 | 0.2 | 3.1 | 2.13 | 0.8 | 0 | 321 |

### Item 6: inter-warp targets (log)

| workload | inter-warp predictions | ahead (%) | already passed the PC (%) | never execute it later (%) | other CTA (%) |
|---|---|---|---|---|---|
| backprop 65536 | 51610 | 54.1 | 45.9 | 0.0 | 0.0 |
| hotspot 1024_2_2__ | 63258 | 60.3 | 39.1 | 0.5 | 0.0 |
| hotspot 512_2_2___ | 15776 | 63.1 | 35.5 | 1.4 | 0.0 |
| lud _i___data_ | 76322 | 65.4 | 34.6 | 0.0 | 0.0 |
| lud _s_256__v | 8506 | 51.5 | 48.5 | 0.0 | 0.0 |
| nw 2048_10 | 0 | n/a | n/a | n/a | n/a |
| histo _i___data_ | 338701 | 48.8 | 51.2 | 0.0 | 0.0 |
| mri-q _i___data_ | 2350 | 46.5 | 53.5 | 0.0 | 0.0 |
| srad_v1 100_0_5_50 | 365778 | 70.1 | 29.7 | 0.2 | 0.0 |

### Item 8: duplicates at offer and predictions born late (log)

| workload | DUP_L1: valid in L1 | DUP_L1: demand in flight | DUP_L1: prefetch in flight | DUP_MSHR: demand | DUP_MSHR: prefetch | DUP_MSHR: other | sectors newly queued | of them, a demand miss already in flight (%) |
|---|---|---|---|---|---|---|---|---|
| backprop 65536 | 17260 | 11621 | 5916 | 0 | 0 | 0 | 48414 | 7.6 |
| hotspot 1024_2_2__ | 47893 | 5079 | 13915 | 0 | 0 | 0 | 115962 | 1.5 |
| hotspot 512_2_2___ | 9343 | 1282 | 3269 | 0 | 0 | 0 | 24294 | 2.9 |
| lud _i___data_ | 6770 | 16683 | 16920 | 0 | 0 | 0 | 100521 | 6.7 |
| lud _s_256__v | 244 | 3127 | 2489 | 0 | 0 | 0 | 12052 | 7.9 |
| nw 2048_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| histo _i___data_ | 277681 | 124722 | 211955 | 0 | 0 | 0 | 1172695 | 9.4 |
| mri-q _i___data_ | 313 | 1671 | 55 | 0 | 0 | 0 | 3713 | 4.9 |
| srad_v1 100_0_5_50 | 165171 | 46649 | 51998 | 0 | 0 | 0 | 379084 | 8.0 |

### Items 9 and 10: misses on resident prefetches; reads never answered (log)

| workload | new misses on a sector holding an unused prefetch (must be 0) | first cases (cycle, SM, sector) | reads with no data back at the end: new / merged into prefetch / merged into demand |
|---|---|---|---|
| backprop 65536 | 0 | – | 0 / 0 / 0 |
| hotspot 1024_2_2__ | 0 | – | 0 / 0 / 0 |
| hotspot 512_2_2___ | 0 | – | 0 / 0 / 0 |
| lud _i___data_ | 0 | – | 0 / 0 / 0 |
| lud _s_256__v | 0 | – | 0 / 0 / 0 |
| nw 2048_10 | 0 | – | 0 / 0 / 0 |
| histo _i___data_ | 0 | – | 149 / 261 / 12 |
| mri-q _i___data_ | 0 | – | 0 / 0 / 0 |
| srad_v1 100_0_5_50 | 0 | – | 310 / 120 / 92 |

### Item 11: Head and Tail churn (whole run)

| workload | Head row evictions per update (%) | Tail evictions per allocation (%) | Tail hit rate (%) | T1 promotions | T1 demotions | warps removed |
|---|---|---|---|---|---|---|
| backprop 65536 | 0.0 | 99.3 | 1.2 | 872 | 794 | 2387 |
| hotspot 1024_2_2__ | 0.0 | 0.0 | 99.7 | 128 | 0 | 0 |
| hotspot 512_2_2___ | 0.0 | 0.0 | 99.2 | 81 | 0 | 0 |
| lud _i___data_ | 2.9 | 55.4 | 76.6 | 19700 | 10926 | 35778 |
| lud _s_256__v | 5.4 | 45.5 | 64.3 | 2638 | 672 | 3173 |
| nw 2048_10 | 0.0 | 44.5 | 14.0 | 0 | 0 | 0 |
| histo _i___data_ | 0.0 | 65.1 | 96.5 | 17136 | 1658 | 5331 |
| mri-q _i___data_ | 0.0 | 4.2 | 82.0 | 1172 | 120 | 1193 |
| srad_v1 100_0_5_50 | 0.0 | 95.3 | 70.4 | 47127 | 5021 | 29339 |

### Item 12: lane filter (whole run, lane mode 1)

| workload | loads observed | excluded (fit no run length, %) | runs of 32 (%) | 16 (%) | 8 (%) | 4 (%) |
|---|---|---|---|---|---|---|
| backprop 65536 | 294917 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| hotspot 1024_2_2__ | 117304 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| hotspot 512_2_2___ | 29240 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| lud _i___data_ | 274304 | 0.0 | 8.9 | 91.1 | 0.0 | 0.0 |
| lud _s_256__v | 35776 | 0.0 | 16.8 | 83.2 | 0.0 | 0.0 |
| nw 2048_10 | 311296 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| histo _i___data_ | 1549151 | 18.8 | 81.2 | 0.0 | 0.0 | 0.0 |
| mri-q _i___data_ | 15552 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| srad_v1 100_0_5_50 | 1552490 | 2.3 | 97.6 | 0.1 | 0.0 | 0.0 |

### Metric: SnakeCharmer late loads (whole run and log)

| workload | sc_ld_ontime | sc_ld_late (first merge per prefetch) | sc_ld_miss (all MISS returns) | cov_on as reported: on / (on + late + miss) | late counted once: on / (on + miss) | SPEC OQ-36 (M = new requests), estimated: on / (on + late + miss x new share) | log: MISS returns that are new / merged into a prefetch / merged into a demand (%) |
|---|---|---|---|---|---|---|---|
| backprop 65536 | 2165 | 119140 | 562826 | 0.3 | 0.4 | 0.5 | 59.3 / 27.9 / 12.8 |
| hotspot 1024_2_2__ | 188117 | 292432 | 501843 | 19.1 | 27.3 | 27.4 | 41.1 / 58.9 / 0.0 |
| hotspot 512_2_2___ | 45572 | 57220 | 126842 | 19.8 | 26.4 | 26.9 | 52.5 / 47.5 / 0.0 |
| lud _i___data_ | 28115 | 269952 | 977628 | 2.2 | 2.8 | 2.9 | 69.8 / 29.0 / 1.2 |
| lud _s_256__v | 925 | 42866 | 130147 | 0.5 | 0.7 | 0.7 | 65.6 / 34.1 / 0.3 |
| nw 2048_10 | 0 | 0 | 1064960 | 0.0 | 0.0 | 0.0 | 100.0 / 0.0 / 0.0 |
| histo _i___data_ | 1629874 | 1938842 | 4647699 | 19.8 | 26.0 | 26.6 | 55.0 / 43.0 / 1.9 |
| mri-q _i___data_ | 29 | 5774 | 62179 | 0.0 | 0.0 | 0.0 | 90.4 / 9.6 / 0.0 |
| srad_v1 100_0_5_50 | 214556 | 555717 | 4185259 | 4.3 | 4.9 | 5.9 | 69.0 / 16.3 / 14.7 |
