# S6: why Snake does not raise IPC (2026-10-16)

Question: Tier 1 (results/final_tier1.md) shows Snake within ±0.3% IPC of the baseline on every app, against the paper's +14% average. Where does the gain go, and can it be fixed?

Data: the Tier 1 runs (1B cap; Snake = exact footprint, default config) plus 22 new runs:
- 8 perfect-memory ceiling runs (`sim_run_snake_gate/S6_ceiling`)
- 8 experiment runs (`S6_scope`, `S6_scope_bp`)
- the synthetic gate (11) and the off-regression (6) for the new option, both PASS; Snake-DT synthetic equals S4b exactly

Build: accel-sim `4502b2d`, gpgpu-sim `e85e7eb1` (the ceiling runs used the S5 frozen build, identical model).

## 1. In most apps the simulated GPU already hides the latency Snake could remove

Per-scheduler cycle breakdown (gpgpu-sim's `Warp Occupancy Distribution`):
- `W0_Scoreboard`: nothing issued because every warp waits on a register, mostly a load. This is the stall a prefetcher removes.
- `Stall`: an instruction was ready but its pipeline was busy.
- `W0_Idle`: no instruction to issue.

| workload | issued | Stall | W0_Idle | W0_Scoreboard, baseline → Snake | cycles, Snake vs baseline |
|---|---|---|---|---|---|
| backprop | 33.1% | 24.9% | 1.3% | 40.7 → 40.8% | +0.20% |
| hotspot 1024 | 71.8 | 24.5 | 0.6 | 3.1 → 3.3 | −0.16% |
| hotspot 512 | 69.4 | 24.1 | 1.7 | 4.7 → 4.7 | −0.13% |
| lud 512 | 22.4 | 28.3 | 33.7 | 15.7 → 15.8 | −0.12% |
| lud 256 | 13.3 | 12.0 | 45.5 | 29.3 → 29.1 | −0.08% |
| nw | 4.9 | 0.2 | 72.1 | 22.8 → 22.8 | 0.00% |
| histo | 19.5 | 4.7 | 2.9 | 72.9 → 69.6 | +1.96% |
| mri-q | 50.4 | 29.8 | 0.5 | 19.3 → 19.3 | −0.09% |
| srad_v1 | 43.7 | 39.4 | 6.5 | 10.5 → 10.3 | +0.21% |

- hotspot loses only 3–5% of scheduler cycles to load waits. Snake raises its L1 hit rate from 0.2% to 27% without changing that share: the other warps (77% occupancy) already hid those loads.
- lud and nw are dominated by idle schedulers (too few warps), which no prefetcher changes.

**Ceiling: perfect memory** (`-gpgpu_perfect_mem 1`, token `PERFECT_MEM`). L1 misses still use the L1D and its miss queue, but the memory system answers at once.

| workload | baseline IPC | perfect-memory IPC | ceiling | paper's Snake gain | Snake now |
|---|---|---|---|---|---|
| hotspot 512 | 4,733 | 4,774 | +0.9% | +13% | +0.1% |
| hotspot 1024 | 6,187 | 6,224 | +0.6% | +13% | +0.2% |
| mri-q | 3,845 | 3,909 | +1.7% | +4% | +0.1% |
| lud 512 | 182 | 190 | +4.8% | +18% | +0.1% |
| lud 256 | 47 | 50 | +4.4% | +18% | +0.1% |
| srad_v1 | 1,546 | 1,895 | +22.5% | +29% | −0.2% |
| histo | 667 | 865 | +29.6% | +33% | −1.9% |
| backprop | 2,494 | 3,657 | +46.7% | +1% | −0.2% |

**In our simulator, no prefetcher could reach the paper's gains on hotspot, mri-q or lud.** Even zero memory latency gives them less than 5%.

## 2. Where there is headroom, it is bandwidth and queueing, not latency

- **backprop.** It is DRAM-congested: 895k DRAM-full stalls, 40% mean channel utilization, 1,015-cycle average memory latency.
  - The perfect-memory run makes **0** DRAM reads, so its +47% mostly removes the bandwidth limit, which a prefetcher cannot do.
  - Every config reads the same **434k** sectors from DRAM: baseline, Snake, and the S6 experiment that made 14% of demand sectors timely.
  - With prefetching, the average memory latency **rises** (1,015 → 1,065 → 1,124 cycles), because prefetches add queueing.
  - The paper's backprop gain is also small (+1%).
- **histo.** 1.85M DRAM-full stalls, and 3.65M L1 reservation failures for MSHR or miss-queue space. 50% of its predicted sectors are wrong (inter-warp), and the wrong prefetches take miss-queue slots from demand: Snake costs −1.9%.
- **srad_v1.** 2.77M L1 reservation failures, and 15.7% of bandwidth windows at ≥ 70% use of the miss-queue drain (one request per cycle). Prefetches compete with demand for the same slots: 36% of its predicted sectors were dropped from a full prefetch queue, and 19% were still queued when the demand came.
- **mri-q.** Its loads come in bursts at kernel start. Demand alone fills the 16-entry miss queue, so prefetches wait about 250 cycles (p50) and are almost never issued in time.

## 3. Snake predicts too close to the use

Lead time = prediction → the matching demand (p50); average memory latency from the baseline.

| workload | memory latency | inter-warp lead | own-warp inter-thread lead | inter-warp share of predictions |
|---|---|---|---|---|
| backprop | 1,015 | 19 | 13 | 97% |
| srad_v1 | 467 | 22 | 25 | 82% |
| histo | 354 | 82 | 134 | 72% |
| hotspot 512 | 347 | 108 | 24 | 85% |

The inter-warp targets are the next warps of the same CTA (OQ-8). They reach the same load only a few to tens of cycles after the predicting warp. So almost every issued prefetch is late: backprop has 113,702 late inter-warp sectors (lead p50 19) against 2,165 timely ones.

**Experiment: `-snake_interwarp_scope`** (OQ-8 extension, default 0 = unchanged):
- 1 = the next warps by global warp id among all resident warps of the kernel
- 2 = the same warp index in the next resident CTAs

Both assume addresses linear in the global warp id across CTAs (1D grids).

| workload | scope | IPC vs baseline | coverage_pred | coverage_paper | accuracy_paper | inter-warp timely: n / lead p50 |
|---|---|---|---|---|---|---|
| backprop (1B) | 0 | −0.20% | 64.6 | 12.4 | 0.2 | 2,165 / 403 |
| | 1 | −0.22% | 71.9 | 14.8 | 3.6 | 34,641 / 446 |
| | **2** | **+0.07%** | 49.2 | 16.2 | **14.0** | **136,760 / 1,602** |
| srad_v1 (100M) | 0 | +0.56% | 84.8 | 22.8 | 6.4 | 22,511 / 371 |
| | 1 | +0.30% | 86.4 | 22.8 | 7.1 | 25,528 / 399 |
| | 2 | +0.04% | 64.3 | 15.9 | 4.5 | 17,810 / 1,180 |
| histo (100M) | 0 | −1.85% | 30.8 | 20.9 | 9.9 | 29,018 / 511 |
| | 1 | −1.85% | identical (one CTA per SM) | | | |
| | 2 | +0.09% | 24.4 | 14.7 | 9.3 | 0 (no other CTA: inter-warp off) |
| hotspot 512 (100M) | 0 | +0.13% | 90.6 | 59.5 | 26.4 | 45,572 / 897 |
| | 1 | +0.10% | 90.8 | 53.6 | 18.6 | 32,107 / 742 |
| | 2 | −4.73% | 50.0 | 18.4 | 0.0 | 0 (2D grid: wrong addresses) |

- **Scope 2 fixes backprop's timeliness:** timely sectors rise 60× (accuracy 0.2% → 14.0%), issued 1,602 cycles ahead. IPC still does not move, because backprop is DRAM-bound (§2).
- histo's +0.09% comes only from switching its harmful inter-warp prefetches off.
- On 2D grids (hotspot) scope 2 predicts wrong addresses. **The default stays 0.**

## 4. Conclusions

1. **The simulator, more than Snake, sets the ceiling.**
   - In our Accel-Sim setup, hotspot, mri-q and lud hide their memory latency with multithreading: even perfect memory gains < 5%.
   - backprop and histo are bound by DRAM queueing, and srad_v1 and mri-q by L1 miss-queue and MSHR throughput. A prefetcher moves the same bytes and cannot add throughput.
   - The paper reports that memory stalls are 55% of stalls and reservation fails 30% of L1 accesses (Figs. 3–5). Its GPU was more latency-bound than ours (SPEC §5, VERSION_DIFF); that is the main reason its gains do not reproduce here.
2. **Snake's own limit is lead time.** Inter-warp predictions within a CTA come ~20 cycles before use against 350–1,000 cycles of latency. Targeting later CTAs (scope 2) does deliver timely prefetches where the grid is 1D and several CTA waves run. But on our apps that does not turn into IPC, for the reasons in point 1.
3. **Issue-path losses are real but secondary.** Queue-full drops, the miss-queue reserve, bursty demand (mri-q) and throttling (9–17% of cycles held) lose many correct predictions. Fixing them adds miss-queue traffic in apps that are already throughput-bound. Tier 2 (queue 64, issue width 2, miss-queue reserve 4, bytes meter) measures them.

**Not fixed.** No Snake change found here raises IPC on our workloads: the headroom that exists is bandwidth, not latency.

**What would change the outcome:** a baseline that is latency-bound like the paper's. Options:
- the paper-era Accel-Sim v1.2.0 configuration (release: streaming L1D, 4,096 MSHRs; VERSION_DIFF)
- workloads or inputs with more exposed latency

Either is a new decision, not a fix to Snake.
