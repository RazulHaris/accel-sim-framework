# Snake prefetcher in Accel-Sim — work log

Implementation of Snake (Mostofi et al., MICRO 2023) in Accel-Sim. It is optional (`-snake_enable`) and off by default. With Snake off the output is identical to the reference build.

All Snake work is on branch `Snake` in both repos:
- accel-sim: pushed to `myfork` (RazulHaris/accel-sim-framework, public).
- gpgpu-sim: pushed to `backup` (RazulHaris/gpgpu-sim-oracle, private).

Documents:
- [SPEC](SPEC.md): the paper and the open questions (OQ-n).
- [VERSION_DIFF](VERSION_DIFF.md): v1.2.0 vs our base, and the trace-decoding issue.
- [CODE_MAP](CODE_MAP.md)
- [SNAKE_PLAN](SNAKE_PLAN.md): the plan, stage status with hashes and gate results (§0, §8).
- [upstream_issue](upstream_issue.md): draft, not posted.

The paper PDF and the planning copy are in `/home/razul/snake/` (outside git).

## Status
Updated with every stage. A stage's gate results and its own commit hashes go into the follow-up commit, because gates run only on committed, clean builds.

**Done:**

| Step | accel-sim | gpgpu-sim | Gate |
|---|---|---|---|
| S0 analysis docs | `84251a0` | base `6c3cf4ff` | — |
| S1 infrastructure (hooks, options, `L1_PREFETCH_R`) | `d49e1f3`, docs/tooling `b930d75` | `e0d23397` | off-regression PASS (5 workloads) |
| S2 training (Head/Tail, T1/T2, inter-/intra-warp; nothing issued) | `e4030cd` | `ace5cc70` | synthetic PASS, off-regression PASS; hotspot/backprop implausible under the §3.4 lane rule → OQ-17 |
| S2b per-PC lane pattern (OQ-17) | `5b4f096` | `b9cdc46a` | synthetic and off-regression PASS; hotspot trains; backprop exposed mis-decoded traces |
| Trace-parser fix for pre-756ed33 traces (not Snake code) | `fe8f37e` | — | exact on all affected base+delta lines |
| Reference `bin_ref_v2` + legacy-trace tooling | `bed3a2b` | — | synthetic PASS; off-regression vs `bin_ref_v2` PASS |
| Docs for the parser fix, upstream issue draft | `61c0dc1` | — | — |
| S2c stateless lane-run filter (OQ-17), `snake_footprint` setting (OQ-9), `interwarp_trained_pcs` | `293aaaa` | `bc22d0dd` | synthetic PASS (both modes); off-regression vs `bin_ref_v2` PASS (6 workloads); hotspot and backprop excluded 0% (all L = 16); hotspot 80/88 promotions; backprop 0 promotions by design (SPEC §8), 2 inter-warp PCs; srad_v1 2,337 promotions, 16 inter-warp PCs |
| S3a prefetch issue (inter-thread chain/promotion + intra-warp; Snake-DT) | `1221702`, `9dc2631`, `d836622` | `028012de`, `5c6f2267`, `2d50a9aa` | synthetic PASS (trace (d) coverage_paper 93.0%); 7 apps × 2 footprints finish, no warp woken, invariant OK; off-regression PASS; `snake_missq_reserve` 8 from the lud sweep. Tables: `results/S3a_gate.md` |
| S3b inter-warp issue (degree 4, resident warps of the CTA), combined lookahead (OQ-29), inter-warp eviction weight (OQ-30), own-warp queue priority (OQ-31) | `bf1fc2f`, `a708e99` | `298c5f94`, `000798a7`, `f1e4f856` | synthetic PASS (Fig. 15 (c) exact with `SNAKE_IWC0`, + {3100, 3200, 3300} by default; trace (d) 90.8%); 9 workloads line + 7 exact finish, no warp woken, invariant OK; off-regression PASS (6). Tables: `results/S3b_gate.md` |
| S3c diagnosis, stats only (prediction funnel, timing, miss queue) | `59e5bee`, `b7cae56` | `02a9d7de` | synthetic PASS and identical to S3b3; Snake-on runs identical to S3b_fp1; off-regression PASS (6); 9 diagnosis runs. Tables: `results/S3c_diag.md` |

**S3c findings (results/S3c_diag.md §10):**
- Issued prefetches are almost never evicted before use. Losses come from late issue (7–29% of predicted sectors), from never issuing in time (still queued, queue full, miss-queue reserve: 7–58%) and from wrong predictions (10–54%, lud highest).
- The L1D miss queue drains one request per cycle, and the interconnect never refused one.
- Inter-thread prefetches start ~20 cycles before their demand (p50).
- Reserve 4 adds 1.7–4.0 pp coverage_paper, but raises demand miss-queue waits and MISS_QUEUE_FULL (lud +25% over the baseline).
- At promotion only warps whose Head is at PC1 get a prediction; later warps without their bit get none (15–57% of loads).
- histo's coverage_pred gap is real (30.9% with the exact footprint).

**S3b findings, carried to S4 (SPEC §8):**
- coverage_pred (exact) is near the paper: hotspot 88%, lud 83% / 73%, srad_v1 81%, mri-q 87%, backprop 72%.
- coverage_paper lags (hotspot 72%, others 10–44%), because many prefetches are dropped for miss-queue slots or with the prefetch queue full. The S5 sweep (queue size, issue width, miss-queue reserve) decides that.
- IPC is within ±0.4% of the baseline except lud 512 line (−1.2%) and histo line (−3.8%), without decoupling or throttling.
- lud 512 line raises demand MISS_QUEUE_FULL by 29% (the exact footprint lowers it).
- The eviction weight (OQ-30) does not help backprop: every entry there holds an inter-warp stride. srad_v1's such evictions fall by ~25%.

**In progress:** nothing.

**Reference builds:** both are in `/home/razul/snake/`, with checksums in their `BUILD_INFO.txt`.
- `bin_ref_v2` is the regression reference from 2026-10-13: the clean base plus the parser fix.
- `bin_ref_paperera` is the clean base with paper-era decoding.

**Run directories:**
- `sim_run_snake_synth/<stage>`: synthetic traces.
- `sim_run_snake_regress/<stage>`: off-regression.
- `sim_run_snake_train/<stage>`: training plausibility.
- `sim_run_snake_refcmp/<name>`: build comparison.
- `sim_run_snake_gate/<stage>`: real-workload gate runs, from S3a on.
- `sim_run_snake_refcache/bin_ref_v2`: cached baseline outputs.

**Next:** decide on the S3c findings (lateness, miss-queue drain, promotion coverage), then S4, throttling and decoupling (SNAKE_PLAN §8).
