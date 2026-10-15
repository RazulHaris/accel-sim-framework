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
| Miss-queue / memory-path config check vs Tab. 1 and Figs. 3–5 (no runs, cached bin_ref_v2 outputs) | `50570b0` (`util/snake/missq_check.py`) | — | Config matches Tab. 1 except L1 latency 20 (28) and L2 ROP latency 160 (212). MISS_QUEUE_FULL causes 71–100% of reservation fails; fail rates and icnt utilisation are below Figs. 3–4. Miss queue stays 16. `results/missq_config_check.md` |
| S3c diagnosis, stats only (prediction funnel, timing, miss queue) | `59e5bee`, `b7cae56` | `02a9d7de` | synthetic PASS and identical to S3b3; Snake-on runs identical to S3b_fp1; off-regression PASS (6); 9 diagnosis runs. Tables: `results/S3c_diag.md` |
| S3d promoted strides for all warps (`-snake_promoted_all_warps 1`, OQ-32) + funnel by source | `976937f` | `97119f36` | synthetic PASS (Fig. 15 unchanged, trace (d) 90.8%); off-regression PASS (6); 10 runs (defaults and `SNAKE_IWC0`), all invariants OK. Tables: `results/S3d.md` |
| S3e paper defaults: inter-warp chain depth 0 (OQ-29), deeper chain steps need bit + trained (OQ-32); PAPER_LAT config | `20e968f`, `131a513` | `53bbfd25` | synthetic PASS (trace (d) gated under `SNAKE_IWC3` 90.8%; default 86.0% reported); off-regression PASS (6); 5 runs, all invariants OK; PAPER_LAT measured L2 hit 212.0 / L1 33.0. Tables: `results/S3e.md` |

**S3e findings (results/S3e.md, SPEC §8):**
- Within noise of S3d (b). IPC: hotspot +0.04%, lud −0.19%, histo −1.6%, mri-q +0.12%, srad_v1 +0.19%; coverage_pred 31–91%.
- Wrong share by source unchanged (lud: promotion 53%; others: inter-warp 96–99.5%). lud demotions 6,898.
- Our L2 hit is 173 cycles vs 212 in the paper; PAPER_LAT reproduces 212 for the final evaluation.
- Discrepancies with the paper (lud, histo, mri-q) and intra-warp's rarity are in SPEC §8.

**S3d findings (results/S3d.md §5):**
- Every load at a promoted PC1 now gets an inter-thread prediction. 33–57% of loads still get none, all at PCs without a promoted entry.
- (a) vs S3c:
  - coverage_pred: hotspot 87.8 → 90.4, srad_v1 80.6 → 85.9, mri-q 87.1 → 93.8.
  - coverage_paper: hotspot 71.1 → 75.7.
  - IPC: within ±0.5%, except histo −2.6%.
- lud: 59% of no-bit verifications fail; T1 demotions 490 → 6,879.
- Most wrong predictions: promotion and inter-warp chain (lud), inter-warp and its chain (histo, mri-q). Most timely: inter-warp and its chain. Own-warp inter-thread prefetches are almost never timely.
- Chain depth 0 gives slightly better IPC than 3 on all five apps.

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

**S4, throttling and decoupling: the gate stopped** (stop rule; SNAKE_PLAN §8, `results/S4.md`). Mode 1 was revised in S4b below.
- **Implemented** (accel-sim `f708b4f`, gpgpu-sim `57103274`):
  - the space and bandwidth triggers (request slots of the L1D miss-queue drain, OQ-10)
  - the chain cap while throttled (OQ-7)
  - mode 1: regions with flips, bulk free, the transfer rule and the L1 cap (OQ-12, OQ-15)
  - mode 2: the isolated buffer (OQ-22)
- **PAPER_LAT (step 0):** L1 knob 23 and L2 ROP 199 measure an L1 hit of 28.0 and an L2 hit of 212.0 cycles (Table 1: 28 / 212). Both are treated as measured values (VERSION_DIFF).
- **Gates:**
  - Synthetic PASS (11 runs; trace (b) under full Snake is throttled 89.4%, 0 issued).
  - Off-regression PASS (6).
  - 15 app runs: all finish, invariants OK, no quota assert.
- **Stop:** mode 1 lowers the demand L1 hit rate vs Snake-DT by more than 2%:
  - hotspot: Snake −19.6%, Snake-T −20.6%. Bulk frees evict 19.5k prefetched sectors before use; 19.0k unused sectors lost protection when their line flipped (OQ-12).
  - lud: Snake −5.1%. Throttling cuts issued prefetches by 30%.
  - IPC stays within 2% of Snake-DT on every app.
- Snake-ISO keeps Snake-DT's hit rate. Snake-DT < Snake-T ≤ Snake holds only for histo (accuracy).

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

**S4b (done, target missed):** a revised reading of mode 1, `-snake_decouple_variant 1` (the default; 0 = S4). Code: gpgpu-sim `3f635b67`; accel-sim `7af4b95`, `d06395d`. The rules (SPEC OQ-12, OQ-15):
- the untrained cap lasts at most 10 cycles per launch
- the prefetch side of a bulk free gives only stale lines
- the flip is sector-aware

Gates:
- Synthetic PASS; off-regression PASS (6); 15 runs, all invariants OK; variant 0 equals S4.
- **Step 1:** the cap re-arms only at an SM's launch start (no bug). 79–94% of the cap time in lud, histo and srad_v1 comes from SM-launches that never promote. Variant 1 cuts the cap from up to 53% of cycles to 0.1–0.3%.
- **§5.7 target missed** on hotspot (hit rate 26.6 vs ISO 32.5 and DT 32.3) and lud (8.2 vs DT 8.4, −3.0%). It is met on histo, mri-q and srad_v1. hotspot's prefetches are used ~900 cycles after issue, so the 50-cycle staleness frees them; lud's gap is throttling.
- Variant 1 is kept; the gap is in SPEC §8. Tables: `results/S4.md` (S4b).

**S5 (done): final-evaluation configs and tooling; nothing launched but 2 smoke runs.** accel-sim `2c3a333`; gpgpu-sim unchanged (`3f635b67`). Details: `results/S5.md`.
- **Configs** (1B instructions, 9 workloads):
  - Tier 1: 21 unique configs, 189 runs. Baseline; Snake exact and line; Snake-DT, Snake-T, s-Snake, Snake-ISO; the PAPER_LAT pair; Tail {20, 40, ∞ = 1024}; popcount at {10, 20, 40}; throttle {25, 75, 100, 150, 200, 300}.
  - Tier 2: 13 configs, 117 runs. Inter-warp degree {1, 2, 8}, miss-queue reserve 4, stale {200, 1000}, bytes meter, chain depth 3, variant 0, queue 64, issue 2, and the 1132 MHz pair.
- **Tools:** `s5_runs.sh` (list / dry / run), `collect_results.py` (build-tag, finish and invariant checks; energy), `compare_paper.py`, `s5_estimate.py`; `run_apps.sh` `DRY=1` / `RESUME=1`.
- **Dry-run:** all 11 run dirs generated with frozen builds (306 jobs), no sim.out. Two smoke runs confirm every new token parses.
- **Estimate** at 15 parallel: Tier 1 56.5 core-hours, ≈ 3.8 h; Tier 2 35.0 core-hours, ≈ 2.3 h (one config = 2.69 core-hours; histo is longest at ≈ 56 min). Linear extrapolation from 100M runs, ±50%.
- **Energy:** AccelWattch is off in our configs, so Fig. 19's normalized energy is unavailable. Snake's own energy is computed (SPEC OQ-16).
- SPEC §8 records hotspot's 897-cycle median prefetch use (paper: within 50 cycles) and lud's request-slot throttling.

**Next:** your approval to launch Tier 1 (`util/snake/s5_runs.sh tier1 run 15`), then Tier 2.
