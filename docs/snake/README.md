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

**Final evaluation, Tier 1 (done 2026-10-16): 189/189 jobs, all checks OK.** Details: `results/final_tier1.md`, per-job CSV `results/final_tier1.csv`.
- **Pre-flight:**
  - peak RSS 2.23 GB per job: 15 parallel ≤ 54% of RAM
  - 160 GB of disk free
  - debug trace off in all 306 configs
  - launch gate held: clean trees, frozen builds, no rebuild
  - AccelWattch costs +55% wall time (> 30%), so energy stays unavailable (SPEC §8)
- **Canary** (baseline + Snake × 9): clean. Each run's first 100M instructions reproduce the gates exactly; backprop's and srad_v1's full-run coverage is lower because of kernels the 100M gates never reached.
- **Ours vs the paper** (7-app averages):
  - coverage_pred 63.2% vs 75.7%; coverage_paper 21.9%
  - accuracy 6.1% vs 69.1%
  - IPC −0.3% vs +14.1%
  - demand L1 hit rate 18.7 → 25.2% (Snake-ISO 26.4) vs 45 → 79 (84)
  - energy unavailable
- **Ablations:** within ±0.2% IPC of each other.
- **Sweeps:** Tail 10 → 1,024 adds 1.2 pp coverage_pred. popcount is 1.9–3.4 pp below lru_popcount. A longer space halt lowers coverage_paper (22.2 → 17.6%).
- **Unlimited Tail (1,024):** 0 evictions except backprop (107,498).
- **Time:** 64.9 core-hours vs 56.5 estimated; 4 h 33 min vs ≈ 3.8 h.
- **Commits:** accel-sim `e89c9d0` (launch tooling, energy decision) and `snake: final evaluation Tier 1 results`; gpgpu-sim unchanged (`3f635b67`).

**S6: why there is no IPC gain (done 2026-10-16).** Details: `results/S6_ipc.md`; SPEC §8.
- **Ceiling.** With perfect memory, hotspot, mri-q and lud gain < 5%: the simulated GPU already hides their latency. backprop (+47%), histo (+30%) and srad_v1 (+22.5%) have headroom.
- **That headroom is bandwidth and queueing.** backprop and histo are DRAM-queue bound; srad_v1 and mri-q are limited by the L1 miss queue and MSHRs. Prefetching moves the same bytes: backprop's DRAM reads are identical in every config.
- **Snake's own limit is lead time:** inter-warp predictions come ~20 cycles before use.
  - Experiment `-snake_interwarp_scope 2` (later CTAs; default 0): backprop accuracy 0.2 → 14.0%, lead 1,602 cycles, but IPC +0.07% (DRAM-bound); hotspot −4.7% (2D grid).
- **Runs:** 8 ceiling, 8 experiment, 11 synthetic (identical to S4b), 6 off-regression (PASS).
- **Not fixable within Snake on these workloads.** A paper-like gain needs a more latency-bound baseline, which is a separate decision.
- **Commits:** accel-sim `40d0a04` (PERFECT_MEM), `4502b2d` (IWS tokens), results commit; gpgpu-sim `e85e7eb1` (`-snake_interwarp_scope`).

**D1: ceilings on exactly Snake's config (done 2026-10-16).** Details: `results/D1_ceilings.md`.
- **S6 confirmed.** The S6 perfect-memory runs match Tier 1's config, traces, 1B cap and build exactly, but covered 8 of 9 workloads (no nw).
  - Re-run on all 9: IPC bit-identical for the 8; nw +19.0%.
- **Source** (gpgpu-sim `7ed7810e`): `-gpgpu_perfect_mem` replaces the SM's memory interface (shader.cc:166-172, shader.h:2779-2796).
  - It removes all interconnect, L2 and DRAM latency, bandwidth and queueing.
  - The L1 latency queue, L1D tags, MSHRs, miss queue (one request per cycle) and the fill path remain, so loads still miss.
- **New `-gpgpu_perfect_l1d`** (gpgpu-sim `7ed7810e`, token `PERFECT_L1D`, default off): every L1-cached global load hits at L1 latency. Off-regression PASS (6); every perfect-L1D run hits 100% of its global loads.
- **Ceilings** (GMEAN over 7 apps; perfect memory / perfect L1 / Snake vs the paper's +14.1%):
  - PAPER_V100: +16.8% / +16.6% / −0.3%
  - PAPER_LAT: +18.4% / +17.9% / −0.2%
  - hotspot, mri-q and lud stay < 5% with a perfect L1; the headroom is in backprop, histo, srad_v1 and nw.
- **Runs:** 42 (18 perfect memory on Tier 1's frozen build, 18 perfect L1D, 6 off-regression).
- **Commits:** accel-sim `ac28769` (token, normalizer) and the D1 results commit; gpgpu-sim `7ed7810e`.

**K1: does the paper report its best-performing kernel? (2026-10-18, no new runs).** Details: `results/K1_best_kernel.md`; script `util/snake/per_kernel.py` (per-launch differences of the per-kernel stats blocks).
- **IPC.** The best kernel matches the paper only for backprop (+1.37% vs +1%, line footprint).
  - histo rises from −1.9% (whole run) to +11.5–12.0% on `histo_intermediates_kernel` (paper +33%).
  - srad_v1 +2.6% (paper +29%), lud ≤ +1.0% (+18%), hotspot ≤ +0.3% (+13%), mri-q ≤ +0.3% (+4%), nw 0 (+5%).
- **Per-kernel ceiling.** Even a perfect L1 gives at most +0.33% on hotspot and +8.5% on lud's best kernel, below the paper. Kernel selection cannot explain those two.
- **Coverage (predicted) of the best kernels is close to the paper:** hotspot 91–93 vs 88, srad_v1 92 vs 87, lud 83–86 vs 91, mri-q 88 vs 72, histo 83 vs 97.
- **Accuracy is not:** 0–27% (Snake-ISO hotspot 32%) vs 60–96%.

**Results workbook (2026-10-19, no new runs).** `results/Snake_results_2026-10-19.xlsx` (replaces 2026-10-18), generated by `util/snake/make_report_xlsx.py`; regenerate after new runs with `PYTHONPATH=<dir with openpyxl> python3 util/snake/make_report_xlsx.py [<date>]` (openpyxl installed in `/home/razul/snake/pylib`, outside git).
- **18 sheets:** Summary (with key findings and 3 charts), Best_kernel (best-performing kernel per workload vs the paper, 2 charts), Paper reference, Tier1_all (189 jobs), Ablations, Sweeps, Ceilings, PerKernel, Diagnostics_100M, Baseline_vs_paper, Trace_fix, S6_scope, Interpretations (32 OQs), Discrepancies (SPEC §8), Open_items, Glossary, Paper_specs (paper_specs.md Parts 1 and 4), Provenance.
- **Every number is read by the script** from raw `sim.out` files or the docs. Missing values are "n/a" with a reason. Every row has a status: Verified (1B final run) / Preliminary (100M gate run) / Diagnostic only / To check later.
- **Cross-checks** against final_tier1.md, D1_ceilings.md, K1_best_kernel.md (§1, §2), S6_ipc.md and final_tier1.csv: 525 values, 0 mismatches (listed in Provenance).
- `missq_check.py` gained a reusable `collect()`; its output is unchanged.

**Paper specifications vs ours (2026-10-19, no runs).** `results/paper_specs.md`: every spec the paper gives (Table 1, §4, Table 2, Snake parameters, §5.5 cost) quoted with page, against the effective option values of our runs (file:line), plus the specs the paper omits.
- **Table 1:** all sizes, ways, lines, banks, MSHR, SM, scheduler and thread counts match under PAPER_V100. The base QV100 differs (1132 MHz, LRR).
- **Measured vs setting:** L1 28 / L2 212 cycles are measured latencies. Our knobs 20 / 160 measure 25.0 / 173.0; PAPER_LAT's 23 / 199 measure 28.0 / 212.0.
- **Paper ambiguous:**
  - DRAM timings are labelled "(ns)", but they equal our config's DRAM-clock cycles at 850 MHz (1 cycle = 1.176 ns).
  - The register-file unit is not given.
  - "64-bank" L2 = 64 sub-partitions.
  - The effective L1 is 96–128 KB per kernel after the shared-memory carve-out.
  - "Instruction" in the 1B cap is undefined; we count thread instructions.
- **Differs:** Accel-Sim version; AccelWattch off; 6 of 9 comparison points not built; 4 ISPASS apps missing; our own Snake deviations (OQ-12/17/30/32).
- **SPEC.md fixes:** register unit; L1I 128 B line; DRAM units; MRQ abbreviation.

**SK_SC: Snake as SnakeCharmer's baseline (done 2026-10-26).** Details: `results/SK_SC.md`; SPEC OQ-33…OQ-36. SnakeCharmer (Tang et al., MICRO 2026, arXiv 2610.30990) uses Snake as its baseline "SK" and sets what the Snake paper leaves open (§III-A).
- **Config `SK_SC`:**
  - 128-entry queue
  - chain depth 10, with promoted deeper steps
  - intra-warp distance 8 (every iteration 1..8)
  - inter-warp: up to 63 resident warps by global warp id
  - confidence 3
  - `SK_SC_CTA` keeps the inter-warp targets in the CTA.
  - Defaults unchanged.
- **Code:** gpgpu-sim `9622012e` (3 options off by default, chain-depth limit 10, `sc_*` stats for SnakeCharmer's metrics); accel-sim `5d55de3` (tokens, rodinia_2.0-ft in `run_apps.sh`), `dc1a98c` (`sk_sc_report.py`).
- **Gates:** off-regression PASS (6); synthetic PASS (11, identical to S6 apart from the new lines); default Snake identical to S4b (5).
- **Runs:** 87 at 100M on PAPER_V100 and on base QV100 (LRR, 1132 MHz, SnakeCharmer's V100):
  - our 7 apps
  - the 6 SnakeCharmer workloads with local traces: b+tree, bfs, particlefilter float/naive, heartwall and streamcluster (the last two with other inputs)
  - SK_SC_CTA on hotspot and srad_v1
- **No A100 (sm_80) traces exist locally.** The oracle A100 runs use the sm_70 `hw_run` traces; the GhOST tarball is sm_75. mst, sssp, gemm, AlexNet and ResNet have no traces. bfs's traces are 24–25% mis-decoded after the parser fix.
- **Build-tag audit** of runs since 2026-10-22: none mixed accel-sim `scap` with gpgpu-sim `Snake`. The only ones are `sim_run_oracle_coalesced`'s (both repos on `oracle-coalesced-miss`).
- **Ours vs SnakeCharmer's SK** (A100: coverage 11.5%, accuracy 23.1%, on time / too early / too late 48.3 / 28.7 / 23.0%). SK_SC, base QV100, its 6 workloads:
  - coverage 16.4% (27.2% with late uses)
  - accuracy 66.2%
  - timeliness 38.8 / 4.7 / 56.5%
- **IPC vs no prefetching** (SnakeCharmer reports none for SK). GMEAN over 15 workloads, SK_SC +1.4% / +3.0% vs Snake +4.3% / +3.7% (PAPER_V100 / QV100). Without particlefilter_naive (+79% / +65%, latency-bound at 5.9% occupancy): −2.6% / −1.0% vs +0.9% / +0.3%.
- **hotspot:** SK_SC −11.0% / −6.3% on PAPER_V100, because inter-warp targets in other CTAs are wrong on its 2-D grid. SK_SC_CTA restores it (+0.15% / −0.02%). On base QV100 there is no drop (+0.4% / +1.1%). In S6, scope 1 at degree 4 gave +0.10% and scope 2 gave −4.7%.
- **Timeliness on our 7 apps** rises on average (on time 21.5 → 28.3% PAPER_V100, 8.2 → 10.8% QV100), mostly from backprop, without IPC (7-app GMEAN −2.95% vs −0.08%).

**SK match on base QV100, Step 1 (done 2026-10-27).** Details: `results/SK_match_qv100.md`; SPEC OQ-37, OQ-38. Why do our timing split and accuracy differ from SnakeCharmer's SK? All runs are on base QV100 (SnakeCharmer's V100: LRR, 1132 MHz), 100M. Workloads: our 7 apps plus b+tree, particlefilter float/naive, heartwall and streamcluster (no bfs).
- **Every SK variant:** SK_SC with same-CTA inter-warp targets (up to 63) and SnakeCharmer's 64-entry per-warp Head (`SNAKE_HEAD64` = `-snake_head_two_warps 0`).
- **Configs:**
  - SK (reserve 8, decoupling on)
  - (a) SK_R0: miss-queue reserve 0
  - (b) SK_R0_D0: reserve 0, decoupling off
- **Code:** tokens only (accel-sim `2a5c57c`), on the SK_SC build `5d55de3` / `9622012e`. The R2 injection-path spec is saved in `R2_SPEC.md`, not implemented.
- **Runs:** 42 plus 1 Fig. 15 check (PASS: the per-warp Head trains the figure's inter-warp stride).
- **Verdict:** neither (a) nor (b) moves towards SnakeCharmer's 48.3 / 28.7 / 23.0% and 23.1% accuracy.
  - On SnakeCharmer's 5 workloads SK gives 42.2 / 0.0 / 57.8% and 72.6%, (a) 40.3 / 0.0 / 59.6% and 73.6%, (b) 40.7 / 0.0 / 59.3% and 72.9%.
  - Too early stays at 0 even with decoupling off: prefetches are late, not early.
- **IPC:** (b) is best (GMEAN over 14: +4.77% vs SK +3.97%; histo −8.4 → −1.2%). (a) costs histo (−11.0%) and raises demand MISS_QUEUE_FULL.
- **Same-CTA targets** fix b+tree (−12.6% → −2.6%). The per-warp Head trains the same inter-warp PCs as the two-warp rows on 11 of 14 workloads.
- **Proposed Step 2 base:** SK_R0_D0.

**Next:** your go-ahead on Phase 1. Tier 2 (117 jobs) also still awaits approval. SK match Steps 2–4 (R2, ceilings, bfs diagnosis) await approval.
