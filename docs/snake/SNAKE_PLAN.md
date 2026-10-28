# Snake — implementation plan (Phase 3)

Paper: Mostofi et al., "Snake: A Variable-length Chain-based Prefetching for GPUs", MICRO 2023 (`docs/snake.pdf`, kept out of git).
Bases: accel-sim-framework `3016c65`, gpgpu-sim `6c3cf4ff` (both on branch `Snake`). All `file:line` references are at those commits.
Companion docs: [SPEC](SPEC.md) · [VERSION_DIFF](VERSION_DIFF.md) · [CODE_MAP](CODE_MAP.md) · [SNAKE_PLAN](SNAKE_PLAN.md)

## 0. Decisions and status

**Decisions (plan approved 2026-10-08):**
- **Base:** implement on accel-sim `3016c65` / gpgpu-sim `6c3cf4ff`, not on the v1.2.0 tag (VERSION_DIFF, Recommendation).
- **Benchmarks:** 7 apps; ISPASS (CP, LPS, LIB, MUM) is deferred (OQ-13).
- **Rodinia version:** 3.1 (OQ-24).
- **Clock:** 1530 MHz for the main comparison, plus one 1132 MHz sensitivity run (OQ-23).
- **Comparisons:** Snake ablations only; no competitor prefetchers (OQ-25).
- **Decoupling default:** `snake_decouple_mode 1`, the unified flag partition (OQ-12).
- **Everything else:** the proposed interpretations of OQ-1…22 apply.

**Status** (each stage = one commit per changed repo; hashes are filled in as stages land):

| Stage | Status | accel-sim | gpgpu-sim | Test result |
|---|---|---|---|---|
| S0 analysis docs (this file and its companions) | done | `84251a0` | `6c3cf4ff` (base) | — |
| S1 infrastructure | done | `d49e1f3` | `e0d23397` | off-regression PASS on 5 workloads (backprop, hotspot ×2, nw, histo): ref == off, off == on, raw cycles/insn/L1/DRAM equal (`sim_run_snake_regress/S1`, 2026-10-08) |
| S2 training only | done | `e4030cd` | `ace5cc70` | synthetic PASS (Fig. 15 Head/Tail and predictions exact; strided trained; random none promoted; `sim_run_snake_synth/S2`). Off-regression PASS on 6 workloads (+ srad_v1): ref == off, off == on, raw stats equal (`sim_run_snake_regress/S2`). Plausibility: srad_v1 plausible (Tail hits 74%, 2331 promotions); hotspot / backprop **not**: the §3.4 lane check excluded 100% / 99.1% of their loads → OQ-17 decision |
| S2b lane-pattern mode (OQ-17) | done | `5b4f096` | `b9cdc46a` | synthetic PASS in lane modes 1 and 0 (`sim_run_snake_synth/S2b`); off-regression PASS on 6 workloads (`sim_run_snake_regress/S2b`); plausibility in `sim_run_snake_train/S2b`: hotspot trains in mode 1 (Tail hits 99%, 80/85 promotions; mode 0 excludes 100%); backprop blocked by mis-decoded trace addresses → trace-parser fix |
| Trace-parser fix, pre-756ed33 traces (not Snake code) | done | `fe8f37e` | — | exact on the 172,032 / 5,314 / 4,000 affected backprop / hotspot / srad_v1 lines; effect on the baseline in VERSION_DIFF "Pre-756ed33 traces" |
| Regression reference `bin_ref_v2` + legacy-trace tooling | done | `bed3a2b` | — | synthetic PASS in both lane modes (`sim_run_snake_synth/S2c`); off-regression against `bin_ref_v2` PASS on 6 workloads (`sim_run_snake_regress/S2c`); plausibility re-run in `sim_run_snake_train/S2c` (S2 notes) |
| Docs: trace-parser fix, upstream issue draft | done | `61c0dc1` | — | — |
| S2c stateless lane-run filter + footprint setting | done | `293aaaa` (gate results: `snake: S2c gate results`) | `bc22d0dd` | synthetic PASS in lane modes 1 and 0 (Fig. 15 exact; random: all 512 scattered loads excluded; `sim_run_snake_synth/S2c2`). Off-regression vs `bin_ref_v2` PASS on 6 workloads, ref outputs reused from S2c (`sim_run_snake_regress/S2c2`). Plausibility (mode 1, the off-regression `on` runs): hotspot 512/1024 excluded 0%, all L=16, Tail hits 99.3%, promotions 80/88, 2 inter-warp PCs, chain depth 1 for 35%/37%; backprop excluded 0%, all L=16, 0 promotions (SPEC §8), interwarp_trained 32,624 on 2 PCs; srad_v1 excluded 2.5%, Tail hits 74.2%, 2,337 promotions, 16 inter-warp (kernel, PC) pairs, chain depth 0/1/2/3: 89.8/9.9/0.3/0.1% |
| S1 docs + regression tooling | done | `b930d75` | — | — |
| S3a issue: inter-thread (chain, promotion) + intra-warp, Snake-DT | done | `1221702`, `9dc2631`, `d836622` (gate results: `snake: S3a gate results`) | `028012de`, `5c6f2267`, `2d50a9aa` | synthetic PASS (trace (d) coverage_paper 93.0% exact and line; (b) 0 issued; (a) saturates without deadlock; Fig. 15 PASS); 7 apps × 2 footprints: all finish, `pf_woken_warps` 0, invariant OK; off-regression vs `bin_ref_v2` PASS (`sim_run_snake_regress/S3a`, `S3a2`); miss-queue reserve 8 from the lud sweep. Tables: `docs/snake/results/S3a_gate.md` |
| S3b inter-warp issue + combined lookahead + inter-warp eviction weight | done | `bf1fc2f`, `a708e99` (gate results: `snake: S3b gate results`) | `298c5f94`, `000798a7`, `f1e4f856` | synthetic PASS (`sim_run_snake_synth/S3b3`): Fig. 15 (c) exact with `SNAKE_IWC0` and figure + {3100, 3200, 3300} by default; trace (d) 90.8%. 7 apps × line (9 runs) and 7 exact (> 5 pp coverage change): all finish, `pf_woken_warps` 0, invariant OK; off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/S3b`). Tables: `docs/snake/results/S3b_gate.md` |
| S3c diagnosis (stats only): prediction funnel, timing, miss queue | done | `59e5bee`, `b7cae56` (results: `snake: S3c diagnosis results`) | `02a9d7de` | synthetic PASS, outputs identical to S3b3 apart from volatile lines; Snake-on runs identical to S3b_fp1 (cycles, issued, matched); off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/S3c`); 9 diagnosis runs (`sim_run_snake_gate/S3c`). Tables: `docs/snake/results/S3c_diag.md` |
| Miss-queue / memory-path config check (no runs) | done | `50570b0` | — | config matches Tab. 1 except L1 latency 20 (28) and L2 ROP latency 160 (212); MISS_QUEUE_FULL causes 71–100% of reservation fails; miss queue stays 16. `docs/snake/results/missq_config_check.md` |
| S3d promoted strides for all warps (OQ-32) + funnel by source | done | `976937f` (results: `snake: S3d results`) | `97119f36` | synthetic PASS (Fig. 15 unchanged, trace (d) 90.8%); off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/S3d`); 10 runs (`sim_run_snake_gate/S3d`), all invariants OK. `docs/snake/results/S3d.md` |
| S3e paper defaults: inter-warp chain depth 0, deeper chain steps need bit + trained; PAPER_LAT config | done | `20e968f`, `131a513` (results: `snake: S3e results`) | `53bbfd25` | synthetic PASS (Fig. 15 exact; trace (d) 90.8% under `SNAKE_IWC3`, default 86.0% reported); off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/S3e`); 5 runs (`sim_run_snake_gate/S3e`), all invariants OK; PAPER_LAT calibrated to L2 hit 212.0 (`sim_run_snake_calib`). `docs/snake/results/S3e.md` |
| S4 throttling + decoupling; PAPER_LAT L1 knob 23 | **gate stopped** (stop rule) | `f708b4f` (results: `snake: S4 results`) | `57103274` | PAPER_LAT measures L1 28.0 / L2 212.0 (`sim_run_snake_calib/paper_lat23`); synthetic PASS (`sim_run_snake_synth/S4`, 11 runs: Snake-DT runs equal S3e; trace (b) under full Snake throttled 89.4%, 0 issued); off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/S4`); 15 runs (`sim_run_snake_gate/S4`) finish, all invariants OK, no quota assert. **Stop:** mode 1 lowers the demand L1 hit rate vs Snake-DT by > 2% on hotspot (Snake −19.6%, Snake-T −20.6%) and lud (Snake −5.1%); IPC within 2%. `docs/snake/results/S4.md` |
| S4b revised decoupling mode 1 (`-snake_decouple_variant 1`) + cap diagnosis | done; §5.7 target missed (kept, gap in SPEC §8) | `7af4b95`, `d06395d` (results: `snake: S4b results`) | `3f635b67` | synthetic PASS (`sim_run_snake_synth/S4b`, 11); off-regression PASS on 6 (`sim_run_snake_regress/S4b`); 5 step-1 runs (`S4b_diag`) + 10 gate runs (`S4b`), all invariants OK; variant 0 = S4 exactly. Hit rate v1 vs ISO / DT: hotspot 26.6 vs 32.5 / 32.3 (miss), lud 8.2 vs 8.3 / 8.4 (−3.0%, miss), histo, mri-q, srad_v1 meet it. `docs/snake/results/S4.md` (S4b) |
| S5 final-evaluation configs + tooling (no runs) | done; Tier 1 run (below) | `2c3a333` (results: `snake: S5 dry-run and estimate`) | — (`3f635b67`) | Tier 1 21 configs / 189 runs, Tier 2 13 / 117; dry-run generated 11 run dirs (306 jobs, frozen build `2c3a333`/`3f635b67`, no sim.out); 2 smoke runs parse every new token; estimate Tier 1 ≈ 3.8 h (56.5 core-hours), Tier 2 ≈ 2.3 h (35.0) at 15 parallel. `docs/snake/results/S5.md` |
| SK_SC: SnakeCharmer's Snake baseline (options, SnakeCharmer metrics, 87 runs) | done | `5d55de3`, `dc1a98c` (results: `snake: SK_SC results`) | `9622012e` | off-regression vs `bin_ref_v2` PASS on 6 (`sim_run_snake_regress/SK_SC`); synthetic PASS (11, identical to S6 apart from the new lines; `sim_run_snake_synth/SK_SC`); default Snake identical to S4b on its 5 PAPER_V100 runs; 87 runs at 100M (`sim_run_snake_skc`), all invariants OK; SK_SC_CTA run because hotspot dropped > 2%. `docs/snake/results/SK_SC.md` |
| SK match Step 1 (base QV100): per-warp Head, same-CTA targets, miss-queue reserve 0, decoupling off | done; Steps 2–4 wait for approval | `2a5c57c` (tokens, R2 spec), results: `snake: SK match Step 1 results` | — (`9622012e`) | no new build (SK_SC build and gates); Fig. 15 with `SNAKE_HEAD64` PASS (`sim_run_snake_synth/SKM_HEAD64`); 42 runs at 100M (`sim_run_snake_skm/qv100`), all invariants OK. Neither reserve 0 nor decoupling off moves the timing split or accuracy towards SnakeCharmer's. `docs/snake/results/SK_match_qv100.md` |
| fix_today (Tier 1 conditions): R2, accuracy filter, single inter-warp target with CTA anchoring | done; no config meets (a) histo ≥ 0 and (b) none below −1% | `22c4c23`, `c09021a`, `4b51495`, `5465d33`, results: `snake: fix_today` commits | `bb20e7aa` (R2), `19e3d746` (OQ-39, OQ-40) | gates: off-regression PASS (R2, P34b), synthetic PASS (R2b, P34, Fig. 15 with R2 and with the filter); 90 runs at PAPER_V100, 1B (`sim_run_snake_fix/{p2,p3,p3b,p4}`), all checks OK. Best GMEAN G8 −0.24% (Tier 1 Snake −0.29%); histo −1.6 to −5.0% everywhere. `docs/snake/results/fix_today.md` |

**Reference build for the off-regression (§9.1), from 2026-10-13:** `/home/razul/snake/bin_ref_v2/` = the clean bases plus the trace-parser fix `fe8f37e` only (cherry-picked as `6c28685` on local branch `snake-ref-v2`; `accelsim-commit-6c28685_modified_0.0`, `gpgpu-sim_git-commit-6c3cf4ff_modified_0.0`; built in the worktree `/home/razul/snake/src_ref_v2`; checksums in its `BUILD_INFO.txt`). See VERSION_DIFF "Pre-756ed33 traces".
**Previous reference, kept as `/home/razul/snake/bin_ref_paperera/` (was `bin_ref`, used for S1–S2b):** it holds `accel-sim.out` plus `lib/libcudart.so`, which contains the gpgpu-sim model. It was built 2026-10-08 from the clean bases (`accelsim-commit-3016c65_modified_0.0`, `gpgpu-sim_git-commit-6c3cf4ff_modified_0.0`); checksums are in `BUILD_INFO.txt`.

**Rule:** a stage's own commit cannot contain its hash or the result of a gate run on it. Both are recorded in the next stage's commit.

**S2 notes (interpretations made while implementing training; each is marked in the code with its OQ number):**
- **T1/T2 codes (OQ-5).** The states are untrained → observed (fewer than `promote_warps` warps) → promoted → trained. Fig. 15 prints them with different codes (see SPEC OQ-5 correction), so the codes are labels only.
- **T1 trained (OQ-5).** A promoted entry becomes trained on repetition, which is either a warp already in the vector matching the same (PC1, PC2, stride) again, or a warp returning to PC1 through the entry while its intra-warp stride is computed (Fig. 15c).
- **Promotion-time predictions (§3.2 ¶4, OQ-27).** When an entry is promoted, every other warp whose Head register is at PC1 and whose bit is not set gets a prediction for PC2 (Fig. 15b, 1800).
- **Demotion (§3.2, OQ-28).** It applies to promoted or trained entries; when more than `demote_removed` warps have been removed, T1 drops to untrained and the warpID vector is cleared.
- **Lane exclusion (OQ-17).** An excluded load also invalidates its warp's Head register, so no stride is computed across it.
- **Plausibility after the trace-parser fix** (`sim_run_snake_train/S2c`, lane mode 1 / lane mode 0):
  - **hotspot 512 / 1024:** excluded 5.7% / 2.4% in mode 1 (100% in mode 0); Tail hits 99.2% / 99.3%; promotions 80 / 88; chain depth 0: 66% / 63%, 1: 34% / 37%. Plausible.
  - **backprop:** excluded 0% in mode 1 (100% in mode 0: the two printed lanes of PC 0x90 are 16 lanes and 4 B apart, a non-integer lane stride); 0 promotions, chain depth 0 only. Not a defect. The 0x90 → 0x180 stride grows by 128 B per warp (0x90 advances 8 B per warp, 0x180 136 B; `hw_run/.../backprop-rodinia-3.1/65536/traces/kernel-1.traceg`), so no two warps share an inter-thread stride. Each lookup allocates a condition-3 entry (31,989 of 32,069), and the 10-entry Tail thrashes (35,281 evictions). Only the per-PC inter-warp strides are learnable (2.33 inter-warp predictions per load). The paper's backprop gain is also small (~1% IPC, Fig. 18).
  - **srad_v1:** unchanged by the fix (no mis-decoded lines in the simulated kernels); mode 1 excludes 3.9%, mode 0 2.6%; Tail hits 74%; 2207 / 2331 promotions; chain depth 0: 90%, 1: 10%, 2: 0.3%, 3: 0.1%. The `reduce` launches have 26 and 40 load PCs against 10 Tail entries (`util/snake/load_pcs.py`), the `srad` kernel 9.
- **Lane mode (OQ-17).** S2b made `-snake_lane_mode 1` a per-PC lane-offset pattern stored in the Head rows (≈ 4 KB per SM). **S2c replaces it with a stateless run filter:** the active lanes must fit base + (i mod L)·lane_stride + ⌊i/L⌋·run_stride for L ∈ {32, 16, 8, 4}, and nothing is stored per PC. `-snake_lane_mode 0` (config token `SNAKE_LANE0`) is the paper-literal rule. Details are in SPEC OQ-17.
- **Footprint setting (OQ-9, S2c).** `-snake_footprint 0` (default) means one 128 B line per prediction, with only its missing sectors requested. `1` keeps the PC2 load's lane descriptor in each Tail entry (≈ 8 B). S2c adds the option and the storage only; S3 issues the requests.
- **Inter-warp PCs (S2c).** `snake_interwarp_trained_pcs` counts the distinct (kernel launch, PC) pairs, over all SMs, that got a trained inter-warp stride. `interwarp_trained` counts training events, which repeat when an entry is evicted and re-allocated (backprop: 32,624 events for 2 PCs in S2c).
- **Step-1 inter-warp check (2026-10-13, no new runs).** The S2c training runs (mode 1) gave `interwarp_trained` = 32,624 for backprop, 160 for each hotspot input and 14,487 for srad_v1. The traces show 2, 2 and 9 load PCs with a fixed per-warp stride, respectively; the srad count is for its `srad` kernel (1 or 2 in the earlier kernels). hotspot's 160 = 2 PCs × 80 SMs.
- **Inter-warp stride (OQ-6).** It is stored in every Tail entry with that PC1, or in a new PC1-only entry (Fig. 15a), and the latest consistent stride replaces an older one. `snake_interwarp_min_warps` is limited to 2 or 3, because the Head keeps two warps per PC. With `-snake_head_two_warps 0`, the candidates are the per-warp registers whose last load was this PC, nearest warp ID first.
- **Intra-warp stride (OQ-26).** It is stored in every Tail entry with that PC1. The chain walk follows the most recently used qualifying entry at each step.
- **Inter-warp prediction targets (OQ-8).** These are the next `interwarp_degree` warps above the current one that were initialised in the same CTA slot. Warp liveness is not tracked in S2.
- **Fig. 15 test layout.** With a single LRR scheduler, warp 0 issues last in each round, so the test CTA has 5 warps: warp 0 only exits, and W1..W4 are warps 1..4. That keeps warp IDs, warpID bits and the per-warp-ID inter-warp stride identical to the figure.
- **Clean builds.** An incremental build of S2 crashed at start-up (null `m_cluster` in `gpgpu_sim::init`), and a clean build of the same sources ran correctly. Every binary that is tested or used for results is therefore built with `make clean` first.

**S1 notes (clarifications of the plan, no change of design):**
- **Injection API deferred.** `l1_cache::snake_prefetch()` (§2) and the `shader_core_ctx` getters `get_warp_cta_slot` / `warp_active` arrive in S3, where they are first used. They are calls *from* Snake, not hooks; every hook in §2/§4 is wired in S1.
- **One hook in place of two.** `on_kernel_switch` is detected inside `on_warp_init` (kernel uid change, OQ-19), so `init_warps` has one hook.
- **App selection.** No new app suite is needed. `run_simulations.py -B` accepts `suite` and `suite:exe` selectors (`util/job_launching/common.py:110-131`), e.g. `rodinia-3.1:backprop-rodinia-3.1`; `suite:exe` runs every input listed for that app (hotspot and lud have two).
  **Correction (S1 docs commit):** `suite:exe:N` is registered by `common.py:113-121` but crashes `run_simulations.py:108` (`TypeError: 'int' object is not subscriptable`), because that entry stores its argument list as a bare string. Do not use it.
- **What "byte-identical" excludes.** gpgpu-sim prints every registered option at start-up (`OptionParser::Print`, `src/option_parser.cc:422`), so the new `-snake_*` options add lines to that dump. The off-regression therefore removes exactly those lines, and the volatile lines (dates, wall-clock, simulation rate, build strings, binary paths), and then requires the rest to be byte-identical.
- **Pin the model library.** In trace mode `run_simulations.py` copies only `accel-sim.out` into `gpgpu-sim-builds/` (`run_simulations.py:450-480`). The timing model lives in `libcudart.so`, which jobs load through `LD_LIBRARY_PATH` at run time, so a rebuild can change the model under running or queued jobs. Every Snake run uses a frozen `accel-sim.out` + `libcudart.so` pair and sets `LD_LIBRARY_PATH` explicitly.

## 1. Design summary
- **One object per SM.** Each `ldst_unit` owns a `snake_prefetcher`. The object is only constructed when `-snake_enable 1`; with Snake off, every hook is a null-pointer check.
- **Training.** `ldst_unit::issue` passes (warp slot, CTA slot, PC, active mask, lane addresses, stream) to `snake_prefetcher::observe()` for L1-cached global loads. Observe does:
  - the lane check (OQ-17): the stateless run filter (L ∈ {32, 16, 8, 4}) by default, or the §3.4 equal-stride rule in lane mode 0
  - the Head update, which produces (q, p, s)
  - Tail match / allocate / bit-set, plus inter-thread, intra-warp and inter-warp training (§3.1)
  - prediction (§3.2): it computes target warp base addresses and expands each one (OQ-9, `-snake_footprint`): by default the 128 B line holding the address (only missing sectors), or with `1` the target load's lane descriptor (PC2's, from the Tail entry, for inter-thread predictions; the triggering load's for inter-warp and intra-warp)
  - results go into a bounded **prefetch queue**, tagged ready at `now + snake_latency` (2 cycles, §5.5)
- **Issue.** `ldst_unit::cycle` calls `snake->tick()` after `L1_latency_queue_cycle()`, so demand requests win the port. Tick:
  - pops ≤`snake_issue_width` ready entries, unless throttled
  - calls `l1_cache::snake_prefetch(addr, sector, …)`
  - that probes tags (drop if HIT, HIT_RESERVED or MSHR-present: "duplicate")
  - otherwise checks MSHR slack and the miss queue (drop otherwise: "dropped"), then sends through `send_read_request` with type **`L1_PREFETCH_R`** and no instruction
- **Return.** The response follows the normal fill. In `writeback` case 4, prefetch mfs are deleted without waking anything.
- **Bookkeeping.** The sector gets `m_snake_pf=1` at fill:
  - first demand hit → **useful (timely)**
  - demand merge into a prefetch's MSHR → **late**
  - eviction with the bit still set → **evicted-unused**
- **Decoupling** (S4, `snake_decouple_mode`):
  - 1 = paper (§3.2): a line-level region flag, set at allocation, and the victim choice in `tag_array::snake_victim`; a demand access to a prefetch-region line flips it; a set without a free line gets a 25% bulk free (side by the 80% transfer rule); 50%-until-trained and 50-cycle confinement cap L1 data (SPEC OQ-12, OQ-15).
  - 2 = isolated buffer (§5.7): 256 lines beside the L1D; a demand that finds its sector there moves it into the L1D (OQ-22).
  - 0 = none (Snake-DT / s-Snake ablations).
- **Throttle** (S4, §3.3): the space event halts issue for 50 cycles; the bandwidth meter (request slots of the L1D miss-queue drain, 70%/50% hysteresis, OQ-10) halts until the level recovers. While throttled the chain depth is capped at 1 (OQ-7).

## 2. File-level change list
**New (gpgpu-sim `src/gpgpu-sim/`)**
- `snake_prefetcher.h/.cc`:
  - `snake_config` (all options, `reg_options(option_parser_t)`)
  - `snake_head_table`, `snake_tail_table`, `snake_pf_queue`, `snake_throttle`
  - `class snake_prefetcher { observe(); tick(); on_fill(); on_demand_hit(); on_late_merge(); on_evict(); on_kernel_switch(); on_warp_init(); print_stats(FILE*); }`
  - `snake_stats` (per SM, aggregated by `gpgpu_sim`)
- `snake_isolated_buffer.h/.cc` (S4, mode 2 only).
- Added to `src/gpgpu-sim/CMakeLists.txt` and the `Makefile` source lists.

**New (accel-sim)**
- `docs/snake/*.md`
- `util/job_launching/configs/define-standard-cfgs.yml` entries: `SNAKE`, `SNAKE_DT`, `SNAKE_T`, `S_SNAKE`, `SNAKE_ISO`, `PAPER_V100` (GTO + 1530 MHz clock), `PAPER_V100_1132` (GTO, stock 1132 MHz; OQ-23 sensitivity run), `SK_SC` and `SK_SC_CTA` (SnakeCharmer's SK settings, SPEC OQ-33), `SNAKE_HEAD64` (per-warp Head, OQ-37), `SNAKE_MQR0`, `SNAKE_DM0` (SK match Step 1), `SNAKE_R2`, `SNAKE_PQARB1`, `SNAKE_PQMSHR1` (R2, OQ-38), `SNAKE_ACCF`, `SNAKE_ACCFB10K` (accuracy filter, OQ-39), `SNAKE_IWD8/16/32` (single inter-warp target with CTA anchoring, OQ-40)
- `util/job_launching/apps/` snake-paper suite
- `util/job_launching/stats/snake_stats.yml`
- `util/snake/gen_synthetic_traces.py` (S2 microbenchmarks)
- `util/snake/compare_paper.py`

**Existing gpgpu-sim files touched (each hook marked `// SNAKE:`)**

| File:function | Hook | Why |
|---|---|---|
| AHM.h:772 `MEM_ACCESS_TYPE_TUP_DEF` | add `L1_PREFETCH_R` | tag that survives the L2 split |
| shader.cc:4645 `update_icnt_stats`; `traffic_breakdown.cc:32` | add case | otherwise `assert(0)` |
| gpu-cache.cc:938 `cache_stats::print_stats` | skip all-zero `L1_PREFETCH_R` rows | byte-identical output with Snake off |
| gpu-sim.cc:328 `shader_core_config::reg_options` | `snake_config::reg_options` | options |
| shader.h `ldst_unit` (≈1449) + ctor shader.cc:2647 | `snake_prefetcher *m_snake` (nullptr when off) | ownership |
| shader.cc:2674 `ldst_unit::issue` | `observe()` | training trigger |
| shader.cc:2912 `ldst_unit::cycle` | `tick()` after `L1_latency_queue_cycle` | issue (demand first) |
| shader.cc:2794 `ldst_unit::writeback` case 4 | drop prefetch mfs | no warp wake-up |
| shader.cc:4733 `icnt_cycle` | skip `memlatstat_read_done` for prefetches | clean latency stats |
| gpu-cache.h/.cc `l1_cache` | new `snake_prefetch()` (probe + `send_read_request`) | injection |
| gpu-cache.h `sector_cache_block`/`line_cache_block` | `m_snake_pf[]`, `m_snake_region` | usefulness and decoupling |
| gpu-cache.cc:353, 371, 422, 439, 450, 464 `tag_array` | callbacks: hit, evict, fill, flush, invalidate | stats and decoupling |
| gpu-cache.cc `tag_array::access` (MISS, SECTOR_MISS) | S4: `snake_victim` (mode 1 regions, L1 cap, bulk free; space trigger), `on_allocate` (region, flip) | decoupling |
| gpu-cache.cc `l1_cache::access`, `baseline_cache::fill`, `l1_cache::snake_prefetch` | S4 mode 2: buffer → L1D move, fill routing, no L1D line for a prefetch | isolated buffer |
| gpu-cache.cc:569 `mshr_table::add` | demand merge into a prefetch entry → late | timeliness |
| shader.h `shader_core_ctx` | `get_warp_cta_slot(w)`, `warp_active(w)` getters | inter-warp targets |
| shader.cc:533/574 `init_warps` | `on_warp_init(w)`, `on_kernel_switch()` | OQ-19/20 |
| gpu-sim.cc:1535 `gpu_print_stat` | Snake stats block | reporting |

**Accel-sim C++:** none expected. PC and warp ID come through `warp_inst_t`.

## 3. Data structures (field ↔ SPEC §2)
```cpp
struct snake_head_warp  { bool valid; new_addr_type pc, addr; };        // per HW warp slot (OQ-1a)
struct snake_head_pcrow { bool valid; new_addr_type pc;                 // OQ-1b, snake_head_entries rows
                          struct { bool v; unsigned wid; new_addr_type addr; } w[2]; uint64_t lru; };
enum snake_train : uint8_t { UNTRAINED=0, OBSERVED=1, PROMOTED=2, TRAINED=3 }; // OQ-5
struct snake_tail_entry {
  bool valid; new_addr_type pc1, pc2;           // PC1, PC2 (pc2 may be invalid: inter-warp-only entry, Fig.15a)
  long long inter_thread; snake_train t1;       // fields 3,4
  std::bitset<MAX_WARP_PER_SHADER> wvec;        // field 5
  unsigned removed_warps;                       // ">2 removed" rule (§3.2)
  long long intra; snake_train t2; unsigned intra_confirm; std::bitset<..> intra_warps; // 6,7
  long long inter_warp; bool inter_warp_valid;  // 8 (no train bits)
  uint64_t lru_stamp;
};
struct snake_pf_req { new_addr_type sector_addr; unsigned target_wid; uint8_t mode /*THREAD,WARP,INTRA*/;
                      uint8_t depth; unsigned long long ready_cycle; unsigned stream; };
```

## 4. Hook points and cost

| Hook | Observes / does | Cost |
|---|---|---|
| `ldst_unit::issue` | `observe()` for each L1-cached global load | O(32) lane check + O(tail=10) CAM + O(depth·10) chain walk |
| `ldst_unit::cycle` | `tick()`: throttle update + ≤`issue_width` L1 probes | O(1) |
| writeback case 4 | discard prefetch mfs | O(1) |
| `tag_array` hit / fill / evict | bit set/clear + counter | O(1) |
| `tag_array::probe` (mode 1) | quota-aware victim | O(assoc), only when Snake is on |
| `mshr_table::add` | late detection | O(1) |

Budget with Snake on: **≤15% slower** simulation. With Snake off: **within run-to-run noise** (target <1%).

## 5. Prefetch request lifecycle
1. **Predict** in `observe`.
   - Order: promotion-time predictions, inter-thread chain (depth ≤ `max_chain_depth` × throttle factor), then inter-warp (≤`interwarp_degree` resident targets, then each target's promoted chain ≤ `interwarp_chain_depth`, OQ-29), then intra-warp.
   - Each target warp base is expanded to sectors (OQ-9) and grouped into **one queue entry per 128 B line** (OQ-4, S3a). Sectors already queued → `pf_dup_queue`.
   - A full queue drops the newest line (`pf_drop_queue_full`, in sectors). From S3b an own-warp line (inter-thread, promotion, intra-warp) first displaces the newest inter-warp line not yet being offered (`pf_queue_displaced`, OQ-31).
2. **Wait** `snake_latency` cycles.
3. **Issue** in `ldst_unit::snake_issue` after the demand stage, when not throttled. Up to `issue_width` lines per cycle (from S3b: the line already being offered, else the oldest ready own-warp line, else the oldest ready inter-warp line; OQ-31), each sector to a bank no demand used this cycle (OQ-4). `l1_cache::snake_prefetch`, per sector:
   - tag probe HIT or HIT_RESERVED → `pf_dup_l1`
   - MSHR has the sector → `pf_dup_mshr`
   - no line can be reserved → `pf_drop_line`; MSHR free ≤ `mshr_reserve` → `pf_drop_mshr`; miss queue full → `pf_drop_miss_queue` (together `pf_drop_resource`)
   - else allocate MSHR plus miss queue → `pf_issued`
   - Never RESERVATION_FAIL-retry (no head-of-line blocking).
4. **Arbitration.** Demand always goes first: tick runs after the demand stage-0 access, and only uses miss-queue slots not needed by demand.
5. **In flight.**
   - A demand miss to the same line merges into the MSHR → `pf_late` (counts toward coverage, not toward paper "accuracy").
   - A demand RESERVATION_FAIL caused by prefetch-held MSHRs → `pf_caused_resfail` (risk monitor).
6. **Fill.** `baseline_cache::fill` → sector `m_snake_pf=1`, `pf_filled`; region = prefetch (mode 1).
7. **First demand hit** → `pf_useful_timely`, clear the bit, flip region to L1 (§3.2).
8. **Eviction or flush with the bit set** → `pf_evicted_unused`.
9. **End of kernel** → still-resident unused lines are counted as `pf_unused_at_end`.

## 6. Config options (gpgpu-sim, `-snake_*`)

| name | type | default | meaning / source |
|---|---|---|---|
| `snake_enable` | bool | 0 | master switch (rule 1) |
| `snake_head_entries` | uint | 32 | Head PC rows (Tab. 3, §5.5) |
| `snake_head_two_warps` | bool | 1 | two warps per PC under GTO (§3.1, §5.5) |
| `snake_tail_entries` | uint | 10 | Tail size (Tab. 3, §5.5, Fig. 20) |
| `snake_tail_evict` | enum str | `lru_popcount` | `lru_popcount` (§3.1) / `popcount` (Fig. 22) / `lru` |
| `snake_tail_lru_group` | uint | 3 | OQ-3 |
| `snake_promote_warps` | uint | 3 | ≥3 warps (§3 intro, §3.1, §3.4) |
| `snake_intra_confirm_warps` | uint | 3 | §3.4 |
| `snake_interwarp_min_warps` | uint | 3 | §3.1 |
| `snake_demote_removed` | uint | 2 | ">2 removed → untrained" (§3.2) |
| `snake_max_chain_depth` | uint | 3 | OQ-7; at most 10 (SK_SC sets 10) |
| `snake_chain_deep_promoted` | bool | 0 | chain steps after the first accept a promoted entry with the warp's bit (SK_SC, OQ-35) |
| `snake_intra_distance` | uint | 1 | intra-warp prefetch distance in iterations (SK_SC 8, OQ-34) |
| `snake_intra_distance_mode` | uint | 0 | 0 = every iteration 1..D, 1 = only iteration D (OQ-34) |
| `snake_interwarp_degree` | uint | 4 | OQ-8 |
| `snake_interwarp_chain_depth` | uint | 0 | promoted inter-thread steps from each inter-warp target; 0 = paper Fig. 15 (OQ-29; S3b–S3d default 3, S3e default 0) |
| `snake_promoted_all_warps` | bool | 1 | the first inter-thread step uses a promoted stride also for warps without their warpID bit, verified at their next load; deeper steps need the bit and a trained entry (S3e); 0 = S3c (OQ-32) |
| `snake_interwarp_weight` | uint | 3 | Tail eviction score added for a trained inter-warp stride; 0 = paper popcount (OQ-30, S3b) |
| `snake_enable_inter_thread` / `_inter_warp` / `_intra_warp` | bool | 1/1/1 | s-Snake = 1/0/0 (§4) |
| `snake_lane_mode` | uint | 1 | 0 = §3.4 equal lane strides; 1 = stateless run filter, L ∈ {32,16,8,4} (deviation, OQ-17) |
| `snake_footprint` | uint | 0 | 0 = one 128 B line per prediction (missing sectors only); 1 = exact lane descriptor per Tail entry (OQ-9) |
| `snake_require_equal_lane_stride` | bool | 1 | §3.4; lane mode 0 only |
| `snake_latency` | uint | 2 | §5.5 |
| `snake_queue_size` | uint | 32 | OQ-4: entries are 128 B lines (S3a) |
| `snake_issue_width` | uint | 1 | OQ-4: lines per cycle (S3a) |
| `snake_pred_window` | uint | 5000 | cycles a prediction counts for coverage_pred (OQ-14) |
| `snake_mshr_reserve` | uint | 8 | OQ-11 |
| `snake_missq_reserve` | uint | 8 | OQ-11: miss-queue slots left free for demand (S3a, from the lud sweep) |
| `snake_pq_path` / `snake_pq_arb` / `snake_pq_mshr` / `snake_pq_size` | uint | 0 / 0 / 0 / 128 | R2 (OQ-38): separate prefetch queue sharing the injection point; demand first or round-robin; MSHR at send or at enqueue |
| `snake_entry_acc_filter` | bool | 0 | OQ-39 accuracy filter; `snake_acc_filter_window` 32, `_min` 0.25, `_age` 1000, `_block` 1000 |
| `snake_interwarp_single` / `snake_interwarp_cta_anchor` | bool | 0 / 0 | OQ-40: only the degree-th resident warp ahead; cross-CTA targets based on their CTA's first warp |
| `snake_l1_filter` | bool | 1 | "not in L1" (§3.2) |
| `snake_throttle_enable` | bool | 1 | Snake-T = 0 (§4) |
| `snake_throttle_space_cycles` | uint | 50 | §3.3, Fig. 23 |
| `snake_bw_high` / `snake_bw_low` | float | 0.70 / 0.50 | §3.3 |
| `snake_bw_window` | uint | 100 | OQ-10 |
| `snake_bw_mode` | uint | 0 | 0 = request slots of the L1D miss-queue drain (peak 1 per cycle); 1 = data bytes against `snake_bw_peak_bytes` (S4, OQ-10) |
| `snake_bw_peak_bytes` | uint | 32 | peak for `snake_bw_mode 1` (S4) |
| `snake_throttle_chain_depth` | uint | 1 | chain depth limit while throttled (S4, OQ-7) |
| `snake_decouple_mode` | uint | 1 | 0 none / 1 unified / 2 isolated (OQ-12) |
| `snake_untrained_l1_frac` | float | 0.50 | §3.2 |
| `snake_throttle_confine_cycles` | uint | 50 | §3.2 |
| `snake_free_frac` | float | 0.25 | §3.2, footnote 2 |
| `snake_transfer_frac` | float | 0.80 | §3.2 |
| `snake_isolated_lines` | uint | 256 | mode 2 size (32 KB; §5.7 gives no size → OQ-22) |
| `snake_decouple_variant` | uint | 1 | mode 1 reading: 0 = S4 literal, 1 = revised (S4b; OQ-12, OQ-15); token `SNAKE_DV0` |
| `snake_untrained_cycles` | uint | 10 | variant 1: untrained cap length per kernel launch (§3.2, §3.5; OQ-15) |
| `snake_pf_stale_cycles` | uint | 50 | variant 1: unused prefetched data older than this is stale (§3.2, footnote 3; OQ-12) |
| `snake_energy_pj_per_access` / `snake_static_mw` | float | 6.4 / 6.0 | §5.5 (offline energy) |
| `snake_debug_trace` | bool | 0 | per-event log for tests |

The ablations come from these flags:
- **Snake-DT** = `decouple_mode 0`, `throttle 0`
- **Snake-T** = `decouple 1`, `throttle 0`
- **s-Snake** = inter-thread only

## 7. Stats (per SM; GPU sum printed as `snake_*`)
- **Issue:** `pf_generated`, `pf_issued` (`_inter_thread`, `_intra_warp`, `_promotion`, from S3b `_inter_warp`, `_inter_warp_chain`), `pf_queue_displaced` (S3b), `pf_dup_queue`, `pf_dup_l1`, `pf_dup_mshr`, `pf_drop_queue_full`, `pf_drop_resource` = `pf_drop_line` + `pf_drop_mshr` + `pf_drop_miss_queue`, `pf_wait_port`, `pf_woken_warps` (must be 0), `pf_drop_throttled` (S4).
- **Prediction (OQ-14):** `pred_sectors`, `demand_pred_matched` (→ coverage_pred), `demand_resfail`.
- **Tail, S3b analysis:** `tail_evict_interwarp_trained` and `tail_evict_interwarp_wbits` (warp bits of those evicted entries).
- **Outcome:** `pf_filled`, `pf_useful_timely`, `pf_late`, `pf_evicted_unused`, `pf_unused_at_end`, `pf_caused_resfail`.
- **S3c diagnosis:**
  - `funnel_<mode>_<class>` for the prediction funnel, with `funnel_invariant` (SPEC OQ-14).
  - `snake_dist_*` / `snake_hist_*` distributions: `pred_to_issue_<mode>`, `queue_wait`, `port_wait`, `lead_issue_<mode>_{late,timely,evicted}`, `lead_pred_<mode>`, `mq_wait_{demand,prefetch}`.
  - Miss-queue counters `mq_*`, plus `pf_drop_mq_reserve` and `promoted_nobit_loads`.
- **S3d:**
  - `funnel_src_<source>_<class>`, with `funnel_src_invariant`.
  - `no_it_pred_loads`, `no_it_pred_promoted_pc`, `pred_inter_thread_nobit`, `nobit_verify_{ok,fail,lost}`.
- **Demand base:** `demand_sector_accesses` (`GLOBAL_ACC_R`, L1-cached).
- **Derived:**
  - `coverage_paper = (timely+late)/demand`
  - `accuracy_paper = timely/demand`
  - `accuracy_std = (timely+late)/issued`
  - `timeliness = timely/(timely+late)`
- **Tables:** `head_updates`, `tail_lookups`, `tail_hits`, `tail_allocs`, `tail_evictions`, `t1_promotions`, `t1_demotions`, `t2_trained`, `interwarp_trained`, `interwarp_trained_pcs`, `warp_excluded_lane_stride` (mode 0), `warp_excluded_lane_runs`, `lane_runs_{32,16,8,4}` (mode 1, OQ-17), `head_row_evictions`, `chain_depth_hist[0..D]`.
- **Per mode:** {inter_thread, inter_warp, intra_warp} × {generated, issued, useful, late, unused}.
- **Throttle (S4):** `throttle_space_events`, `throttle_bw_events`, `throttled_cycles` (`_space`, `_bw`), `bw_windows`, `bw_util_hist_<0..100>`, `pf_held_throttled` (replaces `pf_drop_throttled`), `loads_throttled`, `throttle_chain_cut`, `no_free_space`. The optional time series was not added.
- **Decoupling (S4):** mode 1: `region_flips` (`_sector_miss`), `flip_unused_sectors`, `pf_fill_l1_region`, `pf_line_allocs`, `bulk_frees`, `bulk_free_l1_side` / `_pf_side`, `bulk_free_lines`, `bulk_free_pf_unused`, `l1_quota_victims`, `l1_cap_cycles_untrained` / `_confine`, `quota_checks`, `decouple_fallback`; mode 2: `iso_inserts`, `iso_hits`, `iso_evictions`, `iso_dup`, `iso_transfer_fail`.
- **SnakeCharmer metrics (SK_SC, SPEC OQ-36):** `sc_pf_fills` (every prefetch fill), `sc_ld_ontime`, `sc_ld_late`, `sc_ld_early` (first load to a sector evicted unused, same kernel), `sc_ld_miss` (loads that sent a request), `sc_victim_sectors`, `sc_victim_demand`, `sc_victim_demand_by_pf`. Report: `util/snake/sk_sc_report.py`.
- **No double counting:** each issued prefetch sector ends in exactly one terminal state.
  - At fill, if a demand had merged into its MSHR entry → `late` (bit not set). Otherwise the bit is set, and the bit later resolves to exactly one of `useful_timely` / `evicted_unused` / `unused_at_end`.
  - Invariant `issued == useful_timely + late + evicted_unused + unused_at_end`. It is checked at every stats print and reported as `snake_pf_invariant = OK|FAIL`; `unused_at_end` = invalidated + resident unused + in flight unmerged (`snake_pf_unused_at_end_total`).

## 8. Staged implementation (each stage = one commit per changed repo, `snake: S<n> <desc>`)
- **S1 Infrastructure.** Options, empty `snake_prefetcher` wired to all hooks, `L1_PREFETCH_R` type, print guard, stats skeleton, configs/yml in accel-sim.
  - *Done when:* it builds and the **off-regression** (§9.1) is byte-identical on 4 workloads; with Snake on (no-op) it is also identical apart from the `snake_*` block.
  - **Done 2026-10-08:** gpgpu-sim `e0d23397`, accel-sim `d49e1f3`. Off-regression PASS on 5 workloads (hotspot has two inputs).
- **S2 Training only.** Head/Tail/T1/T2/inter-warp, lane-stride check, demotion, kernel/warp resets, eviction policy, `snake_debug_trace`.
  - *Done when:* synthetic traces (§9.2) give the expected strides and promotions (unit-style assertions on the debug log), and on rodinia hotspot, srad_v1 and backprop `tail_hits/lookups`, promotions and the chain-depth histogram are plausible. Timing is still identical to baseline, since there is no issue.
  - **S2 done 2026-10-13:** gpgpu-sim `ace5cc70`, accel-sim `e4030cd`. Synthetic PASS, off-regression PASS. Deviation: hotspot/backprop were implausible under the §3.4 lane rule → OQ-17.
  - **S2b done 2026-10-13:** gpgpu-sim `b9cdc46a`, accel-sim `5b4f096`. Per-PC lane-pattern mode (OQ-17 deviation). Gates PASS. backprop then exposed the trace-decoding bug: parser fix `fe8f37e`, reference `bin_ref_v2` from `bed3a2b`, docs `61c0dc1`.
- **S2c Stateless lane-run filter + footprint setting** (decided 2026-10-13).
  - Lane mode 1 becomes the run filter (OQ-17), and the S2b per-PC pattern is deleted. `-snake_footprint` gets the setting and the Tail descriptor storage (OQ-9). `snake_interwarp_trained_pcs` is added.
  - *Done when:* Fig. 15 and the synthetic traces pass in both lane modes. Plausibility on hotspot, backprop and srad_v1 (mode 1): exclusion, promotions, `interwarp_trained`(`_pcs`), chain depth. Off-regression against `bin_ref_v2`, reusing the cached S2c reference outputs.
  - **S2c done 2026-10-13:** gpgpu-sim `bc22d0dd`, accel-sim `293aaaa`. synthetic PASS in lane modes 1 and 0 (Fig. 15 exact; random: all 512 scattered loads excluded; `sim_run_snake_synth/S2c2`). Off-regression vs `bin_ref_v2` PASS on 6 workloads, ref outputs reused from S2c (`sim_run_snake_regress/S2c2`). Plausibility (mode 1, the off-regression `on` runs): hotspot 512/1024 excluded 0%, all L=16, Tail hits 99.3%, promotions 80/88, 2 inter-warp PCs, chain depth 1 for 35%/37%; backprop excluded 0%, all L=16, 0 promotions (SPEC §8), interwarp_trained 32,624 on 2 PCs; srad_v1 excluded 2.5%, Tail hits 74.2%, 2,337 promotions, 16 inter-warp (kernel, PC) pairs, chain depth 0/1/2/3: 89.8/9.9/0.3/0.1%. Deviations: the stage's gate results are recorded in a follow-up commit (gates run only on committed builds); the plausibility check reuses the off-regression `on` runs, which are the same jobs, instead of separate training runs; `snake_interwarp_trained_pcs` was added to report distinct trained PCs.
- **S3a Intra-warp + inter-thread issue, no throttling or decoupling.** Then **S3b** adds inter-warp.
  - *Done when:* there are no deadlocks on the full suite at a 100M-instruction cap, the `pf_woken_warps` assert is 0, the invariant holds, and accuracy and coverage print. The strided microbenchmark reaches coverage ≥90%.
- **S4 Throttling + decoupling** (modes 0/1/2).
  - *Done when:* the random microbenchmark shows `throttled_cycles` >50% and `pf_issued/demand` <10%; quotas hold (debug assert); Snake-DT < Snake-T ≤ Snake in accuracy on the suite (§5.2 trend).
- **S3a done (implementation 2026-10-13).** Decisions taken during the stage:
  - Trace (d) `strided-dep` carries the coverage gate (≥ 90% coverage_paper with the exact footprint; if it falls short, coverage_pred ≥ 90% with the shortfall recorded). Trace (a) became a saturation test.
  - The prefetch queue holds one entry per 128 B line (OQ-4); with one sector per entry, trace (d) reached 82.8%.
  - coverage_pred was added (OQ-14).
  - Decoupling and throttling are forced off (`snake_config::init`; gate configs use `SNAKE_DT`).
  - Inter-warp predictions are only counted, not issued (S3b).
  - **Done 2026-10-13.** gpgpu-sim `028012de` (issue path), `5c6f2267` (`-snake_missq_reserve`), `2d50a9aa` (its default 8); accel-sim `1221702` (docs, tooling), `9dc2631` (configs), `d836622` (`run_apps.sh` fix).
  - **Gates:**
    - Synthetic (`sim_run_snake_synth/S3a`): PASS in both footprints. Trace (d) coverage_paper is 93.0% (coverage_pred 97.6%); (b) issues 0; (a) runs without deadlock (demand MISS_QUEUE_FULL 9,024; 16,416 sectors dropped with the prefetch queue full); Fig. 15 PASS.
    - The 7 apps (9 workloads) × line/exact (`sim_run_snake_gate/S3a`): all finish, `pf_woken_warps` 0, invariant OK.
    - Off-regression vs `bin_ref_v2` (ref cached from S2c, `ON=0`): PASS on 6 workloads, both before (`S3a`) and after (`S3a2`) the miss-queue reserve.
    - Baselines for lud ×2 and mri-q were run once into `sim_run_snake_refcache/bin_ref_v2`.
  - **Miss-queue starvation stop (lud) → `-snake_missq_reserve`.** lud dropped most prefetches for miss-queue slots and doubled demand MISS_QUEUE_FULL. A sweep over N ∈ {2, 4, 8} (lud ×2, line; `S3a_mqr`) chose the default 8. lud exact was re-run with it (`S3a_mqr_fp1`): demand MISS_QUEUE_FULL 42,855 → 37,509 (512) and 38,177 → 29,387 (256); `pf_caused_resfail` 0. The other apps were not re-run with N = 8; they dropped < 10% of their prefetches for miss-queue slots.
  - **Findings (SPEC §8):**
    - Coverage is mostly late (no inter-warp lookahead yet), so IPC within ±0.4% is expected.
    - coverage_pred is far below the paper (hotspot 35%, srad_v1 14%); re-check after S3b.
    - Inter-warp Tail entries are evicted with ~0.9 warp bits (backprop) → fix in S3b.
  - **Deviations:**
    - Gate results are in a follow-up commit.
    - Trace (d) replaced (a) as the coverage gate.
    - The queue holds one entry per line.
    - `-snake_missq_reserve` was added.
    - The planned gate count was 16 app runs; it is 18, because lud and hotspot have two inputs.
  - Full tables: `docs/snake/results/S3a_gate.md`.
- **S3b Inter-warp issue** (2026-10-14).
  - Inter-warp predictions are issued: degree 4, targets are the resident warps of the same CTA (OQ-8).
  - Combined lookahead (OQ-29): each target also follows ≤ 3 promoted inter-thread strides from its predicted PC1 address. `-snake_interwarp_chain_depth 0` (`SNAKE_IWC0`) is the paper's Fig. 15.
  - Tail eviction score = popcount + `-snake_interwarp_weight` (3) for a trained inter-warp stride (OQ-30). Weight 0 (`SNAKE_IWW0`) is the paper's rule.
  - Inter-thread priority in the queue and at issue (OQ-31).
  - **Done 2026-10-14.** gpgpu-sim `298c5f94` (issue, lookahead, weight), `000798a7` (queue displacement), `f1e4f856` (issue priority); accel-sim `bf1fc2f` (checks, configs), `a708e99` (Fig. 15 check on offered sectors).
  - **Gates:**
    - Synthetic (`sim_run_snake_synth/S3b3`, 9 runs): PASS. Fig. 15 (c) predictions are exactly {2600, 2700, 2800, 3000} with `SNAKE_IWC0`, and that set plus {3100, 3200, 3300} by default (line and exact). Every predicted sector is offered to the L1D once; 2 / 9 / 16 are dropped for miss-queue slots. Trace (d) coverage_paper 90.8% (S3a 93.0%), (b) issues 0.
    - The 7 apps (9 workloads), line footprint (`sim_run_snake_gate/S3b`, 9 runs), and exact footprint for the 7 workloads whose coverage_paper changed by > 5 pp (`sim_run_snake_gate/S3b_fp1`; nw and histo not run): all finish, `pf_woken_warps` 0, invariant OK.
    - Off-regression vs `bin_ref_v2` (ref cached from S2c, `ON=0`): PASS on 6 workloads (`sim_run_snake_regress/S3b`).
    - Baselines reused from `sim_run_snake_refcache/bin_ref_v2`, none re-run.
  - **Findings (SPEC §8):** coverage_pred is close to the paper (hotspot 88%, lud 83%, srad_v1 81%, mri-q 87%), coverage_paper is not (dropped prefetches); lud 512 line raises demand MISS_QUEUE_FULL by 29%; histo line −3.8% IPC; the eviction weight does not help backprop.
  - **Deviations (each agreed 2026-10-14):**
    - Combined lookahead behind an option, with the Fig. 15 gate run in both modes (OQ-29).
    - The Fig. 15 gate compares the sectors offered to the L1D, since the miss-queue reserve drops some of the burst.
    - Queue displacement and issue priority for own-warp lines (OQ-31), after trace (d) fell to 88.8% and then 89.3%.
    - Gate results are in a follow-up commit; synthetic ran 3 times (`S3b`, `S3b2`, `S3b3`).
  - Full tables: `docs/snake/results/S3b_gate.md`.
- **S3c Diagnosis** (2026-10-14). Stats only; no behaviour change.
  - Prediction funnel per mode, timing distributions (prediction → issue, queue and port wait, issue → first demand per mode and outcome), L1D miss-queue waits and head-of-line counts, `pf_drop_mq_reserve`, `promoted_nobit_loads`.
  - **Done 2026-10-14.** gpgpu-sim `02a9d7de`; accel-sim `59e5bee` (funnel invariant in `check_synthetic.py`), `b7cae56` (dist-line parsing, `run_apps.sh` `ONLY` filter).
  - **Checks:**
    - Synthetic (`sim_run_snake_synth/S3c`, 9 runs): PASS; outputs equal to S3b3 apart from volatile lines.
    - The 4 current-settings diagnosis runs equal S3b_fp1 (cycles, `pf_issued`, `demand_pred_matched`).
    - Off-regression PASS on 6 (`sim_run_snake_regress/S3c`).
  - **Runs** (`sim_run_snake_gate/S3c`, exact footprint, 100M): hotspot 512, srad_v1, mri-q and lud 512 at reserve 8 and 4, plus histo (9).
  - **Findings:** see SPEC §8 and `results/S3c_diag.md` §10.
  - **Deviations:**
    - Synthetic (9) and off-regression (6) runs in addition to the 9 listed, for the synthetic-first rule and the identity check.
    - `run_apps.sh` got an `ONLY` filter to run single inputs.
    - The first synthetic check crashed on the new distribution lines; the parser was fixed and the same outputs were re-checked without re-running.
- **S3d Promoted strides for all warps + funnel by source** (2026-10-14).
  - `-snake_promoted_all_warps` (default 1, OQ-32): a promoted inter-thread entry predicts for any warp at PC1, verified at the warp's next load (§3.2).
  - Diagnostics: the funnel by the source of a record's first prediction; loads without an own inter-thread prediction; verification counts.
  - **Done 2026-10-14.** gpgpu-sim `97119f36`; accel-sim `976937f` (`SNAKE_PAW0` token, per-source funnel check).
  - **Gates:**
    - Synthetic (`sim_run_snake_synth/S3d`, 9 runs): PASS; Fig. 15 (c) unchanged in both modes; trace (d) 90.8%.
    - Off-regression PASS on 6 (`sim_run_snake_regress/S3d`).
    - Runs (`sim_run_snake_gate/S3d`, 10): hotspot 512, srad_v1, mri-q, lud 512, histo × {defaults, `SNAKE_IWC0`}; all finish, every invariant OK.
  - **Findings:** SPEC §8 and `results/S3d.md` §5.
    - No load at a promoted PC1 is left without a prediction; 33–57% of loads still get none.
    - coverage_pred rises by up to 6.7 pp (mri-q), and coverage_paper by up to 4.6 pp (hotspot).
    - lud's demotions rise 14×.
    - Chain depth 0 beats 3 slightly in IPC.
  - **Deviations:**
    - The bit requirement is lifted at every chain step, not only the first. Only the first step is verified, as only it is the next load.
    - The per-source funnel uses the record's first prediction (S3c used the latest).
- **S3e Paper defaults for the chain walk; PAPER_LAT** (2026-10-14).
  - `-snake_interwarp_chain_depth` defaults to 0 (OQ-29).
  - With `-snake_promoted_all_warps 1`, only the first chain step may use an entry without the warp's bit; deeper steps need the bit and a trained entry (OQ-32, §3.2 Fig. 13 paragraph).
  - VERSION_DIFF records the miss-queue check. Config `PAPER_LAT` (L1 knob 28, L2 ROP 199 → measured L2 hit 212.0) is for the final evaluation.
  - SPEC §8 lists the discrepancies with the paper.
  - **Done 2026-10-14.** gpgpu-sim `53bbfd25`; accel-sim `20e968f` (`SNAKE_IWC3`, synthetic Fig. 15 extra config), `131a513` (trace (d) gate on `SNAKE_IWC3`).
  - **Gates:**
    - Synthetic (`sim_run_snake_synth/S3e`, 9 + 1 runs): PASS.
    - Off-regression PASS on 6 (`sim_run_snake_regress/S3e`).
    - Runs (`sim_run_snake_gate/S3e`, 5): all invariants OK.
  - **Results** (results/S3e.md): within noise of S3d (b); lud demotions 6,898 (S3d 6,829–6,879).
  - **Deviations (each agreed 2026-10-14):**
    - Trace (d) fell to 86.0% with the paper default, so its ≥ 90% gate now runs under `SNAKE_IWC3` and the default is reported only; one extra synthetic run.
    - PAPER_LAT was calibrated with the l2_lat microbenchmark: 3 calibration runs in the new run dir `sim_run_snake_calib`, one of which aborted on an sm_75 trace (opcode map) and was redone with an sm_70 build.
    - The L1 knob is 28 as specified, which measures 33; knob 23 would measure 28.
- **S4 Throttling + decoupling** (2026-10-15).
  - **Step 0, PAPER_LAT:** `-gpgpu_l1_latency 23` (measured L1 hit 28) and L2 ROP 199 (measured 212); both Table 1 latencies are treated as measured values (VERSION_DIFF). One l2_lat calibration run (`sim_run_snake_calib/paper_lat23`, `bin_ref_v2`): L2 hit 212.0, L1 hit 28.0 cycles.
  - **Throttling** (SPEC OQ-10, OQ-7): space trigger 50 cycles; bandwidth trigger on the request-slot utilisation of the L1D miss-queue drain over 100-cycle windows, halt at 0.70, resume at 0.50 (`-snake_bw_mode 1`: bytes); while throttled, no issue and chain depth ≤ 1.
  - **Decoupling** (SPEC OQ-12, OQ-15; OQ-22): mode 1 regions, flips, bulk free, transfer rule, L1 cap until trained and while throttled; mode 2 isolated buffer; mode 0 unchanged (Snake-DT). The S3 override that forced modes 0 and no throttling is removed; the defaults are now full Snake (mode 1 + throttle).
  - **Gate plan:** synthetic (Fig. 15 and trace (d) under the S3e configs; trace (b) under full Snake: throttled > 50%, `pf_issued`/demand < 10%); off-regression vs `bin_ref_v2` (6, ref cached from S2c); apps hotspot 512, srad_v1, mri-q, lud 512, histo × {Snake, Snake-T, Snake-ISO}, exact footprint, 100M (15 runs, `sim_run_snake_gate/S4`), Snake-DT reused from S3e; stop if mode 1 lowers the demand L1 hit rate or IPC by more than 2% vs Snake-DT. Report: `util/snake/s4_report.py`.
  - **Implemented 2026-10-15.** gpgpu-sim `57103274`; accel-sim `f708b4f` (docs, tooling, `PAPER_LAT` L1 knob 23).
  - **Gates** (33 runs: 1 calibration, 11 synthetic, 6 off-regression, 15 apps):
    - PAPER_LAT: L1 hit 28.0, L2 hit 212.0 cycles.
    - Synthetic PASS. The 10 Snake-DT runs equal S3e (cycles, instructions, issued, timely, late, matched). Trace (b) under full Snake is throttled 89.4% of its cycles (bandwidth trigger) and issues 0 of 20,480 demand sectors.
    - Off-regression PASS on 6.
    - Apps: all 15 finish; `pf_woken_warps` 0; every invariant OK; no quota assert fired, 0 fallbacks.
    - **Stop rule fired:** mode 1 lowers the demand L1 hit rate by more than 2% (relative) vs Snake-DT on hotspot (Snake 32.3 → 26.0, −19.6%; Snake-T −20.6%) and lud (Snake 8.4 → 8.0, −5.1%). IPC is within 2% of Snake-DT on every app.
  - **Findings** (results/S4.md §6):
    - hotspot: bulk frees evict 19.5k prefetched sectors before use (Snake-DT: 1.3k), which costs 11.4k timely hits. 19.0k unused sectors lost protection when their line flipped on its first demand hit (OQ-12).
    - lud: throttling (19% of cycles) cuts issued prefetches by 30% and timely sectors by 14%.
    - The 50% until-trained cap holds 33–53% of cycles in lud and histo.
    - Snake-ISO keeps Snake-DT's hit rate and accuracy.
    - Snake-DT < Snake-T ≤ Snake holds only for histo (accuracy).
  - **Deviations:**
    - `pf_drop_throttled` renamed `pf_held_throttled`, since nothing is dropped.
    - Synthetic trace (b) runs under full Snake as an extra synthetic config (`$RAND_CFGS`); one run.
  - **Stopped here** by the stop rule; S4 is not done.
- **S4b Revised mode 1** (2026-10-15, after the S4 stop; agreed: mode 1 as in S4 contradicts §5.7).
  - **Step 1:** diagnose the 50%-until-trained cap with new counters (launch → first promotion, launches never promoting, cap cycles split, re-arms); 5 variant-0 Snake re-runs, since the S4 outputs lack these counts. A re-arm at any time other than a kernel launch is a bug, fixed in a separate commit.
  - **Step 2:** `-snake_decouple_variant` 0 = S4, 1 = revised (default): untrained cap ≤ 10 cycles per launch; the prefetch side of a bulk free gives only stale lines (> 50 cycles), the rest from the L1 side; a line flips only when none of its prefetched sectors is fresh (SPEC OQ-12, OQ-15).
  - **Gate:** synthetic (Fig. 15, trace (d), trace (b)); off-regression; 10 runs: hotspot 512, srad_v1, mri-q, lud 512, histo × Snake and Snake-T, variant 1, exact footprint, 100M (`sim_run_snake_gate/S4b`); S4's Snake-ISO and S3e's Snake-DT reused. Target (§5.7): variant 1 Snake's demand L1 hit rate within 5 points of Snake-ISO and no more than 2% below Snake-DT on every app. If missed, keep variant 1, record the gap in SPEC §8 and stop. Report: `util/snake/s4b_report.py`.
  - **Done 2026-10-15.** gpgpu-sim `3f635b67`; accel-sim `7af4b95` (docs, `SNAKE_DV0`, report), `d06395d` (report reads the step-1 runs from their own dir).
  - **Runs** (31): synthetic 11 PASS; off-regression 6 PASS; step-1 5 (`sim_run_snake_gate/S4b_diag`); gate 10 (`sim_run_snake_gate/S4b`). All finish with every invariant OK and no quota assert. Variant 0 equals S4 in issued, timely, late, hits and IPC.
  - **Step 1** (results/S4.md S4b.1):
    - The cap re-arms only at an SM's launch start, so there is no re-arm bug and no fix commit.
    - Promoting SM-launches take a p50 of 57–655 cycles.
    - 79–94% of the cap time in lud, histo and srad_v1 is in SM-launches that never promote.
    - Variant 1 cuts the cap to 0.1–0.3% of cycles.
  - **Target missed** on hotspot (26.6 vs ISO 32.5, DT 32.3) and lud (8.2 vs DT 8.4, −3.0%). hotspot's prefetches are used ~900 cycles after issue, so the 50-cycle staleness frees them; lud's gap is throttling. Variant 1 is kept (SPEC §8).
  - **Deviations:**
    - The step-1 runs went to `S4b_diag` instead of `S4b`, since `run_apps.sh` refuses an existing runs dir and they had to be checked before the gate runs.
    - The first gate launch was refused (dirty tree: an uncommitted report-script edit), so nothing ran. The script was committed (`d06395d`) and the same 10 runs were launched on the same frozen build.
- **S5 Sweep hooks + cleanup.** Sweep configs for Tail {10, 20, 40, ∞}, eviction policy, throttle {25…300} (Figs. 20, 22, 23), the prefetch queue size and issue width (OQ-4, added in S3a; defaults 32 lines and 1 line per cycle stay until then) and `snake_missq_reserve` (OQ-11, default 8); `compare_paper.py`; docs updated.
  - **S5 as run (2026-10-15): final-evaluation configs and tooling; nothing launched.** All configs are on `QV100-SASS-PAPER_V100-1B_INSN` with the 9 workloads (7 apps) of `run_apps.sh`. Configs that coincide are listed once: Snake exact = Tail 10 = lru_popcount at Tail 10 = throttle 50 = the Tier 2 reference. `util/snake/s5_runs.sh <tier> list|dry|run` holds the lists.
    - **Tier 1 (21 configs, 189 runs):**
      - `sim_run_baseline`: baseline; PAPER_LAT baseline
      - `sim_run_snake`: Snake exact (`SNAKE-SNAKE_FP1`); Snake line (`SNAKE`); Snake-ISO; PAPER_LAT Snake exact
      - `sim_run_snake_dt`, `sim_run_snake_t`, `sim_run_ssnake`: Snake-DT, Snake-T, s-Snake (exact)
      - `sim_run_snake_sweep_tail`: Tail 20, 40, ∞ (1024; SPEC OQ-3)
      - `sim_run_snake_sweep_evict`: popcount at Tail 10, 20, 40
      - `sim_run_snake_sweep_throttle`: space halt 25, 75, 100, 150, 200, 300
    - **Tier 2 (13 configs, 117 runs), each one change from Snake exact:**
      - `sim_run_snake_sens`: interwarp_degree 1, 2, 8; missq_reserve 4; pf_stale_cycles 200, 1000; bandwidth in bytes; interwarp_chain_depth 3; decouple_variant 0; queue_size 64; issue_width 2 (the last two as separate configs)
      - `sim_run_baseline_1132`, `sim_run_snake_1132`: the 1132 MHz pair
    - **New config tokens:** `SNAKE_TAIL20/40/INF`, `SNAKE_EVPOP`, `SNAKE_THR25/75/100/150/200/300`, `SNAKE_IWD1/2/8`, `SNAKE_STALE200/1000`, `SNAKE_BWBYTES`, `SNAKE_Q64`, `SNAKE_ISSUE2`.
    - **Tooling:**
      - `run_apps.sh`: `DRY=1` (generate the jobs and freeze the build, launch nothing) and `RESUME=1`
      - `s5_runs.sh`: tier lists; dry-run; one longest-first pool per tier that never re-runs a job with a sim.out
      - `collect_results.py`: per-job CSV; fails a job whose build tags are not `_modified_0.0` or differ within a run dir, or that did not finish or broke an invariant; energy per SPEC OQ-16
      - `compare_paper.py`: SPEC §5 table, Fig. 25 L1 hit rate, ablations, Figs. 20/22/23, Tier 2
      - `s5_estimate.py`: cost estimate
    - Energy: AccelWattch is off in our configs, so Fig. 19's normalized energy is reported as unavailable, with Snake's own energy (SPEC OQ-16).
    - **Done 2026-10-15.** accel-sim `2c3a333` (configs, tooling, docs); gpgpu-sim unchanged.
    - **Dry-run:** 11 run dirs, 189 + 117 jobs, each dir with the frozen clean build (`2c3a333`, `3f635b67`); no sim.out.
    - **Smoke runs (2, `sim_run_snake_synth/S5_smoke`):** all new tokens of each tier stacked on the Fig. 15 trace; both parse, finish and keep the invariants.
    - **Estimate** (`s5_estimate.py`, results/S5.md §3): 2.69 core-hours per config. Tier 1 56.5 core-hours (≈ 3.8 h at 15 parallel); Tier 2 35.0 (≈ 2.3 h). Order: Tier 1, check, then Tier 2.
    - **Deviations:**
      - Tier 2 queue size 64 and issue width 2 are separate configs.
      - Tier 2 gets a new run dir, `sim_run_snake_sens`; Snake-ISO and the PAPER_LAT pair share `sim_run_baseline` and `sim_run_snake`.
      - The first smoke attempt aborted before generating any job (environment not sourced), so it ran nothing.
  - **Final evaluation, Tier 1 (2026-10-15/16; approved with conditions).**
    - **Pre-flight:** 1 RSS run plus 1 power smoke run, plus a paired power off/on of hotspot 512 to measure the slowdown (`sim_run_snake_preflight`, 4 short runs).
      - Peak RSS 1.85 GB (2.23 GB across Tier 1): parallelism 15.
      - Disk 160 GB free; debug trace off in all 306 configs; launch gate held.
      - AccelWattch +55% wall time, > 30%: no POWER configs; energy unavailable (SPEC §8).
    - **Commit `e89c9d0`:** `s5_runs.sh run` checks the launch gate, writes time.txt per job and takes `ONLY_CFGS` for the canary; `collect_results.py` reads time.txt.
    - **Canary** (18 jobs): clean; first 100M instructions identical to the gates.
    - **Tier 1:** 171 more jobs; 189/189 pass `collect_results.py --require-all`.
      - Unlimited Tail evictions: backprop 107,498, others 0.
      - Ours vs the paper (7-app averages): coverage_pred 63.2 vs 75.7%, accuracy 6.1 vs 69.1%, IPC −0.3 vs +14.1%, L1 hit 18.7 → 25.2 vs 45 → 79.
      - Time: 64.9 vs 56.5 core-hours; 4 h 33 min vs ≈ 3.8 h.
      - `results/final_tier1.md`, `results/final_tier1.csv`.
    - **Deviation:** the power slowdown needed a power-off reference under the same load, so 2 runs (paired off/on) were added to the 1 smoke run.
    - **Tier 2 waits for approval.**
- **S6: IPC investigation** (2026-10-16, on request: "why is there no IPC improvement; try to fix it").
  - **Code:** gpgpu-sim `e85e7eb1` adds `-snake_interwarp_scope` (S6 experiment, default 0 = OQ-8 unchanged); the warp-init hook now passes the global warp id. accel-sim `40d0a04` (`PERFECT_MEM`), `4502b2d` (`SNAKE_IWS1/2`); clean build `4502b2d`/`e85e7eb1`.
  - **Gates:** synthetic PASS (`sim_run_snake_synth/S6`, 11; the Snake-DT runs equal S4b); off-regression PASS on 6 (`sim_run_snake_regress/S6`).
  - **Runs:**
    - 8 perfect-memory ceilings (`sim_run_snake_gate/S6_ceiling`, S5 frozen build, 1B, no nw)
    - 6 scope runs at 100M (`S6_scope`: hotspot 512, srad_v1, histo × scope 1, 2)
    - 2 backprop scope runs at 1B (`S6_scope_bp`)
  - **Findings** (results/S6_ipc.md): perfect memory gains < 5% on hotspot, mri-q and lud. backprop and histo are DRAM-queue bound; srad_v1 and mri-q are limited by the miss queue and MSHRs. Inter-warp lead is ~20 cycles; scope 2 makes backprop timely (accuracy 14%) without IPC (DRAM-bound).
  - **Outcome:** no Snake change raises IPC on these workloads; the defaults stay paper-faithful.
  - **Deviation:** an investigation stage with code (an opt-in option) and 22 runs beyond the S5 plan, on the user's request.
- **SK_SC: SnakeCharmer's Snake baseline** (2026-10-26, on request; SPEC OQ-33…OQ-36, results/SK_SC.md).
  - **Code:** gpgpu-sim `9622012e` adds `-snake_intra_distance`, `-snake_intra_distance_mode`, `-snake_chain_deep_promoted` (all off by default), raises the chain-depth limit to 10, and adds the `sc_*` stats (the L1D passes victim valid sectors and the access status). accel-sim `5d55de3` (`SK_SC`, `SK_SC_CTA`; `run_apps.sh` takes rodinia_2.0-ft), `dc1a98c` (`sk_sc_report.py`).
  - **Gates:** off-regression PASS on 6; synthetic PASS (11, identical to S6 apart from the new lines); default Snake identical to S4b (5 runs).
  - **Runs** (`sim_run_snake_skc`, 100M): 7 apps × {Snake, SK_SC} on PAPER_V100 (baseline cached) and × {baseline, Snake, SK_SC} on base QV100; 6 SnakeCharmer workloads × {baseline, Snake, SK_SC} × both; SK_SC_CTA on hotspot ×2 and srad_v1 × both (SK_SC dropped hotspot by > 2%). 87 runs + 17 gate runs.
  - **Findings:** against SnakeCharmer's SK, coverage is of the same order, accuracy ~3× and timeliness mostly late. SK_SC's IPC is below our Snake's (GMEAN without particlefilter_naive −2.6% / −1.0% vs +0.9% / +0.3%). Inter-warp targets in other CTAs cost hotspot up to −11% on PAPER_V100. No A100 traces exist locally.
  - **Defaults unchanged.**
- **SK match on base QV100, Step 1** (2026-10-27, on request; SPEC OQ-37, OQ-38; results/SK_match_qv100.md).
  - **Question:** why our timing split and accuracy differ from SnakeCharmer's SK. Suspects: the miss-queue reserve and decoupling.
  - **Code:** none in gpgpu-sim. accel-sim `2a5c57c` adds `SNAKE_HEAD64` (= `-snake_head_two_warps 0`, SnakeCharmer's 64-entry per-warp Head), `SNAKE_MQR0`, `SNAKE_DM0`, and saves the R2 spec (`docs/snake/R2_SPEC.md`, not implemented).
  - **Configs:** SK = SK_SC + same-CTA targets + per-warp Head; SK_R0 = reserve 0; SK_R0_D0 = reserve 0, decoupling off.
  - **Runs:** 14 workloads (our 7 apps; b+tree, particlefilter ×2, heartwall, streamcluster) × 3 configs, base QV100, 100M = 42 (`sim_run_snake_skm/qv100`), plus 1 Fig. 15 run (PASS). Baselines, Snake and the old SK_SC from `sim_run_snake_skc`.
  - **Findings:** neither suspect explains the gap. Too early stays at 0 with decoupling off, and prefetches stay late (78–80%). SK_R0_D0 has the best IPC (+4.77% over 14) and is the proposed base for Step 2 (R2).
  - **SK_R0_D0 at Tier 1 conditions** (on request): 9 runs, PAPER_V100, 1B (`sim_run_snake_skm/v100p_1b`, SK build, Tier 1's baselines).
    - Whole run −0.52% vs Tier 1 Snake −0.29% (7-app GMEAN).
    - The window / setup / config decomposition (best kernel diagnostic only) is in results/SK_match_qv100.md (`util/snake/kernel_gap.py`).
  - **Steps 2–4** (R2, ceilings, bfs diagnosis) wait for approval. R2 (Step 2) was then implemented and run under fix_today, below.
- **fix_today** (2026-10-27/28, on request; SPEC OQ-38…OQ-40; results/fix_today.md). Tier 1 conditions only (PAPER_V100, 1B, 9 workloads).
  - **Phase 0** (no runs, `util/snake/kernel_diag.py`): SK_R0_D0's histo prescan and final kernels lose to a flood of unused prefetches (inter-thread chains, intra-warp), not to miss-queue blocking.
  - **Phase 1:** R2 (gpgpu-sim `bb20e7aa`).
  - **Phase 3/4 code:** accuracy filter (OQ-39), single inter-warp target and CTA anchoring (OQ-40), inter-warp distance counters (`19e3d746`).
  - **Runs** (`sim_run_snake_fix/{p2,p3,p3b,p4}`):
    - A–C: R2 (27)
    - D, E: filter (18)
    - F, E10: filter on B; 10,000-cycle stop (18)
    - G8/16/32: one target 8/16/32 ahead, anchoring (27)
  - **Outcome:** no config meets (a) and (b). Best GMEAN G8 −0.24% (Tier 1 Snake −0.29%), meets (c) and (d). Closest to (a)/(b): E10 (histo −1.64%).
  - **Session break:** 6 Phase 2 jobs and one off-regression were killed and rerun (partial outputs kept as `sim.out.killed`).
  - *Done when:* the sweep run dirs generate and the plots/tables are reproducible.

## 9. Verification
1. **Off-regression.**
   - Reference: `/home/razul/snake/bin_ref_v2/` (`accel-sim.out` + `lib/libcudart.so`, the clean bases plus the trace-parser fix; outside the repo; `SNAKE_BIN_REF` overrides). Run it with `LD_LIBRARY_PATH=/home/razul/snake/bin_ref_v2/lib`. Until 2026-10-13 (S1–S2b) the reference was the clean-base build, now `bin_ref_paperera`. `util/snake/compare_bins.sh <name> <build a> <build b>` compares two builds on these workloads (`sim_run_snake_refcmp/<name>`).
   - Cached reference outputs: `REF_FROM=<stage>` reuses an earlier stage's `ref` outputs (symlink) when that stage ran the same reference build, so `bin_ref_v2` runs only once (`sim_run_snake_regress/S2c/ref`) until it changes.
   - Runner: `util/snake/off_regression.sh <stage>` (from S1 docs commit on). It refuses a dirty tree or a branch other than `Snake`, freezes the build into `sim_run_snake_regress/<stage>/bin_new`, and runs ref / off / on on rodinia-3.1 backprop, hotspot (2 inputs), nw, srad_v1 (added in S2) and parboil histo, `QV100-SASS-PAPER_V100-100M_INSN`.
   - Compare after `util/snake/regress_norm.py`, which removes only volatile lines (wall-clock, simulation rate, build strings, binary paths) and the added `-snake_*` option-dump lines, and makes two heap-address-dependent debug dumps comparable (pending-request list sorted, MSHR pointers masked). **Must be identical.** The final cycles, instructions, L1D read hits/misses and DRAM reads are also compared raw. Repeat at every stage.
2. **Microbenchmarks.** `gen_synthetic_traces.py` writes accel-sim text traces (no GPU needed):
   - (a) a strided loop (fixed inter-thread chain + intra stride) → S2: the chain, inter-warp and intra-warp strides are trained; S3: coverage ≥90%
   - (b) random gather → S2: nothing is promoted; S4: throttled, little issue
   - (d) `strided-dep`, added in S3a: the loop of (a) with 32 iterations, where each iteration ends with an IADD3 reading the three loaded registers and writing the next iteration's address register. Each warp waits for memory once per iteration. S3: coverage_paper ≥ 90% with the exact footprint. From S3 on, (a) is a saturation test (no deadlock, invariant holds; MISS_QUEUE_FULL and drops reported).
   - (c) the Fig. 15 example (PCs 0x520/0x540, W1..W4 = warps 1..4 of a 5-warp CTA whose warp 0 only exits, one LRR scheduler) → exact Head/Tail state after each phase and predictions 1300 / 1800 / {2600, 2700, 2800, 3000}. This is a direct check of the SPEC.
   - Runner: `util/snake/run_synthetic.sh <stage>` (traces from `gen_synthetic_traces.py`, app suite `snake-synth`, config `QV100-SASS-SYNTH_LRR1-SNAKE-SNAKE_DBG` in lane mode 1 and with `-SNAKE_LANE0` in lane mode 0; every check must pass in both; checks in `check_synthetic.py`).
   - From S3a the synthetic default configs are `QV100-SASS-SYNTH_LRR1-SNAKE_DT-SNAKE_DBG` (line footprint) and `…-SNAKE_FP1` (exact). Every run must finish, keep `pf_woken_warps` at 0 and keep the outcome invariant.
   - Real workloads: `util/snake/run_apps.sh <out dir> <build|current> <configs>` (7 suite apps; the build is frozen; existing jobs are never re-run). Snake gate runs go to `sim_run_snake_gate/<stage>`. Cached `bin_ref_v2` baselines are in `sim_run_snake_refcache/bin_ref_v2/runs`: lud and mri-q were run there once, and the other 6 workloads are symlinks to `sim_run_snake_regress/S2c/ref`. `util/snake/s3_report.py <gate runs> <ref runs>` prints the per-app table.
   - Training plausibility on real traces: `util/snake/training_runs.sh <stage>` runs hotspot, backprop and srad_v1 in both lane modes into `sim_run_snake_train/<stage>` and prints `util/snake/training_report.py`. `util/snake/load_pcs.py <kernel-N.traceg>...` counts each kernel's load PCs and (PC1, PC2) edges against the 10 Tail entries.
   - Fallback: NVBit-trace small CUDA kernels on the local RTX 2080 Ti.
3. **Paper comparison.**
   - Run the suite on `QV100 + PAPER_V100 + 1B_INSN`, baseline vs. SNAKE (and the DT/T/s-Snake ablations).
   - `compare_paper.py` puts per-benchmark coverage, accuracy, IPC gain and energy beside the SPEC §5 table.
   - Every gap >10 pp is explained against the VERSION_DIFF list and the OQs.

## 10. Experiment layout
- **Run dirs (new only; verified that none exist today):** `sim_run_baseline`, `sim_run_snake`, `sim_run_snake_dt`, `sim_run_snake_t`, `sim_run_ssnake`, `sim_run_snake_sweep_{tail,evict,throttle}`, `sim_run_snake_sens` (S5 Tier 2), `sim_run_snake_regress`, `sim_run_snake_synth`, `sim_run_snake_train`, `sim_run_snake_refcmp`, `sim_run_snake_gate`, `sim_run_snake_refcache`, `sim_run_baseline_1132`, `sim_run_snake_1132`, `sim_run_snake_calib` (S3e latency calibration), `sim_run_snake_skc` (SK_SC), `sim_run_snake_skm` (SK match), `sim_run_snake_fix` (fix_today).
- **Never** touch existing `sim_run_*`, `hw_run/` or traces. The traces are read-only `-T` inputs.

```bash
cd /home/razul/accel-sim-framework
# gate: both trees clean (tracked files) — abort otherwise
for r in . gpu-simulator/gpgpu-sim; do
  [ -z "$(git -C $r status --porcelain --untracked-files=no)" ] || { echo "DIRTY $r"; exit 1; }; done
source gpu-simulator/setup_environment.sh release && make -j -C gpu-simulator
# after the build: confirm the tags
./gpu-simulator/bin/release/accel-sim.out 2>&1 | grep -o '_modified_[0-9.]*'   # expect _modified_0.0 twice
CFG=QV100-SASS-PAPER_V100-1B_INSN
for R in "sim_run_baseline:$CFG" "sim_run_snake:$CFG-SNAKE"; do D=${R%%:*}; C=${R#*:}
  python3 util/job_launching/run_simulations.py -B rodinia-3.1 -C $C -T hw_run/rodinia-3.1/11.0 -N ${D}_rod -r $D
  python3 util/job_launching/run_simulations.py -B parboil     -C $C -T hw_run/parboil          -N ${D}_pb  -r $D
done
python3 util/job_launching/job_status.py -N sim_run_snake_rod
python3 util/job_launching/get_stats.py -r sim_run_baseline -s util/job_launching/stats/snake_stats.yml -R > docs/snake/results/baseline.csv
python3 util/job_launching/get_stats.py -r sim_run_snake    -s util/job_launching/stats/snake_stats.yml -R > docs/snake/results/snake.csv
grep -rhoE '(gpgpu-sim_git-commit|accelsim-commit)-[0-9a-f]+_modified_[0-9.]+' sim_run_baseline sim_run_snake | sort -u   # every job must be _modified_0.0
```

- **1132 MHz sensitivity run (OQ-23):** the same loop with `CFG=QV100-SASS-PAPER_V100_1132-1B_INSN` (GTO, stock 1132 MHz clock) into `sim_run_baseline_1132` and `sim_run_snake_1132`.
- Each job's output also records the build strings, so the clean-tree proof is kept with the results.
- **Rule: every run snapshots `libcudart.so` alongside `accel-sim.out`.** The run directory gets a frozen copy of both (e.g. `<run_dir>/bin/accel-sim.out` and `<run_dir>/bin/lib/libcudart.so`), and every job is launched with `LD_LIBRARY_PATH` pointing to that copy. `run_simulations.py` copies only `accel-sim.out` (`run_simulations.py:450-480`), and its `justrun.sh` takes `libcudart.so`, which holds the timing model, from the environment at launch.
- **Rule: no rebuilds while jobs are queued or running.** Before `make`, check that no `accel-sim.out` process is running and no launcher still has jobs queued (`ps`, `screen -ls`, `job_status.py`). Even with snapshots, a queued `justrun.sh` launched outside the snapshot launcher would pick up the new library.
- `-B` is narrowed to the paper apps with `suite:exe` selectors (S1 notes): `rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1,rodinia-3.1:lud-rodinia-3.1,rodinia-3.1:nw-rodinia-3.1` and `parboil:parboil-histo,parboil:parboil-mri-q`.

## 11. Risks
- **Deadlock / livelock:**
  - a prefetch must never RESERVATION_FAIL-retry or block the latency queue (it bypasses it)
  - MSHR reserve for demand
  - prefetch mfs dropped at writeback without consuming the slot
  - a watchdog stat: `gpu_stall_dramfull` and no-progress cycles compared with baseline
- **MSHR starvation:** `mshr_reserve`; watch `pf_caused_resfail` and L1D RESERVATION_FAIL vs. baseline.
- **Stats double counting:** single bit per sector plus the invariant assert; the paper's L1 hit rate is taken from `GLOBAL_ACC_R` rows only.
- **Determinism:**
  - no hashing of pointers
  - no `unordered_map` iteration affecting order
  - a seeded PRNG is not needed
- **Behaviour change when off:** every hook is guarded by `m_snake != nullptr`, and the zero-row print guard; checked by §9.1.
- **Slowdown:** ≤15% with Snake on; measured via `gpgpu_simulation_rate`.
- **Decoupling invasiveness (mode 1)** is the riskiest change to `tag_array`. It is isolated behind `decouple_mode==1`, so mode 2 is a fallback.
- **Kernel-relative PCs** alias across concurrent kernels → OQ-19.

## 12. Open Questions
All of OQ-1…OQ-18 from SPEC §7 carry over, plus:
- **OQ-19 Concurrent kernels and PC aliasing.** Key Snake state by (kernel uid, PC) and flush per SM when an SM starts a CTA of a different kernel. (medium)
- **OQ-20 Warp-slot reuse.** On `init_warps`, invalidate Head[w] (no stride across CTAs) but keep Tail bits. (medium)
- **OQ-21 Scope.** Train and prefetch only on L1-cached global loads (`GLOBAL_ACC_R`, not bypass, not local, texture or const). Stores and atomics are ignored. (high)
- **OQ-22 Isolated-Snake size.** Same capacity as the prefetch half (default 256 lines). (low)
  - **Implemented (S4, 2026-10-15),** `-snake_decouple_mode 2`, `-snake_isolated_lines` 256 (32 KB) per SM: a fully associative LRU buffer with a valid bit per sector, beside the L1D.
    - A prefetch takes an MSHR entry and a miss-queue slot as before, but no L1D line (`pf_drop_line` cannot occur). A sector already in the buffer is a duplicate (`pf_dup_l1`, also `snake_iso_dup`).
    - Its fill goes to the buffer, or, if a demand merged into it (late), to the L1D line that demand reserved.
    - An L1D access (load or store) that misses and finds its sector in the buffer first moves the sector into the L1D (allocated as a fill, the replacement policy's victim; no added latency) and then proceeds, so it hits. The moved sector counts as timely (`snake_iso_hits`); a sector the L1D cannot place stays (`snake_iso_transfer_fail`).
    - A fill into a full buffer evicts the LRU line (its unused sectors: evicted unused) and fires the space trigger (OQ-10).
    - The buffer is invalidated with the L1D between kernels. Regions, bulk free and the L1 cap are mode 1 only. (decided)
- **OQ-23 Clock.** Use the paper's 1530 MHz, or the base's 1132 MHz QV100 calibration? **Decided:** 1530 for the main comparison, plus one 1132 sensitivity run.
- **OQ-24 Rodinia version.** 3.1 (srad_v1) vs. 2.0-ft (srad_v2). **Decided:** 3.1.
- **OQ-25 Competitor prefetchers** (INTRA, INTER, MTA, CTA-Aware, Tree). **Decided:** none for now; Snake ablations only.
- **OQ-26 Intra-warp stride storage (S2).** The stride belongs to the load PC and is kept in every Tail entry with that PC1. The non-consecutive case walks this warp's chain through the most recently used qualifying entry at each step. (medium)
- **OQ-27 Promotion-time targets (S2).** "Issues prefetching requests for all future warps as soon as … promoted" (§3.2) is read as: every other warp whose Head register is at PC1 and whose warpID bit is not yet set. From S3a the target warp must also still be active (`shader_core_ctx::warp_active`). (medium)
- **OQ-17 Lane check.** **Decided 2026-10-13:** by default `snake_lane_mode 1`, a deviation from §3.4: the per-PC lane pattern in S2b, replaced in S2c by the stateless run filter. The paper-literal rule stays as mode 0. See SPEC OQ-17.
- **OQ-9 Footprint.** **Revised 2026-10-13 (S2c):** `snake_footprint` 0 = one 128 B line (default), 1 = exact lane descriptor per Tail entry. See SPEC OQ-9.
- **OQ-32 (S3d):** promoted inter-thread strides serve all warps; see SPEC §7.
- **OQ-33…OQ-36 (SK_SC):** SnakeCharmer's SK settings (inter-warp targets, intra-warp distance, confidence 3) and metrics; see SPEC §7.
- **OQ-37, OQ-38 (SK match):** same-CTA targets and the per-warp Head in every SK variant; R2, the separate prefetch-queue injection path (implemented under fix_today); see SPEC §7.
- **OQ-39, OQ-40 (fix_today):** accuracy filter; single inter-warp target and CTA anchoring (our extensions); see SPEC §7.
- **OQ-29…OQ-31 (S3b):** combined lookahead for inter-warp targets, the inter-warp eviction weight, and inter-thread priority in the prefetch queue; see SPEC §7.
- **OQ-28 Demotion (S2).** A demoted entry (more than `demote_removed` warps removed, §3.2) also clears its warpID vector, so it is re-trained from scratch instead of being re-promoted by the remaining bits. (low)
