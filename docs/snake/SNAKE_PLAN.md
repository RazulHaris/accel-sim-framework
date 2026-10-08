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
| S1 infrastructure | done | `snake: S1 infrastructure` (hash recorded with S2) | `e0d23397` | off-regression result recorded with S2 (see S1 notes) |
| S2 training only | pending | | | |
| S3a/S3b issue (intra + inter-thread, then inter-warp) | pending | | | |
| S4 throttling + decoupling | pending | | | |
| S5 sweeps + cleanup | pending | | | |

**Reference build for the off-regression (§9.1):** `/home/razul/snake/bin_ref/` holds `accel-sim.out` plus `lib/libcudart.so`, which contains the gpgpu-sim model. It was built 2026-10-08 from the clean bases (`accelsim-commit-3016c65_modified_0.0`, `gpgpu-sim_git-commit-6c3cf4ff_modified_0.0`); checksums are in `BUILD_INFO.txt`.

**Rule:** a stage's own commit cannot contain its hash or the result of a gate run on it. Both are recorded in the next stage's commit.

**S1 notes (clarifications of the plan, no change of design):**
- **Injection API deferred.** `l1_cache::snake_prefetch()` (§2) and the `shader_core_ctx` getters `get_warp_cta_slot` / `warp_active` arrive in S3, where they are first used. They are calls *from* Snake, not hooks; every hook in §2/§4 is wired in S1.
- **One hook in place of two.** `on_kernel_switch` is detected inside `on_warp_init` (kernel uid change, OQ-19), so `init_warps` has one hook.
- **App selection.** No new app suite is needed. `run_simulations.py -B` accepts `suite:exe[:argindex]` selectors (`util/job_launching/common.py:110-131`), e.g. `rodinia-3.1:backprop-rodinia-3.1:0`.
- **What "byte-identical" excludes.** gpgpu-sim prints every registered option at start-up (`OptionParser::Print`, `src/option_parser.cc:422`), so the new `-snake_*` options add lines to that dump. The off-regression therefore removes exactly those lines, and the volatile lines (dates, wall-clock, simulation rate, build strings, binary paths), and then requires the rest to be byte-identical.
- **Pin the model library.** In trace mode `run_simulations.py` copies only `accel-sim.out` into `gpgpu-sim-builds/` (`run_simulations.py:450-480`). The timing model lives in `libcudart.so`, which jobs load through `LD_LIBRARY_PATH` at run time, so a rebuild can change the model under running or queued jobs. Every Snake run uses a frozen `accel-sim.out` + `libcudart.so` pair and sets `LD_LIBRARY_PATH` explicitly.

## 1. Design summary
- **One object per SM.** Each `ldst_unit` owns a `snake_prefetcher`. The object is only constructed when `-snake_enable 1`; with Snake off, every hook is a null-pointer check.
- **Training.** `ldst_unit::issue` passes (warp slot, CTA slot, PC, active mask, lane addresses, stream) to `snake_prefetcher::observe()` for L1-cached global loads. Observe does:
  - the lane-stride check (§3.4)
  - the Head update, which produces (q, p, s)
  - Tail match / allocate / bit-set, plus inter-thread, intra-warp and inter-warp training (§3.1)
  - prediction (§3.2): it computes target warp base addresses and expands each into sector addresses with the triggering access's lane stride and active-lane count (OQ-9)
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
  - 1 = paper (§3.2): a line-level region flag and a quota-aware victim choice in `tag_array`; hit → flag flip; 25% bulk free; 50%-until-trained and 50-cycle confinement.
  - 2 = isolated buffer (§5.7).
  - 0 = none (Snake-DT / s-Snake ablations).
- **Throttle** (S4, §3.3): the space event halts for 50 cycles; the bandwidth meter (70%/50% hysteresis) halts until the level recovers. Both also cap the chain depth.

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
- `util/job_launching/configs/define-standard-cfgs.yml` entries: `SNAKE`, `SNAKE_DT`, `SNAKE_T`, `S_SNAKE`, `SNAKE_ISO`, `PAPER_V100` (GTO + 1530 MHz clock), `PAPER_V100_1132` (GTO, stock 1132 MHz; OQ-23 sensitivity run)
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
| gpu-cache.cc:246 `tag_array::probe` | S4: quota-aware victim (mode 1 only) | decoupling |
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
   - Order: inter-thread chain (depth ≤ `max_chain_depth` × throttle factor), then inter-warp (≤`interwarp_degree` targets), then intra-warp.
   - Each target warp base is expanded to sectors and deduplicated in the queue (same sector already queued → `pf_dup_queue`).
   - A full queue drops the newest (`pf_drop_queue_full`).
2. **Wait** `snake_latency` cycles.
3. **Issue** in `tick` when not throttled. `l1_cache::snake_prefetch`:
   - tag probe HIT or HIT_RESERVED → `pf_dup_l1`
   - MSHR has the line → `pf_dup_mshr`
   - MSHR free ≤ `mshr_reserve` or miss queue full → `pf_drop_resource`
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
| `snake_max_chain_depth` | uint | 3 | OQ-7 |
| `snake_interwarp_degree` | uint | 4 | OQ-8 |
| `snake_enable_inter_thread` / `_inter_warp` / `_intra_warp` | bool | 1/1/1 | s-Snake = 1/0/0 (§4) |
| `snake_require_equal_lane_stride` | bool | 1 | §3.4 |
| `snake_latency` | uint | 2 | §5.5 |
| `snake_queue_size` | uint | 32 | OQ-4 |
| `snake_issue_width` | uint | 1 | OQ-4 |
| `snake_mshr_reserve` | uint | 8 | OQ-11 |
| `snake_l1_filter` | bool | 1 | "not in L1" (§3.2) |
| `snake_throttle_enable` | bool | 1 | Snake-T = 0 (§4) |
| `snake_throttle_space_cycles` | uint | 50 | §3.3, Fig. 23 |
| `snake_bw_high` / `snake_bw_low` | float | 0.70 / 0.50 | §3.3 |
| `snake_bw_window` | uint | 100 | OQ-10 |
| `snake_decouple_mode` | uint | 1 | 0 none / 1 unified / 2 isolated (OQ-12) |
| `snake_untrained_l1_frac` | float | 0.50 | §3.2 |
| `snake_throttle_confine_cycles` | uint | 50 | §3.2 |
| `snake_free_frac` | float | 0.25 | §3.2, footnote 2 |
| `snake_transfer_frac` | float | 0.80 | §3.2 |
| `snake_isolated_lines` | uint | 256 | mode 2 size (32 KB; §5.7 gives no size → OQ-22) |
| `snake_energy_pj_per_access` / `snake_static_mw` | float | 6.4 / 6.0 | §5.5 (offline energy) |
| `snake_debug_trace` | bool | 0 | per-event log for tests |

The ablations come from these flags:
- **Snake-DT** = `decouple_mode 0`, `throttle 0`
- **Snake-T** = `decouple 1`, `throttle 0`
- **s-Snake** = inter-thread only

## 7. Stats (per SM; GPU sum printed as `snake_*`)
- **Issue:** `pf_generated`, `pf_issued`, `pf_dup_queue`, `pf_dup_l1`, `pf_dup_mshr`, `pf_drop_queue_full`, `pf_drop_resource`, `pf_drop_throttled`.
- **Outcome:** `pf_filled`, `pf_useful_timely`, `pf_late`, `pf_evicted_unused`, `pf_unused_at_end`, `pf_caused_resfail`.
- **Demand base:** `demand_sector_accesses` (`GLOBAL_ACC_R`, L1-cached).
- **Derived:**
  - `coverage_paper = (timely+late)/demand`
  - `accuracy_paper = timely/demand`
  - `accuracy_std = (timely+late)/issued`
  - `timeliness = timely/(timely+late)`
- **Tables:** `head_updates`, `tail_lookups`, `tail_hits`, `tail_allocs`, `tail_evictions`, `t1_promotions`, `t1_demotions`, `t2_trained`, `interwarp_trained`, `warp_excluded_lane_stride`, `chain_depth_hist[0..D]`.
- **Per mode:** {inter_thread, inter_warp, intra_warp} × {generated, issued, useful, late, unused}.
- **Throttle:** `throttle_space_events`, `throttle_bw_events`, `throttled_cycles`, `bw_util_hist`, and an optional time series (`snake_throttle_log` every N cycles).
- **Decoupling:** `region_flips`, `bulk_frees`, `bulk_free_l1_side` / `_pf_side`, `l1_quota_victims`.
- **No double counting:** each issued prefetch sector ends in exactly one terminal state.
  - At fill, if a demand had merged into its MSHR entry → `late` (bit not set). Otherwise the bit is set, and the bit later resolves to exactly one of `useful_timely` / `evicted_unused` / `unused_at_end`.
  - Invariant `issued == useful_timely + late + evicted_unused + unused_at_end`, asserted at end of kernel in debug builds.

## 8. Staged implementation (each stage = one commit per changed repo, `snake: S<n> <desc>`)
- **S1 Infrastructure.** Options, empty `snake_prefetcher` wired to all hooks, `L1_PREFETCH_R` type, print guard, stats skeleton, configs/yml in accel-sim.
  - *Done when:* it builds and the **off-regression** (§9.1) is byte-identical on 4 workloads; with Snake on (no-op) it is also identical apart from the `snake_*` block.
- **S2 Training only.** Head/Tail/T1/T2/inter-warp, lane-stride check, demotion, kernel/warp resets, eviction policy, `snake_debug_trace`.
  - *Done when:* synthetic traces (§9.2) give the expected strides and promotions (unit-style assertions on the debug log), and on rodinia hotspot, srad_v1 and backprop `tail_hits/lookups`, promotions and the chain-depth histogram are plausible. Timing is still identical to baseline, since there is no issue.
- **S3a Intra-warp + inter-thread issue, no throttling or decoupling.** Then **S3b** adds inter-warp.
  - *Done when:* there are no deadlocks on the full suite at a 100M-instruction cap, the `pf_woken_warps` assert is 0, the invariant holds, and accuracy and coverage print. The strided microbenchmark reaches coverage ≥90%.
- **S4 Throttling + decoupling** (modes 0/1/2).
  - *Done when:* the random microbenchmark shows `throttled_cycles` >50% and `pf_issued/demand` <10%; quotas hold (debug assert); Snake-DT < Snake-T ≤ Snake in accuracy on the suite (§5.2 trend).
- **S5 Sweep hooks + cleanup.** Sweep configs for Tail {10, 20, 40, ∞}, eviction policy, and throttle {25…300} (Figs. 20, 22, 23); `compare_paper.py`; docs updated.
  - *Done when:* the sweep run dirs generate and the plots/tables are reproducible.

## 9. Verification
1. **Off-regression.**
   - Reference: `/home/razul/snake/bin_ref/` (`accel-sim.out` + `lib/libcudart.so`, built from the clean bases; outside the repo). Run it with `LD_LIBRARY_PATH=/home/razul/snake/bin_ref/lib`.
   - Run the ref and the new binary with `-snake_enable 0` on rodinia-3.1 backprop, hotspot and nw plus parboil histo, with `-gpgpu_max_insn 100000000`.
   - Compare with `diff` after stripping wall-clock, simulation-rate and build-string lines. **Must be empty.** Repeat at every stage.
2. **Microbenchmarks.** `gen_synthetic_traces.py` writes accel-sim text traces (no GPU needed):
   - (a) a strided loop (fixed inter-thread chain + intra stride) → coverage ≥90%
   - (b) random gather → throttled, little issue
   - (c) the Fig. 15 example (4 warps, PCs 520/540) → exact Tail state and prefetch queue 2600/2700/2800/3000. This is a direct check of the SPEC.
   - Fallback: NVBit-trace small CUDA kernels on the local RTX 2080 Ti.
3. **Paper comparison.**
   - Run the suite on `QV100 + PAPER_V100 + 1B_INSN`, baseline vs. SNAKE (and the DT/T/s-Snake ablations).
   - `compare_paper.py` puts per-benchmark coverage, accuracy, IPC gain and energy beside the SPEC §5 table.
   - Every gap >10 pp is explained against the VERSION_DIFF list and the OQs.

## 10. Experiment layout
- **Run dirs (new only; verified that none exist today):** `sim_run_baseline`, `sim_run_snake`, `sim_run_snake_dt`, `sim_run_snake_t`, `sim_run_ssnake`, `sim_run_snake_sweep_{tail,evict,throttle}`, `sim_run_snake_regress`, `sim_run_baseline_1132`, `sim_run_snake_1132`.
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
- Each run directory also gets a frozen copy of the build's `libcudart.so`, and jobs are launched with `LD_LIBRARY_PATH` pointing to it (S1 notes).
- `-B` is narrowed to the paper apps through the snake-paper suite entry (exact mechanism verified in S1).

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
- **OQ-23 Clock.** Use the paper's 1530 MHz, or the base's 1132 MHz QV100 calibration? **Decided:** 1530 for the main comparison, plus one 1132 sensitivity run.
- **OQ-24 Rodinia version.** 3.1 (srad_v1) vs. 2.0-ft (srad_v2). **Decided:** 3.1.
- **OQ-25 Competitor prefetchers** (INTRA, INTER, MTA, CTA-Aware, Tree). **Decided:** none for now; Snake ablations only.
