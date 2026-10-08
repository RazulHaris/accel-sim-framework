# Snake — code map of the global-load path (Phase 2)

Paper: Mostofi et al., "Snake: A Variable-length Chain-based Prefetching for GPUs", MICRO 2023 (`docs/snake.pdf`, kept out of git).
Bases: accel-sim-framework `3016c65`, gpgpu-sim `6c3cf4ff` (both on branch `Snake`). All `file:line` references are at those commits.
Companion docs: [SPEC](SPEC.md) · [VERSION_DIFF](VERSION_DIFF.md) · [CODE_MAP](CODE_MAP.md) · [SNAKE_PLAN](SNAKE_PLAN.md)

Roots:
- **G** = gpgpu-sim @ `6c3cf4ff`. `shader.*`, `gpu-cache.*`, `gpu-sim.cc`, `l2cache.cc` and `mem_fetch.*` are under `G/src/gpgpu-sim/`; `abstract_hardware_model.*` (AHM) is under `G/src/`.
- **A** = accel-sim @ `3016c65`.

## 1. Path of a global load
1. **Clock.**
   - `gpgpu_sim::cycle` gpu-sim.cc:1973.
   - Order: core `icnt_cycle` (responses, 1976-1980) → icnt return push (1982-2004) → DRAM (2007-2028) → L2 (2032-2052) → `icnt_transfer` (2059) → `core_cycle` (2063-2070).
   - `simt_core_cluster::core_cycle` shader.cc:4486 → `shader_core_ctx::cycle` shader.cc:3673. That runs writeback → execute (→ `ldst_unit::cycle`, 1811) → read_operands → issue → decode/fetch.
2. **Warp issue.**
   - `scheduler_unit::cycle` shader.cc:1260. Memory ops go to `issue_warp(*m_mem_out, …)` at 1345-1354.
   - Trace mode: `trace_shader_core_ctx::issue_warp` (A/gpu-simulator/trace-driven/trace_driven.cc:660) → `shader_core_ctx::issue_warp` shader.cc:1037.
   - In there: `warp_inst_t::issue` (AHM.cc:53) sets warp, dynamic warp, uid and stream. Then `func_exec_inst` (shader.cc:1053) → trace `func_exec_inst` (trace_driven.cc:644) → `generate_mem_accesses()` (656).
3. **Trace → `warp_inst_t`.**
   - `trace_warp_inst_t::parse_from_trace_struct` trace_driven.cc:155.
   - Sets `pc` (168) and the per-lane addresses `set_addr(i, addrs[i])` (247-251).
   - LDG/LDL → global/local space, `CACHE_ALL` (262-283). `.STRONG.GPU` / `.BYPASS` → `CACHE_GLOBAL` = L1 bypass (281).
4. **Coalescing.**
   - `generate_mem_accesses` AHM.cc:286 → `memory_coalescing_arch` AHM.cc:477. With `coalesce_arch ≥ 40` (QV100 = 70) the segments are 32 B.
   - `memory_coalescing_arch_reduce_and_send` AHM.cc:696 pushes `mem_access_t` into `m_accessq` (747). **One sector per access** (`get_sector_index` asserts `count()==1`, gpu-cache.h:505-506).
5. **LD/ST unit.**
   - `ldst_unit::issue` shader.cc:2674: `m_pending_writes[warp][reg] += accessq_count()` (2686).
   - `ldst_unit::cycle` shader.cc:2843 → `memory_cycle` 2261.
   - Bypass decision 2274-2281. A bypassed access goes straight to `m_icnt->push` (2306-2310).
   - Otherwise `process_memory_access_queue_l1cache` 2063: `m_mf_allocator->alloc` (2074) → `l1_latency_queue[bank][lat-1]` (2083). A busy slot gives `BK_CONF` (2097).
6. **L1D access** at latency-queue stage 0: `ldst_unit::L1_latency_queue_cycle` shader.cc:2122 → `m_L1D->access` (2128).
   - `l1_cache::access` gpu-cache.cc:2001 → `data_cache::access` 1977 (`tag_array::probe`, then `process_tag_probe` 1929, then stats 1988).
   - **HIT** (2136): `--m_pending_writes`; at zero, `releaseRegister` + `warp_inst_complete` (2142-2153).
   - **RESERVATION_FAIL** (2183): the request stays and retries.
   - **MISS / HIT_RESERVED** (2186): the slot is freed.
   - Status enum gpu-cache.h:49; stat mapping `select_stats_status` gpu-cache.cc:722.
7. **Miss.**
   - `rd_miss_base` gpu-cache.cc:1843 checks `miss_queue_full`, then `send_read_request` 1354.
   - MSHR probe/full (1362-1363). Merge = MSHR_HIT (1364-1372). Otherwise a new entry plus `m_miss_queue.push_back` (1374-1388).
   - MSHR: `mshr_table` gpu-cache.h:1024. `probe` .cc:554, `full` 560, `add` 569, `mark_ready` 595, `next_access` 605.
8. **Miss queue → interconnect.**
   - `baseline_cache::cycle` gpu-cache.cc:1215 → `m_memport->push`. `shader_memory_interface` shader.h:2735 → `simt_core_cluster::icnt_inject_request_packet` shader.cc:4615 → `update_icnt_stats` 4639. **Its `switch` ends in `assert(0)` (4681-4682).**
9. **L2 / DRAM.**
   - `memory_sub_partition::push` l2cache.cc:786 (sector split at 718) → `cache_cycle` 465 → `m_L2cache->access` 528.
   - Hit: reply (545-548). Miss: to DRAM (514). `dram_cycle` 306; L2 fill 500.
   - Return: gpu-sim.cc:1984-1995 `icnt_push(…, mf->get_tpc())`.
10. **Response.**
    - `simt_core_cluster::icnt_cycle` shader.cc:4721 → `memlatstat_read_done` + `accept_ldst_unit_response` (4733-4736) → `ldst_unit::fill` 2342, which pushes onto `m_response_fifo`.
    - `ldst_unit::cycle` 2850-2905 → `m_L1D->fill` (2897-2900) → `baseline_cache::fill` gpu-cache.cc:1231 (tag fill 1255-1258, `mark_ready` 1262).
11. **Writeback and wake-up.**
    - `ldst_unit::writeback` shader.cc:2701. Client 4: `m_L1D->access_ready()` → `next_access()` → `m_next_wb = mf->get_inst(); delete mf` (2794-2800).
    - Next cycle, `m_next_wb` is written back: `--m_pending_writes`, `releaseRegister`, `warp_inst_complete` (2703-2744).
    - Global loads `dec_inst_in_pipeline` when they leave the dispatch register (2975).

## 2. Where each identifier is available

| Info | Location |
|---|---|
| PC | `inst_t::pc` AHM.h:1007; `mem_fetch::get_pc()` mem_fetch.h:122 (−1 if no instruction) |
| Warp (HW slot) | `warp_inst_t::warp_id()` AHM.h:1198; `mem_fetch::get_wid()` mem_fetch.h:98 |
| Dynamic warp | `warp_inst_t::dynamic_warp_id()` AHM.h:1206 |
| Per-lane addresses | `warp_inst_t::get_addr(n)` AHM.h:1214, `get_active_mask()` 1111, `data_size` 1049 |
| Coalesced accesses | `accessq_count/back` AHM.h:1223-1226. **`mem_access_t` has no PC or warp ID** (getters at 835-842) |
| SM | `ldst_unit::m_sid` shader.h:1449; `mem_fetch::get_sid()` mem_fetch.h:96 |
| Stream | `warp_inst_t::get_streamID()` AHM.h:1237. L1 stats are keyed by stream (gpu-cache.cc:670) |
| CTA | Only the HW CTA slot: `shd_warp_t::get_cta_id()` shader.h:272, set in `init` 143-147 from `init_warps` shader.cc:574. The logical `ctaid` is not stored. `m_warp` is protected (shader.h:2530), so **a getter is needed** |

**Observation point:** `ldst_unit::issue` (shader.cc:2674). PC, warp ID, active mask, lane addresses and `m_accessq` all exist there, and `m_core` / `m_sid` are reachable.

## 3. Creating and injecting a prefetch
- **Precedent `L1_WR_ALLOC_R`** (`data_cache::wr_miss_wa_fetch_on_write` gpu-cache.cc:1685-1700):
  - builds `new mem_access_t(type, addr, …)` and `new mem_fetch(*ma, NULL /*inst*/, stream, …, wid, sid, tpc, …)`
  - calls `send_read_request`, so it gets an MSHR and the miss queue, goes to L2, and fills
  - its response hits writeback client 4 with an **empty** `m_next_wb`, so no warp is woken (shader.cc:2795-2799)
- **Tag the request with a new access type `L1_PREFETCH_R`** in `MEM_ACCESS_TYPE_TUP_DEF` (AHM.h:772-778), appended before `NUM_`.
  - It survives the L2 sector split. `l2cache.cc:731-776` copies only the access type, so a new mf field would be lost.
  - Required switch cases: `update_icnt_stats` (shader.cc:4645-4682) and `traffic_breakdown.cc:32-51`.
- **Pitfalls:**
  - **The empty instruction's `memory_op` is uninitialized** (AHM.h:940-969). `L1_latency_queue_cycle` dereferences `get_inst().is_load()/out[]` (shader.cc:2139-2201), so **prefetches must not enter `l1_latency_queue`**. Probe L1 directly instead.
  - `memlatstat_read_done` (mem_latency_stat.cc:205) must be skipped for prefetches, or they pollute the latency stats. Keep a real `sid`.
- **Keeping the warp asleep:** in `ldst_unit::writeback` case 4, pop and delete prefetch mfs *without* consuming the writeback slot. MSHR-merged demand requests behind them are still delivered.
  - Late-prefetch hook: a demand merging into a prefetch's MSHR entry (`mshr_table::add`, gpu-cache.cc:569).

## 4. Prefetched-line bookkeeping
- **Line state:**
  - `sector_cache_block` gpu-cache.h:285-503 (per-sector `m_status[4]`, `m_readable[4]`, timestamps 491-503); `line_cache_block` 172-283.
  - Add `bool m_snake_pf[SECTOR_CHUNCK_SIZE]` (set on fill of a prefetch mf; cleared on the first demand hit).
  - Add a line-level `m_snake_region` flag for decoupling.
- **Fill:** `baseline_cache::fill` gpu-cache.cc:1231 → `tag_array::fill` 402/408/439 (the mf is available).
- **First use:** `tag_array::access` HIT / HIT_RESERVED (gpu-cache.cc:353-356); `rd_hit_base` 1819.
- **Eviction:**
  - victim choice in `tag_array::probe` gpu-cache.cc:246-300
  - overwrite at `tag_array::access` MISS, `m_lines[idx]->allocate` (371) → hook "evicted unused" just before it
  - also `tag_array::fill` ON_FILL MISS (422-424), `flush` 450, `invalidate` 464
  - QV100 L1 is ON_MISS (`dl1 …,L:T:m:L:L`, config :152). The adaptive code (shader.cc:3612-3621) only switches *streaming* caches.

## 5. Config options and stats
- **Registration:** `option_parser_register(opp, "-name", OPT_UINT32|OPT_BOOL|OPT_CSTR|OPT_FLOAT|OPT_INT32, &field, "desc", "default")`.
  - `shader_core_config::reg_options` gpu-sim.cc:328-668 (L1D options 353-389). `gpgpu_sim_config::reg_options` 670.
  - Accel-sim's own: `trace_config::reg_options` trace_driven.cc:411.
  - Cache string parsing: `cache_config::init` gpu-cache.h:569-770.
  - Adaptive resize: `shader_core_config::max_cta` shader.cc:3584-3621.
- **Stats:**
  - `shader_core_stats_pod` shader.h:1722; `shader_core_stats::print` shader.cc:618.
  - `gpgpu_sim::print_stats` gpu-sim.cc:1263 → `gpu_print_stat` 1449 (L1 per core via `shader_print_cache_stats` shader.cc:3094; `Total_core_cache_stats_breakdown` 1522-1532; L2 1565-1603).
  - Accel-sim calls this per kernel at `accel-sim.cc:145`.
- **⚠ Output stability:** `cache_stats::print_stats` (gpu-cache.cc:938-944) prints **every** access type × status unconditionally. A new type would add zero rows even with Snake off.
  - **Hook:** skip `L1_PREFETCH_R` rows when they are all zero.
  - `cache_sub_stats` totals (gpu-cache.cc:1021-1046) sum over all types, so prefetches inflate `L1D_total_cache_*`. The paper's L1 hit rate must come from the `GLOBAL_ACC_R` rows.

## 6. Existing prefetch code
- **None** in either repo. `grep -i prefetch` only matches a texture-cache comment (gpu-cache.h:1748), AccelWattch XML fields, and accel-sim's trace-reader thread (A/gpu-simulator/main.cc:11).
- Reuse only the `L1_WR_ALLOC_R` / writeback patterns.
- Do not touch the oracle-branch code (not on this base).

## 7. Extra facts that affect the design
- **Trace PCs are kernel-relative SASS offsets.** Two kernels resident on one SM can alias PCs → **OQ-19**.
- **HW warp slots are reused** across CTAs, so Head rows can be stale → **OQ-20**.
- **The L1 bypass path** (`.STRONG.GPU`, `gmem_skip_L1D`) is never seen by L1. Snake trains only on L1-cached global loads → **OQ-21**.
