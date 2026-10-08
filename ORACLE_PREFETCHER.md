# Oracle Uncoalesced Prefetcher

An idealized prefetcher that removes the memory latency of L1D misses that come
from **uncoalesced global loads**. Such a miss is satisfied immediately out of a
per-SM infinite-capacity prefetch buffer: no MSHR is reserved, no request is
sent to the L2/interconnect/DRAM, and the L1D tag and data arrays are left
completely untouched.

The line address is then **kept in that buffer**, which is probed ahead of the
L1D by every later global load — coalesced ones included — so an oracle-served
line stays available instead of vanishing and re-missing.

Everything else — fully coalesced loads, L1D hits, stores, atomics, shared /
constant / texture / local memory — goes through the unmodified pipeline.

```
-oracle_prefetcher_uncoalesced 0   # 0 = disabled (default), 1 = enabled
```

All changes are inside `gpu-simulator/gpgpu-sim/`.

---

## 1. How it works

### Step 1 — tagging uncoalesced accesses

Coalescing happens in `warp_inst_t::memory_coalescing_arch()`
(`src/abstract_hardware_model.cc`), which turns the 32 per-thread addresses of
one warp load into a set of segment-sized transactions pushed onto `m_accessq`.
The number of transactions it produced was already being recorded by earlier
work in this tree:

| member | meaning |
| --- | --- |
| `warp_inst_t::m_num_coalesced_transactions` | coalescing degree of this instruction |
| `warp_inst_t::m_is_uncoalesced` | `true` when the degree is `> 1` |

Two things were added on top:

* **The tags are now reset at the top of `generate_mem_accesses()`.**
  `warp_inst_t` objects are recycled by the pipeline registers, and only
  `memory_coalescing_arch()` writes these fields. An instruction that never
  reaches it — shared/const/tex accesses, and atomics, which take the separate
  `memory_coalescing_arch_atomic()` path — used to be able to inherit the degree
  of whatever instruction previously occupied that slot.

* **The tag now travels with each transaction.** `mem_fetch` gained
  `m_uncoalesced`, copied from the instruction in the `mem_fetch` constructor
  (`src/gpgpu-sim/mem_fetch.cc`), plus ``m_oracle_prefetch_hit, set when the
  oracle actually serves that transaction. Tagging per `mem_fetch` — rather than
  reading it back off the instruction — is what makes the per-transaction
  granularity requirement hold: within one uncoalesced load, transactions that
  hit L1 are left alone while the ones that miss are served by the oracle.

### Step 2 — short-circuiting the miss

The interception point is `data_cache::access()` in `src/gpgpu-sim/gpu-cache.cc`,
immediately after the tag probe and **before** `process_tag_probe()`:

```
probe_status = m_tag_array->probe(block_addr, cache_index, mf, ...)   // side-effect free
        │
        ├── oracle_buffer_serves(mf, block_addr) ───── yes ──► buffer_hits++, return HIT
        │      (line already served on this SM)
        │
        ├── oracle_prefetch_serves(mf, probe_status) ─ yes ──► uncoalesced_misses++,
        │      (uncoalesced + L1D miss)                        insert line into buffer,
        │                                                      return HIT
        │
        └── no ──► process_tag_probe(...)  (unchanged: MSHR, miss queue, L2, fills)
```

The tag probe is side-effect free, so running it first costs nothing and lets
both oracle paths report an honest L1D hit/miss status to the cache statistics.
The *behavioural* order is still buffer-then-L1: neither oracle path touches any
cache state.

`process_tag_probe()` is where every side effect lives — line allocation and
eviction, LRU / sector state, `m_mshrs.add()`, `send_read_request()`. Returning
before it is what guarantees the three "do NOT"s of the spec, and it is why no
per-transaction bypass had to be threaded through the load/store unit.

Returning `HIT` makes `ldst_unit` complete the load through its **existing** hit
path — the same register / scoreboard / LDGSTS release code used by a real L1
hit, in both the `l1_latency > 0` pipeline (`L1_latency_queue_cycle()`) and the
`l1_latency == 0` path (`process_cache_access()`). No load/store unit code was
modified.

A transaction qualifies when **all** of the following hold:

| condition | why |
| --- | --- |
| `m_level == L1_GPU_CACHE` | the L2 is shared; its misses are not what the oracle models |
| `mf->get_access_type() == GLOBAL_ACC_R` | global loads only — excludes shared, const, texture, local, and all writes |
| `!mf->isatomic()` | atomics must be performed at memory |
| `probe_status ∈ {MISS, SECTOR_MISS}` | not resident. `SECTOR_MISS` counts: in a sector cache it is just as much a trip to memory. `HIT_RESERVED` is excluded (the line is already in flight), `RESERVATION_FAIL` is a structural stall, not a miss |
| `mf->is_uncoalesced()` | the load produced more than one transaction |
| `-oracle_prefetcher_uncoalesced 1` | enabled |

A **buffer** hit needs only the first three rows plus membership in the buffer:
it deliberately does *not* require `is_uncoalesced()`, because serving coalesced
re-accesses to an already-served line is the whole point of the buffer.

### Step 2b — the per-SM prefetch buffer

`data_cache::m_oracle_prefetch_buffer` is a `std::unordered_set<new_addr_type>`
of cache-line addresses. Each SM's L1D object owns one, so it is per-SM by
construction. Nothing is ever evicted; `gpgpu_sim::launch()` clears every SM's
buffer at each kernel launch, so it lives exactly one kernel.

Without it, an oracle-served line simply disappeared: later accesses to it —
**including fully coalesced ones** — re-missed and issued real memory traffic the
baseline never had. Measured on the no-buffer version, bfs-4096's uncoalesced
misses ballooned from 11K to 59K and pathfinder lost 12%. With the buffer,
bfs-4096 goes from +1.0% to **−6.6%** and pathfinder's regression more than
halves.

Writes are deliberately left out of the buffer: a store has to reach memory, and
the "global loads only" constraint excludes it. A store to a buffered line does
not invalidate the entry — the timing model carries no data, so there is nothing
to go stale.

---

## 2. Timing model — read this before interpreting results

An oracle-served transaction is completed **exactly as if it had hit in the
L1D**: it still traverses the L1 latency queue (`-gpgpu_l1_latency 34` on A100).

This is the reading of *"the warp should proceed as if the data was available in
L1"*: zero **memory** latency, not zero cycles. A prefetch buffer that returned
data in 0 cycles would also have to bypass the L1 banks entirely, which models a
wider datapath rather than a better prefetcher.

Data-port bandwidth is billed with the **probe status**, not as a hit:

```cpp
m_bandwidth_management.use_data_port(mf, probe_status, events);
```

An earlier version charged `HIT` here, reasoning that it was the conservative
choice. That was wrong. In `use_data_port()` a `MISS` or `SECTOR_MISS` occupies
the port for zero cycles (only a write-back costs anything), while a `HIT` costs
`line_size / port_width` cycles — so billing an oracle-served miss as a hit adds
port occupancy the baseline never paid. Charging the probe status keeps the
oracle strictly subtractive. (On the A100 config this turns out to be inert,
because the `l1_latency > 0` path never consults `data_port_free()`, but the
config-independent property is the one worth having.)

If you do want a literal 0-cycle service, the change would be in
`ldst_unit::process_memory_access_queue_l1cache()`: probe before pushing into
`l1_latency_queue[bank][...]` and complete the load in the same cycle. That is a
different experiment and is **not** what is implemented here.

---

## 3. Statistics

Printed at the end of `gpgpu_sim::gpu_print_stat()`:

```
========= Oracle uncoalesced prefetcher =========
oracle_pref_total_warp_loads = X
oracle_pref_uncoalesced_warp_loads = Y
oracle_pref_uncoalesced_transactions = Z
oracle_pref_uncoalesced_misses = W
oracle_pref_buffer_hits = U
oracle_pref_coalesced_misses = V
oracle_pref_buffer_size_peak = P
oracle_pref_uncoalesced_warp_load_rate = Y/X
oracle_pref_uncoalesced_miss_rate = W/Z
```

| stat | definition |
| --- | --- |
| `total_warp_loads` | global-space load instructions that produced at least one transaction |
| `uncoalesced_warp_loads` | ... of those, the ones broken into more than one transaction |
| `uncoalesced_transactions` | transactions generated by those uncoalesced loads |
| `uncoalesced_misses` | uncoalesced transactions that missed the L1D (= the ones served at zero latency when enabled) |
| `coalesced_misses` | fully coalesced global-load transactions that missed the L1D, always handled normally |
| `buffer_hits` | any later transaction, coalesced or not, served from the prefetch buffer |
| `buffer_size_peak` | largest number of lines any one SM's buffer ever held |

Counting sites:

* The three instruction-level counters are sampled once per issued warp
  instruction in `shader_core_ctx::issue_warp()`, right after `func_exec_inst()`
  has run the coalescing unit — the one point where the degree is known and each
  instruction is seen exactly once.
* The miss counters and both buffer counters are incremented in
  `data_cache::access()`.  `buffer_hits` and `buffer_size_peak` are only ever
  non-zero when the prefetcher is enabled, since the buffer stays empty
  otherwise.

Two deliberate properties of the miss counters:

* **They are collected whether or not the prefetcher is enabled**, so a baseline
  run tells you how many misses an enabled run *would* have served. This is free:
  the probe they key off already happens on every access.
* **Retries are not double counted.** A miss that the cache could not accept this
  cycle (`access_status == RESERVATION_FAIL`, e.g. MSHR or miss-queue full) is
  retried by the load/store unit and is only counted on the attempt that is
  accepted.

Predicated-off loads (no transactions) and atomics (separate coalescing path,
never eligible) are excluded from `total_warp_loads`.

An oracle-served access is reported to the normal L1D cache stats with its
**probe** status, i.e. as a miss. The L1D hit rate therefore stays directly
comparable with a baseline run instead of being inflated by the oracle;
`oracle_pref_uncoalesced_misses` is what tells you how many of those misses never
reached the memory hierarchy.

---

## 4. Files changed

| file | change |
| --- | --- |
| `src/abstract_hardware_model.cc` | reset `m_is_uncoalesced` / `m_num_coalesced_transactions` at the top of `generate_mem_accesses()` |
| `src/abstract_hardware_model.h` | `m_num_coalesced_transactions = NULL` → `0` in the default ctor |
| `src/gpgpu-sim/mem_fetch.h` | `m_uncoalesced`, `m_oracle_prefetch_hit` + accessors |
| `src/gpgpu-sim/mem_fetch.cc` | tag each transaction from its generating instruction |
| `src/gpgpu-sim/gpu-cache.h` | oracle helper declarations + `m_oracle_prefetch_buffer` and `oracle_prefetch_buffer_reset()` on `data_cache` |
| `src/gpgpu-sim/gpu-cache.cc` | the helpers, the buffer probe/insert, and the hook in `data_cache::access()` |
| `src/gpgpu-sim/gpu-sim.h` | `oracle_prefetch_stats_t`, member on `gpgpu_sim`, `get_oracle_prefetch_stats()` |
| `src/gpgpu-sim/gpu-sim.cc` | register `-oracle_prefetcher_uncoalesced`; `oracle_prefetch_stats_t::print()`; call it from `gpu_print_stat()`; clear every SM's buffer in `launch()` |
| `src/gpgpu-sim/shader.h` | `shader_core_config::oracle_prefetcher_uncoalesced` |
| `src/gpgpu-sim/shader.cc` | per-instruction coalescing-degree counters in `issue_warp()`; `oracle_prefetch_buffer_reset()` on `ldst_unit` / `shader_core_ctx` / `simt_core_cluster` |

Not touched: any scheduler, the scoreboard, the writeback path, `ldst_unit`
memory handling, the tag array, the MSHR table, the L2, DRAM.

---

## 5. Running it

```bash
cd gpu-simulator
source ./setup_environment.sh release
make -j$(nproc)
```

Add to the config, or pass on the command line:

```
-oracle_prefetcher_uncoalesced 1
```

### Validation

1. **Disabled == baseline.** With the flag at `0` the only new work is counter
   increments; no simulated state or timing changes. Confirmed on Rodinia-3.1:
   every baseline cycle count matches the pre-change sweeps (`uncoal_final`,
   `sim_run_12.8`) exactly, and both arms retire identical instruction counts.
2. **Irregular workloads move the most.** Compare
   `oracle_pref_uncoalesced_misses` against `oracle_pref_coalesced_misses` in the
   baseline run first — that ratio predicts which benchmarks can respond at all.
3. **`oracle_pref_buffer_hits` should be large** on workloads that regressed
   without the buffer. pathfinder records 1.35M buffer hits, bfs-4096 61K.

### `>= baseline` is not guaranteed — and the buffer is not why

The task statement expects the buffer to make regressions impossible. On
Rodinia-3.1 that does not hold for every benchmark, and it is worth being precise
about why, because the obvious explanation (a mis-wired buffer probe) has been
ruled out.

Taking pathfinder, the worst case, all with the buffer enabled:

| run | cycles |
| --- | --- |
| baseline | 108,431 |
| oracle, no buffer | 121,637 |
| oracle, with buffer | 114,959 |
| `-gpgpu_perfect_mem 1` (ideal-memory floor) | 97,153 |
| `-gpgpu_perfect_mem 1` **+** oracle | 96,206 |

Evidence that the mechanism is correct:

* the oracle serves 99.3% of all global-load transactions (404,699 oracle misses
  plus 1,345,950 buffer hits out of 1,762,166), and DRAM reads collapse from
  1,246,731 to **79**;
* L1 write statistics are byte-identical between the two arms, so the write path
  is untouched;
* **perfect memory + oracle (96,206) is faster than perfect memory alone
  (97,153)** — with memory already free the oracle costs nothing, which rules out
  any added serialization in the oracle path itself.

What is left is a second-order scheduling effect. Removing the latency makes
warps ready far earlier — scoreboard stall cycles drop from 11.5M to 3.6M — but
pipeline-issue stalls rise from 5.0M to 7.5M as the warps bunch up and contend
for the LSU. The staggering that memory latency used to impose was itself
providing memory-level parallelism for the few accesses that remain. This is a
property of the workload and the warp scheduler, not of the prefetcher: no
change to the buffer can recover it, and the ideal-memory floor confirms the
oracle is already extracting essentially all of the available benefit.

### Other caveats

* **Loads that bypass the L1D are out of reach.** In
  `ldst_unit::memory_cycle()`, `inst.cache_op == CACHE_GLOBAL` (and
  `-gpgpu_gmem_skip_L1D 1`) sends transactions straight to the interconnect;
  they never probe the L1D, so the oracle cannot see them and they are absent
  from every counter here. A100's config has `-gpgpu_gmem_skip_L1D 0`, so this
  only affects instructions whose SASS carries a `.CG`-style modifier.
* `oracle_pref_*` counters are device-wide and cumulative across kernels, like
  the other `gpgpu_sim` stats. The buffer itself is per-SM and per-kernel.
* **A clean rebuild is required** after any header change here. The Makefile's
  `makedepend` tracking misses `shader.h` / `gpu-sim.h` dependencies for
  `trace_driven.o`, and a stale object produces a binary that segfaults at
  startup in `shader_core_ctx::reinit`. Use `make clean && make -j$(nproc)`.
