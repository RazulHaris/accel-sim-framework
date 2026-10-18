# Snake — implementation spec (Phase 1)

Paper: Mostofi et al., "Snake: A Variable-length Chain-based Prefetching for GPUs", MICRO 2023 (`docs/snake.pdf`, kept out of git).
Bases: accel-sim-framework `3016c65`, gpgpu-sim `6c3cf4ff` (both on branch `Snake`). All `file:line` references are at those commits.
Companion docs: [SPEC](SPEC.md) · [VERSION_DIFF](VERSION_DIFF.md) · [CODE_MAP](CODE_MAP.md) · [SNAKE_PLAN](SNAKE_PLAN.md)

Paper page numbers are 728–741. Citations use §, Fig., Tab. and page (p.).

## 1. Mechanism overview
- Snake detects **chains of variable strides between consecutive load PCs (PC_ld) of the same warp**. These are "inter-thread strides", e.g. (-400, 40400, -400) across PC1→PC4 in LPS (§2 Fig. 8, p.732).
- It also detects classic **intra-warp** (same PC, same warp, next iteration) and **inter-warp** (same PC, different warps) fixed strides (§3.1, p.733).
- Three steps (§3 intro, p.732):
  1. **Detection**: Head table → Tail table.
  2. **Prefetching**: from trained Tail entries, for the current warp's next PCs (chains) and for future warps.
  3. **Throttling**: on lack of free space or bandwidth saturation.
- A trained stride is promoted to prefetch for other warps once seen in **≥3 warps** (§3 intro; §3.4).
- **Decoupling**: prefetched data lives in a flagged "prefetch" region of the unified L1/shared space, separate from "normal" L1 data (§3.2, Fig. 14, p.734).
- Inter-thread prefetches have priority over inter-warp prefetches, because they are more accurate (§3.4, p.735).

## 2. Hardware structures (per SM; §5.5 p.738–739, Tab. 3 p.739)

| Structure | Entries / org | Fields (bits) | Index / tag | Replacement |
|---|---|---|---|---|
| **Head table** | 32 rows × 14 B = 448 B (Tab. 3). N = #warps/2 (§5.5) | PC_ld, WID_a, Addr_a, WID_b, Addr_b. "Two warp IDs, two base addresses, one PC" (§5.5). Widths not given. 14 B = 112 b fits **PC 32 + 2×WID 8 + 2×Addr 32** (inferred) | "indexed by warp IDs" (§3.1), but only #warps/2 rows → **OQ-1** | n/a (overwrite) |
| **Tail table** | 10 entries × 32 B = 320 B (Tab. 3, §5.5). Fully associative: "searches 10 available PC1s … at once" (§5.5 Latency) | PC1, PC2, inter-thread stride, T1 (2 b), warpID vector (1 b/warp → 64 b on V100), intra-warp stride, T2 (2 b), inter-warp stride (§3.1 eight fields; Fig. 15). Inferred: 32 b PCs and strides → 28 B + 4 b; ~28 b unaccounted (LRU/valid?) → **OQ-2** | CAM on PC1. Several entries may share PC1, or PC1+PC2, with different strides (§3.1 cond. 3; §3.4 last ¶) | **LRU-group, then fewest 1s in warpID vector** (§3.1 p.733; Fig. 20 vs 22). LRU group size unspecified → **OQ-3** |
| **Prefetch queue** | unspecified (Fig. 15 shows ≥4) | address | FIFO (inferred) | **OQ-4** |
| **Throttle engine** | Fig. 14 | free-space flag, bandwidth meter, 50-cycle timer | — | — |
| **Prefetch space** | part of unified L1 (Fig. 14) | per-line flag prefetch/normal (§3.2) | same indexing as L1D, "each half … k-way set associative" (§3.2) | LRU + 25%-free rule (§3.2) |

T1/T2 encodings: '00' = not trained (§3.2), '10' = promoted (§3.2). Fig. 15 also shows '01' and '11', which are never defined → **OQ-5**.

## 3. Algorithms

### 3.1 Training (per global load; §3.1, Fig. 12 p.733, §3.4 p.735)
Inputs: warp w, PC p, address vector of active lanes.
0. **Warp address.** If all lanes have an equal inter-lane stride, keep a = first active lane's address. Otherwise the warp is excluded from Snake for this load (§3.4). The implementation keeps this rule as lane mode 0; its default (lane mode 1) is a deviation → **OQ-17**.
1. **Head.** Read the previous (q, b) for w. Write Head[w] ← (p, a). Send (w, q, p, s = a − b) to Tail (Fig. 12 step 1).
2. **Tail match.** Create a new entry (q, p, s) if (i) no PC1 == q (step 2), or (ii) PC1 matches but no PC2 == p (step 3), or (iii) PC1 and PC2 match but the stride differs (step 3). Then set bit w in the warpID vector (step 4).
3. **Inter-thread promotion.** When ≥3 warps hold the same (PC1, PC2, s), set T1 → '10' promoted (§3.4). Prefetching starts from this state (§3.2).
4. **Intra-warp stride** (§3.1 "Intra/Inter-warp strides"):
   - If q == p: intra = a − b.
   - Otherwise: walk the chain p → … → q through entries whose bit w is set. Then base_prev(p) = b − Σ strides, and intra = a − base_prev(p).
   - Store intra in the entry with PC1 == p, with T2 ← '00'.
   - Trained when consistent in **3 distinct warps** (§3.4).
5. **Inter-warp stride.** A fixed stride between **≥3 warps** executing the same PC (§3.1 last ¶). It has no train bits: it is written only once trained (§3.1).
   - Needs other warps' addresses for PC p. That is the purpose of the 2-warps-per-PC Head columns under GTO (§3.1 ¶1; §5.5).
   - Fig. 15a: W1..W4 → 1000..1300 at PC 520 gives inter-warp stride 100, so the stride is per warp-ID step → **OQ-6**.

### 3.2 Prediction (§3.2 p.733–734, Fig. 13, Fig. 15c)
On a load (w, p, a):
1. **Inter-thread.**
   - For Tail entries with PC1 == p, trained, and bit w set: prefetch a + s, which is warp w's address at its next PC2.
   - **Chain:** from PC2, find an entry with PC1 == PC2, bit w set and trained; prefetch a + s + s₂, and so on (Fig. 13).
   - Depth is "controlled by the throttling mechanism" (§3.2). No numeric limit is given → **OQ-7**.
2. **Inter-warp.**
   - If the inter-warp stride is set: prefetch a + k·S_iw for future warps.
   - Fig. 15c: W1 at 2500 prefetches 2600/2700/2800 for W2–W4 and 3000 for W1@PC540.
   - "All future warps" is not bounded → **OQ-8**.
3. **Intra-warp.** If T2 is trained: prefetch a + S_intra (next iteration).
4. **Rules.**
   - Inter-thread has priority over inter-warp (§3.4).
   - Issue only if the line is not already in L1 (§3.2 ¶1).
   - "When a PC_ld is encountered for the first time … issues prefetching requests for all future warps as soon as … promoted" (§3.2 ¶4).
5. **Verification during prefetching.** Compare each warp's actual PC2 and stride against the Tail entry. On mismatch, clear bit w. If **more than 2 warps** are removed, T1 → '00' and the entry returns to detection (§3.2 ¶5).
6. **Prefetch footprint.** The paper predicts only a warp base address. How it becomes memory requests is not stated → **OQ-9** (`-snake_footprint`: one line per prediction by default, or an exact lane descriptor).
7. **Latency.** 2 cycles for the detection and prefetch pipeline (§5.5 Latency). Training takes 3 cycles on average, at most 10 (§3.2 ¶7).

### 3.3 Chain length (§3.2, Fig. 13)
- Variable. A chain is extended by Tail look-ups while trained entries with bit w set exist.
- It is bounded only by Tail capacity (10 entries) and by throttling. No explicit maximum → **OQ-7**.

### 3.4 Throttling (§3.3 p.734, §5.4 Fig. 23 p.738)
- **T-a, space.** When the unified memory has **no free space**, halt prefetching for **50 cycles**. Sensitivity: 25–300 cycles (Fig. 23); 50 gives 75% accuracy with 2% coverage loss.
- **T-b, bandwidth.** When measured bandwidth reaches **70%** of theoretical peak, halt until it falls to **50%**.
  - Which link: L1↔L2 interconnect, inferred from the Fig. 4 context.
  - Measurement window and per-SM vs. global → **OQ-10**.
- Decision frequency: unspecified (OQ-10).
- Reported effect: up to 20% lower early-eviction rate (§3.3); throttling costs about 2% coverage (§5.1).

### 3.5 Memory decoupling (§3.2 ¶6–8 p.734, Fig. 14, §5.7 Fig. 25 p.739)
- After the shared-memory carve-out, the remaining unified space is split into a **prefetch space** ("upper") and **L1 data space** ("lower"). Each half is indexed like the L1D, k-way set-associative.
- Lines are distinguished by a **flag**. Both sides expand freely until the unified memory is full.
- **Until the prefetcher is trained**, L1 data may use at most **50%**.
- **While throttled**, L1 data is confined to its designated space for up to **50 cycles**.
- **Prefetch-space hit** → the line becomes L1 data by flipping the flag, with no data movement.
- **When there is no free space**, free **25%** of the unified space by LRU:
  - If >**80%** of prefetched data has been transferred to L1, evict older L1 data.
  - Otherwise, evict older prefetched entries.
  - The 25% value is tied to "L1 hit rate up to 75%" (footnote 2).
- When there is no free space, Snake keeps prefetched data until an L1 miss occurs (§3.5 tiling ¶).
- **Isolated-Snake** (§5.7) uses a separate buffer instead. Its L1 hit rate is 84%, vs. 79% for Snake and 45% for baseline (Fig. 25).
- Interaction with **MSHRs, miss queue and demand misses: not described** → **OQ-11**.
- "Free space" definition and the scope of the 25% rule (whole cache vs. per set) → **OQ-12**.

## 4. Placement (Fig. 14, §3.2)
- Snake observes **instructions at the LD/ST unit** (warp ID, PC, address) and feeds Head → Tail → Throttle engine.
- Prefetched data is written into the unified L1 (prefetch side). Requests go to L2 on L1 miss.
- The injection point is not specified. The natural point is the L1D miss path → **OQ-11**.

## 5. Evaluation setup (§4 p.736, Tab. 1, Tab. 2)
- **Simulator:** Accel-Sim v1.2.0 with AccelWattch v1.0. The **V100** model has 80 SMs, 1530 MHz, GTO, and 4 schedulers/SM.
- **Per SM:** 2048 threads, 64K registers.
- **Unified cache:** 128 KB, 256-way, 128 B lines, 4 banks, 28-cycle.
- **MSHR:** 512 entries, merge 8.
- **Other caches:** L1I 128 KB 16-way; constant 64 KB 8-way 64 B.
- **L2:** 96 KB/sub-partition, 24-way, 128 B, 64 banks, 212-cycle.
- **DRAM (ns):** tCCD=1 tRRD=3 tRCD=12 tRAS=28 tRP=12 tRC=40 tCL=12 tWL=2 tCDLR=3 tWR=10 tCCDL=2 tRTPL=3.
- **Run length:** to completion or **1 B instructions**.
- **Benchmarks (Tab. 2), 11 total; inputs not given → OQ-13:**
  - ISPASS: CP, LPS, LIB, MUM.
  - Rodinia: backprop, hotspot, srad, lud, nw.
  - Parboil: histo, mri-q.
- **Comparison points (§4):** INTRA, INTER, MTA, CTA-Aware, Tree (64 KB chunks), s-Snake (chains only), Snake-DT (no decoupling or throttling), Snake-T (decoupling, no throttling), Snake+CTA.
- **Metrics (§4):**
  - IPC and energy.
  - **Coverage** = correctly predicted addresses / total demand addresses.
  - **"Accuracy"** = *timely* correctly predicted addresses / total demand addresses. This is timely coverage, not the usual useful/issued → **OQ-14**.
- **Headline numbers** (values read off bar charts, ±3 pp):

| Bench | Coverage (Fig.16) | Accuracy (Fig.17) | IPC gain (Fig.18) | Energy norm. (Fig.19) |
|---|---|---|---|---|
| CP | ~62% | ~57% | ~1% | ~1.00 |
| LPS | ~100% | ~98% | ~6% | ~0.90 |
| LIB | ~93% | ~97% | **60%** | ~0.36 |
| MUM | ~96% | ~96% | ~11% | ~0.90 |
| Backprop | ~83% | ~68% | ~1% | ~0.92 |
| Hotspot | ~88% | ~85% | ~13% | ~0.86 |
| Srad | ~87% | ~83% | **29%** | ~0.71 |
| lud | ~91% | ~82% | ~18% | ~0.82 |
| nw | ~12% | ~10% | ~5% | ~0.95 |
| Histo | ~97% | ~96% | **33%** | ~0.64 |
| mri-q | ~72% | ~60% | ~4% | ~0.96 |
| **Avg / GMEAN** | **80%** | **75%** | **17%** | **0.82 (−17%)** |

- **Memory-path check (2026-10-14):** our config matches Tab. 1 except `-gpgpu_l1_latency` 20 (28) and `-gpgpu_l2_rop_latency` 160 (212). The baselines reproduce the paper's "miss-queue congestion dominates reservation fails" (§2: 71–100% MISS_QUEUE_FULL), at lower fail rates and interconnect utilisation than Figs. 3–4 (results/missq_config_check.md, VERSION_DIFF).
- **Other reference points:**
  - Baseline L1 hit rate 45% → Snake 79% → Isolated-Snake 84% (§5.7).
  - Snake beats Snake-DT by 13% and Snake-T by 7% IPC (§5.2).
  - Without decoupling, accuracy drops 50% (§5.1).
  - Tail = 10 entries costs 8% coverage vs. unlimited (Fig. 20).
  - Motivation numbers: reservation fails 30% of L1D accesses (Fig. 3); L1↔L2 bandwidth utilisation 33% (Fig. 4); memory stalls 55% of stalls (Fig. 5).

## 6. Hardware cost (§5.5, Tab. 3, Fig. 21)
- **Storage:** Head 448 B + Tail 320 B = **768 B/SM**. Fig. 21 compares MTA ~550 B, CTA ~700 B, Snake(10) ~730 B, Snake(20) ~1000 B, Snake(40) ~1550 B.
- **Area:** <1% of the V100 die (815 mm²); CACTI 7, 22 nm scaled to 12 nm.
- **Power:** 6.4 pJ/access and 6 mW static (DC, NanGate 28 nm → 12 nm); <1% overhead.
- **Latency:** 2 cycles.

## 7. Open Questions
Format: interpretation (confidence).
- **OQ-1 Head table organisation.**
  - Model it logically as (a) a per-warp register {last PC, last addr} for inter-thread and intra-warp detection, and (b) per-PC "last two warps" {(wid, addr)×2} for inter-warp detection under GTO.
  - Expose `snake_head_entries` = 32 as the capacity of (b). (low)
- **OQ-2 Field widths.** Use 64-bit addresses and strides in the simulator, since Accel-Sim addresses are 64-bit. Report the paper's 32-bit storage figure unchanged. (medium)
- **OQ-3 Tail eviction.** "LRU group" = the `snake_tail_lru_group` (default 3) least-recently-used entries; evict the one with the fewest set bits. Paper silent on group size. (low)
  - **"Unlimited" Tail (Fig. 20; S5, 2026-10-15):** `-snake_tail_entries 1024` (config `SNAKE_TAILINF`). The Tail is cleared at every kernel switch (OQ-19). The most Tail allocations per SM-launch in our 100M runs average 449 (backprop; ≈ 620 over its full 137M instructions). So 1,024 entries rarely evict, without changing the code. (decided)
- **OQ-4 Prefetch queue.** **Revised 2026-10-13 (S3a): one entry per 128 B line**, the paper's prefetch unit (Fig. 15 lists one address per warp).
  - `snake_queue_size` = 32 line entries, FIFO.
    - A predicted line already queued absorbs the new sectors; sectors already pending count as `pf_dup_queue`.
    - A new line with the queue full is dropped, newest first; its sectors count in `pf_drop_queue_full`.
  - Each entry becomes ready `snake_latency` (2) cycles after its prediction.
  - `snake_issue_width` = 1 line per cycle. When it issues, the line's sectors not yet offered go to the L1D, one per L1D bank and only to banks no demand access used in that cycle, and only while the data port is free. A sector whose bank is busy waits; each sector is offered once.
  - A prefetch line therefore never gets more L1D or miss-queue bandwidth per cycle than a demand access to the same line: ≤ 4 sector accesses across the 4 banks, each subject to the demand's miss-queue test. Demand still goes first, since issue runs after the demand stage.
  - **Why per line.** With one sector per entry, the S3a test trace (d) overflowed the queue (5,695 sectors dropped) and reached only 82.8% coverage_paper. With one line per entry it reaches 93.0%.
  - **S3b:** a full queue and the issue order now favour own-warp lines over inter-warp lines (OQ-31).
  - Queue size and issue width are S5 sweep parameters. (decided)
- **OQ-5 T1/T2 encoding.** 00 = untrained → 01 = observed (≥1 warp) → 10 = promoted (≥3 warps, prefetch for others) → 11 = trained (repetition confirmed; also allows chaining). Text and figure partly conflict. (low)
  - **Correction (S2):** Fig. 15 does not use these codes. It prints the promoted entry of phase (b) as '01', the trained one of (c) as '11', and a one-warp entry and a new intra stride as '00'. The *states* match the order above; only the printed codes differ, so the codes are treated as internal labels (SNAKE_PLAN S2 notes).
- **OQ-6 Inter-warp stride.** Normalised per warp-ID distance: (a_j − a_i)/(w_j − w_i), equal across ≥3 warps, as in Fig. 15a. (medium)
- **OQ-7 Chain depth.** Config `snake_max_chain_depth`, default 3. Paper gives no number; Fig. 13 shows depth 2. Throttling can reduce it to 0. (low)
  - **S4:** §3.2 says the chain length is "controlled by the throttling mechanism", without a value. While throttled (OQ-10), the inter-thread chain stops at `-snake_throttle_chain_depth` (default 1): the warp's next load is still predicted, the deeper steps are not. Chain steps cut this way are counted in `snake_throttle_chain_cut`. (low)
- **OQ-8 Inter-warp targets.** Prefetch for the next `snake_interwarp_degree` (default 4) warp IDs above w that are resident on the SM and in the same CTA. "All future warps" is unbounded. (low)
  - **S6 experiment, `-snake_interwarp_scope`** (default 0 = this reading): 1 = the next warps by global warp id among all resident warps of the kernel; 2 = the same warp index in the next resident CTAs, i.e. "all future warps" across CTAs. Both assume addresses linear in the global warp id (1D grids). Results: results/S6_ipc.md §3; kept off by default.
  - **Implemented (S3b):** the targets are the next 4 warp slots above w that were initialised in the same HW CTA slot and have not exited (`shader_core_ctx::warp_active`); exited warps are skipped, not counted. Each target's address is a + (w₂ − w)·S_iw, with the triggering load's lane descriptor (OQ-9).
- **OQ-9 Prefetch footprint.** **Revised 2026-10-13 (S2c):** selected by `-snake_footprint`. In S2c the option and the storage exist; requests are issued from S3 on.
  - **0 = line (default).** One 128 B line per predicted warp address: the line holding the address, requesting only the sectors that are missing in L1. Nothing extra is stored. Storage stays the paper's **768 B** per SM (Head 448 B + Tail 320 B, Tab. 3).
  - **1 = exact.** Each Tail entry keeps the lane descriptor of its PC2 load, from the last warp that took the edge:
    - run length L (2 b), first active lane (5 b), lane stride (24 b) and run stride (32 b), i.e. 63 bits ≈ **8 B per Tail entry**, **+80 B** for 10 entries (848 B per SM). The simulator keeps 64-bit fields.
    - Inter-thread predictions target PC2, so they replay the PC2 descriptor of the entry that produced them.
    - Inter-warp and intra-warp predictions target the triggering load's own PC, so they use that load's descriptor and need no storage.
    - The active mask is not stored, so S3 replays all 32 lanes of the descriptor, anchored so that the predicted address is the first active lane's.
  - **S3a:** both footprints group their sectors by 128 B line into queue entries (OQ-4). Line mode requests all sectors of the line, and those already in L1 or in flight are dropped at issue (`pf_dup_l1`, `pf_dup_mshr`). Without a descriptor, exact mode falls back to the line.
  - (decided; replaces the earlier "triggering access's lane stride and active-lane count" expansion)
- **OQ-10 Bandwidth throttle.**
  - Per-SM injected L1→icnt bytes per window, divided by the per-SM share of peak L1↔L2 bandwidth.
  - Window `snake_bw_window` = 100 cycles; hysteresis 70%/50%.
  - Evaluated at the end of each window. (low)
  - **Implemented (S4, 2026-10-15): request slots, not bytes.** The paper says "bandwidth". In our model the L1↔L2 limit an SM sees is the L1D miss-queue drain: at most one request per cycle, and the interconnect never refused one (S3c, results/S3c_diag.md §2). So the default meter (`-snake_bw_mode 0`) is the request-slot utilisation of that drain: requests sent from the SM's L1D miss queue (demand, prefetch and writes) per cycle, against a peak of 1.
    - Per SM, over windows of `snake_bw_window` (100) cycles, evaluated at the first Snake tick after each window ends.
    - Prefetch issue halts at ≥ `snake_bw_high` (0.70) and resumes at ≤ `snake_bw_low` (0.50).
    - `-snake_bw_mode 1` is the bytes variant: the data bytes of the requests sent (32 B per sector read, the write size for writes) against `-snake_bw_peak_bytes` (32) per cycle.
    - Stats: `snake_throttle_bw_events`, `snake_throttled_cycles_bw`, `snake_bw_util_hist_<0..100>` (windows by utilisation, in tenths).
  - **Space trigger (§3.3).** "When the unified memory has no free space", issue halts for `snake_throttle_space_cycles` (50); a new event restarts the 50 cycles. No free space is: mode 1, an allocation whose L1D set has no free line (the bulk-free event, OQ-12); mode 2, the isolated buffer full at a prefetch fill (OQ-22); mode 0, a full set at any L1D allocation.
  - **While throttled:** nothing is issued; the queue keeps its lines and new predictions still enter it. Cycles in which a ready line was held back are counted in `snake_pf_held_throttled` (the S3 placeholder `pf_drop_throttled` is renamed, since nothing is dropped). The chain is capped (OQ-7) and L1 data is confined (OQ-15). `snake_throttled_cycles` / `snake_hook_tick` is the throttled share of an SM's cycles; `_space` and `_bw` split it by trigger (a cycle with both counts in both). (decided)
- **OQ-11 MSHR and miss queue.**
  - A prefetch sector allocates its line or sector, an L1 MSHR entry and a miss-queue slot through `send_read_request`, like a demand read miss, as an `L1_PREFETCH_R` read with no instruction. Demand misses can merge into it ("late prefetch").
  - **Implemented (S3a, `l1_cache::snake_prefetch`):** a sector is offered once and never retried.
    - Already valid or reserved in L1 → `pf_dup_l1`; an MSHR entry exists → `pf_dup_mshr`.
    - No line can be reserved in the set → `pf_drop_line`.
    - Free MSHR entries ≤ `snake_mshr_reserve` (8) → `pf_drop_mshr`.
    - Miss queue full by the demand read-miss test (`miss_queue_full(1)`), or fewer than `snake_missq_reserve` slots would stay free for demand after the prefetch → `pf_drop_miss_queue`.
    - **`snake_missq_reserve` = 8 (decided 2026-10-13).** With the miss-queue test alone, lud (input 512, line footprint) dropped more prefetches for miss-queue slots than it issued (212,435 vs 176,706). It also more than doubled demand MISS_QUEUE_FULL (42,855 → 95,207), so demand no longer always won (SNAKE_PLAN §5).
      - The default is the smallest N in {2, 4, 8} that keeps lud's demand MISS_QUEUE_FULL within +10% of the baseline on both inputs (line footprint).
      - N = 2 gives +97% and N = 4 gives +50% on input 512; N = 8 gives +4.9% (512) and −9% (256), with `pf_caused_resfail` = 0.
      - The S5 sweep varies it.
    - `pf_drop_resource` is the sum of the last three.
    - **S3c:** `pf_drop_mq_reserve` counts the miss-queue drops in which a demand read miss would still have fit, i.e. only the reserve refused the prefetch (≈ 98% of `pf_drop_miss_queue` in the S3c runs).
  - **Miss-queue size (2026-10-14).** The paper gives no miss-queue size or drain rate; Accel-Sim's 16 is kept (the same in dev, release and our base; results/missq_config_check.md).
  - **Miss-queue drain (S3c, from the code).** The L1D miss queue (16 entries) sends at most one request per core cycle, when `icnt_has_buffer` holds. With `-network_mode 2` that is the SM's request-network input buffer below 512 packets, whatever the packet size. In every S3c run the buffer never refused, so the drain rate of one request per cycle is the limit (results/S3c_diag.md §2).
  - **`pf_caused_resfail`:** a demand reservation failure counts as caused by Snake when the MSHR table was full while prefetches held entries, or the miss queue was full while it held prefetches. Failures for lack of a free line in the set are not attributed.
  - Demand always wins arbitration (OQ-4). (medium)
- **OQ-12 Decoupling semantics.**
  - Implement in the L1D tag array: a per-line `prefetched` flag plus a per-set quota-aware victim choice.
  - "No free space" = the fill's set has no invalid, unreserved line.
  - "Free 25%" = evict ⌈25%⌉ of that set's evictable lines (LRU), from the side chosen by the >80%-transferred rule. That rule uses a per-SM counter, reset each window.
  - Also offer `snake_decouple_mode`: 0 = none (Snake-DT), 1 = unified flag partition (paper), 2 = isolated buffer (§5.7). (low)
  - **Implemented (S4, 2026-10-15), mode 1** (`tag_array::snake_victim`, `snake_prefetcher::on_allocate` / `on_tag_hit`):
    - **Region per line.** Every L1D line has one flag: L1 region or prefetch region. The L1D allocates on a miss, so the flag is set when the line is reserved: by a prefetch → prefetch region, by any other access → L1 region.
    - **Flip.** A demand access (any non-prefetch access: a hit, a pending hit or a sector miss) to a prefetch-region line moves the whole line to the L1 region, with no data movement (§3.2). A sector miss flips too, because demand data then lives in the line (`snake_region_flips_sector_miss`).
    - **Sectors (decided 2026-10-15: follow the paper, one flag per entry).** A flip moves the line's other prefetched sectors to the L1 region as well; they lose the region's protection while still unused. They are counted in `snake_flip_unused_sectors`; prefetched sectors filled into a line that is already in the L1 region (a sector added to an L1 line, or a line flipped while the prefetch was in flight) in `snake_pf_fill_l1_region`.
    - **Victims.** A demand allocation never evicts a prefetch-region line; prefetch-region lines leave only by a bulk free (or an invalidation). Both regions grow into free lines.
    - **No free space** = the allocation's set has no free line. Then the space trigger fires (OQ-10) and a **bulk free** evicts ⌈`snake_free_frac` (25%) × the set's evictable lines⌉ by LRU, all from one side:
      - the L1 side if more than `snake_transfer_frac` (80%) of the prefetched lines moved to the L1 region, otherwise the prefetch side;
      - the transfer share is (flips) / (prefetch-region allocations) counted since the previous bulk free on that SM: the "window" of the earlier text is the interval between bulk frees;
      - if the chosen side has no evictable line, the other side is used;
      - evictable is as for the replacement policy (no reserved line); a modified line is freed only in a write-through cache, as the L1D is, since it needs no writeback.
      The allocation then takes a freed line.
    - **L1 cap** (OQ-15 and the throttle confinement): a demand allocation in a set whose L1-region lines reach ⌊`snake_untrained_l1_frac` × assoc⌋ evicts the set's LRU evictable L1-region line, even if free lines exist; it fires no bulk free. If no L1-region line is evictable, the normal path is taken, so the cap never causes a reservation failure.
    - **Debug asserts** (always compiled in): the victim is never reserved; a demand victim is never a valid prefetch-region line; a capped victim is an L1-region line. Each check is counted in `snake_quota_checks`. `snake_decouple_fallback` counts a bulk free that could free nothing (only possible in a write-back cache).
    - The set is the unit of every rule; the QV100 L1D has 4 sets of up to 256 ways, so a set is a quarter of the unified space.
  - Mode 2: SNAKE_PLAN §12 OQ-22. (decided)
  - **Revised (S4b, 2026-10-15): `-snake_decouple_variant 1`, the new default; 0 keeps the S4 reading above.** S4's mode 1 contradicted §5.7, where decoupled Snake's L1 hit rate is within ~5 points of Isolated-Snake and decoupling raises the hit rate (results/S4.md §6). Two rules change:
    - **Bulk free, "older prefetched data entries".** §3.2 says "typically, the data is consumed within 50 cycles" (footnote 3), so older means unused for more than `-snake_pf_stale_cycles` (50). A prefetch-region line is stale when it has had no activity (allocation, prefetch fill, demand access) for more than 50 cycles. When the 80% rule picks the prefetch side, only stale prefetched lines are freed, by LRU of that activity. If fewer than 25% are stale, the rest comes from the L1 side, by LRU. When it picks the L1 side, L1 lines are freed as before. Fresh prefetched lines are freed only if nothing else can be (`snake_bulk_free_lines_pf_fresh`).
    - **Sector-aware flip.** The paper's per-entry flag ignores sectors. A line now moves to the L1 region only when none of its prefetched sectors is still fresh, i.e. each has been demanded, or was filled more than 50 cycles ago, and none is still in flight. Until then the line stays in the prefetch region, so its unused prefetched sectors stay protected (`snake_flips_deferred`). A deferred line flips at a later demand access that finds no fresh sector; until then it leaves only by a bulk free (once stale) or an invalidation.
    - Counters: bulk-free victims by side (`snake_bulk_free_lines_l1`, `_pf_stale`, `_pf_fresh`), unused prefetched sectors evicted from prefetch-region lines (`snake_evict_pf_protected`) or from L1-region lines (`snake_evict_pf_unprotected`). (decided)
    - **Result (S4b):** the sector-aware flip works: unused sectors evicted from L1-region lines fall to ≤ 253 per run. But the 50-cycle staleness frees prefetched lines that are used later: hotspot's timely inter-warp prefetches are used a p50 of 897 cycles after issue. The §5.7 target is missed on hotspot and lud (§8).
- **OQ-13 Inputs and traces.**
  - Use Accel-Sim's default `define-all-apps.yml` arguments.
  - Rodinia version: paper unstated. Local traces exist for both 2.0-ft (srad_v2) and 3.1 (srad_v1).
  - **ISPASS CP/LPS/LIB/MUM traces are not available locally.** They're also absent from the public V100 1.1.0 trace set. CP is commented out ("compile issues"), and only LPS is built in gpu-app-collection. The local GPU is an RTX 2080 Ti (SM75), not a V100.
  - **Decided (2026-10-01):** ISPASS is deferred. The first evaluation uses 7 apps: rodinia-3.1 backprop / hotspot / srad_v1 / lud / nw and parboil histo / mri-q. See VERSION_DIFF for the effect on the average.
- **OQ-14 Metric definitions.** **Revised 2026-10-13 (S3a).** All metrics are reported, all at the granularity of L1D sector accesses. Demand = `GLOBAL_ACC_R` accesses that did not fail with RESERVATION_FAIL (`snake_demand_sector_accesses`).
  - **coverage_pred** = demand sectors that a prediction generated earlier named, whether issued or not, within `snake_pred_window` (5000) cycles / demand sectors. Each prediction matches one demand access.
  - **coverage_paper** = (timely + late) / demand sectors.
  - **accuracy_paper** = timely / demand sectors.
  - **accuracy_std** = (timely + late) / issued.
  - **The paper's terms.** Its "coverage" ("correctly predicted addresses / total demand addresses", §4) most likely means **coverage_pred**: predictions, independent of issue. Its "accuracy" ("timely correctly predicted addresses") means timely prefetches, i.e. **accuracy_paper**. Both readings are kept: coverage_paper is what the issued prefetches achieved, coverage_pred what the predictor named.
  - **Outcomes of an issued sector** (each ends in exactly one; the invariant is printed as `snake_pf_invariant`):
    - **late:** a demand merged into its MSHR entry before the fill.
    - **timely:** the first demand access (load or store) hit the filled sector.
    - **evicted_unused:** evicted, or found stale at a later fill, with its bit still set.
    - **unused_at_end:** dropped by an L1D invalidation between kernels, still resident and unused, or still in flight without a merge at the end of the run.
  - **Prediction funnel (S3c, diagnosis only).** Each prediction record (one per predicted sector in the window, `snake_pred_sectors`) ends in exactly one class: matched by a demand (still queued, dropped queue-full / miss-queue / line-MSHR, duplicate L1 / MSHR, issued late / evicted / timely / other), wrong (no demand within the window; issued or not) or live at the end (`snake_funnel_invariant`).
    - A record takes the mode of its latest prediction and the result of its latest attempt, except that once issued it stays issued and its outcome is that of the issued sector.
    - Lead times (`snake_dist_lead_issue_*`) use the mode of the queue entry that issued the sector.
    - **S3d:** `snake_funnel_src_*` is the same funnel by the source of the record's **first** prediction. It is used for the per-source attribution in results/S3d.md.
    - Definitions and results: results/S3c_diag.md §1.
  - (decided)
- **OQ-15 50%-until-trained.** Applies per kernel launch, from launch until the first Tail entry reaches promoted. (low)
  - **Implemented (S4):** per SM. The flag is cleared when the SM starts a different kernel (OQ-19) and set at the SM's first T1 promotion. Until then, L1 data is capped at 50% of each set (OQ-12, "L1 cap").
  - **Confinement while throttled (§3.2):** the same cap applies from the start of each throttle episode for at most `snake_throttle_confine_cycles` (50) cycles. Cycles under each cap: `snake_l1_cap_cycles_untrained` and `_confine`. (decided)
  - **Revised (S4b, 2026-10-15), `-snake_decouple_variant 1`:** the untrained cap lasts at most `-snake_untrained_cycles` (10) Snake cycles of each kernel launch on an SM, or until the first promotion if that is earlier.
    - **Why.** §3.2 gives training "an average of 3 cycles (up to 10 cycles)", and §3.5 "the designated 10-cycle period, which represents the maximum required time for training Snake". A pattern confirmed across 3 warps cannot take 10 cycles, so these must be the table-update pipeline time. Tying "trained" to the first promotion, as variant 0 does, keeps the cap on for hundreds of cycles or more, and again at every launch (step-1 data: results/S4.md, S4b).
    - **Step-1 data (S4b, variant 0; results/S4.md S4b.1).** SM-launches that promote take a p50 of 57–655 Snake cycles to do so (max 941). Most cap time is in SM-launches that never promote: lud 93%, histo 94%, srad_v1 79% of their cap cycles. The cap was active 33% (lud), 53% (histo) and 16% (srad_v1) of Snake cycles. It is re-armed only when an SM starts a launch (0 switches back to a launch, never on demotion). Variant 1 cuts it to 0.1–0.3%.
    - Diagnosis counters (both variants): `snake_dist_launch_to_promotion` (Snake cycles from an SM's launch start to its first promotion), `snake_sm_launches_no_promo`, `snake_untrained_cycles_promoted` / `_never` (untrained cycles of launches that do / do not promote), `snake_kernel_switch_revisit` (an SM switching back to a launch it ran before, which would re-arm the cap within a launch). (decided)
- **OQ-16 Energy.** AccelWattch is not modified. Add Snake energy offline: accesses × 6.4 pJ + 6 mW × time, from Snake stats. (medium)
  - **Implemented (S5, `util/snake/collect_results.py`):**
    - Snake energy = accesses × `-snake_energy_pj_per_access` (6.4 pJ) + `-snake_static_mw` (6 mW) × SMs (80) × simulated time (`gpu_tot_sim_cycle` / core clock).
    - Accesses = Head updates + Tail look-ups + prediction steps (`snake_head_updates` + `snake_tail_lookups` + `snake_pred_*`), one table access each.
    - Normalized energy (Fig. 19) = (GPU energy with Snake + Snake energy) / baseline GPU energy. GPU energy = AccelWattch's last `gpu_tot_avg_power` (`accelwattch_power_report.log`) × simulated time.
    - **Unavailable in our configs:** the QV100 config runs with `-power_simulation_enabled 0`, so no AccelWattch output exists. `compare_paper.py` then reports Snake's own energy and "unavailable" for Fig. 19. Turning AccelWattch on for the final runs would be a new config token (not in Tier 1 or 2). (decided)
- **OQ-17 Which warps Snake uses (lane check).** **Decided 2026-10-13: a deliberate deviation from §3.4, selected by `-snake_lane_mode`.**
  - **Why.** §3.4 says the inter-lane strides of a warp are "consistently equal" and excludes warps where they are not. In the local Rodinia traces this is false for 2-D thread blocks. With 16×16 blocks each warp covers two 16-lane rows, e.g. backprop PC 0x180 has lane deltas 4 ×15, 8, 4 ×15. The paper-literal rule excluded 100% of hotspot's and 99.1% of backprop's eligible loads in the S2 gate, so neither trains, yet the paper reports ~88% and ~83% coverage for them (Fig. 16).
  - **Lane mode 0 (paper-literal, §3.4).** Lanes = active lanes only. A load is used only if every active lane is the same distance per lane index from its predecessor; a single active lane has stride 0. `-snake_require_equal_lane_stride 0` disables the check (mode 0 only). The footprint follows OQ-9.
  - **Lane mode 1 (stateless run filter, default; revised 2026-10-13, S2c).**
    - A warp is used (trained on, and a trigger for predictions) if its active-lane addresses fit base + (i mod L)·lane_stride + ⌊i / L⌋·run_stride, where i is the lane index, for some run length L ∈ {32, 16, 8, 4}, tried in that order. L = 32 is the §3.4 rule; L = 16 covers two 16-lane rows of a 16×16 block (e.g. backprop PC 0x180: lane deltas 4 ×15, 8, 4 ×15).
    - The lane stride comes from the first run with two active lanes, or 0 if none. The run stride comes from the first active lane in a run other than the first active lane's, or 0 if none. Every active lane is then checked exactly.
    - Otherwise the load is excluded and counted in `warp_excluded_lane_runs`, and the warp's Head register is invalidated, as in mode 0. Accepted run lengths are counted in `lane_runs_{32,16,8,4}`.
    - Training uses the first active lane's address as the warp address, as in mode 0. The fitted (L, first lane, lane stride, run stride) is the load's lane descriptor (OQ-9).
  - **Storage.** The filter is stateless: nothing is stored per PC. The per-PC 32-offset pattern of S2b (≈ 4 KB per SM) was removed in S2c. Storage is the paper's 768 B per SM, plus ≈ 8 B per Tail entry only with `-snake_footprint 1` (OQ-9).
  - (decided)
- **OQ-29 Combined lookahead for inter-warp targets (S3b, decided 2026-10-14).** §3.2 "prefetch for all future warps" is read as: an inter-warp target k also gets the inter-thread chain that follows its predicted PC1 address.
  - From a + k·S_iw, up to `-snake_interwarp_chain_depth` (default 3) steps are followed, each through a Tail entry for the current PC with a PC2 and T1 ≥ promoted (promotion is what lets a stride serve warps other than those that trained it, §3.2). The target's warpID bit is not required; an entry that holds it is preferred, otherwise the most recently used. Each step's address uses the entry's PC2 lane descriptor (OQ-9).
  - Order of one load's predictions (§3.4: inter-thread before inter-warp): promotion-time predictions, the warp's own chain, all inter-warp target addresses, then each target's chain, then intra-warp.
  - **Fig. 15 (c).** The figure's queue holds 2600, 2700, 2800 and 3000. Its 520→540 entry is trained with all four warp bits, so with the chain W2..W4 also get 3100, 3200 and 3300. `-snake_interwarp_chain_depth 0` (config `SNAKE_IWC0`) is the paper-literal behaviour and reproduces the figure exactly; the synthetic gate checks both.
  - **S3e (2026-10-14): the default is 0, the paper's.** Our extension was net-negative in S3d: IPC was lower with depth 3 than with 0 on all five apps tested, and on lud it added 280k predicted sectors, 83% of them wrong (results/S3d.md). The option stays (config `SNAKE_IWC3`). The synthetic trace (d) ≥ 90% gate runs with `SNAKE_IWC3`; the default gives 86.0%, reported only.
  - (decided)
- **OQ-30 Tail eviction weight for an inter-warp stride (S3b, decided 2026-10-14).** The eviction score is popcount(warpID vector) + `-snake_interwarp_weight` (default 3) if the entry holds a trained inter-warp stride; the lowest score in the LRU group (OQ-3) is evicted, ties to the older. `popcount` uses the same score; `lru` ignores it.
  - **Why.** Training an inter-warp stride already needed 3 agreeing warps (§3.1), but an inter-warp-only entry (Fig. 15a) has no warpID bits, so the plain popcount rule evicted it first (S3a: backprop evicted 32,137 such entries with 0.9 bits on average; SPEC §8).
  - Weight 0 (config `SNAKE_IWW0`) is the paper's rule.
  - (decided)
- **OQ-31 Inter-thread priority in the prefetch queue (S3b, decided 2026-10-14).** §3.4 gives inter-thread strides priority over inter-warp ones. The order inside one load (OQ-29) was not enough: in synthetic trace (d) the queued inter-warp lines pushed out and delayed later own-warp lines, and coverage_paper fell from 93.0% (S3a) to 88.8%.
  - **Classes.** Own-warp lines (inter-thread chain, promotion, intra-warp) come before inter-warp lines (inter-warp targets and their chains). An own-warp prediction that merges into a queued inter-warp line raises that line to the own-warp class.
  - **Displacement.** With the queue full, a new own-warp line displaces the newest queued inter-warp line not yet being offered. Its sectors count in `pf_drop_queue_full` and `pf_queue_displaced`. A new inter-warp line with the queue full is dropped (newest dropped, as before).
  - **Issue.** Each issue slot takes the line already partly offered to the L1D, else the oldest ready own-warp line, else the oldest ready inter-warp line. Within a class the queue stays FIFO.
  - With displacement alone, trace (d) reached 89.3%; with displacement and issue priority, 90.8%.
  - (decided)
- **OQ-32 Promoted inter-thread strides serve all warps (S3d, decided 2026-10-14).**
  - **Paper basis.**
    - §3 intro (p.732): Snake "promotes the discovered chains of strides to prefetch for other warps if they are detected at least for three warps".
    - §3.2 (p.733, SIMT paragraph): "if a consistent stride is detected across multiple threads, it is highly probable that all threads observe the same stride … Snake confirms the stride's consistency across multiple warps before promoting it to issue prefetch requests for future warps".
    - §3.2 (p.734, verification): "each warp's PC2 and Inter-thread stride are compared with their corresponding values in the Tail table. If they do not match, that particular warp is removed from the warpID vector. If more than two warps are removed … 'not-trained'".
  - **Implementation** (`-snake_promoted_all_warps 1`, default):
    - A load at PC1 by a warp without the entry's warpID bit also gets the **first** inter-thread step from a promoted entry. The entry with the warp's bit is preferred, otherwise the most recently used one.
    - **S3e:** every deeper step needs the warp's bit set and the entry trained. §3.2 (Fig. 13 paragraph, p.734): Snake "revisits the Tail table to locate an entry that possesses the same PC1 value as the current entry's PC2 value, while the corresponding bit in warpID vector is set and the status is trained". In S3d the bit requirement was lifted at every step; S3e changes the results only within noise (results/S3e.md).
    - The warp's next load verifies the first step. A match sets its bit through training (Fig. 12 step 4). A mismatch counts as a removed warp of that entry (`warps_removed`, `nobit_verify_fail`), with demotion after more than `-snake_demote_removed` removals (OQ-28).
    - A Head invalidated in between (OQ-17, OQ-20) leaves the step unverified (`nobit_verify_lost`).
    - The one-shot promotion-time predictions (OQ-27) are unchanged.
  - `0` is the S3c behaviour: only warps holding their bit, plus OQ-27, got inter-thread predictions. S3c measured 15–57% of loads at a promoted PC1 without the bit (results/S3c_diag.md §3).
  - Results: results/S3d.md.
  - (decided)
- **OQ-18 Scheduler.** Paper uses GTO. The base QV100 config says `-gpgpu_scheduler lrr` (`configs/tested-cfgs/SM7_QV100/gpgpusim.config:134`, despite the "Greedy then oldest" comment at :133). Run both baseline and Snake with GTO via a `GTO` extra-param config. (high)

OQ-19…OQ-28 (simulator-specific questions, the decisions on clock, Rodinia version and competitors, and the S2 training interpretations) are in [SNAKE_PLAN §12](SNAKE_PLAN.md). OQ-29…OQ-31 (S3b) are above.

## 8. Design limitations observed in our traces
### Discrepancies with the paper (S3e, exact footprint, 100M; paper values from §5 above)

| app | paper coverage / accuracy / IPC | ours coverage_pred / coverage_paper / accuracy_paper / IPC | likely cause |
|---|---|---|---|
| lud | ~91% / ~82% / +18% | 83.6% / 45.7% / 3.3% / −0.2% | Predictions are plentiful but half are wrong: 53% of the wrong ones come from promotion, i.e. inter-thread strides shared by 3 warps but not by the others (59% of no-bit verifications fail, ~6,900 demotions). Correct ones arrive late (inter-thread ~20 cycles ahead, S3c). |
| histo | ~97% / ~96% / +33% | 30.9% / 25.0% / 8.8% / −1.6% | The prediction gap is real: 37% of predicted sectors match no demand, and inter-warp gives 96% of the wrong ones. The data-dependent histogram accesses have no fixed strides for most loads (33% of loads get no inter-thread prediction), and the baseline L1 hit rate is 63%. |
| mri-q | ~72% / ~60% / +4% | 90.8% / 9.5% / 0.0% / +0.1% | Predicted, but not issued in time: 54–82% of each source's predictions are still queued or dropped when the demand comes. Demand keeps the 16-entry miss queue near full (drained 1 per cycle), so the 8-slot reserve blocks prefetches (S3c, results/missq_config_check.md). |

- **Common factors** (VERSION_DIFF):
  - Our GPU hides less latency: the L2 hit is 173 cycles vs the paper's 212, and the reservation-fail rates and L1↔L2 traffic are lower than in Figs. 3–4.
  - Decoupling and throttling are still off (S4).
  - We run 100M instructions instead of 1B, and the inputs may differ (OQ-13).
- **Intra-warp barely fires** in all five apps: 0–45 intra-warp predictions and 0–181 trained T2 strides per run, against 5k–155k observed loads (results/S3e.md).
  - This is consistent with the paper's §2: "memory-bound GPGPU applications replace deep loops with parallel programming", hence "using intra-warp stride opportunities is unlikely".

### Remaining gap after S4b (decoupling mode 1, variant 1; results/S4.md S4b)
- **Target (§5.7):** decoupled Snake's demand L1 hit rate within 5 points of Isolated-Snake and no more than 2% below Snake-DT. Variant 1 misses it on two apps:
  - **hotspot**: 26.6% vs Snake-ISO 32.5% and Snake-DT 32.3% (−6.0 pts, −17.7%)
  - **lud**: 8.2% vs Snake-DT 8.4% (−3.0%)
  - It holds on histo, mri-q and srad_v1.
- **hotspot.** Its demand hits are timely prefetches, used a p50 of 897 cycles (p90 1,699) after issue. The 50-cycle staleness of §3.2 ("typically, the data is consumed within 50 cycles") marks them stale long before use. 9.7k stale prefetched lines are freed with 15.6k unused sectors. The paper's GPU likely consumed prefetches sooner. In ours, inter-warp lookahead runs far ahead of demand (S3c), and the 128 KB L1D held hotspot's prefetches without any decoupling.
- **lud.** The gap is throttling, not decoupling: Snake-T v1 is within 1% of Snake-DT. Throttling holds issue for 19% of cycles and cuts issued prefetches by 30%.
- Per the S4b decision, variant 1 is kept and no further revision was made.
- **Paper-vs-simulator timing discrepancy (recorded in S5).** In hotspot, useful prefetches are used a median of **897 cycles** after issue (p90 1,699; `snake_dist_lead_issue_inter_warp_timely`, S4b). The paper says prefetched data is "typically consumed within 50 cycles" (§3.2, footnote 3). The 50-cycle stale rule (OQ-12, variant 1) therefore evicts good prefetches in our simulator. This is a timing discrepancy between the paper and our simulator, not a fault in the rule. Tier 2 measures `-snake_pf_stale_cycles` 200 and 1000.
- **Energy (Fig. 19) stays unavailable (S5 pre-flight, 2026-10-15).** The base repo has AccelWattch for QV100 SASS (`Accelwattch_SASS_SIM`, `accelwattch_sass_sim.xml`). With it on, hotspot 512 (baseline, run side by side with the same job with power off) takes 66.0 s instead of 42.5 s: **+55% wall time** (+63% user CPU), with identical cycles (19,686). That is above the 30% limit set for the final runs, so no POWER configs were added. `compare_paper.py` reports Snake's own energy (OQ-16) and "unavailable" for normalized energy.
- **lud's remaining §5.7 miss comes from request-slot throttling** (OQ-10): Snake-T is within 1% of Snake-DT, and throttling cuts issued prefetches by 30%. Tier 2 measures the bytes-based meter (`SNAKE_BWBYTES`).

### Why Snake does not raise IPC in our simulator (S6, results/S6_ipc.md)
- **Little latency to hide.** With perfect memory (`-gpgpu_perfect_mem`), IPC rises only +0.6–0.9% (hotspot), +1.7% (mri-q) and +4.4–4.8% (lud), against the paper's Snake gains of +13%, +4% and +18%. Load waits (`W0_Scoreboard`) are 3–5% of scheduler cycles in hotspot: other warps already hide the latency.
- **Headroom elsewhere is bandwidth.**
  - backprop (+47% ceiling) and histo (+30%) are DRAM-queue bound (0.9M and 1.8M DRAM-full stalls). Prefetching reads the same 434k DRAM sectors (backprop) and raises the average memory latency (1,015 → 1,124 cycles).
  - srad_v1 (+22.5%) and mri-q are limited by the L1 miss queue and MSHRs: one request per cycle, and bursty demand.
- **Lead time.** Inter-warp predictions within a CTA come ~20 cycles before use (latency 350–1,000).
  - `-snake_interwarp_scope 2` (same warp index in later resident CTAs; S6 experiment, default 0) makes backprop's prefetches timely: accuracy 0.2 → 14.0%, lead 1,602 cycles. IPC stays flat (+0.07%) because backprop is DRAM-bound.
  - Scope 2 is wrong on 2D grids (hotspot −4.7%).
- The paper's GPU was evidently more latency-bound (its Figs. 3–5: memory stalls 55% of stalls); that, more than the prefetcher, explains the IPC gap.
- **D1 (results/D1_ceilings.md): the ceilings on exactly Tier 1's config, all 9 workloads.**
  - The S6 perfect-memory numbers are confirmed bit for bit; nw is new (+19.0%).
  - `-gpgpu_perfect_mem` removes everything beyond the SM (interconnect, L2 and DRAM latency, bandwidth and queueing) but keeps the L1 latency queue, L1D misses, MSHRs and the one-per-cycle miss queue.
  - The new `-gpgpu_perfect_l1d` makes every L1-cached global load hit at L1 latency.
  - **Perfect-L1 gains:** hotspot +0.1–0.3%, mri-q +2%, lud +3–4%; srad_v1 +21%, nw +17%, histo +39%, backprop +42%.
  - GMEAN: +16.6% (PAPER_V100) and +17.9% (PAPER_LAT), against the paper's +14.1% and Snake's −0.3% / −0.2%.
  - Both ceilings also remove the loads' memory traffic, so for the DRAM- or miss-queue-bound apps they are upper bounds that a real prefetcher cannot reach.
- **K1: per-kernel view (results/K1_best_kernel.md).** If the paper reported each benchmark's best-performing kernel:
  - Our best kernels match its predicted coverage within about ±15 pp, and backprop's IPC gain (+1.4% vs +1%).
  - They do not match accuracy (≤ 27–32% vs 60–96%) or the other IPC gains. Best kernel vs paper: histo +12% vs +33%, srad_v1 +2.6% vs +29%, lud +1.0% vs +18%, hotspot +0.3% vs +13%.
  - For hotspot and lud even a perfect L1 on the best kernel stays below the paper (+0.33%, +8.5%), so kernel selection does not explain the gap.

### Observations per stage
- **S4 (results/S4.md).**
  - Mode 1 costs hotspot a fifth of its demand L1 hits (32.3 → 25.6–26.0%). Its hits are timely prefetches. Bulk frees evict 19.5k prefetched sectors before use (Snake-DT: 1.3k), close to the 19.0k unused sectors that lost protection when their line flipped on its first demand hit (OQ-12, one flag per line). The 128 KB L1D held hotspot's prefetches without decoupling (S3c).
  - Throttling is mostly the bandwidth trigger: 1.2–19% of cycles. In lud it cuts issued prefetches by 30%.
  - The per-launch 50% until-trained cap (OQ-15) holds for 33–53% of cycles in lud and histo.
  - Snake-ISO (mode 2) keeps Snake-DT's hit rate and accuracy; its buffer rarely overflows.
- **S3d (results/S3d.md).**
  - With OQ-32 every load at a promoted PC1 gets an inter-thread prediction. 33–57% of loads still get none, all at PCs without a promoted entry.
  - Own-warp inter-thread prefetches are rarely wrong but almost never timely. Timely prefetches come from the inter-warp source and its chain (OQ-29), which also produce most of the wrong ones.
  - In lud, most wrong predictions come from promotion and the no-bit path: 59% of its no-bit verifications fail, and T1 demotions rise from 490 to 6,879.
  - Chain depth 0 gives slightly better IPC than 3 on all five apps tested.
- **S3c diagnosis (results/S3c_diag.md, exact footprint).**
  - Issued prefetches are almost never evicted before use. The losses are lateness and non-issue.
    - Issued but late: 7–29% of predicted sectors.
    - Never issued in time (still queued, queue-full or miss-queue drops): 7–58% (mri-q 58%).
    - Wrong: 10–54% (lud 54%, mostly promotion and inter-warp-chain predictions).
  - The miss queue drains one request per cycle, and the interconnect never blocks it.
  - Lead times show why prefetches are late: inter-thread prefetches start ~20 cycles before their demand.
  - At promotion, Snake prefetches only for the warps whose Head is at PC1 at that moment. Later warps without their warpID bit get no inter-thread prediction; such loads are 15–57% of loads in hotspot, srad_v1, lud and mri-q.
  - histo's coverage_pred gap is real (30.9% exact).
- **S3b: coverage_pred approaches the paper; coverage_paper does not.** With inter-warp issue and the combined lookahead (OQ-29), coverage_pred (exact footprint) is hotspot 88%, lud 83% / 73%, srad_v1 81%, mri-q 87%, backprop 72% (paper coverage ~88 / 91 / 87 / 72 / 83%, Fig. 16). coverage_paper is 72% for hotspot but 29–44% for srad_v1, lud and backprop and 10% for mri-q. The difference is mostly prefetches dropped before issue: `pf_drop_resource` (miss-queue slots, OQ-11) is 36k–269k sectors, and lines dropped with the prefetch queue full are 40k–683k sectors. The S5 sweep over queue size, issue width and miss-queue reserve is where this is decided. Timeliness rose (accuracy_paper hotspot 31–33%, srad_v1 8%, histo 10%, vs ≤ 7% in S3a). IPC is within ±0.4% of the baseline except lud 512 (line, −1.2%) and histo (line, −3.8%); decoupling and throttling (S4) are still off.
- **S3b: lud 512 with the line footprint raises demand MISS_QUEUE_FULL** from 42,855 to 55,360 (+29%; S3a with MQR8: +4.9%). The exact footprint lowers it (23,751). The line footprint requests 4 sectors per prediction against ≤ 2 for lud's loads, so miss-queue pressure grows with the inter-warp volume.
- **S3b: the inter-warp eviction weight (OQ-30) does not help backprop.** backprop still evicts ~29,000 entries holding a trained inter-warp stride, with 0.9 warp bits on average (S3a 32,137). Every backprop Tail entry has PC1 0x90 or 0x180, and the stride is stored in every entry with that PC1 (OQ-6), so all candidates get the same +3. srad_v1 drops from ~14,000 to ~10,400; lud and histo are unchanged.
- **S3a: coverage is mostly late.** (S3b update above.) Almost every covered demand sector is a demand merging into a prefetch still in flight (accuracy_paper ≤ 1% on hotspot, lud, srad_v1 and mri-q; 7% on histo). Inter-thread chains and intra-warp prefetches run ahead of their own warp by at most a few loads, and inter-warp prefetching, which runs ahead for other warps, starts in S3b. IPC within ±0.4% of the baseline is therefore expected in S3a. Re-check after S3b.
- **S3a: coverage_pred is far below the paper's coverage** (Fig. 16): hotspot 35–38% (exact) vs ~88%, srad_v1 14–16% vs ~87%, lud 19–21% vs ~91%, histo 20–23% vs ~97%. Inter-warp predictions are not issued yet and do not count. Re-check after S3b.
- **S3a: inter-warp Tail entries are evicted nearly empty.** backprop evicts 32,137 entries holding a trained inter-warp stride with 0.9 warp bits on average (srad_v1 ~14,000 with 1.8). Inter-warp-only entries (Fig. 15a) have no warpID bits, so the popcount eviction (OQ-3) removes them first. To be fixed in S3b.
- **backprop (S2, after the trace-parser fix).** In `bpnn_layerforward` (kernel 1), the inter-thread stride from PC 0x90 to PC 0x180 grows by 128 B per warp. 0x90 (`input_cuda[index_in]`) advances 8 B per warp and 0x180 (`input_hidden_cuda[index]`) 136 B (`hw_run/rodinia-3.1/11.0/backprop-rodinia-3.1/65536/traces/kernel-1.traceg`).
  - No two warps share the (0x90, 0x180, stride) triple, so T1 is never promoted and no chain forms; every lookup allocates a new condition-3 entry and the 10-entry Tail thrashes.
  - This is a limitation of the design, which promotes an inter-thread stride only when ≥ 3 warps share it (§3, §3.4), not an implementation error.
  - Both PCs have fixed per-warp inter-warp strides (8 B and 136 B per warp-ID step), which Snake does learn. That is consistent with the paper's small backprop gain (~1% IPC, Fig. 18).
