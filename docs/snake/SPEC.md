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
0. **Warp address.** If all lanes have an equal inter-lane stride, keep a = first active lane's address. Otherwise the warp is excluded from Snake for this load (§3.4).
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
6. **Prefetch footprint.** The paper predicts only a warp base address. Expanding it to the whole warp's lines (base + lane·lane_stride) is not stated → **OQ-9**.
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
- **OQ-4 Prefetch queue.**
  - 32 entries, FIFO; drop the newest when full.
  - Issue ≤1 prefetch/cycle, only in cycles where the L1D port was not used by a demand access. (low)
- **OQ-5 T1/T2 encoding.** 00 = untrained → 01 = observed (≥1 warp) → 10 = promoted (≥3 warps, prefetch for others) → 11 = trained (repetition confirmed; also allows chaining). Fig. 15 matches this. Text and figure partly conflict. (low)
- **OQ-6 Inter-warp stride.** Normalised per warp-ID distance: (a_j − a_i)/(w_j − w_i), equal across ≥3 warps, as in Fig. 15a. (medium)
- **OQ-7 Chain depth.** Config `snake_max_chain_depth`, default 3. Paper gives no number; Fig. 13 shows depth 2. Throttling can reduce it to 0. (low)
- **OQ-8 Inter-warp targets.** Prefetch for the next `snake_interwarp_degree` (default 4) warp IDs above w that are resident on the SM and in the same CTA. "All future warps" is unbounded. (low)
- **OQ-9 Prefetch footprint.** Regenerate the predicted warp's full footprint (active-lane count and lane stride of the triggering access) and coalesce it into sectors and lines exactly like a demand load. (medium)
- **OQ-10 Bandwidth throttle.**
  - Per-SM injected L1→icnt bytes per window, divided by the per-SM share of peak L1↔L2 bandwidth.
  - Window `snake_bw_window` = 100 cycles; hysteresis 70%/50%.
  - Evaluated at the end of each window. (low)
- **OQ-11 MSHR and miss queue.**
  - A prefetch allocates an L1 MSHR entry, so demand misses can merge into it ("late prefetch").
  - It is issued only if MSHR free entries > `snake_mshr_reserve` (default 8) and the miss queue has space.
  - Otherwise it is dropped and counted.
  - Demand always wins arbitration. (medium)
- **OQ-12 Decoupling semantics.**
  - Implement in the L1D tag array: a per-line `prefetched` flag plus a per-set quota-aware victim choice.
  - "No free space" = the fill's set has no invalid, unreserved line.
  - "Free 25%" = evict ⌈25%⌉ of that set's evictable lines (LRU), from the side chosen by the >80%-transferred rule. That rule uses a per-SM counter, reset each window.
  - Also offer `snake_decouple_mode`: 0 = none (Snake-DT), 1 = unified flag partition (paper), 2 = isolated buffer (§5.7). (low)
- **OQ-13 Inputs and traces.**
  - Use Accel-Sim's default `define-all-apps.yml` arguments.
  - Rodinia version: paper unstated. Local traces exist for both 2.0-ft (srad_v2) and 3.1 (srad_v1).
  - **ISPASS CP/LPS/LIB/MUM traces are not available locally.** They're also absent from the public V100 1.1.0 trace set. CP is commented out ("compile issues"), and only LPS is built in gpu-app-collection. The local GPU is an RTX 2080 Ti (SM75), not a V100.
  - **Decided (2026-10-01):** ISPASS is deferred. The first evaluation uses 7 apps: rodinia-3.1 backprop / hotspot / srad_v1 / lud / nw and parboil histo / mri-q. See VERSION_DIFF for the effect on the average.
- **OQ-14 Metric definitions.** Report the paper's definitions and the standard ones:
  - coverage_paper = (timely + late useful) / demand accesses
  - accuracy_paper = timely useful / demand accesses
  - accuracy_std = useful / issued
  - Granularity: sector accesses at L1D. (medium)
- **OQ-15 50%-until-trained.** Applies per kernel launch, from launch until the first Tail entry reaches promoted. (low)
- **OQ-16 Energy.** AccelWattch is not modified. Add Snake energy offline: accesses × 6.4 pJ + 6 mW × time, from Snake stats. (medium)
- **OQ-17 Divergent and partial warps.** Lanes = active lanes only. The "equal stride" check tolerates a single active lane, with stride 0. (medium)
- **OQ-18 Scheduler.** Paper uses GTO. The base QV100 config says `-gpgpu_scheduler lrr` (`configs/tested-cfgs/SM7_QV100/gpgpusim.config:134`, despite the "Greedy then oldest" comment at :133). Run both baseline and Snake with GTO via a `GTO` extra-param config. (high)

OQ-19…OQ-25 (simulator-specific questions and the decisions on clock, Rodinia version and competitors) are in [SNAKE_PLAN §12](SNAKE_PLAN.md).
