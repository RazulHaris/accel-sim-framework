# Snake — version check vs Accel-Sim v1.2.0 (Phase 0)

Paper: Mostofi et al., "Snake: A Variable-length Chain-based Prefetching for GPUs", MICRO 2023 (`docs/snake.pdf`, kept out of git).
Bases: accel-sim-framework `3016c65`, gpgpu-sim `6c3cf4ff` (both on branch `Snake`). All `file:line` references are at those commits.
Companion docs: [SPEC](SPEC.md) · [VERSION_DIFF](VERSION_DIFF.md) · [CODE_MAP](CODE_MAP.md) · [SNAKE_PLAN](SNAKE_PLAN.md)

## Commits

| repo | paper version (v1.2.0) | my base | commits between |
|---|---|---|---|
| accel-sim-framework | tag `v1.2.0` = `ed6d38b` (2021-10-19, release branch, PR #74) | `3016c65` (2026-03-23, #521); `git describe` = v1.3.0-72 | 1531 (`v1.2.0..3016c65`) |
| gpgpu-sim (release pairing) | `971dd69e`, tip of `release-accelwattch` (2021-10-19). v1.2.0 `gpu-simulator/setup_environment.sh:42-46` checks out that branch. gpgpu-sim has no tags | `6c3cf4ff` (2026-01-22, #133) | 265 |
| gpgpu-sim (dev at tag date) | `6b244a5d` (2021-10-17, #237) | `6c3cf4ff` | 151 |

- **Ancestry:** v1.2.0 is **not an ancestor** of the base. It is a release branch split off at v1.1.0 (`4c2bf09`) plus AccelWattch.
- **The release gpgpu-sim branch** split from dev on 2020-10-18. It lacks the whole 2021 L1 rework.

### Which v1.2.0 did the paper actually use?
Paper Tab. 1 lists "Unified cache 128KB, 256-way, 128B, 4-bank" and "MSHR 512 entries, 8 merge".
- **Base / dev `6b244a5d`:** `-gpgpu_cache:dl1 S:4:128:64,…,A:512:8` with `-gpgpu_unified_l1d_size 128`. The adaptive carve-out turns this into 4 sets × 256 ways = 128 KB, with 512 MSHRs and merge 8.
- **Release `971dd69e`:** `S:1:128:256,L:L:s:N:L,A:256:8`, a streaming 32 KB cache. The MSHR count is overridden to 4096 and merge to 64 (`gpu-cache.h:562-566`), and there is no unified-L1 option.

**→ The paper's L1 matches dev / my base, not the release tag** (high confidence).

The paper's GTO scheduler and 1530 MHz clock match **no** shipped QV100 config:

| commit | QV100 scheduler | QV100 clock |
|---|---|---|
| release `971dd69e` | gto | 1447 |
| dev `6b244a5d` | lrr | 1447 |
| base `6c3cf4ff` | lrr | 1132 |

So the authors overrode them, or 1530 MHz is the nominal V100 boost clock quoted in the paper.

## Per-subsystem changes (v1.2.0 → base)

| Subsystem | Change | Affects Snake |
|---|---|---|
| L1D `gpu-cache.*` | Release: streaming, ON_FILL, 4096 MSHRs, and a read to a line pending for a *different* instruction is sent as a separate request rather than merged (`gpu-cache.cc:316-323`). Base: on-miss, 512/8 MSHR, adaptive 128 KB unified (`f2a7d9ce`, `09f10eb4`, `1ee03f01`, `f7833519`, `a2ba2f57`, `e3d186bb`), write-through with write-allocate (`7fac247e`), byte masks, `tag_array::fill` reservation-fail return (`cb6060a6`), sector size fixed at 32 B (#57) | **Yes.** It decides whether a demand load merges with an in-flight prefetch ("late prefetch"), MSHR pressure, and the decoupling space. The base matches the paper's Tab. 1 |
| LD/ST and coalescer | `07f77e1c`: block address became `new_addr_type`; release truncates to 32 bits. #127/#133: L1 bypass path pushes up to `l1_banks` per cycle (`shader.cc:2261`). LDGSTS (`a0c12f5d`). Sub-core dispatch (`585dcf5d`) | **Yes.** Address arithmetic for strides; bypass loads are not seen by L1 |
| icnt / L2 / DRAM | iSLIP rewrite changes arbitration order (`bc268aab` #67, `local_interconnect.cc:193-260`). L2 byte-mask and deadlock fixes (`40077df9`). `mem_fetch` gains a stream ID. `dram.cc` is formatting only | **Maybe.** Prefetch latency and bandwidth (throttle T-b) |
| Warp scheduling | QV100, TITANV and RTX2060 configs gto → lrr (`a8256e50`, `84c4f46f`, Aug 2021). New `rrr` | **Yes.** Inter-warp stride detection and the Head two-warps-per-PC design assume GTO (§3.1) |
| Trace front end (accel-sim) | Tracer v3 → v5 (optional line info, `trace_parser.cc:144-156`). `.BYPASS` L1 bypass (#466). Multi-stream kernel window (`accel-sim.cc:51-66`). EXIT deadlock fix (#503) | **No / Low.** PC and warp ID are available in both versions |
| Configs | QV100 clock 1447 → 1132 (`8ca01b07`); `perfect_inst_const_cache` 0 → 1; `max_concurrent_kernel` 8 → 128; `trace.config` int/sp latency 4,2 → 2,2. Unchanged: L2 `S:32:128:24`, `l1_latency 20`, `l2_rop_latency 160`, `dram_latency 100`, DRAM timing | **Yes.** Clock ratio moves the benefit of hiding latency |
| Stats / AccelWattch | AccelWattch is equivalent (`84c6cf45` ≈ `d90d7ab0`). The base excludes MSHR_HIT from L1D TOTAL_ACCESS (`a374b330`, `gpu-cache.cc:945`). Output is per stream (`38b4df56`) | **No for the model. Maybe** for comparing stats and energy |

## Why my numbers may differ from the paper even with a correct implementation
1. **Scheduler:** the base uses lrr; the paper used GTO. **Mitigation:** run with `-gpgpu_scheduler gto`.
2. **Clock:** the base runs QV100 at 1132 MHz; the paper states 1530 MHz.
3. **Latency mismatches:**
   - Paper "28-cycle" L1 vs. config `l1_latency 20`.
   - Paper "212-cycle" L2 vs. `l2_rop_latency 160`. That could be measured end to end.
4. **Interconnect:** iSLIP arbitration change (#67) and multi-bank bypass (#127).
5. **Kernel concurrency:** `max_concurrent_kernel` 128 plus the stream window can overlap kernels.
6. **Execution latencies:** retuned `trace.config` values and the perfect const/inst cache expose a different share of memory latency.
7. **Traces:**
   - Unknown trace and CUDA versions in the paper.
   - Local traces are the V100 1.1.0 set (CUDA 11.0).
   - ISPASS traces are absent (OQ-13), so **LIB, LPS, MUM and CP are missing** from the first evaluation.
   - LIB is the paper's largest gain (+60%, Fig. 18), and LPS (~6%) and MUM (~11%) also contribute.
   - **The 7-app geometric mean is therefore expected to be below the paper's 17%**, even with a faithful implementation. Compare per benchmark, not only the average.
8. **Stat accounting:** MSHR_HIT handling and the per-stream format.
9. **Unknown implementation details:** everything in OQ-1…OQ-18. This is probably the largest source of difference.

## Recommendation: implement on my current base (`3016c65` / `6c3cf4ff`)
- **The paper's L1 matches the base.** The base's L1 model is the one in the paper's Tab. 1. The literal v1.2.0 release tag has a *different* streaming L1, so "reproducing on v1.2.0" would be less faithful in the subsystem Snake lives in.
- **Configs are closer too:** the remaining config gaps (GTO, clock) are overridable flags.
- **The base is maintained:** it has bug fixes (EXIT deadlock, coalescer 64-bit addresses) and newer traces.
- **Tradeoff:** residual differences (#67 iSLIP, #127, the stats format, the 1132 vs 1447/1530 clock) cannot be removed by config, so the per-benchmark numbers will not match exactly.
- **Paper-faithful config:** add a `PAPER_V100` extra-params config (`-gpgpu_scheduler gto` + `-gpgpu_clock_domains 1530.0:1530.0:1530.0:850.0`). Combine it with the existing `1B_INSN` (`-gpgpu_max_insn 1000000000`, §4) as `QV100-SASS-PAPER_V100-1B_INSN`. Use the **same** config for baseline and Snake.
- **Sensitivity (decided 2026-10-08):** one extra baseline + Snake pair at the base's calibrated 1132 MHz (`PAPER_V100_1132`: GTO only, stock clock).
- **Optional:** if a closer match matters, a later port to dev `6b244a5d` is possible. Snake's code is self-contained, so it would be a small port, but it is not recommended up front.
