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

## Pre-756ed33 traces (trace-parser fix, 2026-10-13)
- **What.** Before tracer commit `756ed33` (2020-11-06, "Resolve wrong address generation issue in tracer"; in v1.3.0 and the base), the tracer printed `active & predicate` as a line's mask but compressed the addresses over `active` alone (`475a282` `tracer_tool.cu:529`/`:548`). The parser decodes with the printed mask, so lanes of predicated memory instructions get wrong addresses. Details and a reproduction: [upstream_issue.md](upstream_issue.md) (draft, not posted).
- **Our traces are affected.** All local Rodinia 3.1 / Parboil traces are tracer version 3, nvbit 1.4, dated 2020-08-09, i.e. pre-`756ed33`.
- **The paper era had it too.** v1.2.0 (`ed6d38b`) has both the pre-fix tracer (`tracer_tool.cu:549`, `mask(ma->active_mask)`) and the same parser. So the paper's Accel-Sim v1.2.0 simulations very likely decoded these lines the same wrong way, whatever traces they used: pre-fix traces were mis-decoded by the parser, and traces the authors made with the v1.2.0 tracer were written wrongly. Upstream `dev` (`0db0445`) still does not handle pre-fix traces.
- **Fix (accel-sim `fe8f37e`, separate from Snake).** For tracer version ≤ 3, a base+delta line with a partial mask and exactly 31 deltas (a fully active warp) is decoded over all 32 lanes, keeping the printed lanes. This is exact; every other line is parsed as before. Each run prints `accel-sim: pre-756ed33 base+delta trace lines re-decoded = N`.
- **References.** `bin_ref_v2` (clean bases + `fe8f37e` cherry-picked as `6c28685` on local branch `snake-ref-v2`) is the regression reference from 2026-10-13. The old clean-base build is kept as `bin_ref_paperera`, with the paper-era decoding.
- **Scan of the 7 suite apps** (`util/snake/scan_legacy_traces.py`, all inputs and kernels). "Harmless" lines carry extra deltas but decode correctly: either the mask is empty, or lane 0 is the only printed lane. All remaining base+stride gap lines are shared-memory `LDS`/`STS` in hotspot (masks like `fffcfffc`). They cannot be fixed from the trace, because the first active lane is unknown, and they affect bank-conflict timing only, not global/L1 traffic.

| app | kernel files | memory lines (global) | base+delta fixed | base+delta harmless / wrong | list_all mismatch | base+stride gap: global / other | still mis-decoded after fix |
|---|---|---|---|---|---|---|---|
| backprop-rodinia-3.1 | 2 | 1048583 (425991) | 172032 (16.41%) | 0 / 0 | 0 | 0 / 0 | 0 (0.00%) |
| hotspot-rodinia-3.1 | 2 | 1452024 (202960) | 5314 (0.37%) | 0 / 0 | 0 | 0 / 114168 | 114168 (7.86%) |
| srad_v1-rodinia-3.1 | 502 | 49597440 (22440040) | 4000 (0.01%) | 200 / 0 | 0 | 0 / 0 | 0 (0.00%) |
| lud-rodinia-3.1 | 140 | 2706808 (423144) | 0 (0.00%) | 144 / 0 | 0 | 0 / 0 | 0 (0.00%) |
| nw-rodinia-3.1 | 255 | 3686400 (573440) | 0 (0.00%) | 16384 / 0 | 0 | 0 / 0 | 0 (0.00%) |
| parboil-histo | 80 | 6565000 (5215880) | 0 (0.00%) | 0 / 0 | 0 | 0 / 0 | 0 (0.00%) |
| parboil-mri-q | 4 | 21792 (21792) | 0 (0.00%) | 0 / 0 | 0 | 0 / 0 | 0 (0.00%) |

- **Effect on the baseline,** `bin_ref_paperera` → `bin_ref_v2` (`util/snake/compare_bins.sh paperera_vs_v2`, `sim_run_snake_refcmp/paperera_vs_v2`, 2026-10-13):

| workload (100M insn, PAPER_V100) | cycles | L1D read hits | L1D read misses | DRAM reads | lines re-decoded |
|---|---|---|---|---|---|
| backprop 65536 | 26,170 → 25,995 (−0.67%) | 28,310 → 6,830 (−75.9%) | 49,798 → 44,172 (−11.3%) | 150,193 → 142,210 (−5.3%) | 172,032 |
| hotspot 512 | 19,697 → 19,686 (−0.06%) | 888 → 322 (−63.7%) | 88,430 → 85,256 (−3.6%) | 67,612 → 65,536 (−3.1%) | 1,673 |
| hotspot 1024 | 19,458 → 19,451 (−0.04%) | 1,414 → 761 (−46.2%) | 103,174 → 102,283 (−0.9%) | 79,422 → 78,173 (−1.6%) | 1,004 |
| srad_v1 | 60,676 (0) | 130,839 (0) | 54,659 (0) | 28,984 (0) | 40 |
| nw 2048_10 | 1,515,147 (0) | 11,233 (0) | 460,910 (0) | 519,049 (0) | 0 |
| parboil histo | 141,169 (0) | 991,495 (0) | 147,213 (0) | 172,871 (0) | 0 |

  Cycles change by less than 0.7%. The wild lane addresses produced spurious L1 hits, because many warps hit the same wild line, and spurious misses. Without them, backprop's L1D read hits fall by 76% and its DRAM reads by 5%. hotspot 1024 also simulated a few more instructions within the 100M cap (100,001,955 → 100,006,626).

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
8a. **Trace decoding:** the paper's v1.2.0 simulations most likely mis-decoded predicated memory instructions (see "Pre-756ed33 traces" above). We fix the base+delta case, which changes the L1 hit and DRAM counts of backprop and hotspot.
   - **The paper's baseline L1 hit rates for backprop and hotspot were therefore likely inflated.** The wild lane addresses of predicated loads repeat across warps, so they hit the same wild line in L1.
   - In `paperera_vs_v2` (100M instructions, GLOBAL_ACC_R reads), the L1D read hit rate drops from 36.2% to 13.4% for backprop (hits 28,310 → 6,830). For hotspot 512 it drops from 0.99% to 0.38%, and for hotspot 1024 from 1.35% to 0.74%.
   - The paper's baseline L1 hit rate (45% on average, §5.7) and the headroom Snake had on these two apps may therefore differ from what a correctly decoded baseline shows. That holds whether the authors used pre-fix traces or traced with the v1.2.0 tracer. Compare L1 hit rates with the paper per app, not as a correction to the mean.
9. **Unknown implementation details:** everything in OQ-1…OQ-18. This is probably the largest source of difference.

## Recommendation: implement on my current base (`3016c65` / `6c3cf4ff`)
- **The paper's L1 matches the base.** The base's L1 model is the one in the paper's Tab. 1. The literal v1.2.0 release tag has a *different* streaming L1, so "reproducing on v1.2.0" would be less faithful in the subsystem Snake lives in.
- **Configs are closer too:** the remaining config gaps (GTO, clock) are overridable flags.
- **The base is maintained:** it has bug fixes (EXIT deadlock, coalescer 64-bit addresses) and newer traces.
- **Tradeoff:** residual differences (#67 iSLIP, #127, the stats format, the 1132 vs 1447/1530 clock) cannot be removed by config, so the per-benchmark numbers will not match exactly.
- **Paper-faithful config:** add a `PAPER_V100` extra-params config (`-gpgpu_scheduler gto` + `-gpgpu_clock_domains 1530.0:1530.0:1530.0:850.0`). Combine it with the existing `1B_INSN` (`-gpgpu_max_insn 1000000000`, §4) as `QV100-SASS-PAPER_V100-1B_INSN`. Use the **same** config for baseline and Snake.
- **Sensitivity (decided 2026-10-08):** one extra baseline + Snake pair at the base's calibrated 1132 MHz (`PAPER_V100_1132`: GTO only, stock clock).
- **Optional:** if a closer match matters, a later port to dev `6b244a5d` is possible. Snake's code is self-contained, so it would be a small port, but it is not recommended up front.
