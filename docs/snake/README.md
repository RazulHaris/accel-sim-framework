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

**S3a findings, carried to S3b (SPEC §8):**
- Coverage is mostly late, since there is no inter-warp lookahead yet, so IPC within ±0.4% is expected.
- coverage_pred is far below the paper (hotspot 35%, srad_v1 14%); re-check after S3b.
- Inter-warp Tail entries are evicted with ~0.9 warp bits (backprop); fix in S3b.
- The non-lud apps were not re-run with miss-queue reserve 8, because they dropped < 10% of their prefetches for miss-queue slots.

**In progress:** nothing.

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

**Next:** S3b, issue inter-warp prefetches (SNAKE_PLAN §8), using the S3a `tail_evict_interwarp_trained` analysis.
