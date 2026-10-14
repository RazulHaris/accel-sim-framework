# S3e: paper defaults for the chain walk; PAPER_LAT (2026-10-14)

Build: gpgpu-sim `53bbfd25`, accel-sim `20e968f` (frozen in `sim_run_snake_gate/S3e/bin`; the later accel-sim commit `131a513` changes test scripts only). QV100-SASS-PAPER_V100, 100M instructions, Snake-DT, exact footprint, new defaults:
- `-snake_interwarp_chain_depth 0` (paper, OQ-29)
- `-snake_promoted_all_warps 1`, now for the first chain step only (OQ-32)

Tables: `util/snake/s3d_report.py -p sim_run_snake_gate/S3d/runs:QV100-SASS-PAPER_V100-100M_INSN-SNAKE_DT-SNAKE_FP1-SNAKE_IWC0 sim_run_snake_gate/S3e/runs sim_run_snake_refcache/bin_ref_v2/runs`; `prev` = S3d (b).

## 0. Checks

- **Synthetic** (`sim_run_snake_synth/S3e`, 9 + 1 runs): PASS.
  - Fig. 15 (c) is exactly {2600, 2700, 2800, 3000} with the default, and adds {3100, 3200, 3300} with `SNAKE_IWC3`.
  - **Trace (d), exact footprint:** 90.8% with `SNAKE_IWC3` (gated, ≥ 90%). With the default it is 86.0%, reported only. Agreed 2026-10-14: the inter-warp chain is what ran ahead on this trace, and the OQ-32 change does not fire here.
  - The extra run (trace (d) × `SNAKE_IWC3`) was made by hand into the same directory with the frozen binary, so the 9 unchanged runs were not repeated. `run_synthetic.sh` now does this via `$DEP_CFGS`.
- **Off-regression** (`sim_run_snake_regress/S3e`, `ON=0`, ref from S2c): PASS on 6 workloads.
- **The 5 runs** (`sim_run_snake_gate/S3e`): all finish, `pf_woken_warps` 0, every invariant OK.

## 1. Results (vs S3d (b) in parentheses)

### Coverage, IPC and demand MISS_QUEUE_FULL vs the bin_ref_v2 baseline

| run | done / invariants | coverage_pred | coverage_paper | accuracy_paper | dIPC | demand MISS_QUEUE_FULL base -> Snake | issued |
|---|---|---|---|---|---|---|---|
| hotspot | yes / OK | 90.6 (90.6) | 76.4 (76.3) | 32.1 (32.1) | +0.04% (+0.12%) | 17562 -> 4796 (4779) | 135918 |
| lud | yes / OK | 83.6 (83.6) | 45.7 (46.2) | 3.3 (3.3) | -0.19% (-0.18%) | 42855 -> 28607 (28713) | 432154 |
| histo | yes / OK | 30.9 (30.7) | 25.0 (24.9) | 8.8 (9.1) | -1.60% (-1.39%) | 112252 -> 70655 (70242) | 446155 |
| mri-q | yes / OK | 90.8 (90.8) | 9.5 (9.5) | 0.0 (0.0) | +0.12% (+0.12%) | 28484 -> 25566 (25566) | 2008 |
| srad_v1 | yes / OK | 84.7 (84.9) | 30.1 (30.0) | 6.8 (6.8) | +0.19% (+0.53%) | 84436 -> 39647 (40193) | 121662 |

### Share of all wrong and all timely records by source (%), S3e

| run | wrong records | wrong: inter-thread, own warp | wrong: inter-thread, at promotion | wrong: intra-warp | wrong: inter-warp | wrong: inter-warp chain (OQ-29) | timely records | timely: inter-thread, own warp | timely: inter-thread, at promotion | timely: intra-warp | timely: inter-warp | timely: inter-warp chain (OQ-29) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotspot | 21249 | 0.5 | 0.0 | 0.0 | 99.5 | 0.0 | 55468 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| lud | 515580 | 24.8 | 52.7 | 0.0 | 22.5 | 0.0 | 18098 | 22.4 | 7.9 | 0.0 | 69.7 | 0.0 |
| histo | 350646 | 3.5 | 0.1 | 0.0 | 96.3 | 0.0 | 138187 | 73.9 | 2.3 | 0.0 | 23.8 | 0.0 |
| mri-q | 9996 | 3.8 | 0.1 | 0.0 | 96.1 | 0.0 | 0 | - | - | - | - | - |
| srad_v1 | 47732 | 0.7 | 2.6 | 0.0 | 96.7 | 0.0 | 27024 | 0.5 | 0.0 | 0.0 | 99.5 | 0.0 |

S3d (b), for comparison (from `results/S3d.md` §2.2):

| run | wrong records | own inter-thread | promotion | intra | inter-warp | chain | timely records | own inter-thread | promotion | intra | inter-warp | chain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotspot | 21,249 | 0.5 | 0.0 | 0.0 | 99.5 | 0 | 55,470 | 0.0 | 0.0 | 0.0 | 100.0 | 0 |
| lud | 517,069 | 24.6 | 53.1 | 0.0 | 22.3 | 0 | 18,064 | 25.2 | 7.9 | 0.0 | 66.9 | 0 |
| histo | 368,046 | 3.4 | 0.1 | 0.0 | 96.5 | 0 | 142,877 | 70.6 | 2.3 | 0.0 | 27.1 | 0 |
| mri-q | 9,996 | 3.8 | 0.1 | 0.0 | 96.1 | 0 | 0 | – | – | – | – | – |
| srad_v1 | 48,346 | 0.7 | 3.5 | 0.0 | 95.7 | 0 | 27,031 | 0.6 | 0.0 | 0.0 | 99.4 | 0 |

### Loads without an own inter-thread prediction (OQ-32)

| run | loads observed | no own inter-thread prediction | of which at a PC with a promoted entry | loads at a promoted PC1 without the bit | predictions without the bit | verified ok / failed / lost | T1 demotions |
|---|---|---|---|---|---|---|---|
| hotspot | 29240 | 14814 (50.7%) | 0 (0.0%) | 4294 (14.7%) | 4294 | 4240 / 0 / 0 | 0 |
| lud | 141898 | 80774 (56.9%) | 0 (0.0%) | 57706 (40.7%) | 57706 | 23741 / 33959 / 0 | 6898 |
| histo | 154495 | 51285 (33.2%) | 0 (0.0%) | 1133 (0.7%) | 1133 | 1133 / 0 / 0 | 67 |
| mri-q | 5312 | 2380 (44.8%) | 0 (0.0%) | 2932 (55.2%) | 2932 | 2464 / 312 / 0 | 2 |
| srad_v1 | 101032 | 51965 (51.4%) | 0 (0.0%) | 35798 (35.4%) | 35798 | 32751 / 570 / 260 | 84 |

**T1 demotions (lud 512):** S3c 490 → S3d (a) 6,879 → S3d (b) 6,829 → S3e 6,898. Other apps: srad_v1 1 / 92 / 81 / 84, histo 110 / 109 / 67 / 67, mri-q 0 / 2 / 2 / 2, hotspot 0.

## 2. Findings

- **S3e is within noise of S3d (b).**
  - Requiring the warp's bit and a trained entry at deeper steps barely changes anything: the no-bit predictions were almost all depth 1. `pred_inter_thread_nobit` now equals the depth-1 count (lud 57,706), against 57,467 at all depths in S3d (b).
  - IPC: hotspot +0.04% (+0.12), lud −0.19% (−0.18), histo −1.60% (−1.39), mri-q +0.12% (+0.12), srad_v1 +0.19% (+0.53).
  - coverage_pred and coverage_paper are within ±0.5 pp.
- **Wrong share by source is unchanged.** Promotion gives 53% of lud's wrong predictions; inter-warp gives 96% of histo's, mri-q's and srad_v1's, and 99.5% of hotspot's.
- **lud's demotions stay about 6,900.** They come from the first-step verification of no-bit predictions (34k of 58k fail), which S3e keeps.
- **Intra-warp barely fires:** 0–45 predictions and 0–181 trained T2 strides per run, against 5k–155k observed loads (SPEC §8).

## 3. PAPER_LAT calibration (`sim_run_snake_calib`, `bin_ref_v2`, Snake off)

- **Benchmark.** `l2_lat` from gpu-app-collection GPU_Microbenchmark (hw_def `volta_QV100`), built for sm_70 and traced with NVBit 1.7.6 on the local RTX 2080 Ti. Its kernels:
  - kernel 1: 32,768 dependent `LDG.E.64.STRONG.GPU` loads (L1 bypass)
  - kernel 2: 32,768 dependent `LDG.E.64.STRONG.CTA` loads (L1)
- **Latency** = kernel cycles / 32,768.
- **First attempt aborted.** A first trace, built for sm_75, aborted at the `shader.cc:1490` specialized-unit assert: binary version 75 selects the Turing opcode map. The log is in `failed_sm75/`.

| config | `-gpgpu_l1_latency` | `-gpgpu_l2_rop_latency` | L2 hit (cycles) | L1 hit (cycles) |
|---|---|---|---|---|
| current QV100 + PAPER_V100 (`base`) | 20 | 160 | 173.0 | 25.0 |
| **PAPER_LAT** (`paper_lat`) | 28 | 199 | **212.0** | 33.0 |
| paper Table 1 | — | — | 212 | 28 |
| native RTX 2080 Ti (sm_70 build), for reference | — | — | 213.0 | 32.0 |

- **ROP setting.** The ROP latency was chosen as 160 + (212 − 173) = 199, since L1-bypass loads do not pass the L1 pipeline. PAPER_LAT then measures 212.0.
- **L1 setting.** The simulator adds about 5 cycles to `-gpgpu_l1_latency`. With 28, as specified, the L1 hit measures 33. If Table 1's 28 is a measured latency, the matching knob is 23 (open question for the final evaluation).
- **Use.** Config token `PAPER_LAT` (`define-standard-cfgs.yml`), for the final evaluation only; not used in any S3e run.
