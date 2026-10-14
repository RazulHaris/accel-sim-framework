# S3d: promoted strides for all warps; funnel by source (2026-10-14)

Build: gpgpu-sim `97119f36`, accel-sim `976937f` (frozen in `sim_run_snake_gate/S3d/bin`). QV100-SASS-PAPER_V100, 100M instructions, Snake-DT, exact footprint, `-snake_missq_reserve` 8.
- **(a) new defaults:** `-snake_promoted_all_warps 1`, `-snake_interwarp_chain_depth 3`.
- **(b) paper's inter-warp:** `SNAKE_IWC0`, chain depth 0.

Tables: `util/snake/s3d_report.py -p sim_run_snake_gate/S3c/runs:QV100-SASS-PAPER_V100-100M_INSN-SNAKE_DT-SNAKE_FP1 sim_run_snake_gate/S3d/runs sim_run_snake_refcache/bin_ref_v2/runs`. `prev` is the S3c run at (a)'s settings with `promoted_all_warps` 0 behaviour.

## 0. Checks

- **Synthetic** (`sim_run_snake_synth/S3d`, 9 runs): PASS in all three configs. Fig. 15 (c) is unchanged (default and `SNAKE_IWC0`), and trace (d) coverage_paper is 90.8%. The synthetic traces do not exercise the no-bit path (`pred_inter_thread_nobit` 0), since every warp trains its bits there.
- **Off-regression** (`sim_run_snake_regress/S3d`, `ON=0`, ref from S2c): PASS on 6 workloads.
- **The 10 runs** (`sim_run_snake_gate/S3d`): all finish, `pf_woken_warps` 0, and the outcome, funnel and per-source funnel invariants are OK.

## 1. Change (OQ-32)

- With `-snake_promoted_all_warps 1` (default), a load at PC1 gets the inter-thread chain even when the warp holds none of the entry's warpID bits.
  - An entry with the warp's bit is still preferred; otherwise the most recently used entry is taken.
  - Each chain step keeps its T1 requirement: promoted at depth 1, trained deeper.
- **Verification (§3.2):** the warp's next load checks the first step. If PC2 and the stride match, training sets its bit. If not, it counts as a removed warp of that entry, and the entry is demoted after more than `-snake_demote_removed` removals.
- `0` keeps the S3c behaviour: one-shot promotion predictions only.
- **Source of a record (funnel_src):** the source of the prediction that created it. In S3c the funnel used the latest prediction's mode.

## 2. Funnel by prediction source

"Never issued in time" = still queued, dropped with the queue full, dropped at the miss queue, or dropped for a line or MSHR.

### 2.1 Per source, % of that source's records (% of that source's records)

| run | source | records (% of all) | wrong | late | timely | never issued in time | dup L1/MSHR | evicted / other | live |
|---|---|---|---|---|---|---|---|---|---|
| hotspot / (b) IWC0 | inter-thread, own warp | 18688 (7.9) | 0.6 | 87.5 | 0.0 | 11.4 | 0.0 | 0.0 | 0.5 |
| hotspot / (b) IWC0 | inter-thread, at promotion | 105 (0.0) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| hotspot / (b) IWC0 | inter-warp | 216955 (92.0) | 9.7 | 27.7 | 25.6 | 10.2 | 0.1 | 0.0 | 26.6 |
| hotspot / (a) default | inter-thread, own warp | 18824 (8.1) | 0.5 | 87.4 | 0.0 | 11.4 | 0.1 | 0.0 | 0.5 |
| hotspot / (a) default | inter-warp | 113117 (48.4) | 10.2 | 21.1 | 26.4 | 16.5 | 0.2 | 0.0 | 25.6 |
| hotspot / (a) default | inter-warp chain (OQ-29) | 101800 (43.6) | 9.3 | 34.3 | 25.2 | 4.0 | 0.2 | 0.0 | 27.0 |
| lud / (b) IWC0 | inter-thread, own warp | 151818 (14.7) | 83.9 | 2.5 | 3.0 | 2.0 | 0.1 | 0.0 | 8.6 |
| lud / (b) IWC0 | inter-thread, at promotion | 367958 (35.5) | 74.6 | 11.0 | 0.4 | 4.9 | 0.0 | 0.0 | 9.0 |
| lud / (b) IWC0 | inter-warp | 516108 (49.8) | 22.3 | 37.0 | 2.3 | 33.0 | 2.7 | 0.0 | 2.7 |
| lud / (a) default | inter-thread, own warp | 75852 (6.3) | 80.7 | 4.0 | 2.8 | 3.3 | 0.2 | 0.0 | 9.0 |
| lud / (a) default | inter-thread, at promotion | 360950 (30.0) | 75.6 | 10.3 | 0.3 | 5.1 | 0.0 | 0.0 | 8.6 |
| lud / (a) default | inter-warp | 484082 (40.3) | 20.5 | 34.9 | 1.4 | 38.5 | 2.1 | 0.0 | 2.5 |
| lud / (a) default | inter-warp chain (OQ-29) | 280643 (23.4) | 82.9 | 3.3 | 1.4 | 3.4 | 0.1 | 0.0 | 8.8 |
| histo / (b) IWC0 | inter-thread, own warp | 239759 (24.5) | 5.2 | 46.9 | 42.1 | 1.2 | 1.4 | 0.0 | 3.2 |
| histo / (b) IWC0 | inter-thread, at promotion | 105501 (10.8) | 0.4 | 63.3 | 3.2 | 30.5 | 2.6 | 0.0 | 0.0 |
| histo / (b) IWC0 | inter-warp | 632922 (64.7) | 56.1 | 10.8 | 6.1 | 6.2 | 1.8 | 0.0 | 18.9 |
| histo / (a) default | inter-thread, own warp | 87548 (7.5) | 2.6 | 45.7 | 43.8 | 2.6 | 2.7 | 0.0 | 2.5 |
| histo / (a) default | inter-thread, at promotion | 106108 (9.1) | 1.4 | 62.1 | 3.2 | 31.3 | 2.0 | 0.0 | 0.0 |
| histo / (a) default | intra-warp | 4 (0.0) | 0.0 | 0.0 | 0.0 | 75.0 | 25.0 | 0.0 | 0.0 |
| histo / (a) default | inter-warp | 439296 (37.7) | 63.2 | 10.0 | 4.3 | 9.1 | 1.8 | 0.0 | 11.6 |
| histo / (a) default | inter-warp chain (OQ-29) | 532066 (45.7) | 43.9 | 16.7 | 17.2 | 0.7 | 0.3 | 0.0 | 21.2 |
| mri-q / (b) IWC0 | inter-thread, own warp | 3142 (10.7) | 12.1 | 9.8 | 0.0 | 78.1 | 0.0 | 0.0 | 0.0 |
| mri-q / (b) IWC0 | inter-thread, at promotion | 1136 (3.9) | 0.7 | 16.9 | 0.0 | 82.4 | 0.0 | 0.0 | 0.0 |
| mri-q / (b) IWC0 | inter-warp | 25008 (85.4) | 38.4 | 6.0 | 0.0 | 55.6 | 0.0 | 0.0 | 0.0 |
| mri-q / (a) default | inter-thread, own warp | 2614 (8.7) | 14.5 | 11.6 | 0.0 | 73.9 | 0.0 | 0.0 | 0.0 |
| mri-q / (a) default | inter-thread, at promotion | 1136 (3.8) | 0.7 | 16.9 | 0.0 | 82.4 | 0.0 | 0.0 | 0.0 |
| mri-q / (a) default | inter-warp | 12064 (40.0) | 35.7 | 10.5 | 0.0 | 53.7 | 0.0 | 0.0 | 0.0 |
| mri-q / (a) default | inter-warp chain (OQ-29) | 14368 (47.6) | 38.6 | 1.7 | 0.0 | 59.7 | 0.0 | 0.0 | 0.0 |
| srad_v1 / (b) IWC0 | inter-thread, own warp | 51790 (12.5) | 0.7 | 17.1 | 0.3 | 41.4 | 40.2 | 0.0 | 0.2 |
| srad_v1 / (b) IWC0 | inter-thread, at promotion | 19101 (4.6) | 8.9 | 1.5 | 0.0 | 72.2 | 15.1 | 0.0 | 2.3 |
| srad_v1 / (b) IWC0 | inter-warp | 342044 (82.8) | 13.5 | 24.6 | 7.9 | 30.7 | 16.2 | 0.0 | 7.1 |
| srad_v1 / (a) default | inter-thread, own warp | 43985 (10.5) | 0.8 | 15.4 | 0.4 | 41.5 | 41.7 | 0.0 | 0.2 |
| srad_v1 / (a) default | inter-thread, at promotion | 19647 (4.7) | 9.9 | 1.3 | 0.0 | 71.9 | 13.4 | 0.0 | 3.4 |
| srad_v1 / (a) default | inter-warp | 225376 (54.0) | 13.7 | 26.3 | 5.2 | 34.6 | 15.2 | 0.0 | 5.0 |
| srad_v1 / (a) default | inter-warp chain (OQ-29) | 128425 (30.8) | 11.5 | 20.3 | 15.2 | 26.1 | 16.2 | 0.0 | 10.7 |

### 2.2 Share of all wrong and all timely records by source (%)

| run | wrong records | wrong: inter-thread, own warp | wrong: inter-thread, at promotion | wrong: intra-warp | wrong: inter-warp | wrong: inter-warp chain (OQ-29) | timely records | timely: inter-thread, own warp | timely: inter-thread, at promotion | timely: intra-warp | timely: inter-warp | timely: inter-warp chain (OQ-29) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotspot / (b) IWC0 | 21249 | 0.5 | 0.0 | 0.0 | 99.5 | 0.0 | 55470 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| hotspot / (a) default | 21137 | 0.5 | 0.0 | 0.0 | 54.7 | 44.9 | 55537 | 0.0 | 0.0 | 0.0 | 53.8 | 46.2 |
| lud / (b) IWC0 | 517069 | 24.6 | 53.1 | 0.0 | 22.3 | 0.0 | 18064 | 25.2 | 7.9 | 0.0 | 66.9 | 0.0 |
| lud / (a) default | 666346 | 9.2 | 41.0 | 0.0 | 14.9 | 34.9 | 13969 | 15.3 | 8.4 | 0.0 | 47.7 | 28.5 |
| histo / (b) IWC0 | 368046 | 3.4 | 0.1 | 0.0 | 96.5 | 0.0 | 142877 | 70.6 | 2.3 | 0.0 | 27.1 | 0.0 |
| histo / (a) default | 515022 | 0.4 | 0.3 | 0.0 | 53.9 | 45.4 | 152260 | 25.2 | 2.2 | 0.0 | 12.5 | 60.1 |
| mri-q / (b) IWC0 | 9996 | 3.8 | 0.1 | 0.0 | 96.1 | 0.0 | 0 | - | - | - | - | - |
| mri-q / (a) default | 10250 | 3.7 | 0.1 | 0.0 | 42.1 | 54.1 | 0 | - | - | - | - | - |
| srad_v1 / (b) IWC0 | 48346 | 0.7 | 3.5 | 0.0 | 95.7 | 0.0 | 27031 | 0.6 | 0.0 | 0.0 | 99.4 | 0.0 |
| srad_v1 / (a) default | 47851 | 0.7 | 4.1 | 0.0 | 64.4 | 30.8 | 31424 | 0.5 | 0.0 | 0.0 | 37.3 | 62.2 |

## 3. Loads without an own inter-thread prediction (OQ-32)

| run | loads observed | no own inter-thread prediction | of which at a PC with a promoted entry | loads at a promoted PC1 without the bit | predictions without the bit | verified ok / failed / lost | T1 demotions |
|---|---|---|---|---|---|---|---|
| hotspot / (b) IWC0 | 29240 | 14822 (50.7%) | 0 (0.0%) | 4286 (14.7%) | 4330 | 4240 / 0 / 0 | 0 |
| hotspot / (a) default | 29240 | 14860 (50.8%) | 0 (0.0%) | 4248 (14.5%) | 4248 | 4240 / 0 / 0 | 0 |
| lud / (b) IWC0 | 141896 | 81090 (57.1%) | 0 (0.0%) | 57423 (40.5%) | 57467 | 23708 / 33707 / 0 | 6829 |
| lud / (a) default | 141886 | 80834 (57.0%) | 0 (0.0%) | 56900 (40.1%) | 56931 | 23333 / 33549 / 0 | 6879 |
| histo / (b) IWC0 | 154516 | 51288 (33.2%) | 0 (0.0%) | 1150 (0.7%) | 2646 | 1150 / 0 / 0 | 67 |
| histo / (a) default | 154509 | 51321 (33.2%) | 0 (0.0%) | 1187 (0.8%) | 2647 | 1169 / 18 / 0 | 109 |
| mri-q / (b) IWC0 | 5312 | 2380 (44.8%) | 0 (0.0%) | 2932 (55.2%) | 2932 | 2464 / 312 / 0 | 2 |
| mri-q / (a) default | 5312 | 2380 (44.8%) | 0 (0.0%) | 2932 (55.2%) | 2932 | 2464 / 312 / 0 | 2 |
| srad_v1 / (b) IWC0 | 101032 | 51921 (51.4%) | 0 (0.0%) | 36028 (35.7%) | 37029 | 32948 / 572 / 262 | 81 |
| srad_v1 / (a) default | 101032 | 50023 (49.5%) | 0 (0.0%) | 37756 (37.4%) | 38853 | 34514 / 607 / 282 | 92 |

## 4. Summary vs the bin_ref_v2 baseline (prev in parentheses)

| run | done / invariants | coverage_pred | coverage_paper | accuracy_paper | dIPC | demand MISS_QUEUE_FULL base -> Snake | issued |
|---|---|---|---|---|---|---|---|
| hotspot / (b) IWC0 | yes / OK | 90.6 (87.8) | 76.3 (71.1) | 32.1 (33.1) | +0.12% (-0.15%) | 17562 -> 4779 (9181) | 135839 |
| hotspot / (a) default | yes / OK | 90.4 (87.8) | 75.7 (71.1) | 32.2 (33.1) | +0.04% (-0.15%) | 17562 -> 4841 (9181) | 134665 |
| lud / (b) IWC0 | yes / OK | 83.6 (82.9) | 46.2 (43.3) | 3.3 (3.1) | -0.18% (-0.26%) | 42855 -> 28713 (23751) | 435368 |
| lud / (a) default | yes / OK | 83.9 (82.9) | 42.3 (43.3) | 2.5 (3.1) | -0.47% (-0.26%) | 42855 -> 31473 (23751) | 462515 |
| histo / (b) IWC0 | yes / OK | 30.7 (30.9) | 24.9 (24.9) | 9.1 (9.5) | -1.39% (-1.77%) | 112252 -> 70242 (72110) | 453432 |
| histo / (a) default | yes / OK | 30.8 (30.9) | 24.9 (24.9) | 9.7 (9.5) | -2.59% (-1.77%) | 112252 -> 70288 (72110) | 464530 |
| mri-q / (b) IWC0 | yes / OK | 90.8 (87.1) | 9.5 (9.5) | 0.0 (0.2) | +0.12% (+0.16%) | 28484 -> 25566 (25562) | 2008 |
| mri-q / (a) default | yes / OK | 93.8 (87.1) | 9.5 (9.5) | 0.0 (0.2) | +0.01% (+0.16%) | 28484 -> 25566 (25562) | 2008 |
| srad_v1 / (b) IWC0 | yes / OK | 84.9 (80.6) | 30.0 (29.2) | 6.8 (7.7) | +0.53% (+0.31%) | 84436 -> 40193 (39609) | 121557 |
| srad_v1 / (a) default | yes / OK | 85.9 (80.6) | 31.0 (29.2) | 7.9 (7.7) | +0.37% (+0.31%) | 84436 -> 39008 (39609) | 125384 |

T1 demotions and warps removed, S3c → S3d (a): lud 490 → 6,879 and 1,545 → 21,838; srad_v1 1 → 92 and 37 → 634; mri-q 0 → 2 and 0 → 312; histo and hotspot unchanged.

## 5. Findings

- **The fix reaches every load at a promoted PC1.**
  - Loads at a PC with a promoted entry but no own inter-thread prediction: 0 in every run. In S3c these were the `promoted_nobit_loads`: 15% (hotspot), 40% (lud), 55% (mri-q), 37% (srad_v1), 0.8% (histo) of loads.
  - **Still without an own inter-thread prediction:** 33% (histo) to 57% (lud) of loads. All of them are at PCs with no promoted entry, i.e. no inter-thread stride shared by 3 warps.
- **Verification results.** The no-bit predictions verify well in hotspot (4,240 ok / 0 failed), srad_v1 (34,514 / 607) and mri-q (2,464 / 312). In lud, 59% fail (23,333 ok / 33,549 failed), so lud's T1 demotions rise from 490 to 6,879: its strides differ between warps (cf. backprop, SPEC §8).
- **Effect, (a) vs S3c:**
  - coverage_pred rises: hotspot 87.8 → 90.4, srad_v1 80.6 → 85.9, mri-q 87.1 → 93.8, lud 82.9 → 83.9; histo unchanged at 30.8.
  - coverage_paper: hotspot 71.1 → 75.7, srad_v1 29.2 → 31.0, lud 43.3 → 42.3; mri-q and histo unchanged.
  - IPC: hotspot −0.15 → +0.04%, srad_v1 +0.31 → +0.37%, mri-q +0.16 → +0.01%, lud −0.26 → −0.47%, histo −1.77 → −2.59%.
  - Demand MISS_QUEUE_FULL: hotspot 9,181 → 4,841; lud 23,751 → 31,473; histo and srad_v1 about the same.
- **(b), chain depth 0, against (a):**
  - IPC is slightly better in all five: hotspot +0.12 vs +0.04%, srad_v1 +0.53 vs +0.37%, mri-q +0.12 vs +0.01%, lud −0.18 vs −0.47%, histo −1.39 vs −2.59%.
  - Coverage is about the same. The chain adds coverage_pred only in mri-q (90.8 → 93.8) and srad_v1 (84.9 → 85.9).
  - In lud the chain adds 280k records, 83% of them wrong, and lowers coverage_paper (46.2 with depth 0 vs 42.3).
- **Which source is wrong, which is timely:**

  | app | most wrong, (a) | most wrong, (b) | most timely, (a) | most timely, (b) |
  |---|---|---|---|---|
  | lud 512 | promotion 41.0%, inter-warp chain 34.9% | promotion 53.1% | inter-warp 47.7% | inter-warp 66.9% |
  | histo | inter-warp 53.9%, chain 45.4% | inter-warp 96.5% | inter-warp chain 60.1%, own inter-thread 25.2% | own inter-thread 70.6% |
  | mri-q | inter-warp chain 54.1%, inter-warp 42.1% | inter-warp 96.1% | none timely | none timely |
  | hotspot 512 | inter-warp 54.7%, chain 44.9% | inter-warp 99.5% | inter-warp 53.8%, chain 46.2% | inter-warp 100% |
  | srad_v1 | inter-warp 64.4%, chain 30.8% | inter-warp 95.7% | chain 62.2%, inter-warp 37.3% | inter-warp 99.4% |

  - **Own-warp inter-thread predictions are rarely wrong (≤ 3%) except in lud (81%) and mri-q (15%), but almost never timely** outside histo: hotspot 87% late and 0% timely; srad_v1 41% never issued in time and 42% duplicates in L1.
  - **Timely prefetches come from the sources that run ahead:** inter-warp and its chain.
  - **lud's wrong predictions** come from inter-thread strides that 3 warps share but the others do not (promotion, then the no-bit path).
