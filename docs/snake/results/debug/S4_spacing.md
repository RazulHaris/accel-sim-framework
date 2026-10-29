# Debug plan Stage 4: what the programs allow (2026-10-29)

**Status:** done, no new runs. Sources:
- The Stage 3 logs (`sim_run_snake_debug/s3`, SMs 0, 16, 32, 48, 64, Tier 1 Snake).
- For 4d, the existing D1 perfect-L1 runs (`sim_run_snake_gate/D1_pl1d`) against `sim_run_baseline`.

Script: `util/snake/debug/s4_spacing.py`. Yardstick: each run's median demand miss latency (a new L1D miss to its data leaving the L1D).

## TLDR

- **4a (H3 holds on 5 of 7 apps): a warp issues its loads back to back.**
  - The median gap between a warp's consecutive loads is 2-12 cycles everywhere except histo (363).
  - The share of gaps at least one miss latency long is 0-13% on backprop, hotspot, nw, mri-q and srad_v1.
  - A chain prefetch for PC_i+1 triggered at PC_i therefore cannot be timely there.
  - The exceptions are lud (41%, bimodal gaps: back-to-back loads, then about 500-cycle phases) and histo (53%).
- **4b (H4 holds): the next few warps reach a load PC almost together, often in reverse order.**
  - For warps w+1 to w+7 of the same CTA (Snake's inter-warp targets are the next 4), the median gap is 1-184 cycles.
  - 0-5% of these gaps reach the miss latency on backprop, lud and mri-q; 15-41% on srad_v1, histo and hotspot.
  - Only warps further away (8-15 in the same CTA, or in other CTAs) have the lead: median 115-1,013 cycles, 28-87% of gaps reaching the latency on srad_v1, histo and hotspot.
- **4c:** loads with a constant intra-warp stride appear only in histo (median iteration 272 cycles, 42% at least the miss latency) and srad_v1 (one PC, 643 cycles).
- **4d: the headroom sits in kernels where Snake issues almost nothing:**
  - backprop: `adjust_weights` is 54% of the baseline cycles with perfect-L1 +120%. There 82% of Snake's prediction records are never issued and 0% are timely.
  - srad_v1: `srad2` (+47%) has 89% not issued; `reduce` (+24%) 56%.
  - histo: `intermediates` (+131%) has 5% timely and 34% late; prescan's records go 76% unused.
  - nw: both kernels (+16-17%) have no prediction at all (S5 item 11).

## Summary across workloads

Gap = cycles; "≥ latency" = share of gaps at least the run's median miss latency (%). "–": no pairs of that kind in the log.

| workload | median miss latency | 4a gap p50 | 4a gaps >= latency (%) | 4b d=1 p50 (same CTA) | >= latency (%) | 4b d=4-7 p50 | >= latency (%) | 4b d=8-15 p50 | >= latency (%) | 4b other CTA p50 | >= latency (%) | 4c iteration p50 | >= latency (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| backprop | 1227 | 8 | 12.6 | 2 | 0.0 | 13 | 0.0 | – | – | 113 | 0.0 | – | – |
| hotspot 1024 | 220 | 2 | 0.0 | -80 | 30.6 | 184 | 40.8 | – | – | 488 | 68.8 | – | – |
| hotspot 512_ | 419 | 2 | 0.0 | 5 | 20.7 | 183 | 19.8 | – | – | – | – | – | – |
| lud _i__ | 322 | 12 | 41.2 | 2 | 1.5 | 53 | 5.0 | – | – | 200 | 25.5 | – | – |
| lud _s_2 | 249 | 10 | 40.6 | 2 | 0.2 | 16 | 0.1 | – | – | – | – | – | – |
| nw | 372 | 7 | 11.0 | – | – | – | – | – | – | – | – | – | – |
| histo | 335 | 363 | 53.0 | 9 | 11.0 | 63 | 21.6 | 115 | 28.3 | – | – | 272 | 42.0 |
| mri-q | 401 | 5 | 0.0 | 1 | 0.0 | -1 | 0.0 | 33 | 0.0 | – | – | – | – |
| srad_v1 | 553 | 9 | 4.9 | 7 | 3.3 | 132 | 14.5 | 333 | 35.7 | 1013 | 87.4 | 643 | 100.0 |

## 4d Per-kernel perfect-L1 ceilings (D1) and Snake's funnel per kernel (Stage 3 log)

Kernels with at least 1% of the baseline cycles; the funnel is in % of the kernel's prediction records. "Not issued" = queue full, still queued at the demand, already in L1 or in flight, or no resource. "No demand" = no demand within the 5,000-cycle window.

### backprop 65536

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z24bpnn_adjust_weights_cudaPfiS_i | 1 | 53.7 | 120.0 | 35257 | 0.0 | 5.5 | 0.0 | 82.0 | 12.5 |
| _Z22bpnn_layerforward_CUDAPfS_S_S_ | 1 | 46.3 | 0.3 | 9328 | 0.8 | 59.7 | 0.0 | 37.4 | 2.1 |

### nw 2048_10

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z20needle_cuda_shared_1PiS_iiii | 128 | 50.1 | 16.3 | 0 | n/a | n/a | n/a | n/a | n/a |
| _Z20needle_cuda_shared_2PiS_iiii | 127 | 49.9 | 16.9 | 0 | n/a | n/a | n/a | n/a | n/a |

### histo _i___data_de

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z17histo_main_kernelP6uchar4jjjjj | 9 | 66.1 | 23.5 | 367247 | 21.8 | 15.1 | 0.3 | 12.2 | 50.6 |
| _Z26histo_intermediates_kernelP5ui | 9 | 17.1 | 131.1 | 106696 | 4.9 | 33.6 | 0.0 | 22.7 | 38.8 |
| _Z18histo_final_kerneljjjjPjS_S_S_ | 9 | 10.2 | 71.5 | 83357 | 12.6 | 26.3 | 0.1 | 23.1 | 38.0 |
| _Z20histo_prescan_kernelPjiS_ | 10 | 6.5 | 28.7 | 85893 | 1.4 | 8.8 | 0.0 | 14.2 | 75.6 |

### srad_v1 100_0_5_502_

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z6reduceliiPfS_ | 20 | 46.2 | 24.1 | 41758 | 6.4 | 22.1 | 0.0 | 55.8 | 15.7 |
| _Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_ | 9 | 23.1 | 6.8 | 132938 | 7.4 | 11.5 | 0.0 | 67.7 | 13.3 |
| _Z5srad2fiilPiS_S_S_PfS0_S0_S0_S0_ | 9 | 17.3 | 47.0 | 150284 | 0.1 | 2.2 | 0.0 | 88.9 | 8.8 |
| _Z7preparelPfS_S_ | 10 | 12.1 | 10.7 | 75703 | 0.0 | 7.6 | 0.0 | 36.9 | 55.5 |
| _Z7extractlPf | 1 | 1.3 | 6.9 | 1712 | 0.2 | 54.8 | 0.0 | 44.9 | 0.0 |

## Per workload

Runs: `sim_run_snake_debug/s3/runs`, config `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SNAKE_LC16` (Tier 1 Snake, PAPER_V100, 1B; log on SMs 0, 16, 32, 48, 64).

### Workload: backprop 65536

Median demand miss latency (new L1D miss to data back, these SMs): **1227** cycles (n = 20959).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 2 _Z24bpnn_adjust_weights_cudaPf | 1 | 2 | 7 | 14 | 1508 | 12384 | 14.3 |
| 1 _Z22bpnn_layerforward_CUDAPfS_ | 207 | 363 | 467 | 570 | 775 | 2040 | 2.0 |
| **all kernels** | 2 | 3 | 8 | 363 | 1419 | 14424 | 12.6 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -59 | -13 | 2 | 18 | 78 | 16212 | 0.0 |
| 1 | no | -39 | -15 | 13 | 25 | 27 | 7 | 0.0 |
| 2-3 | yes | -95 | -14 | 5 | 34 | 184 | 25476 | 0.0 |
| 2-3 | no | -10 | 12 | 38 | 438 | 444 | 35 | 0.0 |
| 4-7 | yes | -62 | -6 | 13 | 60 | 251 | 23160 | 0.0 |
| 4-7 | no | 12 | 31 | 56 | 444 | 462 | 154 | 0.0 |
| 8-15 | no | 42 | 63 | 80 | 475 | 492 | 252 | 0.0 |
| 16-31 | no | -26 | 30 | 70 | 124 | 305 | 1988 | 0.0 |
| 32-63 | no | -26 | 66 | 113 | 174 | 391 | 2508 | 0.0 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

#### 4d Per-kernel perfect-L1 ceiling (D1) and Snake's funnel (Stage 3 log, % of records)

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z24bpnn_adjust_weights_cudaPfiS_i | 1 | 53.7 | 120.0 | 35257 | 0.0 | 5.5 | 0.0 | 82.0 | 12.5 |
| _Z22bpnn_layerforward_CUDAPfS_S_S_ | 1 | 46.3 | 0.3 | 9328 | 0.8 | 59.7 | 0.0 | 37.4 | 2.1 |

### Workload: hotspot 1024_2_2___d

Median demand miss latency (new L1D miss to data back, these SMs): **220** cycles (n = 13061).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 1 _Z14calculate_tempiPfS_S_iiiif | 2 | 2 | 2 | 4 | 8 | 3665 | 0.0 |
| **all kernels** | 2 | 2 | 2 | 4 | 8 | 3665 | 0.0 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -1561 | -736 | -80 | 900 | 1865 | 6406 | 30.6 |
| 2-3 | yes | -1785 | -1327 | 34 | 1441 | 1844 | 10040 | 42.6 |
| 4-7 | yes | -1368 | 32 | 184 | 344 | 1859 | 9092 | 40.8 |
| 16-31 | no | -497 | 109 | 488 | 1062 | 1420 | 128 | 68.8 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

### Workload: hotspot 512_2_2___da

Median demand miss latency (new L1D miss to data back, these SMs): **419** cycles (n = 4189).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 1 _Z14calculate_tempiPfS_S_iiiif | 2 | 2 | 2 | 4 | 8 | 908 | 0.0 |
| **all kernels** | 2 | 2 | 2 | 4 | 8 | 908 | 0.0 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -1088 | -176 | 5 | 281 | 1180 | 1586 | 20.7 |
| 2-3 | yes | -1135 | -249 | 34 | 612 | 1352 | 2482 | 28.8 |
| 4-7 | yes | -526 | 66 | 183 | 330 | 1316 | 2222 | 19.8 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

### Workload: lud _i___data_51

Median demand miss latency (new L1D miss to data back, these SMs): **322** cycles (n = 42243).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 3 _Z12lud_internalPfii | 2 | 4 | 265 | 885 | 1341 | 944 | 49.5 |
| 6 _Z12lud_internalPfii | 3 | 6 | 193 | 502 | 731 | 928 | 43.8 |
| 9 _Z12lud_internalPfii | 2 | 5 | 239 | 686 | 1036 | 800 | 46.6 |
| 12 _Z12lud_internalPfii | 2 | 6 | 221 | 633 | 897 | 752 | 47.3 |
| 15 _Z12lud_internalPfii | 2 | 5 | 214 | 656 | 968 | 704 | 46.9 |
| 18 _Z12lud_internalPfii | 2 | 6 | 227 | 617 | 843 | 704 | 46.9 |
| **all kernels** | 2 | 4 | 12 | 571 | 878 | 11726 | 41.2 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -43 | -9 | 2 | 22 | 75 | 13587 | 1.5 |
| 2-3 | yes | -33 | -4 | 14 | 58 | 125 | 21351 | 2.8 |
| 4-7 | yes | -9 | 18 | 53 | 89 | 198 | 19410 | 5.0 |
| 16-31 | no | 17 | 58 | 221 | 274 | 296 | 84 | 8.3 |
| 32-63 | no | 58 | 98 | 200 | 323 | 449 | 876 | 25.5 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

### Workload: lud _s_256__v

Median demand miss latency (new L1D miss to data back, these SMs): **249** cycles (n = 5282).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 3 _Z12lud_internalPfii | 2 | 4 | 318 | 615 | 721 | 240 | 50.0 |
| 6 _Z12lud_internalPfii | 2 | 6 | 303 | 511 | 659 | 208 | 50.0 |
| 9 _Z12lud_internalPfii | 2 | 4 | 294 | 481 | 609 | 160 | 50.0 |
| 12 _Z12lud_internalPfii | 2 | 6 | 255 | 438 | 549 | 144 | 50.0 |
| 15 _Z12lud_internalPfii | 2 | 6 | 262 | 365 | 503 | 112 | 50.0 |
| 18 _Z12lud_internalPfii | 2 | 6 | 245 | 336 | 423 | 96 | 49.0 |
| **all kernels** | 2 | 4 | 10 | 375 | 554 | 1529 | 40.6 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -22 | -6 | 2 | 6 | 32 | 1659 | 0.2 |
| 2-3 | yes | -49 | -12 | 4 | 23 | 59 | 2607 | 0.1 |
| 4-7 | yes | -63 | -14 | 16 | 47 | 74 | 2370 | 0.1 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

### Workload: nw 2048_10

Median demand miss latency (new L1D miss to data back, these SMs): **372** cycles (n = 66560).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 116 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 458 | 144 | 11.1 |
| 119 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 449 | 144 | 11.1 |
| 121 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 474 | 144 | 11.1 |
| 123 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 441 | 144 | 11.1 |
| 124 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 401 | 144 | 11.1 |
| 125 _Z20needle_cuda_shared_1PiS_ii | 4 | 7 | 7 | 7 | 475 | 144 | 11.1 |
| **all kernels** | 4 | 7 | 7 | 7 | 415 | 18432 | 11.0 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

#### 4d Per-kernel perfect-L1 ceiling (D1) and Snake's funnel (Stage 3 log, % of records)

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z20needle_cuda_shared_1PiS_iiii | 128 | 50.1 | 16.3 | 0 | n/a | n/a | n/a | n/a | n/a |
| _Z20needle_cuda_shared_2PiS_iiii | 127 | 49.9 | 16.9 | 0 | n/a | n/a | n/a | n/a | n/a |

### Workload: histo _i___data_de

Median demand miss latency (new L1D miss to data back, these SMs): **335** cycles (n = 160061).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 15 _Z17histo_main_kernelP6uchar4j | 65 | 188 | 315 | 494 | 685 | 9234 | 48.1 |
| 27 _Z17histo_main_kernelP6uchar4j | 65 | 188 | 317 | 500 | 695 | 9216 | 48.6 |
| 11 _Z17histo_main_kernelP6uchar4j | 65 | 193 | 316 | 506 | 688 | 9144 | 48.0 |
| 23 _Z17histo_main_kernelP6uchar4j | 65 | 193 | 322 | 494 | 678 | 9144 | 48.6 |
| 35 _Z17histo_main_kernelP6uchar4j | 67 | 198 | 305 | 490 | 687 | 9144 | 47.1 |
| 31 _Z17histo_main_kernelP6uchar4j | 79 | 241 | 406 | 588 | 750 | 6888 | 61.1 |
| **all kernels** | 60 | 191 | 363 | 587 | 842 | 93728 | 53.0 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -325 | -72 | 9 | 105 | 363 | 23694 | 11.0 |
| 2-3 | yes | -402 | -113 | 23 | 193 | 496 | 42838 | 16.0 |
| 4-7 | yes | -403 | -87 | 63 | 281 | 626 | 67515 | 21.6 |
| 8-15 | yes | -336 | -32 | 115 | 387 | 747 | 62411 | 28.3 |
| 16-31 | yes | -89 | 43 | 110 | 300 | 625 | 4464 | 22.4 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| 58 | 114 | 272 | 893 | 1164 | 59682 | 42.0 |

#### 4d Per-kernel perfect-L1 ceiling (D1) and Snake's funnel (Stage 3 log, % of records)

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z17histo_main_kernelP6uchar4jjjjj | 9 | 66.1 | 23.5 | 367247 | 21.8 | 15.1 | 0.3 | 12.2 | 50.6 |
| _Z26histo_intermediates_kernelP5ui | 9 | 17.1 | 131.1 | 106696 | 4.9 | 33.6 | 0.0 | 22.7 | 38.8 |
| _Z18histo_final_kerneljjjjPjS_S_S_ | 9 | 10.2 | 71.5 | 83357 | 12.6 | 26.3 | 0.1 | 23.1 | 38.0 |
| _Z20histo_prescan_kernelPjiS_ | 10 | 6.5 | 28.7 | 85893 | 1.4 | 8.8 | 0.0 | 14.2 | 75.6 |

### Workload: mri-q _i___data_sm

Median demand miss latency (new L1D miss to data back, these SMs): **401** cycles (n = 3586).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 2 _Z12ComputeQ_GPUiiPfS_S_S_S_ | 2 | 4 | 6 | 30 | 57 | 256 | 0.0 |
| 3 _Z12ComputeQ_GPUiiPfS_S_S_S_ | 1 | 3 | 5 | 13 | 56 | 256 | 0.0 |
| 4 _Z12ComputeQ_GPUiiPfS_S_S_S_ | 1 | 3 | 5 | 14 | 52 | 256 | 0.0 |
| 1 _Z17ComputePhiMag_GPUPfS_S_i | 2 | 2 | 5 | 41 | 42 | 16 | 0.0 |
| **all kernels** | 1 | 4 | 5 | 14 | 55 | 784 | 0.0 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -70 | -12 | 1 | 7 | 73 | 870 | 0.0 |
| 2-3 | yes | -90 | -49 | 1 | 45 | 103 | 1374 | 0.0 |
| 4-7 | yes | -122 | -65 | -1 | 72 | 137 | 1284 | 0.0 |
| 8-15 | yes | -14 | 8 | 33 | 47 | 73 | 72 | 0.0 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| n/a | | | | | 0 | |

### Workload: srad_v1 100_0_5_502_

Median demand miss latency (new L1D miss to data back, these SMs): **553** cycles (n = 181413).

#### 4a Load spacing in a warp (consecutive observed loads, cycles)

| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|
| 6 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 1 | 4 | 11 | 26 | 200 | 4320 | 4.7 |
| 11 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 1 | 4 | 10 | 26 | 170 | 4320 | 4.2 |
| 16 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 1 | 4 | 10 | 24 | 151 | 4320 | 3.5 |
| 31 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 2 | 4 | 10 | 25 | 201 | 4176 | 4.9 |
| 36 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 2 | 4 | 10 | 28 | 153 | 4032 | 4.4 |
| 41 _Z5srad2fiilPiS_S_S_PfS0_S0_S0 | 1 | 4 | 10 | 25 | 154 | 4032 | 4.7 |
| **all kernels** | 2 | 3 | 9 | 24 | 331 | 79650 | 4.9 |

#### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w

| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | yes | -115 | -26 | 7 | 77 | 252 | 90885 | 3.3 |
| 2-3 | yes | -76 | -4 | 42 | 156 | 428 | 163569 | 6.6 |
| 4-7 | yes | 4 | 43 | 132 | 354 | 658 | 254262 | 14.5 |
| 8-15 | yes | 51 | 131 | 333 | 705 | 967 | 217748 | 35.7 |
| 32-63 | no | 423 | 875 | 1013 | 1129 | 1449 | 2560 | 87.4 |

#### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)

| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |
|---|---|---|---|---|---|---|
| 643 | 643 | 643 | 643 | 643 | 352 | 100.0 |

#### 4d Per-kernel perfect-L1 ceiling (D1) and Snake's funnel (Stage 3 log, % of records)

| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | issued, unused or other | not issued | no demand |
|---|---|---|---|---|---|---|---|---|---|
| _Z6reduceliiPfS_ | 20 | 46.2 | 24.1 | 41758 | 6.4 | 22.1 | 0.0 | 55.8 | 15.7 |
| _Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_ | 9 | 23.1 | 6.8 | 132938 | 7.4 | 11.5 | 0.0 | 67.7 | 13.3 |
| _Z5srad2fiilPiS_S_S_PfS0_S0_S0_S0_ | 9 | 17.3 | 47.0 | 150284 | 0.1 | 2.2 | 0.0 | 88.9 | 8.8 |
| _Z7preparelPfS_S_ | 10 | 12.1 | 10.7 | 75703 | 0.0 | 7.6 | 0.0 | 36.9 | 55.5 |
| _Z7extractlPf | 1 | 1.3 | 6.9 | 1712 | 0.2 | 54.8 | 0.0 | 44.9 | 0.0 |

