# Debug plan Stage 3: prefetch lifecycle trace (2026-10-29)

**Status:** done.
- **Runs:** 9 Tier 1 runs (PAPER_V100, 1B, `-SNAKE-SNAKE_FP1-SNAKE_LC16`) in `sim_run_snake_debug/s3` (20:49-21:11, all exit 0). Build accel-sim `ec13114` + gpgpu-sim `1558559d`.
- **Logging:** SMs 0, 16, 32, 48 and 64 (`-snake_lc_stride 16`, default 0 = off); 58 MB of gzipped logs in all.
- **Checks:**
  - Gates DBG3b: Snake-off byte-identical on the 6 check workloads; synthetic suite with Fig. 15 passed.
  - Logging on vs off: identical output on all 11 synthetic runs.
  - All 9 runs reproduce the original Tier 1 runs exactly: every one of the 324 counters both print, and IPC, are equal.
- **Analysis:** `util/snake/debug/s3_lifecycle.py`; the event format is in its docstring.

## TLDR

- **3a (H2 confirmed): most predictions never become prefetches, and of those that do, most are late.**
  - Of the prediction records a demand later matched (the records behind coverage_pred), 62% are never issued (mean of the 6 apps that predict; nw makes no predictions):
    - 21% dropped because the 32-line queue is full;
    - 26% still queued when their demand arrives (38% of those while throttled);
    - 6% already in L1 or in flight;
    - 9% find no line, MSHR or miss-queue slot (mostly the 8-slot miss-queue reserve).
  - The 38% issued split into 11% timely, 26% late and 1% evicted first.
  - So "63% predicted, 26% issued" is mostly queue capacity and queue wait, then the reserves.
- **3b: timeliness is out of reach by construction for most predictions.**
  - The median lead from prediction to its demand is 12-30 cycles for most sources and apps (inter-warp: 82-198 on histo and hotspot).
  - The median prefetch latency is 175-2,596 cycles.
  - Queue wait is secondary (median 3-76 cycles), except mri-q (113-322) and srad_v1's promotion and intra-warp predictions (371-599).
- **3c (H7 rejected): a demand merged into an in-flight prefetch does not wait longer because of the merge.**
  - After the prefetch's fill, its data comes back in 4-10 cycles (median; p90 12-46, max 186). That is the L1D response pop, which the prefetch's own response uses first (S5 item 10).
  - The merged demand's medians are below the new misses' in every kernel but one: backprop's `adjust_weights`, 1,382 vs 1,369 cycles. There the prefetches it merges into go to DRAM (96% L2 misses) and were issued only cycles earlier, while new misses include L2 hits.

## Where matched predictions end (3a, records closed by a demand, %)

Mean over an app's inputs; from the per-workload record tables below.

| app | matched (% of records) | not issued: queue full | not issued: still queued at demand | not issued: in L1 or in flight | not issued: no line / MSHR / miss-queue slot | issued: timely | issued: late | issued: evicted first | throttled share of still-queued |
|---|---|---|---|---|---|---|---|---|---|
| backprop | 89.8 | 16.0 | 39.4 | 19.4 | 6.2 | 0.2 | 18.7 | 0.0 | 26.6 |
| hotspot | 76.0 | 12.6 | 5.7 | 0.5 | 6.4 | 28.0 | 42.3 | 4.4 | 70.1 |
| lud | 56.6 | 19.9 | 27.3 | 1.1 | 11.7 | 2.2 | 37.8 | 0.0 | 31.7 |
| histo | 49.7 | 4.6 | 8.2 | 4.0 | 14.7 | 30.2 | 37.8 | 0.4 | 29.3 |
| mri-q | 69.2 | 25.7 | 53.9 | 0.0 | 9.4 | 0.0 | 11.0 | 0.0 | 31.6 |
| srad_v1 | 80.3 | 45.2 | 23.7 | 12.3 | 4.2 | 3.9 | 10.7 | 0.0 | 38.9 |
| **mean, 6 apps** | 70.3 | 20.7 | 26.4 | 6.2 | 8.8 | 10.7 | 26.4 | 0.8 | 38.0 |

## Lead time, queue wait and memory latency (3b, medians in cycles)

| workload | source | available lead (demand − latest prediction) | queue wait (offered − queued) | memory latency (filled − issued) | n (lead) |
|---|---|---|---|---|---|
| backprop 65536 | inter-thread | 12 | 76 | 2596 | 1311 |
| backprop 65536 | promotion | 695 | 78 | 1110 | 1043 |
| backprop 65536 | inter-warp | 18 | 6 | 356 | 37637 |
| hotspot 1024_2 | inter-thread | 23 | 4 | 186 | 23884 |
| hotspot 1024_2 | intra-warp | 22 | 36 | 191 | 774 |
| hotspot 1024_2 | promotion | 9 | 29 | 176 | 30 |
| hotspot 1024_2 | inter-warp | 198 | 20 | 316 | 16345 |
| hotspot 512_2_ | inter-thread | 24 | 3 | 240 | 4556 |
| hotspot 512_2_ | inter-warp | 95 | 17 | 320 | 5144 |
| lud _i___data_ | inter-thread | 25 | 12 | 256 | 6361 |
| lud _i___data_ | promotion | 28 | 24 | 263 | 4340 |
| lud _i___data_ | inter-warp | 20 | 13 | 201 | 42801 |
| lud _s_256__v | inter-thread | 24 | 11 | 247 | 675 |
| lud _s_256__v | promotion | 26 | 10 | 181 | 602 |
| lud _s_256__v | inter-warp | 18 | 13 | 175 | 5055 |
| histo _i___dat | inter-thread | 132 | 11 | 215 | 106939 |
| histo _i___dat | intra-warp | 71 | 7 | 1026 | 4 |
| histo _i___dat | promotion | 161 | 16 | 700 | 18398 |
| histo _i___dat | inter-warp | 82 | 15 | 378 | 196180 |
| mri-q _i___dat | inter-thread | 49 | 322 | 221 | 996 |
| mri-q _i___dat | promotion | 38 | 178 | 201 | 185 |
| mri-q _i___dat | inter-warp | 30 | 113 | 184 | 2306 |
| srad | inter-thread | 24 | 18 | 191 | 117323 |
| srad | intra-warp | 27 | 371 | – | 599 |
| srad | promotion | 27 | 599 | 408 | 8274 |
| srad | inter-warp | 22 | 11 | 186 | 201469 |

Notes on reading the per-workload tables:
- **Record vs sector level.**
  - A prediction record is one predicted sector within the OQ-14 window. A later prediction of the same sector updates the record; it does not add one.
  - The sector-level table counts every predicted sector, repeats included. That is why many offers are DUP_L1 (a repeat of a sector already prefetched) while the record table shows almost no "in L1".
- **DUP_L1 covers sectors that are valid or reserved** (a line waiting for its fill). The split between them is in S5_audit.md item 8.
- **"unknown" source:** an offer or fill whose queue entry was created before the sector's prediction event could be tied to it (0-4% of offers; srad_v1 4.1%, lud 2.2-2.9%).
- **3c "merged into demand":** a second demand merging into another demand's MSHR entry.

Runs: `sim_run_snake_debug/s3/runs`, config `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SNAKE_LC16`. The log covers SMs 0, 16, 32, 48 and 64; the whole-run line is all 80 SMs.

## Workload: backprop 65536

Log events: 243066. Whole run: predicted sectors per demand sector 0.73, coverage_pred 64.6%, issued per demand 0.12, timely + late 12.4%, timely 0.2%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 1419 | 10.8 | 11.2 | 29.4 | 33.8 | 0.0 | 3.2 | 0.0 | 11.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| promotion | 2047 | 12.4 | 4.7 | 14.5 | 57.5 | 0.0 | 0.5 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 10.2 |
| inter-warp | 41119 | 14.6 | 9.5 | 26.5 | 14.8 | 0.0 | 5.9 | 0.2 | 17.8 | 0.0 | 0.0 | 0.0 | 10.7 |
| **all** | 44585 | 14.4 | 9.4 | 26.0 | 17.4 | 0.0 | 5.6 | 0.2 | 16.8 | 0.0 | 0.0 | 0.0 | 10.3 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 2555 | 1051 | 367 | 1137 | 1989 | 178 | 1719 | 0 | 0 | 0 | 92 | 178 | 100.0% | 0.0% |
| promotion | 3145 | 351 | 703 | 2091 | 2210 | 6 | 2179 | 0 | 0 | 0 | 25 | 6 | 100.0% | 0.0% |
| inter-warp | 112707 | 50425 | 17096 | 45186 | 42146 | 7381 | 30085 | 0 | 0 | 0 | 4680 | 7374 | 98.8% | 0.9% |
| **all** | 118407 | 51827 | 18166 | 48414 | 47171 | 7577 | 34797 | 0 | 0 | 0 | 4797 | 7577 | 98.9% | 1.1% |

Unused prefetched sectors lost: evicted 0, invalid at refill 0, at a kernel end 0. Dropped from the queue: displaced 1243, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 2 | 5 | 43 | 117 | 305 | 826 |
| unknown | memory latency (filled - issued) | 173 | 373 | 549 | 713 | 737 | 19 |
| inter-thread | available lead (demand - latest prediction) | 2 | 5 | 12 | 37 | 1605 | 1311 |
| inter-thread | queue wait (offered - queue entry created) | 6 | 36 | 76 | 206 | 370 | 1989 |
| inter-thread | memory latency (filled - issued) | 1043 | 1255 | 2596 | 3154 | 3460 | 178 |
| promotion | available lead (demand - latest prediction) | 55 | 272 | 695 | 1385 | 2440 | 1043 |
| promotion | queue wait (offered - queue entry created) | 33 | 51 | 78 | 236 | 343 | 2210 |
| promotion | memory latency (filled - issued) | 825 | 1108 | 1110 | 1111 | 1936 | 6 |
| inter-warp | available lead (demand - latest prediction) | 4 | 9 | 18 | 26 | 58 | 37637 |
| inter-warp | queue wait (offered - queue entry created) | 2 | 3 | 6 | 62 | 279 | 42146 |
| inter-warp | memory latency (filled - issued) | 305 | 321 | 356 | 530 | 1718 | 7374 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 2 _Z24bpnn_adjust_weights_cuda | new miss | 16913 | 844 | 1369 | 3093 | 1659 | 4621 | – |
| 2 _Z24bpnn_adjust_weights_cuda | merged into prefetch | 2513 | 699 | 1382 | 3237 | 1779 | 4617 | 15.0% |
| 2 _Z24bpnn_adjust_weights_cuda | merged into demand | 3875 | 778 | 1307 | 3107 | 1597 | 4307 | 10.2% |
| 1 _Z22bpnn_layerforward_CUDAPf | new miss | 4046 | 323 | 360 | 430 | 373 | 756 | – |
| 1 _Z22bpnn_layerforward_CUDAPf | merged into prefetch | 7330 | 231 | 324 | 396 | 322 | 741 | 6.8% |
| 1 _Z22bpnn_layerforward_CUDAPf | merged into demand | 647 | 89 | 330 | 407 | 292 | 704 | 8.7% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 175 | 335 | 698 | 5 | 9 | 15 |
| miss | 9668 | 335 | 1962 | 5 | 15 | 77 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians **higher** in kernel 2 (1382 vs 1369); wake-up after the fill p50 5, p90 15, max 77 cycles.

## Workload: hotspot 1024_2_2___d

Log events: 375364. Whole run: predicted sectors per demand sector 1.50, coverage_pred 93.3%, issued per demand 0.85, timely + late 69.1%, timely 27.0%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 4614 | 4.0 | 14.1 | 1.6 | 0.2 | 0.0 | 5.3 | 0.0 | 73.2 | 0.0 | 0.0 | 0.3 | 1.3 |
| intra-warp | 428 | 0.0 | 18.9 | 0.9 | 0.0 | 0.0 | 13.1 | 0.0 | 66.6 | 0.0 | 0.0 | 0.5 | 0.0 |
| promotion | 212 | 2.8 | 0.0 | 2.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.8 | 91.5 |
| inter-warp | 57768 | 5.4 | 2.0 | 0.7 | 0.8 | 0.0 | 4.4 | 19.8 | 26.1 | 3.4 | 0.0 | 0.3 | 37.3 |
| **all** | 63022 | 5.2 | 3.0 | 0.8 | 0.7 | 0.0 | 4.5 | 18.1 | 29.7 | 3.1 | 0.0 | 0.3 | 34.6 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 30874 | 6123 | 771 | 23980 | 27070 | 11408 | 14785 | 0 | 0 | 0 | 877 | 11235 | 99.1% | 57.0% |
| intra-warp | 3204 | 880 | 24 | 2300 | 1082 | 258 | 800 | 0 | 0 | 0 | 24 | 258 | 95.0% | 50.8% |
| promotion | 918 | 0 | 664 | 254 | 254 | 40 | 212 | 0 | 0 | 0 | 2 | 40 | 0.0% | 100.0% |
| inter-warp | 120822 | 4789 | 26605 | 89428 | 85087 | 24756 | 50871 | 0 | 0 | 0 | 9460 | 24185 | 28.6% | 32.5% |
| **all** | 155818 | 11792 | 28064 | 115962 | 113969 | 36652 | 66887 | 0 | 0 | 0 | 10430 | 36652 | 51.1% | 40.3% |

Unused prefetched sectors lost: evicted 6324, invalid at refill 0, at a kernel end 145. Dropped from the queue: displaced 1993, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 13 | 30 | 65 | 84 | 134 | 476 |
| unknown | memory latency (filled - issued) | 170 | 179 | 246 | 348 | 418 | 934 |
| inter-thread | available lead (demand - latest prediction) | 19 | 21 | 23 | 24 | 28 | 23884 |
| inter-thread | queue wait (offered - queue entry created) | 2 | 3 | 4 | 9 | 46 | 27070 |
| inter-thread | memory latency (filled - issued) | 167 | 170 | 186 | 331 | 398 | 11235 |
| intra-warp | available lead (demand - latest prediction) | 21 | 21 | 22 | 22 | 22 | 774 |
| intra-warp | queue wait (offered - queue entry created) | 5 | 9 | 36 | 53 | 74 | 1082 |
| intra-warp | memory latency (filled - issued) | 169 | 172 | 191 | 355 | 428 | 258 |
| promotion | available lead (demand - latest prediction) | 2 | 4 | 9 | 11 | 13 | 30 |
| promotion | queue wait (offered - queue entry created) | 7 | 15 | 29 | 40 | 48 | 254 |
| promotion | memory latency (filled - issued) | 167 | 169 | 176 | 180 | 182 | 40 |
| inter-warp | available lead (demand - latest prediction) | 12 | 37 | 198 | 427 | 1630 | 16345 |
| inter-warp | queue wait (offered - queue entry created) | 7 | 11 | 20 | 47 | 78 | 85087 |
| inter-warp | memory latency (filled - issued) | 171 | 185 | 316 | 380 | 431 | 24185 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 1 _Z14calculate_tempiPfS_S_iii | new miss | 13061 | 173 | 220 | 571 | 311 | 1002 | – |
| 1 _Z14calculate_tempiPfS_S_iii | merged into prefetch | 18747 | 129 | 185 | 372 | 225 | 905 | 0.5% |
| 1 _Z14calculate_tempiPfS_S_iii | merged into demand | 2 | 62 | 63 | 63 | 62 | 63 | 0.0% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 7573 | 150 | 171 | 5 | 13 | 49 |
| miss | 11174 | 279 | 394 | 4 | 11 | 79 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 4, p90 12, max 79 cycles.

## Workload: hotspot 512_2_2___da

Log events: 81355. Whole run: predicted sectors per demand sector 1.36, coverage_pred 90.6%, issued per demand 0.70, timely + late 59.5%, timely 26.4%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 1146 | 15.2 | 13.4 | 1.0 | 0.0 | 0.0 | 3.5 | 0.0 | 66.3 | 0.0 | 0.0 | 0.3 | 0.2 |
| inter-warp | 10014 | 14.9 | 1.8 | 2.0 | 0.0 | 0.0 | 5.4 | 27.3 | 30.2 | 3.9 | 0.0 | 0.2 | 14.3 |
| **all** | 11160 | 14.9 | 3.0 | 1.9 | 0.0 | 0.0 | 5.2 | 24.5 | 33.9 | 3.5 | 0.0 | 0.2 | 12.9 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 5312 | 478 | 639 | 4195 | 4671 | 2038 | 2486 | 0 | 0 | 0 | 147 | 2010 | 98.2% | 45.9% |
| inter-warp | 29868 | 2064 | 7705 | 20099 | 19373 | 5879 | 11376 | 0 | 0 | 0 | 2118 | 5769 | 30.2% | 31.9% |
| **all** | 35180 | 2542 | 8344 | 24294 | 24106 | 7943 | 13894 | 0 | 0 | 0 | 2269 | 7943 | 47.7% | 35.2% |

Unused prefetched sectors lost: evicted 1278, invalid at refill 0, at a kernel end 146. Dropped from the queue: displaced 188, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 33 | 37 | 57 | 93 | 130 | 62 |
| unknown | memory latency (filled - issued) | 174 | 203 | 283 | 354 | 387 | 164 |
| inter-thread | available lead (demand - latest prediction) | 23 | 23 | 24 | 26 | 30 | 4556 |
| inter-thread | queue wait (offered - queue entry created) | 2 | 2 | 3 | 7 | 55 | 4671 |
| inter-thread | memory latency (filled - issued) | 167 | 171 | 240 | 370 | 474 | 2010 |
| inter-warp | available lead (demand - latest prediction) | 10 | 23 | 95 | 301 | 992 | 5144 |
| inter-warp | queue wait (offered - queue entry created) | 6 | 10 | 17 | 38 | 73 | 19373 |
| inter-warp | memory latency (filled - issued) | 171 | 186 | 320 | 379 | 432 | 5769 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 1 _Z14calculate_tempiPfS_S_iii | new miss | 4189 | 180 | 419 | 705 | 427 | 1069 | – |
| 1 _Z14calculate_tempiPfS_S_iii | merged into prefetch | 3788 | 90 | 198 | 430 | 248 | 1092 | 2.1% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 1238 | 148 | 175 | 5 | 12 | 54 |
| miss | 2550 | 280 | 475 | 4 | 12 | 46 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 4, p90 12, max 54 cycles.

## Workload: lud _i___data_51

Log events: 501148. Whole run: predicted sectors per demand sector 1.68, coverage_pred 82.2%, issued per demand 0.51, timely + late 28.4%, timely 2.7%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 13982 | 0.7 | 0.9 | 0.8 | 0.0 | 0.0 | 1.4 | 2.4 | 2.4 | 0.1 | 0.0 | 54.7 | 36.6 |
| promotion | 35790 | 4.1 | 1.2 | 0.8 | 0.0 | 0.0 | 2.8 | 0.7 | 11.8 | 0.0 | 0.0 | 19.3 | 59.4 |
| inter-warp | 58829 | 25.4 | 7.3 | 11.9 | 1.7 | 0.0 | 6.1 | 2.0 | 21.5 | 0.0 | 0.0 | 0.0 | 24.1 |
| **all** | 108601 | 15.2 | 4.4 | 6.8 | 0.9 | 0.0 | 4.4 | 1.6 | 15.9 | 0.0 | 0.0 | 13.4 | 37.3 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 24340 | 4037 | 3226 | 17077 | 19732 | 9651 | 5178 | 0 | 0 | 0 | 4903 | 9234 | 17.3% | 95.2% |
| promotion | 42428 | 3934 | 17770 | 20724 | 24546 | 12798 | 2850 | 0 | 0 | 0 | 8898 | 12675 | 43.5% | 92.9% |
| inter-warp | 152644 | 43428 | 46496 | 62720 | 44604 | 10674 | 30492 | 0 | 0 | 0 | 3438 | 10468 | 89.4% | 97.4% |
| **all** | 219412 | 51399 | 67492 | 100521 | 91509 | 33573 | 40373 | 0 | 0 | 0 | 17563 | 33573 | 51.3% | 95.1% |

Unused prefetched sectors lost: evicted 1241, invalid at refill 0, at a kernel end 13327. Dropped from the queue: displaced 9012, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 9 | 33 | 78 | 297 | 449 | 2627 |
| unknown | memory latency (filled - issued) | 181 | 203 | 246 | 343 | 435 | 1196 |
| inter-thread | available lead (demand - latest prediction) | 23 | 24 | 25 | 30 | 37 | 6361 |
| inter-thread | queue wait (offered - queue entry created) | 2 | 3 | 12 | 113 | 267 | 19732 |
| inter-thread | memory latency (filled - issued) | 175 | 199 | 256 | 344 | 453 | 9234 |
| promotion | available lead (demand - latest prediction) | 23 | 24 | 28 | 48 | 184 | 4340 |
| promotion | queue wait (offered - queue entry created) | 6 | 9 | 24 | 39 | 74 | 24546 |
| promotion | memory latency (filled - issued) | 178 | 202 | 263 | 367 | 469 | 12675 |
| inter-warp | available lead (demand - latest prediction) | 6 | 13 | 20 | 36 | 85 | 42801 |
| inter-warp | queue wait (offered - queue entry created) | 3 | 6 | 13 | 37 | 63 | 44604 |
| inter-warp | memory latency (filled - issued) | 169 | 176 | 201 | 263 | 373 | 10468 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 3 _Z12lud_internalPfii | new miss | 3978 | 274 | 445 | 679 | 464 | 1007 | – |
| 3 _Z12lud_internalPfii | merged into prefetch | 1413 | 147 | 269 | 451 | 286 | 969 | 1.3% |
| 3 _Z12lud_internalPfii | merged into demand | 84 | 36 | 145 | 353 | 177 | 433 | 0.0% |
| 9 _Z12lud_internalPfii | new miss | 3471 | 224 | 329 | 536 | 352 | 663 | – |
| 9 _Z12lud_internalPfii | merged into prefetch | 1062 | 85 | 173 | 302 | 188 | 535 | 0.0% |
| 9 _Z12lud_internalPfii | merged into demand | 33 | 75 | 208 | 224 | 182 | 230 | 0.0% |
| 6 _Z12lud_internalPfii | new miss | 2835 | 212 | 280 | 412 | 294 | 515 | – |
| 6 _Z12lud_internalPfii | merged into prefetch | 1485 | 89 | 176 | 324 | 195 | 490 | 2.6% |
| 6 _Z12lud_internalPfii | merged into demand | 69 | 14 | 66 | 360 | 126 | 451 | 5.8% |
| 12 _Z12lud_internalPfii | new miss | 2971 | 226 | 351 | 564 | 373 | 744 | – |
| 12 _Z12lud_internalPfii | merged into prefetch | 1201 | 99 | 193 | 397 | 222 | 694 | 2.3% |
| 12 _Z12lud_internalPfii | merged into demand | 22 | 254 | 318 | 388 | 319 | 396 | 0.0% |
| 15 _Z12lud_internalPfii | new miss | 3056 | 230 | 352 | 516 | 362 | 623 | – |
| 15 _Z12lud_internalPfii | merged into prefetch | 1017 | 101 | 187 | 352 | 210 | 563 | 1.3% |
| 18 _Z12lud_internalPfii | new miss | 2936 | 229 | 329 | 495 | 343 | 644 | – |
| 18 _Z12lud_internalPfii | merged into prefetch | 1137 | 95 | 192 | 387 | 218 | 574 | 0.9% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 16963 | 163 | 304 | 11 | 38 | 133 |
| miss | 564 | 324 | 503 | 5 | 37 | 79 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 10, p90 38, max 133 cycles.

## Workload: lud _s_256__v

Log events: 55386. Whole run: predicted sectors per demand sector 1.19, coverage_pred 74.9%, issued per demand 0.42, timely + late 33.4%, timely 0.7%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 828 | 0.0 | 1.0 | 2.7 | 0.0 | 0.0 | 14.4 | 2.7 | 8.8 | 0.0 | 0.0 | 31.5 | 39.0 |
| promotion | 2152 | 0.0 | 0.2 | 1.4 | 0.0 | 0.0 | 5.5 | 1.3 | 33.2 | 0.0 | 0.0 | 23.8 | 34.6 |
| inter-warp | 6933 | 8.1 | 6.8 | 21.5 | 0.2 | 0.0 | 9.8 | 0.3 | 28.2 | 0.0 | 0.0 | 0.0 | 25.1 |
| **all** | 9913 | 5.7 | 4.9 | 15.5 | 0.2 | 0.0 | 9.3 | 0.7 | 27.7 | 0.0 | 0.0 | 7.8 | 28.3 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 1768 | 646 | 0 | 1122 | 1768 | 491 | 451 | 0 | 0 | 0 | 826 | 491 | 42.4% | 90.0% |
| promotion | 2864 | 336 | 160 | 2368 | 2704 | 1417 | 399 | 0 | 0 | 0 | 888 | 1417 | 62.5% | 76.1% |
| inter-warp | 17012 | 6636 | 1814 | 8562 | 7026 | 1638 | 4833 | 0 | 0 | 0 | 555 | 1595 | 98.2% | 95.4% |
| **all** | 21644 | 7618 | 1974 | 12052 | 11762 | 3587 | 5860 | 0 | 0 | 0 | 2315 | 3587 | 76.5% | 87.1% |

Unused prefetched sectors lost: evicted 0, invalid at refill 0, at a kernel end 773. Dropped from the queue: displaced 290, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 8 | 27 | 51 | 136 | 151 | 264 |
| unknown | memory latency (filled - issued) | 172 | 177 | 190 | 203 | 242 | 84 |
| inter-thread | available lead (demand - latest prediction) | 20 | 23 | 24 | 26 | 31 | 675 |
| inter-thread | queue wait (offered - queue entry created) | 2 | 3 | 11 | 64 | 149 | 1768 |
| inter-thread | memory latency (filled - issued) | 185 | 203 | 247 | 304 | 426 | 491 |
| promotion | available lead (demand - latest prediction) | 23 | 24 | 26 | 31 | 81 | 602 |
| promotion | queue wait (offered - queue entry created) | 3 | 5 | 10 | 24 | 32 | 2704 |
| promotion | memory latency (filled - issued) | 169 | 174 | 181 | 299 | 378 | 1417 |
| inter-warp | available lead (demand - latest prediction) | 4 | 11 | 18 | 27 | 44 | 5055 |
| inter-warp | queue wait (offered - queue entry created) | 3 | 6 | 13 | 44 | 132 | 7026 |
| inter-warp | memory latency (filled - issued) | 167 | 170 | 175 | 190 | 243 | 1595 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 3 _Z12lud_internalPfii | new miss | 1015 | 268 | 345 | 470 | 359 | 696 | – |
| 3 _Z12lud_internalPfii | merged into prefetch | 413 | 136 | 238 | 386 | 252 | 618 | 1.7% |
| 6 _Z12lud_internalPfii | new miss | 815 | 221 | 300 | 369 | 295 | 415 | – |
| 6 _Z12lud_internalPfii | merged into prefetch | 417 | 94 | 177 | 266 | 180 | 356 | 0.0% |
| 9 _Z12lud_internalPfii | new miss | 626 | 190 | 260 | 347 | 263 | 379 | – |
| 9 _Z12lud_internalPfii | merged into prefetch | 317 | 96 | 157 | 211 | 158 | 329 | 0.0% |
| 12 _Z12lud_internalPfii | new miss | 571 | 202 | 235 | 322 | 249 | 336 | – |
| 12 _Z12lud_internalPfii | merged into prefetch | 287 | 101 | 167 | 219 | 163 | 285 | 0.0% |
| 15 _Z12lud_internalPfii | new miss | 404 | 209 | 240 | 310 | 250 | 317 | – |
| 15 _Z12lud_internalPfii | merged into prefetch | 264 | 112 | 155 | 202 | 156 | 284 | 0.0% |
| 18 _Z12lud_internalPfii | new miss | 290 | 200 | 219 | 268 | 225 | 284 | – |
| 18 _Z12lud_internalPfii | merged into prefetch | 246 | 120 | 163 | 196 | 161 | 265 | 0.0% |
| 18 _Z12lud_internalPfii | merged into demand | 24 | 133 | 180 | 230 | 184 | 232 | 0.0% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 2494 | 152 | 202 | 9 | 32 | 79 |
| miss | 257 | 279 | 389 | 7 | 24 | 57 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 9, p90 32, max 79 cycles.

## Workload: nw 2048_10

Log events: 153758. Whole run: predicted sectors per demand sector 0.00, coverage_pred 0.0%, issued per demand 0.00, timely + late 0.0%, timely 0.0%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a% | n/a% |

Unused prefetched sectors lost: evicted 0, invalid at refill 0, at a kernel end 0. Dropped from the queue: displaced 0, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 116 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 170 | 384 | 1013 | 503 | 1303 | – |
| 119 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 168 | 387 | 523 | 364 | 721 | – |
| 121 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 168 | 387 | 535 | 370 | 710 | – |
| 123 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 169 | 396 | 546 | 377 | 748 | – |
| 124 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 170 | 376 | 1012 | 504 | 1319 | – |
| 125 _Z20needle_cuda_shared_1PiS_ | new miss | 520 | 169 | 401 | 557 | 378 | 714 | – |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 n/a, p90 n/a, max n/a cycles.

## Workload: histo _i___data_de

Log events: 3195719. Whole run: predicted sectors per demand sector 0.70, coverage_pred 34.4%, issued per demand 0.27, timely + late 23.6%, timely 10.8%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 144624 | 3.4 | 1.0 | 0.6 | 2.0 | 0.0 | 12.1 | 40.5 | 24.6 | 0.4 | 0.0 | 11.6 | 3.8 |
| intra-warp | 5 | 0.0 | 0.0 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 80.0 | 0.0 |
| promotion | 66556 | 0.9 | 3.7 | 2.3 | 3.0 | 0.0 | 23.5 | 5.9 | 59.7 | 0.0 | 0.0 | 0.6 | 0.4 |
| inter-warp | 436069 | 2.1 | 0.9 | 3.8 | 1.8 | 0.0 | 3.2 | 8.0 | 10.6 | 0.1 | 0.0 | 2.7 | 66.8 |
| **all** | 647254 | 2.3 | 1.2 | 2.9 | 2.0 | 0.0 | 7.3 | 15.0 | 18.8 | 0.2 | 0.0 | 4.5 | 45.9 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 392119 | 24834 | 1466 | 365819 | 386658 | 137556 | 130973 | 0 | 0 | 0 | 118129 | 137528 | 31.6% | 60.2% |
| intra-warp | 277 | 3 | 0 | 274 | 274 | 16 | 258 | 0 | 0 | 0 | 0 | 16 | 0.0% | 0.0% |
| promotion | 75367 | 3760 | 706 | 70901 | 71126 | 16783 | 7428 | 0 | 0 | 0 | 46915 | 16684 | 84.8% | 25.0% |
| inter-warp | 1235271 | 217527 | 282043 | 735701 | 636710 | 95410 | 475658 | 0 | 0 | 0 | 65642 | 95186 | 67.0% | 60.2% |
| **all** | 1703034 | 246124 | 284215 | 1172695 | 1094944 | 249815 | 614358 | 0 | 0 | 0 | 230771 | 249487 | 48.7% | 57.9% |

Unused prefetched sectors lost: evicted 13922, invalid at refill 0, at a kernel end 16838. Dropped from the queue: displaced 77653, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 83 | 130 | 188 | 249 | 608 | 176 |
| unknown | memory latency (filled - issued) | 182 | 191 | 200 | 208 | 222 | 73 |
| inter-thread | available lead (demand - latest prediction) | 74 | 81 | 132 | 276 | 338 | 106939 |
| inter-thread | queue wait (offered - queue entry created) | 3 | 4 | 11 | 45 | 103 | 386658 |
| inter-thread | memory latency (filled - issued) | 179 | 190 | 215 | 388 | 1088 | 137528 |
| intra-warp | available lead (demand - latest prediction) | 71 | 71 | 71 | 72 | 72 | 4 |
| intra-warp | queue wait (offered - queue entry created) | 4 | 5 | 7 | 11 | 18 | 274 |
| intra-warp | memory latency (filled - issued) | 343 | 400 | 1026 | 1060 | 1133 | 16 |
| promotion | available lead (demand - latest prediction) | 39 | 75 | 161 | 339 | 565 | 18398 |
| promotion | queue wait (offered - queue entry created) | 4 | 8 | 16 | 33 | 55 | 71126 |
| promotion | memory latency (filled - issued) | 182 | 376 | 700 | 1311 | 1769 | 16684 |
| inter-warp | available lead (demand - latest prediction) | 14 | 33 | 82 | 183 | 321 | 196180 |
| inter-warp | queue wait (offered - queue entry created) | 3 | 6 | 15 | 100 | 249 | 636710 |
| inter-warp | memory latency (filled - issued) | 193 | 243 | 378 | 686 | 1380 | 95186 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 23 _Z17histo_main_kernelP6uchar | new miss | 12893 | 178 | 262 | 414 | 287 | 946 | – |
| 23 _Z17histo_main_kernelP6uchar | merged into prefetch | 7167 | 26 | 130 | 291 | 145 | 673 | 1.7% |
| 23 _Z17histo_main_kernelP6uchar | merged into demand | 425 | 63 | 268 | 539 | 290 | 862 | 24.5% |
| 27 _Z17histo_main_kernelP6uchar | new miss | 12464 | 177 | 267 | 426 | 288 | 907 | – |
| 27 _Z17histo_main_kernelP6uchar | merged into prefetch | 7544 | 30 | 133 | 312 | 152 | 638 | 1.8% |
| 27 _Z17histo_main_kernelP6uchar | merged into demand | 452 | 44 | 208 | 437 | 225 | 654 | 11.3% |
| 15 _Z17histo_main_kernelP6uchar | new miss | 12615 | 175 | 258 | 415 | 281 | 770 | – |
| 15 _Z17histo_main_kernelP6uchar | merged into prefetch | 7210 | 28 | 130 | 307 | 151 | 656 | 2.3% |
| 15 _Z17histo_main_kernelP6uchar | merged into demand | 409 | 42 | 189 | 453 | 225 | 677 | 13.2% |
| 11 _Z17histo_main_kernelP6uchar | new miss | 12899 | 176 | 255 | 426 | 286 | 895 | – |
| 11 _Z17histo_main_kernelP6uchar | merged into prefetch | 6819 | 28 | 134 | 298 | 149 | 692 | 1.1% |
| 11 _Z17histo_main_kernelP6uchar | merged into demand | 428 | 70 | 228 | 488 | 257 | 752 | 16.6% |
| 35 _Z17histo_main_kernelP6uchar | new miss | 12870 | 175 | 248 | 432 | 285 | 986 | – |
| 35 _Z17histo_main_kernelP6uchar | merged into prefetch | 6723 | 27 | 131 | 302 | 148 | 686 | 1.9% |
| 35 _Z17histo_main_kernelP6uchar | merged into demand | 422 | 86 | 268 | 552 | 293 | 877 | 21.8% |
| 19 _Z17histo_main_kernelP6uchar | new miss | 10550 | 178 | 284 | 442 | 302 | 874 | – |
| 19 _Z17histo_main_kernelP6uchar | merged into prefetch | 4939 | 31 | 146 | 327 | 167 | 776 | 2.3% |
| 19 _Z17histo_main_kernelP6uchar | merged into demand | 510 | 71 | 250 | 551 | 285 | 779 | 23.9% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 56524 | 135 | 321 | 6 | 20 | 70 |
| miss | 68423 | 443 | 1408 | 3 | 6 | 67 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 4, p90 13, max 70 cycles.

## Workload: mri-q _i___data_sm

Log events: 20530. Whole run: predicted sectors per demand sector 1.29, coverage_pred 88.0%, issued per demand 0.09, timely + late 9.3%, timely 0.0%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 872 | 4.1 | 41.3 | 36.6 | 0.0 | 0.0 | 8.0 | 0.0 | 4.4 | 0.0 | 0.0 | 0.0 | 5.6 |
| promotion | 359 | 0.0 | 1.9 | 32.0 | 0.0 | 0.0 | 16.4 | 0.0 | 25.6 | 0.0 | 0.0 | 0.0 | 24.0 |
| inter-warp | 3807 | 22.6 | 6.0 | 22.4 | 0.0 | 0.0 | 5.2 | 0.0 | 6.6 | 0.0 | 0.0 | 0.0 | 37.2 |
| **all** | 5038 | 17.8 | 11.8 | 25.5 | 0.0 | 0.0 | 6.5 | 0.0 | 7.6 | 0.0 | 0.0 | 0.0 | 30.8 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 2144 | 450 | 532 | 1162 | 1544 | 73 | 1271 | 0 | 0 | 0 | 200 | 73 | 100.0% | 57.5% |
| promotion | 468 | 108 | 52 | 308 | 388 | 92 | 237 | 0 | 0 | 0 | 59 | 92 | 100.0% | 79.3% |
| inter-warp | 9400 | 3861 | 3296 | 2243 | 1005 | 217 | 531 | 0 | 0 | 0 | 257 | 217 | 100.0% | 69.1% |
| **all** | 12012 | 4419 | 3880 | 3713 | 2937 | 382 | 2039 | 0 | 0 | 0 | 516 | 382 | 100.0% | 69.4% |

Unused prefetched sectors lost: evicted 0, invalid at refill 0, at a kernel end 0. Dropped from the queue: displaced 776, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| inter-thread | available lead (demand - latest prediction) | 23 | 29 | 49 | 75 | 103 | 996 |
| inter-thread | queue wait (offered - queue entry created) | 4 | 130 | 322 | 375 | 407 | 1544 |
| inter-thread | memory latency (filled - issued) | 180 | 191 | 221 | 348 | 395 | 73 |
| promotion | available lead (demand - latest prediction) | 22 | 24 | 38 | 62 | 75 | 185 |
| promotion | queue wait (offered - queue entry created) | 2 | 2 | 178 | 379 | 382 | 388 |
| promotion | memory latency (filled - issued) | 180 | 187 | 201 | 220 | 343 | 92 |
| inter-warp | available lead (demand - latest prediction) | 6 | 15 | 30 | 52 | 74 | 2306 |
| inter-warp | queue wait (offered - queue entry created) | 3 | 5 | 113 | 251 | 254 | 1005 |
| inter-warp | memory latency (filled - issued) | 168 | 172 | 184 | 314 | 353 | 217 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 2 _Z12ComputeQ_GPUiiPfS_S_S_S_ | new miss | 1163 | 452 | 810 | 1092 | 786 | 1195 | – |
| 2 _Z12ComputeQ_GPUiiPfS_S_S_S_ | merged into prefetch | 117 | 79 | 281 | 388 | 264 | 488 | 0.0% |
| 3 _Z12ComputeQ_GPUiiPfS_S_S_S_ | new miss | 1159 | 238 | 341 | 445 | 343 | 480 | – |
| 3 _Z12ComputeQ_GPUiiPfS_S_S_S_ | merged into prefetch | 121 | 121 | 165 | 189 | 161 | 204 | 0.0% |
| 4 _Z12ComputeQ_GPUiiPfS_S_S_S_ | new miss | 1160 | 242 | 355 | 446 | 352 | 479 | – |
| 4 _Z12ComputeQ_GPUiiPfS_S_S_S_ | merged into prefetch | 120 | 120 | 158 | 184 | 157 | 211 | 0.0% |
| 1 _Z17ComputePhiMag_GPUPfS_S_i | new miss | 104 | 340 | 358 | 379 | 359 | 411 | – |
| 1 _Z17ComputePhiMag_GPUPfS_S_i | merged into prefetch | 24 | 290 | 313 | 344 | 315 | 355 | 0.0% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 265 | 153 | 179 | 5 | 14 | 20 |
| miss | 117 | 301 | 385 | 3 | 10 | 21 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 4, p90 13, max 21 cycles.

## Workload: srad_v1 100_0_5_502_

Log events: 1888658. Whole run: predicted sectors per demand sector 1.06, coverage_pred 84.8%, issued per demand 0.13, timely + late 12.5%, timely 3.5%.

### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of records closed by a demand (U) or by the window (X); records still open at the end are not counted

| source (first prediction) | records | queue full | queued at demand, throttled | queued at demand, not throttled | in L1 | in flight | no line / MSHR / miss-queue slot | timely | late | evicted before demand | issued, other | no demand: issued | no demand: not issued |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 45777 | 30.9 | 8.4 | 22.8 | 22.1 | 0.0 | 3.3 | 0.2 | 10.6 | 0.0 | 0.0 | 0.1 | 1.7 |
| intra-warp | 64 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| promotion | 23095 | 27.4 | 29.5 | 14.4 | 3.7 | 0.0 | 1.7 | 0.0 | 1.2 | 0.0 | 0.0 | 0.0 | 21.9 |
| inter-warp | 339022 | 37.6 | 5.8 | 9.9 | 8.7 | 0.0 | 3.5 | 3.8 | 8.8 | 0.0 | 0.0 | 0.3 | 21.7 |
| **all** | 407958 | 36.3 | 7.4 | 11.6 | 9.9 | 0.0 | 3.4 | 3.1 | 8.6 | 0.0 | 0.0 | 0.2 | 19.5 |

### 3a Funnel, sector level (counts)

| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | late merge at fill | L2 hit of fills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inter-thread | 200255 | 42725 | 76991 | 80539 | 106423 | 12577 | 85231 | 0 | 0 | 0 | 8615 | 12408 | 98.5% | 99.8% |
| intra-warp | 1037 | 68 | 959 | 10 | 69 | 0 | 69 | 0 | 0 | 0 | 0 | 0 | n/a% | n/a% |
| promotion | 42723 | 8046 | 18242 | 16435 | 20261 | 378 | 19434 | 0 | 0 | 0 | 449 | 354 | 89.8% | 94.4% |
| inter-warp | 1113946 | 306119 | 525727 | 282100 | 205985 | 35092 | 146570 | 0 | 0 | 0 | 24323 | 33969 | 62.1% | 96.9% |
| **all** | 1357961 | 356958 | 621919 | 379084 | 347112 | 48898 | 263818 | 0 | 0 | 0 | 34396 | 48770 | 71.7% | 97.7% |

Unused prefetched sectors lost: evicted 21, invalid at refill 0, at a kernel end 900. Dropped from the queue: displaced 31661, accuracy filter 0 sectors.

### 3b Lead time (cycles)

| source | measure | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|---|
| unknown | queue wait (offered - queue entry created) | 7 | 28 | 66 | 153 | 481 | 14374 |
| unknown | memory latency (filled - issued) | 202 | 280 | 434 | 600 | 749 | 2039 |
| inter-thread | available lead (demand - latest prediction) | 20 | 21 | 24 | 31 | 53 | 117323 |
| inter-thread | queue wait (offered - queue entry created) | 2 | 3 | 18 | 185 | 526 | 106423 |
| inter-thread | memory latency (filled - issued) | 168 | 173 | 191 | 434 | 572 | 12408 |
| intra-warp | available lead (demand - latest prediction) | 21 | 21 | 27 | 42 | 58 | 599 |
| intra-warp | queue wait (offered - queue entry created) | 53 | 111 | 371 | 421 | 1253 | 69 |
| promotion | available lead (demand - latest prediction) | 5 | 16 | 27 | 58 | 170 | 8274 |
| promotion | queue wait (offered - queue entry created) | 13 | 69 | 599 | 727 | 1548 | 20261 |
| promotion | memory latency (filled - issued) | 171 | 188 | 408 | 588 | 714 | 354 |
| inter-warp | available lead (demand - latest prediction) | 6 | 13 | 22 | 54 | 152 | 201469 |
| inter-warp | queue wait (offered - queue entry created) | 4 | 6 | 11 | 43 | 92 | 205985 |
| inter-warp | memory latency (filled - issued) | 169 | 174 | 186 | 255 | 506 | 33969 |

### 3c Demand read latency, access to data back (cycles), top kernels by reads

| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |
|---|---|---|---|---|---|---|---|---|
| 16 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 13204 | 456 | 624 | 939 | 659 | 1148 | – |
| 16 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 607 | 56 | 401 | 605 | 361 | 731 | 0.0% |
| 16 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2952 | 75 | 322 | 699 | 360 | 1070 | 1.2% |
| 6 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 13334 | 422 | 616 | 828 | 620 | 1027 | – |
| 6 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 498 | 89 | 398 | 628 | 378 | 940 | 0.6% |
| 6 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2874 | 65 | 287 | 636 | 323 | 860 | 0.4% |
| 11 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 13307 | 430 | 607 | 838 | 623 | 1041 | – |
| 11 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 503 | 116 | 439 | 629 | 402 | 909 | 1.8% |
| 11 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2845 | 68 | 299 | 638 | 334 | 932 | 0.8% |
| 31 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 12988 | 436 | 618 | 820 | 626 | 1069 | – |
| 31 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 386 | 97 | 441 | 659 | 398 | 783 | 0.0% |
| 31 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2785 | 76 | 304 | 629 | 332 | 936 | 0.9% |
| 36 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 12516 | 437 | 655 | 974 | 689 | 1124 | – |
| 36 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 428 | 114 | 392 | 615 | 365 | 768 | 0.0% |
| 36 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2746 | 87 | 346 | 742 | 381 | 1076 | 0.9% |
| 41 _Z5srad2fiilPiS_S_S_PfS0_S0_ | new miss | 12393 | 425 | 631 | 902 | 651 | 1163 | – |
| 41 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into prefetch | 579 | 125 | 450 | 683 | 435 | 1069 | 4.1% |
| 41 _Z5srad2fiilPiS_S_S_PfS0_S0_ | merged into demand | 2666 | 63 | 330 | 698 | 357 | 1000 | 0.3% |

Demands merged into a prefetch, split at the prefetch's fill (cycles; all kernels):

| prefetch in L2 | n | merge to fill p50 | p90 | fill to data back p50 | p90 | max |
|---|---|---|---|---|---|---|
| hit | 41377 | 157 | 498 | 8 | 47 | 186 |
| miss | 1342 | 384 | 984 | 4 | 10 | 45 |

3c check (a demand merged into a prefetch must not wait longer than a new miss): medians not higher; wake-up after the fill p50 8, p90 46, max 186 cycles.

