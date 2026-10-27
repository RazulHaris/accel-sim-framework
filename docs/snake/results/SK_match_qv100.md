# SK match on base QV100 (2026-10-27)

Why do our SK_SC numbers differ from SnakeCharmer's SK (results/SK_SC.md)? Ours: on time / too early / too late 38.8 / 4.7 / 56.5%, accuracy 66.2%. Theirs (A100): 48.3 / 28.7 / 23.0%, accuracy 23.1%. No sm_80 traces exist locally, so every run here is on base QV100, SnakeCharmer's V100 setup (LRR, 1132 MHz), at 100M instructions.

**Workloads (14):**
- SnakeCharmer's apps without bfs: b+tree, particlefilter float and naive, heartwall†, streamcluster† († = input differs from SnakeCharmer Table II)
- our 7 apps (9 workloads)

Steps 2–4 (R2 injection path, ceilings, bfs) wait for approval; the R2 spec is in [R2_SPEC.md](../R2_SPEC.md). SK_R0_D0 against Tier 1's Snake at Tier 1 conditions (PAPER_V100, 1B), with the window / setup / config decomposition: the section after Step 1.

## Step 1: miss-queue reserve and decoupling (done)

### Configs
Every SK variant is SK_SC (OQ-33…OQ-35) with two changes:
- **Same-CTA inter-warp targets, up to 63** (`SK_SC_CTA`): cross-CTA targets gave wrong addresses on hotspot's 2-D grid (SPEC OQ-37).
- **SnakeCharmer's Head table** (`SNAKE_HEAD64` = `-snake_head_two_warps 0`):
  - Table III lists the Head as "10 bytes per entry, 64 entries", and §III-A indexes it by warp ID.
  - §III-A learns inter-warp strides "when consecutive warps execute the same load PC".
  - Inter-warp training therefore reads the 64 per-warp Head registers (nearest warp IDs whose last load was this PC) instead of the 32 per-PC rows of two warps (OQ-1b), which go unused.
  - An existing option, so there is no simulator change.

| label | tokens on `QV100-SASS-100M_INSN-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64` | miss-queue reserve | decoupling |
|---|---|---|---|
| SK | — | 8 | mode 1 (variant 1) |
| SK_R0 (a) | `SNAKE_MQR0` | 0 | mode 1 |
| SK_R0_D0 (b) | `SNAKE_MQR0-SNAKE_DM0` | 0 | off (throttling stays on) |

**Shown alongside**, from the SK_SC stage (same build, no new runs):
- the no-prefetch baselines
- our Snake (`SNAKE-SNAKE_FP1`)
- SK_SC (old): cross-CTA targets, two-warp Head
- SK_SC_CTA (old): same-CTA targets, two-warp Head; hotspot and srad_v1 only

### Build, runs, checks
- **Build:** the SK_SC build, unchanged: accel-sim `5d55de3`, gpgpu-sim `9622012e` (both `_modified_0.0`). Step 1 adds only config tokens (accel-sim `2a5c57c`), so the SK_SC gates (off-regression PASS on 6, synthetic PASS on 11) apply as they are.
- **Runs:** 42 (`sim_run_snake_skm/qv100`, 14 workloads × 3 configs; frozen build copied from `sim_run_snake_skc/qv100_7apps/bin`). The baselines are the SK_SC stage's base QV100 Snake-off runs.
- **Fig. 15 with `SNAKE_HEAD64`** (1 run, `sim_run_snake_synth/SKM_HEAD64`): PASS. The inter-warp stride 100 trains in phase (a), and the Head, the Tail and the predictions 1300 / 1800 / {2600, 2700, 2800, 3000} match the figure exactly.
- **All 42 runs** finish, keep `pf_woken_warps` at 0, and pass the outcome and funnel invariants.
- **Tables:** `util/snake/sk_match_report.py $PWD/sim_run_snake_skm/qv100/runs $PWD/sim_run_snake_skc`.

### Verdict: neither (a) nor (b) moves the timing split or accuracy towards SnakeCharmer's
| SnakeCharmer apps (5) | on time | too early | too late | accuracy_sc | distance to SK | IPC vs no prefetch |
|---|---|---|---|---|---|---|
| SnakeCharmer SK (A100) | 48.3 | 28.7 | 23.0 | 23.1 | 0 | n/a |
| SK | 42.2 | 0.0 | 57.8 | 72.6 | 119.1 | +13.60% |
| SK_R0 (a) | 40.3 | 0.0 | 59.6 | 73.6 | 123.8 | +13.61% |
| SK_R0_D0 (b) | 40.7 | 0.0 | 59.3 | 72.9 | 122.5 | +13.96% |
| SK_SC (old) | 43.2 | 5.6 | 51.2 | 59.5 | 92.9 | +11.13% |

Distance = the sum of the absolute differences from SnakeCharmer's four numbers, in points. Over all 14 workloads: SK 162.6, (a) 164.8, (b) 166.5.

- **(a) Reserve 0** moves slightly away. On time falls 1–2 points, too late rises, accuracy stays (76.3 → 76.4 over 14).
  - It costs IPC on our apps: 7-app GMEAN −1.03 → −1.41%; histo −8.4 → −11.0%.
  - Demand MISS_QUEUE_FULL rises: −14.1% → −6.7% vs the baseline over 14; histo goes from −30% to +20% of its baseline, heartwall to +67%.
- **(b) Reserve 0 + decoupling off** also moves slightly away. On time falls, too early goes to 0.0, accuracy changes little (76.3 → 75.5).
  - It gives the best IPC: GMEAN over 14 +4.77% (SK +3.97%), 7 apps −0.00% (SK −1.03%), mostly histo (−8.4 → −1.2%) and hotspot 512 (+0.4 → +2.6%).
  - Prefetch-induced evictions fall from 16.4% to 10.7%: decoupling mode 1's bulk frees are what evict demand data for prefetches (hotspot 27.9% → 1.9%).
- **Too early stays at 0 even with decoupling off**, so our protected prefetch region is not why nothing is too early. Prefetches arrive late, not early: inter-warp targets in the same CTA reach the load ~20 cycles after the prediction (S3c, S6). Over 14 workloads 78–80% of used prefetches are late in all three Step 1 configs.
- **Accuracy stays ~73–76%** in all three Step 1 configs: our sector-exact predictions are mostly right. Only the old cross-CTA SK_SC came "closer" to SnakeCharmer (distance 92.9), and only through wrong prefetches (b+tree: accuracy 4.3%, 28% too early, IPC −12.6%). That is not a match.
- **Neither suspect explains the gap.** A cause not tested here: the prefetch footprint.
  - SnakeCharmer's 23% sector accuracy would fit a predictor that fetches whole 128 B lines for scattered accesses.
  - Ours is sector-exact (`SNAKE_FP1`); our default footprint 0 fetches the line's missing sectors.
- **Better of (a) / (b) for Step 2:** (b) SK_R0_D0, on IPC (+4.77% vs +3.71% over 14) and with no miss-queue reserve left to remove, which suits R2. On the matching criterion (b) is slightly better on SnakeCharmer's apps (122.5 vs 123.8) and slightly worse over 14 (166.5 vs 164.8).

### Same-CTA targets and the per-warp Head
- **SK vs SK_SC (old)** changes both the inter-warp scope (same CTA) and the Head. On SnakeCharmer's apps IPC rises from +11.13% to +13.60%. b+tree goes from −12.6% to −2.6% (accuracy 4.3 → 71.9%).
- **The per-warp Head trains the same inter-warp strides as the two-warp Head.** `interwarp_trained_pcs` is equal on 11 of 14 workloads and slightly lower on the other 3: heartwall 14 vs 15, particlefilter_float 128 vs 129, streamcluster 2,410 vs 2,434.
  - The training events (re-training after Tail evictions) are fewer on b+tree, particlefilter_float, heartwall and streamcluster.
  - Against SK_SC_CTA (old), where only the Head differs, hotspot is identical (160 events, 2 PCs) and srad_v1 is within 3% (7,194 vs 7,357).
- **IPC with the per-warp Head is lower on hotspot 512** (+0.36% vs +2.42% for SK_SC_CTA old) and higher on srad_v1 (−0.78% vs −1.33%).

### Means
Arithmetic over workloads (IPC: geometric mean of speedups; demand MISS_QUEUE_FULL: summed over the set vs the baselines' sum). Distance = |on time − 48.3| + |too early − 28.7| + |too late − 23.0| + |accuracy − 23.1| (SnakeCharmer's SK, A100), in points.

| set | config | n | IPC vs no prefetch | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions | demand MISS_QUEUE_FULL vs base | distance to SK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SnakeCharmer apps | Snake | 5 | +12.09% | 13.1 | 19.7 | 74.5 | 39.0 | 0.1 | 61.0 | 12.5 | -6.06% | 127.3 |
| SnakeCharmer apps | SK_SC (old) | 5 | +11.13% | 17.9 | 22.1 | 59.5 | 43.2 | 5.6 | 51.2 | 8.5 | -1.17% | 92.9 |
| SnakeCharmer apps | SK | 5 | +13.60% | 17.8 | 23.6 | 72.6 | 42.2 | 0.0 | 57.8 | 11.2 | -5.45% | 119.1 |
| SnakeCharmer apps | SK_R0 | 5 | +13.61% | 17.8 | 25.0 | 73.6 | 40.3 | 0.0 | 59.6 | 12.0 | -3.28% | 123.8 |
| SnakeCharmer apps | SK_R0_D0 | 5 | +13.96% | 17.8 | 24.2 | 72.9 | 40.7 | 0.0 | 59.3 | 12.3 | -3.11% | 122.5 |
| 7 apps | Snake | 9 | -0.43% | 2.6 | 19.0 | 86.8 | 8.2 | 0.1 | 91.7 | 18.1 | -19.11% | 201.1 |
| 7 apps | SK_SC (old) | 9 | -1.12% | 3.3 | 17.6 | 63.0 | 10.8 | 0.6 | 88.6 | 18.1 | -18.42% | 171.1 |
| 7 apps | SK | 9 | -1.03% | 3.1 | 18.6 | 78.7 | 9.2 | 0.6 | 90.1 | 19.0 | -18.83% | 189.9 |
| 7 apps | SK_R0 | 9 | -1.41% | 3.1 | 20.8 | 78.1 | 8.6 | 0.7 | 90.7 | 18.1 | -8.60% | 190.5 |
| 7 apps | SK_R0_D0 | 9 | -0.00% | 1.2 | 14.5 | 77.2 | 7.0 | 0.0 | 93.0 | 10.1 | -8.91% | 194.0 |
| all | Snake | 14 | +3.87% | 6.4 | 19.2 | 82.1 | 20.0 | 0.1 | 79.9 | 16.2 | -14.49% | 172.7 |
| all | SK_SC (old) | 14 | +3.09% | 8.5 | 19.2 | 61.7 | 23.2 | 2.5 | 74.2 | 14.9 | -12.31% | 141.1 |
| all | SK | 14 | +3.97% | 8.3 | 20.4 | 76.3 | 21.9 | 0.4 | 77.7 | 16.4 | -14.10% | 162.6 |
| all | SK_R0 | 14 | +3.71% | 8.3 | 22.3 | 76.4 | 20.8 | 0.4 | 78.8 | 16.1 | -6.71% | 164.8 |
| all | SK_R0_D0 | 14 | +4.77% | 7.1 | 18.0 | 75.5 | 20.0 | 0.0 | 80.0 | 10.7 | -6.85% | 166.5 |
| SnakeCharmer | SK (A100, 14 apps) | 14 | | 11.5 | | 23.1 | 48.3 | 28.7 | 23.0 | | | 0.0 |

### Per workload
Columns: IPC vs no prefetching; coverage_sc (on-time only, with late); accuracy_sc; on time / too early / too late among used prefetches; prefetch-induced demand evictions as a share of victim sectors; demand (`GLOBAL_ACC_R`) MISS_QUEUE_FULL reservation fails and their change vs the baseline ("base" rows give the baseline count). "-" = nothing to divide.

| workload | config | IPC vs no prefetch | coverage_sc on-time | coverage_sc +late | accuracy_sc | on time | too early | too late | pf-induced evictions | demand MISS_QUEUE_FULL vs base |
|---|---|---|---|---|---|---|---|---|---|---|
| backprop | base | | | | | | | | | 129365 |
| backprop | Snake | -0.25% | 0.0 | 21.5 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 76014 (-41.24%) |
| backprop | SK_SC (old) | -0.85% | 3.7 | 28.7 | 99.6 | 12.8 | 0.0 | 87.2 | 0.0 | 54335 (-58.00%) |
| backprop | SK | -0.11% | 0.0 | 21.6 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 75705 (-41.48%) |
| backprop | SK_R0 | -0.32% | 0.0 | 25.1 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 85203 (-34.14%) |
| backprop | SK_R0_D0 | -0.32% | 0.0 | 25.1 | 100.0 | 0.0 | 0.0 | 100.0 | - | 85203 (-34.14%) |
| hotspot 1024 | base | | | | | | | | | 118672 |
| hotspot 1024 | Snake | +0.06% | 0.4 | 27.1 | 98.2 | 1.5 | 0.0 | 98.5 | 29.3 | 94111 (-20.70%) |
| hotspot 1024 | SK_SC (old) | +0.44% | 0.4 | 22.6 | 54.5 | 2.0 | 0.1 | 98.0 | 27.3 | 96038 (-19.07%) |
| hotspot 1024 | SK_SC_CTA (old) | +0.26% | 0.4 | 27.2 | 97.7 | 1.6 | 0.0 | 98.4 | 27.6 | 94244 (-20.58%) |
| hotspot 1024 | SK | +0.03% | 0.6 | 26.8 | 94.4 | 2.4 | 0.0 | 97.6 | 27.9 | 97079 (-18.20%) |
| hotspot 1024 | SK_R0 | +0.51% | 0.6 | 29.4 | 92.2 | 1.9 | 0.0 | 98.1 | 30.4 | 93352 (-21.34%) |
| hotspot 1024 | SK_R0_D0 | +0.30% | 0.2 | 11.8 | 87.8 | 1.7 | 0.0 | 98.3 | 1.9 | 107137 (-9.72%) |
| hotspot 512 | base | | | | | | | | | 110222 |
| hotspot 512 | Snake | +0.79% | 0.4 | 26.0 | 96.5 | 1.6 | 0.0 | 98.4 | 27.8 | 88040 (-20.12%) |
| hotspot 512 | SK_SC (old) | +1.13% | 0.5 | 20.5 | 53.5 | 2.4 | 0.2 | 97.5 | 27.0 | 93173 (-15.47%) |
| hotspot 512 | SK_SC_CTA (old) | +2.42% | 0.5 | 25.8 | 95.0 | 2.1 | 0.0 | 97.9 | 25.6 | 87999 (-20.16%) |
| hotspot 512 | SK | +0.36% | 0.6 | 26.1 | 93.6 | 2.4 | 0.0 | 97.6 | 27.8 | 87714 (-20.42%) |
| hotspot 512 | SK_R0 | +0.47% | 0.6 | 27.9 | 94.8 | 2.1 | 0.0 | 97.9 | 26.3 | 88199 (-19.98%) |
| hotspot 512 | SK_R0_D0 | +2.64% | 0.2 | 13.9 | 89.0 | 1.4 | 0.0 | 98.6 | 3.3 | 94979 (-13.83%) |
| lud 512 | base | | | | | | | | | 371436 |
| lud 512 | Snake | +0.02% | 1.5 | 14.8 | 44.0 | 10.1 | 0.1 | 89.8 | 29.9 | 344659 (-7.21%) |
| lud 512 | SK_SC (old) | -1.20% | 0.6 | 7.8 | 10.2 | 7.5 | 0.8 | 91.7 | 29.6 | 361356 (-2.71%) |
| lud 512 | SK | -0.27% | 0.9 | 11.1 | 19.6 | 8.4 | 0.2 | 91.4 | 33.2 | 347888 (-6.34%) |
| lud 512 | SK_R0 | -0.61% | 0.8 | 12.1 | 18.7 | 6.9 | 0.2 | 92.8 | 29.8 | 384690 (+3.57%) |
| lud 512 | SK_R0_D0 | -0.25% | 0.8 | 7.4 | 13.3 | 10.5 | 0.3 | 89.3 | 27.4 | 372674 (+0.33%) |
| lud 256 | base | | | | | | | | | 59257 |
| lud 256 | Snake | +0.03% | 0.6 | 22.9 | 72.3 | 2.6 | 0.0 | 97.4 | - | 35785 (-39.61%) |
| lud 256 | SK_SC (old) | -0.01% | 0.4 | 21.0 | 39.3 | 1.8 | 0.0 | 98.2 | - | 41301 (-30.30%) |
| lud 256 | SK | +0.06% | 0.6 | 22.7 | 63.0 | 2.4 | 0.0 | 97.6 | - | 35241 (-40.53%) |
| lud 256 | SK_R0 | +0.02% | 0.5 | 27.2 | 61.6 | 1.7 | 0.0 | 98.3 | - | 47707 (-19.49%) |
| lud 256 | SK_R0_D0 | +0.02% | 0.5 | 27.2 | 61.6 | 1.7 | 0.0 | 98.3 | - | 47707 (-19.49%) |
| nw | base | | | | | | | | | 0 |
| nw | Snake | +0.00% | 0.0 | 0.0 | - | - | - | - | - | 0 |
| nw | SK_SC (old) | +0.00% | 0.0 | 0.0 | - | - | - | - | - | 0 |
| nw | SK | +0.00% | 0.0 | 0.0 | - | - | - | - | - | 0 |
| nw | SK_R0 | +0.00% | 0.0 | 0.0 | - | - | - | - | - | 0 |
| nw | SK_R0_D0 | +0.00% | 0.0 | 0.0 | - | - | - | - | - | 0 |
| histo | base | | | | | | | | | 104780 |
| histo | Snake | -3.78% | 20.4 | 42.2 | 87.3 | 48.0 | 0.5 | 51.5 | 20.8 | 73550 (-29.81%) |
| histo | SK_SC (old) | -8.66% | 23.6 | 42.1 | 70.7 | 53.9 | 3.9 | 42.2 | 22.4 | 74065 (-29.31%) |
| histo | SK | -8.43% | 24.5 | 43.0 | 70.1 | 54.2 | 4.9 | 41.0 | 22.2 | 73381 (-29.97%) |
| histo | SK_R0 | -10.98% | 24.8 | 45.0 | 67.3 | 52.3 | 5.1 | 42.5 | 19.6 | 125477 (+19.75%) |
| histo | SK_R0_D0 | -1.24% | 8.9 | 24.1 | 75.3 | 37.0 | 0.0 | 63.0 | 3.7 | 113460 (+8.28%) |
| mri-q | base | | | | | | | | | 28988 |
| mri-q | Snake | +0.17% | 0.0 | 9.0 | 100.0 | 0.0 | 0.0 | 100.0 | - | 25354 (-12.54%) |
| mri-q | SK_SC (old) | +0.20% | 0.0 | 9.0 | 100.0 | 0.0 | 0.0 | 100.0 | - | 25056 (-13.56%) |
| mri-q | SK | +0.20% | 0.0 | 9.0 | 100.0 | 0.0 | 0.0 | 100.0 | - | 25344 (-12.57%) |
| mri-q | SK_R0 | +0.25% | 0.0 | 12.0 | 100.0 | 0.0 | 0.0 | 100.0 | - | 25388 (-12.42%) |
| mri-q | SK_R0_D0 | +0.25% | 0.0 | 12.0 | 100.0 | 0.0 | 0.0 | 100.0 | - | 25388 (-12.42%) |
| srad_v1 | base | | | | | | | | | 199313 |
| srad_v1 | Snake | -0.87% | 0.1 | 7.2 | 96.1 | 1.8 | 0.0 | 98.2 | 1.0 | 170116 (-14.65%) |
| srad_v1 | SK_SC (old) | -0.78% | 0.4 | 6.7 | 76.7 | 5.8 | 0.0 | 94.2 | 2.0 | 170076 (-14.67%) |
| srad_v1 | SK_SC_CTA (old) | -1.33% | 0.2 | 6.9 | 88.2 | 3.4 | 0.0 | 96.6 | 1.9 | 169626 (-14.89%) |
| srad_v1 | SK | -0.78% | 0.3 | 7.1 | 88.8 | 4.0 | 0.0 | 96.0 | 2.6 | 168370 (-15.52%) |
| srad_v1 | SK_R0 | -1.46% | 0.3 | 8.7 | 90.3 | 3.8 | 0.0 | 96.2 | 2.6 | 175571 (-11.91%) |
| srad_v1 | SK_R0_D0 | -1.38% | 0.3 | 8.7 | 90.4 | 3.8 | 0.0 | 96.2 | 14.2 | 175550 (-11.92%) |
| b+tree | base | | | | | | | | | 43042 |
| b+tree | Snake | -1.87% | 0.3 | 12.8 | 76.4 | 2.7 | 0.0 | 97.3 | 21.5 | 31787 (-26.15%) |
| b+tree | SK_SC (old) | -12.63% | 1.1 | 4.2 | 4.3 | 19.4 | 27.9 | 52.7 | 10.1 | 58704 (+36.39%) |
| b+tree | SK | -2.57% | 0.4 | 11.5 | 71.9 | 3.5 | 0.0 | 96.5 | 18.1 | 34246 (-20.44%) |
| b+tree | SK_R0 | -1.98% | 0.4 | 13.9 | 76.0 | 2.7 | 0.0 | 97.3 | 18.0 | 36391 (-15.45%) |
| b+tree | SK_R0_D0 | -0.89% | 0.3 | 10.7 | 70.8 | 3.2 | 0.0 | 96.8 | 3.5 | 37570 (-12.71%) |
| heartwall † | base | | | | | | | | | 2645 |
| heartwall † | Snake | +0.22% | 0.0 | 0.4 | 6.5 | 0.0 | 0.0 | 100.0 | - | 2278 (-13.88%) |
| heartwall † | SK_SC (old) | -0.10% | 0.0 | 0.6 | 4.9 | 0.0 | 0.0 | 100.0 | - | 2273 (-14.06%) |
| heartwall † | SK | +0.08% | 0.1 | 0.7 | 2.5 | 14.2 | 0.0 | 85.8 | - | 2694 (+1.85%) |
| heartwall † | SK_R0 | -0.13% | 0.1 | 1.0 | 3.1 | 11.3 | 0.0 | 88.7 | - | 4408 (+66.65%) |
| heartwall † | SK_R0_D0 | -0.13% | 0.1 | 1.0 | 3.1 | 11.3 | 0.0 | 88.7 | - | 4408 (+66.65%) |
| particlefilter_float | base | | | | | | | | | 22775 |
| particlefilter_float | Snake | +8.68% | 19.2 | 24.5 | 90.3 | 78.2 | 0.3 | 21.5 | 15.9 | 22939 (+0.72%) |
| particlefilter_float | SK_SC (old) | +8.83% | 22.4 | 29.5 | 90.4 | 75.8 | 0.1 | 24.0 | 15.6 | 24461 (+7.40%) |
| particlefilter_float | SK | +8.83% | 22.4 | 29.5 | 90.4 | 75.8 | 0.1 | 24.0 | 15.6 | 24461 (+7.40%) |
| particlefilter_float | SK_R0 | +8.63% | 25.0 | 32.3 | 91.2 | 77.2 | 0.2 | 22.7 | 18.1 | 22518 (-1.13%) |
| particlefilter_float | SK_R0_D0 | +9.09% | 24.8 | 31.6 | 92.9 | 78.4 | 0.0 | 21.6 | 21.2 | 22391 (-1.69%) |
| particlefilter_naive | base | | | | | | | | | 1581 |
| particlefilter_naive | Snake | +65.60% | 45.2 | 58.4 | 99.3 | 77.3 | 0.0 | 22.7 | - | 342 (-78.37%) |
| particlefilter_naive | SK_SC (old) | +78.49% | 65.4 | 74.0 | 98.0 | 88.4 | 0.0 | 11.6 | - | 341 (-78.43%) |
| particlefilter_naive | SK | +78.49% | 65.4 | 74.0 | 98.0 | 88.4 | 0.0 | 11.6 | - | 341 (-78.43%) |
| particlefilter_naive | SK_R0 | +78.45% | 62.8 | 75.1 | 97.7 | 83.6 | 0.0 | 16.4 | - | 329 (-79.19%) |
| particlefilter_naive | SK_R0_D0 | +78.45% | 62.8 | 75.1 | 97.7 | 83.6 | 0.0 | 16.4 | - | 329 (-79.19%) |
| streamcluster † | base | | | | | | | | | 544477 |
| streamcluster † | Snake | -0.02% | 0.9 | 2.4 | 100.0 | 36.7 | 0.0 | 63.3 | 0.0 | 519929 (-4.51%) |
| streamcluster † | SK_SC (old) | -0.01% | 0.8 | 2.4 | 100.0 | 32.2 | 0.0 | 67.8 | 0.0 | 521556 (-4.21%) |
| streamcluster † | SK | -0.11% | 0.7 | 2.3 | 100.0 | 29.0 | 0.0 | 71.0 | 0.0 | 519256 (-4.63%) |
| streamcluster † | SK_R0 | -0.26% | 0.8 | 2.8 | 100.0 | 26.8 | 0.0 | 73.2 | 0.0 | 530720 (-2.53%) |
| streamcluster † | SK_R0_D0 | -0.26% | 0.8 | 2.8 | 100.0 | 26.8 | 0.0 | 73.2 | - | 530720 (-2.53%) |

### Inter-warp training: per-warp Head (SK) vs two-warp Head
`interwarp_trained` counts training events (re-training after a Tail eviction counts again). `interwarp_trained_pcs` counts distinct (kernel launch, PC) pairs. `pred_inter_warp` counts inter-warp predictions, which depend on the target scope.

`interwarp_trained` counts training events (re-training after a Tail eviction counts again); `interwarp_trained_pcs` counts distinct (kernel launch, PC) pairs; `pred_inter_warp` the inter-warp predictions.

| workload | Head | config | interwarp_trained | interwarp_trained_pcs | pred_inter_warp | pf_issued_inter_warp |
|---|---|---|---|---|---|---|
| backprop | two-warp | SK_SC (old) | 11593 | 2 | 1397754 | 78019 |
| backprop | per-warp (64) | SK | 12626 | 2 | 200928 | 56292 |
| hotspot 1024 | two-warp | SK_SC (old) | 160 | 2 | 483054 | 73929 |
| hotspot 1024 | two-warp | SK_SC_CTA (old) | 160 | 2 | 122216 | 36784 |
| hotspot 1024 | per-warp (64) | SK | 160 | 2 | 121910 | 37884 |
| hotspot 512 | two-warp | SK_SC (old) | 160 | 2 | 434174 | 53583 |
| hotspot 512 | two-warp | SK_SC_CTA (old) | 160 | 2 | 100709 | 29276 |
| hotspot 512 | per-warp (64) | SK | 160 | 2 | 100709 | 31121 |
| lud 512 | two-warp | SK_SC (old) | 10752 | 21 | 3293616 | 142468 |
| lud 512 | per-warp (64) | SK | 10923 | 21 | 402247 | 34344 |
| lud 256 | two-warp | SK_SC (old) | 2691 | 45 | 171782 | 49087 |
| lud 256 | per-warp (64) | SK | 2791 | 45 | 80508 | 21503 |
| nw | two-warp | SK_SC (old) | 0 | 0 | 0 | 0 |
| nw | per-warp (64) | SK | 0 | 0 | 0 | 0 |
| histo | two-warp | SK_SC (old) | 1261 | 20 | 1252185 | 143139 |
| histo | per-warp (64) | SK | 1262 | 20 | 1257291 | 156304 |
| mri-q | two-warp | SK_SC (old) | 526 | 7 | 26806 | 1366 |
| mri-q | per-warp (64) | SK | 668 | 7 | 15430 | 1318 |
| srad_v1 | two-warp | SK_SC (old) | 7335 | 16 | 2030679 | 17656 |
| srad_v1 | two-warp | SK_SC_CTA (old) | 7357 | 16 | 572025 | 14832 |
| srad_v1 | per-warp (64) | SK | 7194 | 16 | 560354 | 14941 |
| b+tree | two-warp | SK_SC (old) | 98407 | 29 | 2721135 | 408643 |
| b+tree | per-warp (64) | SK | 79059 | 29 | 495516 | 55978 |
| heartwall | two-warp | SK_SC (old) | 1476 | 15 | 7861 | 2509 |
| heartwall | per-warp (64) | SK | 879 | 14 | 6055 | 5833 |
| particlefilter_float | two-warp | SK_SC (old) | 17252 | 129 | 260124 | 14555 |
| particlefilter_float | per-warp (64) | SK | 8765 | 128 | 193225 | 14551 |
| particlefilter_naive | two-warp | SK_SC (old) | 263 | 36 | 93271 | 946 |
| particlefilter_naive | per-warp (64) | SK | 263 | 36 | 93271 | 946 |
| streamcluster | two-warp | SK_SC (old) | 19580 | 2434 | 468003 | 10330 |
| streamcluster | per-warp (64) | SK | 8960 | 2410 | 294864 | 9728 |

### Reproduce
```bash
Q=QV100-SASS-100M_INSN; B=$Q-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64
APPS=rodinia-3.1:backprop-rodinia-3.1,rodinia-3.1:hotspot-rodinia-3.1,rodinia-3.1:srad_v1-rodinia-3.1,rodinia-3.1:lud-rodinia-3.1,rodinia-3.1:nw-rodinia-3.1,parboil:parboil-histo,parboil:parboil-mri-q,rodinia-3.1:b+tree-rodinia-3.1,rodinia-3.1:particlefilter_float-rodinia-3.1,rodinia-3.1:particlefilter_naive-rodinia-3.1,rodinia_2.0-ft:heartwall-rodinia-2.0-ft,rodinia_2.0-ft:streamcluster-rodinia-2.0-ft \
  util/snake/run_apps.sh sim_run_snake_skm/qv100 sim_run_snake_skc/qv100_7apps/bin "$B,$B-SNAKE_MQR0,$B-SNAKE_MQR0-SNAKE_DM0" 14
python3 util/snake/sk_match_report.py $PWD/sim_run_snake_skm/qv100/runs $PWD/sim_run_snake_skc
```
Fig. 15 check: `check_synthetic.py sim_run_snake_synth/SKM_HEAD64/runs sim_run_snake_synth/SKM_HEAD64/traces QV100-SASS-SYNTH_LRR1-SNAKE_DT-SNAKE_DBG-SNAKE_HEAD64 snake-synth-fig15` (log `/home/razul/snake/synthetic_SKM_HEAD64.log`).

## Best Snake (SK_R0_D0) vs Tier 1's Snake: window, setup and config (2026-10-27)
SK_R0_D0's best kernels on base QV100 at 100M gained less than Tier 1 Snake's (K1_best_kernel.md: PAPER_V100, 1B). Is that the Snake config, the GPU setup or the 100M window? SK_R0_D0 was run at Tier 1 conditions to separate the three.

**Best-kernel results are diagnostic only.** A kernel's gain does not predict the run's: under SK_R0_D0 histo's best kernel gains +42% while the whole run loses 3.4%. The whole-run gain is the result.

### Runs and method
- **Runs:**
  - 9 (our 7 apps), `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0`, in `sim_run_snake_skm/v100p_1b`.
  - Build: the frozen SK build `5d55de3` / `9622012e` (`sim_run_snake_skc/qv100_7apps/bin`, both `_modified_0.0`), as in Step 1.
  - All 9 finished (22 min, 9 in parallel). `collect_results.py`: 0 failed checks (build tags, outcome and funnel invariants, no warp woken by a prefetch).
- **Baseline:** Tier 1's no-prefetch runs (`sim_run_baseline`, build `2c3a333` / `3f635b67`).
  - The SK build with Snake off equals `bin_ref_v2` (off-regression).
  - Within the first 100M, every complete launch (cycles, instructions) of Tier 1's baseline and Snake runs equals that of the SK build's 100M runs, on 9 of 9 workloads.
- **Steps.** Every run except SK_R0_D0 at 1B already existed:
  - Tier 1 Snake (exact footprint), PAPER_V100, 1B.
  - The same runs, restricted to the launches that start within the first 100M instructions (the "window").
  - Snake, PAPER_V100, 100M (`sim_run_snake_skc/v100p_7apps`; baseline `sim_run_snake_refcache/bin_ref_v2`).
  - Snake, base QV100, 100M (`sim_run_snake_skc/qv100_7apps`).
  - SK_R0_D0 at 1B, its window, and SK_R0_D0 on base QV100 at 100M (Step 1).
- **Per kernel, as in K1** (`per_kernel.py`):
  - Launches pair in order, and a kernel is all launches of a function.
  - The best kernel has the largest gain among kernels with at least 1% of the baseline cycles.
  - At 1B the launch cut by the cap is excluded, as in K1.
  - At 100M that launch is kept and marked `*`: it is the only launch of backprop and hotspot 1024, and it holds mri-q's and srad_v1's main kernels.
- **GMEAN** is over the 7 apps, an app's inputs combined first, as in final_tier1.md and K1. Step 1's tables average over workloads instead: SK_R0_D0's 7-app GMEAN is −0.00% there and −0.20% here.

### Whole run (primary)
| workload | Tier 1 Snake | SK_R0_D0 | difference |
|---|---|---|---|
| backprop | -0.20% | +0.16% | +0.36 pp |
| hotspot 1024 | +0.16% | +0.14% | -0.03 pp |
| hotspot 512 | +0.13% | -0.09% | -0.22 pp |
| lud 512 | +0.12% | -0.11% | -0.23 pp |
| lud 256 | +0.08% | +0.09% | +0.01 pp |
| nw | +0.00% | +0.00% | +0.00 pp |
| histo | -1.93% | -3.44% | -1.51 pp |
| mri-q | +0.09% | +0.21% | +0.12 pp |
| srad_v1 | -0.21% | -0.53% | -0.32 pp |
| **GMEAN** (7 apps) | -0.29% | -0.52% | -0.23 pp |
| GMEAN (9 workloads) | -0.20% | -0.40% | -0.21 pp |

- **At Tier 1 conditions SK_R0_D0 is below Tier 1's Snake:** −0.52% vs −0.29%.
  - histo decides it: −3.44% vs −1.93%. srad_v1 also drops (−0.53% vs −0.21%).
  - backprop is the only clear gain (+0.16% vs −0.20%).
- **The best Snake on base QV100 at 100M is not the best at Tier 1 conditions.** The SK config adds +0.42 pp on base QV100 at 100M but −0.23 pp on PAPER_V100 at 1B.
- **particlefilter_naive** is not a Tier 1 workload, so the split with and without it does not apply to this comparison. On SK_R0_D0's 14 workloads (base QV100, 100M), the reference is:

| GMEAN over workloads | n | whole run | best kernel (diagnostic) |
|---|---|---|---|
| all | 14 | +4.77% | +9.26% |
| without particlefilter_naive | 13 | +0.57% | +5.21% |
| our 7 apps | 9 | -0.00% | +1.35% |
| SnakeCharmer apps | 5 | +13.96% | +25.09% |

### Best kernel (diagnostic only)
| workload | Tier 1 Snake: best kernel (share of cycles) | gain | SK_R0_D0: best kernel (share) | gain |
|---|---|---|---|---|
| backprop | `bpnn_layerforward_CUDA` (46.3%) | +0.07% | `bpnn_adjust_weights_cuda` (53.7%) | +0.29% |
| hotspot 1024 | `calculate_temp` (100.0%) | +0.16% | `calculate_temp` (100.0%) | +0.14% |
| hotspot 512 | `calculate_temp` (100.0%) | +0.13% | `calculate_temp` (100.0%) | -0.09% |
| lud 512 | `lud_internal` (21.3%) | +0.59% | `lud_perimeter` (27.1%) | +0.05% |
| lud 256 | `lud_internal` (18.4%) | +0.43% | `lud_internal` (18.4%) | +0.51% |
| nw | `needle_cuda_shared_1` (50.1%) | +0.00% | `needle_cuda_shared_1` (50.1%) | +0.00% |
| histo | `histo_intermediates_kernel` (17.1%) | +11.48% | `histo_intermediates_kernel` (17.1%) | +42.17% |
| mri-q | `ComputeQ_GPU` (97.6%) | +0.10% | `ComputeQ_GPU` (97.6%) | +0.22% |
| srad_v1 | `extract` (1.3%) | +2.56% | `extract` (1.3%) | +3.00% |
| **GMEAN** (7 apps) | | +2.05% | | +5.72% |

- **histo** decides the GMEAN, and its best kernel is the clearest case where a kernel's gain misleads.
  - `histo_intermediates_kernel` gains +42.17%: +8.1% in its first launch, +43.5–47.4% in the eight after it. With Tier 1 Snake it gains +7.1% and then +9.7–13.9%.
  - The other kernels pay for it: `histo_prescan_kernel` −46.5% (Snake −10.8%), `histo_final_kernel` −23.8% (Snake −2.8%), `histo_main_kernel` +0.07% (Snake −3.88%, 66% of the cycles). The whole run ends at −3.44%.
- **lud 512's** best kernel changes from `lud_internal` (Snake +0.59%, SK_R0_D0 −0.57%) to `lud_perimeter` (+0.05%).
- **backprop's** changes to `bpnn_adjust_weights_cuda` (+0.29%; Snake −0.43%).

### Window, setup and config
| step | whole run | best kernel (diagnostic) |
|---|---|---|
| Tier 1 Snake: PAPER_V100, 1B | -0.29% | +2.05% |
| Tier 1 Snake, launches started in the first 100M | -0.16% | +1.48% |
| Snake: PAPER_V100, 100M | -0.14% | +1.47% |
| Snake: base QV100, 100M | -0.62% | +0.72% |
| SK_R0_D0: PAPER_V100, 1B | -0.52% | +5.72% |
| SK_R0_D0 1B, launches started in the first 100M | -0.46% | +1.61% |
| SK_R0_D0: base QV100, 100M | -0.20% | +1.51% |

Effects, in points (whole run / best kernel):
- window, Tier 1 Snake (T1 -> T1 win): +0.13 / -0.57
- window vs 100M runs (T1 win -> P100; complete launches identical): +0.01 / -0.01
- setup, Snake (P100 -> Q100): -0.48 / -0.75
- config at base QV100, 100M (Q100 -> SK Q100): +0.42 / +0.79
- config at Tier 1 conditions (T1 -> SK 1B): -0.23 / +3.67
- window, SK_R0_D0 (SK 1B -> SK 1B win): +0.06 / -4.11
- setup, SK_R0_D0 (SK 1B win -> SK Q100): +0.26 / -0.11

- **Why SK_R0_D0's best kernels on base QV100 at 100M (+1.51%) were below Tier 1 Snake's (+2.05%):** the window and the setup, not the config.
  - **Snake:** the window costs −0.57 pp, because 100M holds only histo's first `histo_intermediates_kernel` launch. The setup costs −0.75 pp. The config adds +0.79 pp on base QV100.
  - **SK_R0_D0:** the config adds +3.67 pp at Tier 1 conditions. Then the window removes −4.11 pp, because the eight later histo launches carry the gain. The setup costs only −0.11 pp.
- **For the whole run the picture is different.** The window effects are small (+0.13 / +0.06 pp) and base QV100 costs Snake −0.48 pp. The config helps on base QV100 (+0.42 pp) and hurts at Tier 1 conditions (−0.23 pp).

### Per kernel, step by step (diagnostic only)
`*` = includes the launch cut by the 100M cap. Launches = at 1B / within the first 100M.

| workload | kernel (best in) | launches | T1 | T1 win | P100 | Q100 | SK 1B | SK 1B win | SK Q100 |
|---|---|---|---|---|---|---|---|---|---|
| backprop | `bpnn_layerforward_CUDA` (T1, SK Q100) | 1/1 | +0.07 | +0.07 | -0.10* | -0.25* | +0.01 | +0.01 | -0.32* |
| backprop | `bpnn_adjust_weights_cuda` (SK 1B) | 1/0 | -0.43 | n/a | n/a | n/a | +0.29 | n/a | n/a |
| hotspot 1024 | `calculate_temp` (T1, SK 1B, SK Q100) | 1/1 | +0.16 | +0.16 | +0.13* | +0.06* | +0.14 | +0.14 | +0.30* |
| hotspot 512 | `calculate_temp` (T1, SK 1B, SK Q100) | 1/1 | +0.13 | +0.13 | +0.13 | +0.79 | -0.09 | -0.09 | +2.64 |
| lud 512 | `lud_internal` (T1) | 31/7 | +0.59 | +1.00 | +0.99* | +0.08* | -0.57 | -2.33 | -1.07* |
| lud 512 | `lud_perimeter` (SK 1B, SK Q100) | 31/7 | -0.00 | -0.03 | -0.03 | +0.01 | +0.05 | +0.20 | +0.18 |
| lud 256 | `lud_internal` (T1, SK 1B, SK Q100) | 15/15 | +0.43 | +0.43 | +0.43 | +0.19 | +0.51 | +0.51 | +0.12 |
| nw | `needle_cuda_shared_1` (T1, SK 1B, SK Q100) | 128/128 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| histo | `histo_intermediates_kernel` (T1, SK 1B, SK Q100) | 9/1 | +11.48 | +7.07 | +7.07 | +4.34 | +42.17 | +8.06 | +6.38 |
| mri-q | `ComputeQ_GPU` (T1, SK 1B, SK Q100) | 2/1 | +0.10 | -0.12 | +0.14* | +0.28* | +0.22 | +0.10 | +0.40* |
| srad_v1 | `extract` (T1, SK 1B, SK Q100) | 1/1 | +2.56 | +2.56 | +2.56 | +0.18 | +3.00 | +3.00 | +2.63 |

### Reproduce
```bash
C=QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0
util/snake/run_apps.sh sim_run_snake_skm/v100p_1b sim_run_snake_skc/qv100_7apps/bin "$C" 9
python3 util/snake/kernel_gap.py
```
Logs: `/home/razul/snake/skm_v100p_1b.log` and `/home/razul/snake/kernel_gap.md`.

## Step 2: R2 injection path (waiting for approval)
Spec: [R2_SPEC.md](../R2_SPEC.md), SPEC OQ-38. Proposed base: SK_R0_D0.

## Step 3: ceilings (waiting)

## Step 4: bfs trace diagnosis (waiting)
