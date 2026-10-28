# Backtrace of the MTP slide's IPC numbers (2026-10-28)

**The slide, as given:** "Snake predicts 63% of addresses (76%), but only 6% arrive in time, so IPC is flat (1.7% vs 14.9%)".
- The slide number, title, legend and per-app values were left blank in the request.
- So the match below rests on the sentence's five numbers. The per-workload table in Step 2 shows what the graph's bars would be under the method found; compare them with the slide to confirm.

No new simulations. Script: `util/snake/slide_backtrace.py` (about 4 minutes, reads existing runs).

## Step 1: the source

### Search (nothing stores the slide's averages)
- **Searched:**
  - `docs/snake/results/` (md, csv; the xlsx files read with openpyxl: `Snake_results_2026-10-19.xlsx`, `Snake_results_2026-10-27.xlsx`, `slides/Snake_presentation_2026-10-27.xlsx`)
  - `util/snake/` scripts and their outputs in `/home/razul/snake/`
  - every xlsx under `/home/razul`, and the slides / PNG folders
  - all 179 Claude session scratchpads under `/tmp/claude-1001/-home-razul/*/scratchpad`, and the 16 session transcripts
  - `git log --all -S / -G` on every branch of both repos (Snake, dev, scap, oracle, oracle-coalesced-miss, oracle-sector-miss, snake-ref-v2; gpgpu-sim also oracle-latency-hiding, oracle-traffic-elimination)
- **Result:** no file, transcript or commit holds 1.7 next to 14.9, or 1.72 / 14.89 / 1.017 / 0.017 as an IPC average. The hits for 14.9% are unrelated:
  - SCAP sector-mask shares (`07632ab`, `b0c310a`)
  - Snake S3c / S4 diagnostics
  - SK_match's eviction share
  - AccelWattch / DeepBench workbooks
- **No match is from the A100 oracle work.** The oracle branches have no IPC table with either number.

### Candidates recomputed (Tier 1 Snake = exact footprint, our 9 workloads)
| measurement | conditions | averaging | coverage_pred | timely (accuracy_paper) | IPC gain, arithmetic | IPC gain, geometric |
|---|---|---|---|---|---|---|
| whole run | PAPER_V100, 1B (Tier 1) | 7 apps | 63.2 **← slide 63** | 6.1 **← slide 6** | -0.29 | -0.29 |
| whole run | PAPER_V100, 1B (Tier 1) | 9 workloads | 68.1 | 7.9 | -0.19 | -0.20 |
| best kernel | PAPER_V100, 1B (Tier 1) | 7 apps | 73.1 | 5.2 | +2.12 | +2.05 |
| best kernel | PAPER_V100, 1B (Tier 1) | 9 workloads | 76.5 **← slide 76** | 7.2 | +1.72 **← slide 1.7** | +1.66 **← slide 1.7** |
| whole run | PAPER_V100, 100M | 7 apps | 64.0 | 6.4 **← slide 6** | -0.14 | -0.14 |
| whole run | PAPER_V100, 100M | 9 workloads | 68.6 | 8.0 | -0.08 | -0.08 |
| best kernel | PAPER_V100, 100M | 7 apps | 73.4 | 4.5 | +1.50 | +1.47 |
| best kernel | PAPER_V100, 100M | 9 workloads | 76.5 **← slide 76** | 6.5 | +1.26 | +1.24 |
| whole run | base QV100, 100M | 7 apps | 64.9 | 1.7 | -0.61 | -0.62 |
| whole run | base QV100, 100M | 9 workloads | 70.0 | 1.5 | -0.42 | -0.43 |
| best kernel | base QV100, 100M | 7 apps | 73.9 | 0.6 | +0.73 | +0.72 |
| best kernel | base QV100, 100M | 9 workloads | 77.7 | 0.7 | +0.63 | +0.62 |
| best kernel over 7 variants | PAPER_V100, 1B (Tier 1) | 7 apps | 67.6 | 3.9 | +2.48 | +2.41 |
| best kernel over 7 variants | PAPER_V100, 1B (Tier 1) | 9 workloads | 69.1 | 5.4 | +2.06 | +2.00 |
| **paper** (SPEC §5, Figs. 16–18) | paper | 7 apps | 75.7 **← slide 76** | 69.1 | +14.71 | +14.14 |
| **paper** (SPEC §5, Figs. 16–18) | paper | 9 workloads | 78.8 | 72.3 | +14.89 **← slide 14.9** | +14.43 |
| **paper**, its own 11 apps | paper | 11 apps | 80.1 (table: 80) | 75.6 (table: 75) | +16.45 | +15.32 (table: +17) |

(The paper's "11 apps" row is the SPEC §5 table: mean over its 11 rows, and its printed Avg / GMEAN row.)

### The match
- **IPC "1.7% vs 14.9%" = the column means of the K1 table, unweighted over its 9 rows.**
  - **Ours, 1.7%:** the best kernel's IPC gain, arithmetic mean over the 9 workloads = **+1.72%** (geometric +1.66% also rounds to 1.7, but the paper's geometric mean over the same rows is 14.43, not 14.9).
  - **Paper, 14.9%:** the paper's per-app IPC gain (SPEC §5, Fig. 18, read off the bar chart), with each app's value repeated for each of its inputs, arithmetic mean over the same 9 rows = **+14.89%**.
- **Coverage "63%" and "only 6% arrive in time"** are the whole-run 7-app means of `final_tier1.md` §3 (63.2 and 6.1).
- **"(76%)" is either:**
  - the paper's coverage in that same table (75.7, 7-app mean; the "ours (paper)" pattern of the IPC pair), or
  - our best-kernel coverage_pred over the 9 workloads (76.5).

  The sentence does not decide which.
- **The slide mixes two measurements:** whole-run, 7-app means for coverage and accuracy, and best-kernel, 9-workload means for IPC.

| item | value |
|---|---|
| file | `docs/snake/results/K1_best_kernel.md` §1, column "IPC gain: paper / best kernel / whole run" (also the `Best_kernel` sheet of `Snake_results_2026-10-19.xlsx`). Coverage and accuracy: `docs/snake/results/final_tier1.md` §3 |
| commit | K1: accel-sim `5c92662` (2026-10-18, "K1: best-performing kernel vs the paper's per-benchmark numbers"). Tier 1: `03ad5bc` (2026-10-16) |
| script | `util/snake/per_kernel.py` (K1), `util/snake/compare_paper.py` (final_tier1 §3); the averages: `util/snake/slide_backtrace.py` |
| runs | no prefetching: `sim_run_baseline`, `QV100-SASS-PAPER_V100-1B_INSN`; Snake: `sim_run_snake`, `QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1` |
| build | `accelsim-commit-2c3a333_modified_0.0`, `gpgpu-sim_git-commit-3f635b67_modified_0.0` (Tier 1 frozen build) |
| GPU setup | PAPER_V100: QV100 config, 80 SMs, GTO, 1530 MHz; traces `hw_run` (sm_70) |
| instruction cap | 1B (whole run if shorter) |
| metric | per kernel: kernel IPC = instructions / cycles over all complete launches of the kernel function (the launch cut by the 1B cap excluded); gain = Snake / no-prefetch kernel IPC − 1 |
| kernel selection | per workload, the largest gain among kernels with ≥ 1% of the no-prefetch run's cycles (Snake exact footprint only, not across variants) |
| baseline | Tier 1's no-prefetch run of the same workload |
| averaging | unweighted arithmetic mean over the 9 workloads (hotspot and lud count twice), for ours and for the paper |
| apps / workloads | backprop, hotspot (1024, 512), lud (512, 256), nw, histo, mri-q, srad_v1 |

**Rejected candidates:**
- whole-run IPC: −0.29 / −0.20%
- best kernel over 7 apps: +2.12% / +2.05%
- best kernel over all seven Tier 1 variants (K1 §2): +2.06% / +2.48%
- 100M window: +1.26% / +1.50%
- base QV100: +0.63% / +0.73%
- paper over 7 apps: +14.71% / +14.14%
- paper's own 11-app average: +17%

## Step 2: the same method for SK_R0_D0 and the fix_today configs
The runs exist under identical conditions (PAPER_V100, 1B, the same `sim_run_baseline` baseline), so no new runs are needed.
- SK_R0_D0: `sim_run_snake_skm/v100p_1b`, build `5d55de3` / `9622012e`.
- fix_today configs: `sim_run_snake_fix/{p2,p3,p3b,p4}`, builds `c09021a` / `bb20e7aa` (A–C) and `4b51495` / `19e3d746` (D–G32).

Normalized IPC of each workload's best kernel (1 = the same kernel without prefetching); last row the slide's average:

| workload | paper (app) | Tier 1 Snake | SK_R0_D0 | A | B | C | D | E | F | E10 | G8 | G16 | G32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| backprop | 1.01 | 1.0007 `bpnn_layerforward_CUDA` | 1.0029 `bpnn_adjust_weights_cuda` | 1.0005 `bpnn_adjust_weights_cuda` | 1.0027 `bpnn_adjust_weights_cuda` | 0.9995 `bpnn_layerforward_CUDA` | 1.0007 `bpnn_layerforward_CUDA` | 1.0001 `bpnn_layerforward_CUDA` | 1.0021 `bpnn_layerforward_CUDA` | 1.0001 `bpnn_layerforward_CUDA` | 1.0030 `bpnn_adjust_weights_cuda` | 0.9982 `bpnn_layerforward_CUDA` | 1.0044 `bpnn_adjust_weights_cuda` |
| hotspot 1024 | 1.13 | 1.0016 `calculate_temp` | 1.0014 `calculate_temp` | 1.0011 `calculate_temp` | 1.0008 `calculate_temp` | 1.0014 `calculate_temp` | 1.0019 `calculate_temp` | 1.0020 `calculate_temp` | 1.0011 `calculate_temp` | 1.0030 `calculate_temp` | 1.0017 `calculate_temp` | 1.0013 `calculate_temp` | 1.0010 `calculate_temp` |
| hotspot 512 | 1.13 | 1.0013 `calculate_temp` | 0.9991 `calculate_temp` | 0.9990 `calculate_temp` | 1.0017 `calculate_temp` | 1.0005 `calculate_temp` | 1.0020 `calculate_temp` | 1.0007 `calculate_temp` | 1.0024 `calculate_temp` | 1.0007 `calculate_temp` | 1.0017 `calculate_temp` | 1.0033 `calculate_temp` | 1.0009 `calculate_temp` |
| lud 512 | 1.18 | 1.0059 `lud_internal` | 1.0005 `lud_perimeter` | 1.0003 `lud_perimeter` | 1.0000 `lud_diagonal` | 1.0000 `lud_diagonal` | 1.0065 `lud_internal` | 1.0004 `lud_perimeter` | 1.0000 `lud_diagonal` | 1.0002 `lud_perimeter` | 1.0001 `lud_perimeter` | 1.0001 `lud_perimeter` | 1.0001 `lud_perimeter` |
| lud 256 | 1.18 | 1.0043 `lud_internal` | 1.0051 `lud_internal` | 1.0047 `lud_internal` | 1.0054 `lud_internal` | 1.0050 `lud_internal` | 1.0043 `lud_internal` | 1.0051 `lud_internal` | 1.0054 `lud_internal` | 1.0051 `lud_internal` | 1.0054 `lud_internal` | 1.0056 `lud_internal` | 1.0056 `lud_internal` |
| nw | 1.05 | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` |
| histo | 1.33 | 1.1148 `histo_intermediates_kernel` | 1.4217 `histo_intermediates_kernel` | 1.1197 `histo_intermediates_kernel` | 1.4332 `histo_intermediates_kernel` | 1.4223 `histo_intermediates_kernel` | 1.1121 `histo_intermediates_kernel` | 1.3185 `histo_intermediates_kernel` | 1.3211 `histo_intermediates_kernel` | 1.2826 `histo_intermediates_kernel` | 1.2359 `histo_intermediates_kernel` | 1.2068 `histo_intermediates_kernel` | 1.2095 `histo_intermediates_kernel` |
| mri-q | 1.04 | 1.0010 `ComputeQ_GPU` | 1.0022 `ComputeQ_GPU` | 1.0024 `ComputeQ_GPU` | 1.0019 `ComputePhiMag_GPU` | 1.0015 `ComputeQ_GPU` | 1.0010 `ComputeQ_GPU` | 1.0022 `ComputeQ_GPU` | 1.0019 `ComputePhiMag_GPU` | 1.0022 `ComputeQ_GPU` | 1.0012 `ComputeQ_GPU` | 1.0016 `ComputePhiMag_GPU` | 1.0016 `ComputePhiMag_GPU` |
| srad_v1 | 1.29 | 1.0256 `extract` | 1.0300 `extract` | 1.0221 `extract` | 1.0232 `extract` | 1.0178 `extract` | 1.0256 `extract` | 1.0300 `extract` | 1.0232 `extract` | 1.0300 `extract` | 1.0248 `extract` | 1.0021 `reduce` | 1.0014 `reduce` |
| **arithmetic mean of gains, 9 workloads** | +14.89% | +1.72% | +5.14% | +1.66% | +5.21% | +4.98% | +1.71% | +3.99% | +3.97% | +3.60% | +3.04% | +2.43% | +2.49% |

- **Same method, SK_R0_D0 gives +5.14%** where the slide's Tier 1 Snake gives +1.72% (paper +14.89%). B (SK_R0_D0 + R2) is the highest at +5.21%.
- The best kernels' coverage_pred / timely accuracy:

| workload | paper | Tier 1 Snake | SK_R0_D0 |
|---|---|---|---|
| backprop | 83 / 68 | 71.9 / 0.9 | 49.8 / 0.1 |
| hotspot 1024 | 88 / 85 | 93.3 / 27.0 | 93.8 / 8.1 |
| hotspot 512 | 88 / 85 | 90.6 / 26.4 | 90.7 / 16.2 |
| lud 512 | 91 / 82 | 86.2 / 2.8 | 0.0 / 0.0 |
| lud 256 | 91 / 82 | 82.5 / 0.8 | 83.6 / 0.9 |
| nw | 12 / 10 | 0.0 / 0.0 | 0.0 / 0.0 |
| histo | 97 / 96 | 83.3 / 7.0 | 84.0 / 9.9 |
| mri-q | 72 / 60 | 88.4 / 0.0 | 88.8 / 0.9 |
| srad_v1 | 87 / 83 | 92.0 / 0.1 | 92.1 / 0.0 |
| arithmetic mean, 9 workloads | 78.8 / 72.3 | 76.5 / 7.2 | 64.8 / 4.0 |

## Step 3: the difference, per workload (same method)
| workload | paper (app value) | Tier 1 Snake (slide source) | SK_R0_D0 | difference (points) | attribution |
|---|---|---|---|---|---|
| backprop | 1.01 | 1.0007 `bpnn_layerforward_CUDA` | 1.0029 `bpnn_adjust_weights_cuda` | +0.22 | CONFIG; the rule picks another kernel (`layerforward` → `adjust_weights`) |
| hotspot 1024 | 1.13 | 1.0016 `calculate_temp` | 1.0014 `calculate_temp` | -0.02 | CONFIG (±0.02 points) |
| hotspot 512 | 1.13 | 1.0013 `calculate_temp` | 0.9991 `calculate_temp` | -0.22 | CONFIG |
| lud 512 | 1.18 | 1.0059 `lud_internal` | 1.0005 `lud_perimeter` | -0.54 | CONFIG; SK makes `lud_internal` lose (−0.57%), so the rule picks `lud_perimeter` |
| lud 256 | 1.18 | 1.0043 `lud_internal` | 1.0051 `lud_internal` | +0.08 | CONFIG |
| nw | 1.05 | 1.0000 `needle_cuda_shared_1` | 1.0000 `needle_cuda_shared_1` | +0.00 | none (no prediction in either) |
| histo | 1.33 | 1.1148 `histo_intermediates_kernel` | 1.4217 `histo_intermediates_kernel` | +30.69 | CONFIG (same kernel: +11.48% → +42.17%) |
| mri-q | 1.04 | 1.0010 `ComputeQ_GPU` | 1.0022 `ComputeQ_GPU` | +0.12 | CONFIG |
| srad_v1 | 1.29 | 1.0256 `extract` | 1.0300 `extract` | +0.44 | CONFIG (same kernel) |
| **arithmetic mean of gains, 9 workloads** | +14.89% | +1.72% | +5.14% | +3.42 | CONFIG (histo alone: +3.41 of +3.42) |

- **METHOD:** identical (same selection rule, averaging and baseline runs, same script). It contributes nothing to the Tier 1 → SK_R0_D0 difference.
  - Two workloads change kernel (backprop, lud 512). That is the selection rule responding to the config, not a change of method.
- **CONDITIONS:** identical: PAPER_V100, 1B, the same sm_70 traces and the same no-prefetch runs. Only the build differs, and neither build change is a source:
  - With Snake off, the SK build's output equals `bin_ref_v2` (off-regression, SK_SC.md §3).
  - Within the first 100M, every complete launch of Tier 1's runs equals the SK build's 100M runs on 9 of 9 workloads (SK_match_qv100.md).
- **CONFIG accounts for the whole difference.** The SK settings (SK_SC: queue 128, chain depth 10, intra-warp 1..8; same-CTA targets up to 63; per-warp Head; miss-queue reserve 0; decoupling off) change histo's `histo_intermediates_kernel` from +11.48% to +42.17%. That alone adds +3.41 of the +3.42 points in the 9-workload mean.
- **Differences between the slide's two numbers (METHOD):** our 1.7% and the paper's 14.9% use the same averaging, but they are not the same kind of number.
  - The paper's values are per benchmark, read off bar charts (±3 pp), and repeated for both inputs of hotspot and lud. The paper does not say whether they are whole-run or best-kernel numbers; K1 tested the best-kernel reading.
  - Ours is a per-kernel best case. Our whole-run value under the same averaging is −0.19% (9 workloads), −0.29% (7 apps, Tier 1's headline).
  - So "IPC is flat (1.7%)" overstates our side by about 2 points against the whole run.

### Best kernel over the seven Tier 1 variants (candidate c, not the slide's)
| workload | variant | kernel | IPC gain |
|---|---|---|---|
| backprop | line | `bpnn_adjust_weights_cuda` | +1.37% |
| hotspot 1024 | line | `calculate_temp` | +0.18% |
| hotspot 512 | Snake-ISO | `calculate_temp` | +0.32% |
| lud 512 | PAPER_LAT | `lud_internal` | +0.85% |
| lud 256 | PAPER_LAT | `lud_internal` | +1.03% |
| nw | exact | `needle_cuda_shared_1` | +0.00% |
| histo | PAPER_LAT | `histo_intermediates_kernel` | +12.00% |
| mri-q | PAPER_LAT | `ComputeQ_GPU` | +0.29% |
| srad_v1 | exact | `extract` | +2.56% |

## Reproduce
```bash
PYTHONPATH=/home/razul/snake/pylib python3 util/snake/slide_backtrace.py
```
