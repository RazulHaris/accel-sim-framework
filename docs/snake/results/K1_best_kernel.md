# K1: does the paper report its best-performing kernel? (2026-10-18)

Hypothesis (from the mentor): the paper's per-benchmark numbers (SPEC §5; Figs. 16 coverage, 17 accuracy, 18 IPC gain) may come from each benchmark's best-performing kernel, not the whole run.

The test: for each Tier 1 workload, take the kernel with Snake's largest IPC gain and compare its IPC gain, coverage, accuracy and L1 hit rate with the paper's values for that benchmark. Then check that kernel's perfect-L1 ceiling (D1), i.e. whether *any* prefetcher could reach the paper's value there.

**No new runs.** The data is the Tier 1 runs (1B cap; `sim_run_baseline`, `sim_run_snake`, `_dt`, `_t`, `sim_run_ssnake`) and the D1 perfect-L1D runs.
- gpgpu-sim prints a stats block at the end of each kernel launch: cycles, instructions and IPC of that launch, then cumulative L1D and Snake counters. A launch's counters are the differences of consecutive blocks.
- Baseline and Snake launches pair by launch uid and kernel name; every pair matched.
- The last launch of a run stopped by the 1B cap (histo, mri-q, srad_v1) is partial and excluded.
- "Kernel" = a kernel function, all its launches summed.
- Best = largest IPC gain among kernels with ≥ 1% of the workload's baseline cycles, so tiny kernels cannot win on noise. The best single launch is reported too.
- Script: `util/snake/per_kernel.py`.

## 1. Best kernel, Snake (exact footprint, the reference config)

| workload | best kernel (share of baseline cycles) | IPC gain: paper / best kernel / whole run | perfect-L1 ceiling of that kernel | coverage: paper / best kernel (pred, paper-def) | accuracy: paper / best kernel | L1 hit, best kernel: base → Snake |
|---|---|---|---|---|---|---|
| backprop | `bpnn_layerforward_CUDA` (46.3%) | +1% / +0.07% / −0.20% | +0.28% | 83 / 71.9, 43.7 | 68 / 0.9 | 3.4 → 5.7 |
| hotspot 1024 | `calculate_temp` (100%) | +13% / +0.16% / +0.16% | +0.13% | 88 / 93.3, 69.1 | 85 / 27.0 | 0.7 → 27.9 |
| hotspot 512 | `calculate_temp` (100%) | +13% / +0.13% / +0.13% | +0.33% | 88 / 90.6, 59.5 | 85 / 26.4 | 0.2 → 26.6 |
| lud 512 | `lud_internal` (21.3%) | +18% / +0.59% / +0.12% | +8.45% | 91 / 86.2, 29.8 | 82 / 2.8 | 4.4 → 7.1 |
| lud 256 | `lud_internal` (18.4%) | +18% / +0.43% / +0.08% | +6.04% | 91 / 82.5, 36.8 | 82 / 0.8 | 0.0 → 0.8 |
| nw | `needle_cuda_shared_1` (50.1%) | +5% / 0.00% / 0.00% | +16.34% | 12 / 0.0, 0.0 | 10 / 0.0 | 1.5 → 1.5 |
| histo | `histo_intermediates_kernel` (17.1%) | +33% / **+11.48%** / −1.95% | +131.1% | 97 / 83.3, 52.2 | 96 / 7.0 | 1.5 → 8.0 |
| mri-q | `ComputeQ_GPU` (97.6%) | +4% / +0.10% / +0.09% | +2.03% | 72 / 88.4, 9.2 | 60 / 0.0 | 0.0 → 0.0 |
| srad_v1 | `extract` (1.3%) | +29% / +2.56% / −0.19% | +6.88% | 87 / 92.0, 50.8 | 83 / 0.1 | 0.0 → 0.1 |

**Best single launch** (≥ 0.1% of the cycles):
- histo: launch 14 (`histo_intermediates_kernel`), +13.93%
- srad_v1: launch 10 (`srad`), +2.68%
- lud 512: launch 9 (`lud_internal`), +2.06%
- lud 256: launch 36 (`lud_internal`), +1.20%
- mri-q: launch 3 (`ComputeQ_GPU`), +0.32%
- the others: as in the table

## 2. Best kernel across all seven Tier 1 Snake variants

Variants: exact footprint, line footprint, Snake-DT, Snake-T, s-Snake, Snake-ISO, and PAPER_LAT (Snake exact vs the PAPER_LAT baseline). For each workload, the best kernel gain over all of them:

| workload | best gain (variant, kernel) | paper | accuracy of that kernel (paper) |
|---|---|---|---|
| backprop | **+1.37%** (line, `bpnn_adjust_weights`) | +1% | 0.1 (68) |
| hotspot 1024 | +0.18% (line) | +13% | 7.2 (85) |
| hotspot 512 | +0.32% (Snake-ISO) | +13% | 32.3 (85) |
| lud 512 | +0.85% (PAPER_LAT, `lud_internal`) | +18% | 2.1 (82) |
| lud 256 | +1.03% (PAPER_LAT, `lud_internal`) | +18% | 0.5 (82) |
| nw | 0.00% (all) | +5% | 0 (10) |
| histo | +12.00% (PAPER_LAT, `histo_intermediates_kernel`) | +33% | 6.2 (96) |
| mri-q | +0.29% (PAPER_LAT, `ComputeQ_GPU`) | +4% | 0.0 (60) |
| srad_v1 | +2.56% (exact / line / ISO, `extract`) | +29% | 0.1 (83) |

## 3. Perfect-L1 ceiling per kernel (D1 runs, PAPER_V100)

The most a prefetcher could gain in each kernel:

| workload | kernel: share of cycles, ceiling |
|---|---|
| backprop | `adjust_weights` 53.7%, +120.0%; `layerforward` 46.3%, +0.3% |
| hotspot 1024 / 512 | `calculate_temp` 100%, +0.13% / +0.33% |
| lud 512 | `diagonal` 51.7%, +1.0%; `perimeter` 27.1%, +5.1%; `internal` 21.3%, +8.5% |
| lud 256 | `diagonal` +1.1%; `perimeter` +4.9%; `internal` +6.0% |
| nw | `shared_1` +16.3%; `shared_2` +16.9% |
| histo | `main` 66.1%, +23.5%; `intermediates` 17.1%, +131.1%; `final` 10.2%, +71.5%; `prescan` 6.5%, +28.7% |
| mri-q | `ComputeQ` 97.6%, +2.0%; `ComputePhiMag` 2.4%, +7.7% |
| srad_v1 | `reduce` 46.2%, +24.1%; `srad` 23.1%, +6.8%; `srad2` 17.3%, +47.0%; `prepare` 12.1%, +10.7%; `extract` 1.3%, +6.9% |

Under PAPER_LAT the best ceilings are similar:
- histo `intermediates` +132%, backprop `adjust_weights` +120%, srad_v1 `srad2` +47%
- lud `internal` +9.5% / +7.4%, mri-q `ComputePhiMag` +8.2%, nw +18.2%
- hotspot ≤ +0.23%

## 4. How well the best kernel matches the paper

- **IPC gain: no match, except backprop.**
  - backprop's best kernel (+1.37%, line footprint; +0.07% exact) is at the paper's +1%.
  - On every other app the best kernel stays far below the paper:
    - histo +11.5–12.0% vs +33%
    - srad_v1 +2.6% vs +29%
    - lud ≤ +1.0% vs +18%
    - hotspot ≤ +0.3% vs +13%
    - mri-q ≤ +0.3% vs +4%
    - nw 0% vs +5%
  - Kernel selection does lift histo from −1.9% to +11.5%, about a third of the paper's value.
- **For hotspot and lud the hypothesis cannot explain the gap at all.** Even a perfect L1 gains at most +0.33% on hotspot's only kernel and +8.5% on lud's best kernel, against the paper's +13% and +18%. No kernel of these apps has the paper's headroom in our simulator.
  - For mri-q (`ComputePhiMag` +7.7%), nw (+16–17%), srad_v1 (`srad2` +47%) and histo (`intermediates` +131%), some kernel *could* match the paper's value. Snake gets only a small part of it.
- **Coverage (predicted): close to the paper on the best kernels.**
  - hotspot 90.6–93.3 vs 88; srad_v1 92.0 vs 87; lud 82.5–86.2 vs 91; mri-q 88.4 vs 72
  - backprop 71.9 vs 83; histo 83.3 vs 97 (the whole run: 34%)
  - nw 0 vs 12: Snake makes no predictions there.
  - The paper's "coverage" is most plausibly our coverage_pred (SPEC OQ-14), and on the best kernels it is within about ±15 pp.
- **Accuracy (timely): no match on any kernel.** The best kernels reach 0–27% (best: hotspot 26–27%, Snake-ISO 32%), against the paper's 60–96%. This is the prefetch lead-time gap found in S6.
- **L1 hit rate:** the best kernels rise by 0.1–27 pp (hotspot 0.2 → 26.6%). The paper's average goes 45 → 79%.

**Verdict:** reporting the best-performing kernel brings our predicted coverage close to the paper on most apps. It recovers part of histo's IPC gain (+11.5–12% vs +33%), and backprop matches either way. It does not reproduce the paper's accuracy or its IPC gains on hotspot, lud, srad_v1, mri-q or nw. For hotspot and lud even a perfect L1 on their best kernel stays below the paper's gain, so kernel selection cannot be the main reason for the gap. The S6 / D1 explanation stands: in our simulator those kernels have little exposed latency, and Snake's prefetches are rarely timely.

## 5. Per-kernel detail, Snake exact (PAPER_V100)

#### backprop / 65536 (2 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z24bpnn_adjust_weights_cudaPfiS_iS_S_` | 1 | 53.7% | -0.43% | 62.7 | 4.1 | 0.0 | 51.9 -> 52.0 |
| `_Z22bpnn_layerforward_CUDAPfS_S_S_ii` | 1 | 46.3% | +0.07% | 71.9 | 43.7 | 0.9 | 3.4 -> 5.7 |
| **whole run (compared launches)** | 2 | 100% | -0.20% | 64.6 | 12.4 | 0.2 | 41.7 -> 42.3 |

#### hotspot / 1024_2_2___data_temp_102 (1 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z14calculate_tempiPfS_S_iiiiffffff` | 1 | 100.0% | +0.16% | 93.3 | 69.1 | 27.0 | 0.7 -> 27.9 |
| **whole run (compared launches)** | 1 | 100% | +0.16% | 93.3 | 69.1 | 27.0 | 0.7 -> 27.9 |

#### hotspot / 512_2_2___data_temp_512_ (1 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z14calculate_tempiPfS_S_iiiiffffff` | 1 | 100.0% | +0.13% | 90.6 | 59.5 | 26.4 | 0.2 -> 26.6 |
| **whole run (compared launches)** | 1 | 100% | +0.13% | 90.6 | 59.5 | 26.4 | 0.2 -> 26.6 |

#### lud / _i___data_512_dat (94 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z12lud_diagonalPfii` | 32 | 51.7% | +0.00% | 0.0 | 0.0 | 0.0 | 0.0 -> 0.0 |
| `_Z13lud_perimeterPfii` | 31 | 27.1% | -0.00% | 0.0 | 0.0 | 0.0 | 0.0 -> 0.0 |
| `_Z12lud_internalPfii` | 31 | 21.3% | +0.59% | 86.2 | 29.8 | 2.8 | 4.4 -> 7.1 |
| **whole run (compared launches)** | 94 | 100% | +0.12% | 82.2 | 28.4 | 2.7 | 4.2 -> 6.8 |

#### lud / _s_256__v (46 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z12lud_diagonalPfii` | 16 | 54.2% | +0.00% | 0.0 | 0.0 | 0.0 | 0.0 -> 0.0 |
| `_Z13lud_perimeterPfii` | 15 | 27.4% | +0.00% | 0.0 | 0.0 | 0.0 | 0.0 -> 0.0 |
| `_Z12lud_internalPfii` | 15 | 18.4% | +0.43% | 82.5 | 36.8 | 0.8 | 0.0 -> 0.8 |
| **whole run (compared launches)** | 46 | 100% | +0.08% | 74.9 | 33.4 | 0.7 | 0.0 -> 0.7 |

#### nw / 2048_10 (255 launches compared)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z20needle_cuda_shared_1PiS_iiii` | 128 | 50.1% | +0.00% | 0.0 | 0.0 | 0.0 | 1.5 -> 1.5 |
| `_Z20needle_cuda_shared_2PiS_iiii` | 127 | 49.9% | +0.00% | 0.0 | 0.0 | 0.0 | 1.5 -> 1.5 |
| **whole run (compared launches)** | 255 | 100% | +0.00% | 0.0 | 0.0 | 0.0 | 1.5 -> 1.5 |

#### histo / _i___data_default_input_ (37 launches compared, last launch at the 1B cap excluded)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z17histo_main_kernelP6uchar4jjjjjPjS1_S1_` | 9 | 66.1% | -3.88% | 23.9 | 18.0 | 10.8 | 69.6 -> 79.8 |
| `_Z26histo_intermediates_kernelP5uint2jjjP6uchar4` | 9 | 17.1% | +11.48% | 83.3 | 52.2 | 7.0 | 1.5 -> 8.0 |
| `_Z18histo_final_kerneljjjjPjS_S_S_` | 9 | 10.2% | -2.82% | 87.6 | 54.4 | 17.7 | 0.5 -> 18.3 |
| `_Z20histo_prescan_kernelPjiS_` | 10 | 6.5% | -10.83% | 85.4 | 35.3 | 4.7 | 50.5 -> 56.7 |
| **whole run (compared launches)** | 37 | 100% | -1.95% | 34.3 | 23.5 | 10.8 | 59.3 -> 69.5 |

#### mri-q / _i___data_small_input_32 (3 launches compared, last launch at the 1B cap excluded)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z12ComputeQ_GPUiiPfS_S_S_S_` | 2 | 97.6% | +0.10% | 88.4 | 9.2 | 0.0 | 0.0 -> 0.0 |
| `_Z17ComputePhiMag_GPUPfS_S_i` | 1 | 2.4% | +0.00% | 96.9 | 18.8 | 0.0 | 0.0 -> 0.0 |
| **whole run (compared launches)** | 3 | 100% | +0.09% | 88.5 | 9.4 | 0.0 | 0.0 -> 0.0 |

#### srad_v1 / 100_0_5_502_458 (49 launches compared, last launch at the 1B cap excluded)

| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |
|---|---|---|---|---|---|---|---|
| `_Z6reduceliiPfS_` | 20 | 46.2% | +0.19% | 95.0 | 31.9 | 7.2 | 0.4 -> 7.6 |
| `_Z4sradfiilPiS_S_S_PfS0_S0_S0_fS0_S0_` | 9 | 23.1% | -1.20% | 79.8 | 18.1 | 7.3 | 41.3 -> 56.1 |
| `_Z5srad2fiilPiS_S_S_PfS0_S0_S0_S0_S0_` | 9 | 17.3% | -0.60% | 85.3 | 2.2 | 0.1 | 12.2 -> 12.3 |
| `_Z7preparelPfS_S_` | 10 | 12.1% | +0.60% | 93.2 | 15.2 | 0.0 | 50.0 -> 50.0 |
| `_Z7extractlPf` | 1 | 1.3% | +2.56% | 92.0 | 50.8 | 0.1 | 0.0 -> 0.1 |
| **whole run (compared launches)** | 49 | 100% | -0.19% | 84.9 | 12.6 | 3.5 | 25.6 -> 32.0 |

