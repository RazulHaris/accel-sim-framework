# D1: performance ceilings on exactly Snake's config (2026-10-16)

Step 0 before Phase 1: verify the S6 perfect-memory ceiling on exactly the Tier 1 configuration, and add a perfect-L1 ceiling. (The user's earlier perfect-memory runs on an A100 config are not comparable.)

**Verdict: the S6 numbers are confirmed.**
- All 8 S6 perfect-memory runs match Tier 1's configuration exactly.
- The D1 re-runs reproduce their IPC bit for bit.
- S6 had skipped nw, which is now measured: +19.0%.

## 1. The S6 perfect-memory runs (`sim_run_snake_gate/S6_ceiling`, no new runs)

| item | S6 | Tier 1 Snake runs | match |
|---|---|---|---|
| config | `QV100-SASS-PAPER_V100-1B_INSN-PERFECT_MEM` | `QV100-SASS-PAPER_V100-1B_INSN[-SNAKE-SNAKE_FP1]` | yes: each `gpgpusim.config` differs from Tier 1's baseline only by the appended `#PERFECT_MEM` / `-gpgpu_perfect_mem 1` (checked with diff for all 8) |
| extra params | `-gpgpu_scheduler gto`, `-gpgpu_clock_domains 1530.0:1530.0:1530.0:850.0` (PAPER_V100), `-gpgpu_max_insn 1000000000` (1B_INSN), `-gpgpu_perfect_mem 1` | the same without the last | yes |
| traces | `hw_run/rodinia-3.1/11.0/<app>/<args>/traces`, `hw_run/parboil/<app>/<args>/traces` (V100 traces) | the same | yes |
| `-gpgpu_max_insn` | 1000000000 (config and option dump) | 1000000000 | yes |
| build tags | `accelsim-commit-2c3a333_modified_0.0`, `gpgpu-sim_git-commit-3f635b67_modified_0.0` (Tier 1's frozen `sim_run_baseline/bin`) | the same | yes |
| trace-parser fix (bin_ref_v2 decoding) | `fe8f37e` is an ancestor of `2c3a333`; backprop prints "pre-756ed33 base+delta trace lines re-decoded = 172032", as in Tier 1 | the same | yes |
| workloads | 8: backprop, hotspot ×2, lud ×2, histo, mri-q, srad_v1 (**no nw**) | 9 | **no** |

Because one item did not match (8 of 9 workloads), perfect memory was re-run on all 9 workloads of the exact Tier 1 config (§4).

## 2. What `-gpgpu_perfect_mem` removes (gpgpu-sim `7ed7810e`)

- **Option:** gpu-sim.cc:395-397, "enable perfect memory mode (no cache miss)". The description is misleading: L1 misses still happen.
- **The SM's memory interface is swapped.** shader.cc:166-172 creates `perfect_memory_interface` instead of `shader_memory_interface`. The L1D gets this interface as its memory port (`m_icnt`, shader.cc:2713).
- **`perfect_memory_interface`** (shader.h:2779-2796):
  - `push()` puts the request straight into the cluster's response FIFO (`push_response_fifo`, shader.h:2659-2661)
  - `full()` is true only when that FIFO holds `-gpgpu_n_cluster_ejection_buffer_size` entries (shader.h:2656-2658)
  - Stores are acknowledged at once (shader.cc:2959, `store_ack` shader.cc:4144-4149).
- **Removed:** everything beyond the SM. That covers interconnect, L2 and DRAM **latency, bandwidth and queueing**: the request never enters the network, and DRAM reads are 0 in these runs.
- **Kept:**
  - the **L1 latency queue**: global accesses enter `l1_latency_queue` (shader.cc:2138) and reach the L1D in `L1_latency_queue_cycle` (shader.cc:2177)
  - the **L1D** itself: tag lookup, hit or miss, MSHR allocation, reservation failures
  - the **L1D miss queue**, drained at most one request per cycle through `m_memport->full()` / `push()` in `baseline_cache::cycle` (gpu-cache.cc:1403-1410)
  - the response path: the cluster FIFO, then one response per cycle into the LD/ST unit (`simt_core_cluster::icnt_cycle`, shader.cc:4827), then the L1D fill
- **So it is a "zero-cost memory system" ceiling, not a perfect L1:** loads still miss in the L1D and pay the miss-queue, MSHR and fill path.

## 3. The perfect-L1 ceiling: `-gpgpu_perfect_l1d`

- **Implementation** (gpgpu-sim `7ed7810e`):
  - option at gpu-sim.cc:398-402 (`shader_core_config::gpgpu_perfect_l1d`, default 0)
  - in `L1_latency_queue_cycle` (shader.cc:2183-2193), an L1-cached global load (`GLOBAL_ACC_R`) leaving the latency queue goes to `l1_cache::perfect_hit` (gpu-cache.cc:2277) instead of `access`
  - `perfect_hit` counts a hit, uses the data port like a hit and returns HIT, so the load completes at L1 latency. Tags, MSHRs, miss queue and memory are untouched.
  - Stores, atomics and L1-bypassing loads (`.cg`, `CACHE_GLOBAL`) are unchanged.
- **It models a perfect prefetcher:** every load hits. It also removes the loads' memory traffic, which a real prefetcher still has to fetch.
- **Token:** `PERFECT_L1D` (accel-sim `ac28769`).
- **Off-regression** (`sim_run_snake_regress/D1`, `ON=0`, ref from S2c): **PASS ref == off on 6 workloads**, raw stats equal. `regress_norm.py` drops the one new option-dump line `-gpgpu_perfect_l1d 0`, as it does for `-snake_*`.
- **Sanity:** every perfect-L1D run hits 100.00% of its `GLOBAL_ACC_R` accesses (§6).

## 4. Runs (42, all finished, `collect_results.py`: 0 failed checks)

| | runs | dir | build |
|---|---|---|---|
| perfect memory, PAPER_V100 + PAPER_LAT | 18 (9 workloads × 2) | `sim_run_snake_gate/D1_pmem` | Tier 1's frozen build `2c3a333` / `3f635b67` |
| perfect L1D, PAPER_V100 + PAPER_LAT | 18 | `sim_run_snake_gate/D1_pl1d` | `ac28769` / `7ed7810e` (clean) |
| off-regression | 6 | `sim_run_snake_regress/D1` | `ac28769` / `7ed7810e` |

Baseline and Snake (exact footprint) are Tier 1's runs: `sim_run_baseline`, `sim_run_snake`, configs `…-1B_INSN[-PAPER_LAT]` and `…[-PAPER_LAT]-SNAKE-SNAKE_FP1`. Gain = IPC / baseline IPC − 1. GMEAN over the 7 apps (an app's inputs first). Table: `util/snake/d1_report.py`.

## 5. Results

### PAPER_V100 (QV100, GTO, 1530 MHz)

| workload | baseline IPC | perfect memory gain | perfect L1 gain | Snake gain | paper's Snake gain |
|---|---|---|---|---|---|
| backprop / 65536 | 2493.5 | +46.7% | +41.7% | -0.2% | +1% |
| hotspot / 1024_2_2___data_temp | 6186.6 | +0.6% | +0.1% | +0.2% | +13% |
| hotspot / 512_2_2___data_temp_ | 4733.4 | +0.9% | +0.3% | +0.1% | +13% |
| lud / _i___data_512_dat | 181.8 | +4.8% | +3.6% | +0.1% | +18% |
| lud / _s_256__v | 47.5 | +4.4% | +3.0% | +0.1% | +18% |
| nw / 2048_10 | 58.4 | +19.0% | +16.6% | +0.0% | +5% |
| histo / _i___data_default_in | 667.0 | +29.6% | +39.2% | -1.9% | +33% |
| mri-q / _i___data_small_inpu | 3844.6 | +1.7% | +2.2% | +0.1% | +4% |
| srad_v1 / 100_0_5_502_458 | 1546.5 | +22.5% | +20.6% | -0.2% | +29% |
| **GMEAN** (7 apps) | | +16.8% | +16.6% | -0.3% | +14.1% |

### PAPER_LAT (PAPER_V100 + L1 23 / L2 ROP 199: measured L1 28, L2 212 cycles)

| workload | baseline IPC | perfect memory gain | perfect L1 gain | Snake gain | paper's Snake gain |
|---|---|---|---|---|---|
| backprop / 65536 | 2488.4 | +47.0% | +41.8% | -0.2% | +1% |
| hotspot / 1024_2_2___data_temp | 6192.6 | +0.6% | -0.0% | -0.0% | +13% |
| hotspot / 512_2_2___data_temp_ | 4720.9 | +1.5% | +0.2% | +0.1% | +13% |
| lud / _i___data_512_dat | 179.8 | +5.8% | +4.3% | +0.2% | +18% |
| lud / _s_256__v | 46.9 | +5.5% | +3.7% | +0.2% | +18% |
| nw / 2048_10 | 57.5 | +20.9% | +17.9% | +0.0% | +5% |
| histo / _i___data_default_in | 644.2 | +34.2% | +43.9% | -1.6% | +33% |
| mri-q / _i___data_small_inpu | 3843.4 | +1.8% | +1.9% | +0.3% | +4% |
| srad_v1 / 100_0_5_502_458 | 1498.8 | +26.1% | +23.7% | +0.1% | +29% |
| **GMEAN** (7 apps) | | +18.4% | +17.9% | -0.2% | +14.1% |

## 6. Checks

### S6 vs D1 perfect memory (PAPER_V100): identical

| workload | S6 IPC | D1 IPC | same |
|---|---|---|---|
| backprop / 65536 | 3656.7520 | 3656.7520 | yes |
| hotspot / 1024_2_2___data_temp | 6224.2080 | 6224.2080 | yes |
| hotspot / 512_2_2___data_temp_ | 4773.8496 | 4773.8496 | yes |
| lud / _i___data_512_dat | 190.4778 | 190.4778 | yes |
| lud / _s_256__v | 49.5415 | 49.5415 | yes |
| nw / 2048_10 | not run | 69.5579 | - |
| histo / _i___data_default_in | 864.5900 | 864.5900 | yes |
| mri-q / _i___data_small_inpu | 3908.6580 | 3908.6580 | yes |
| srad_v1 / 100_0_5_502_458 | 1894.6754 | 1894.6754 | yes |

### Perfect L1D: global-load hit rate and build tags

- backprop / 65536 -PERFECT_L1D: hit 974863 / 974863 (100.00%), insn 148636000
- backprop / 65536 -PAPER_LAT-PERFECT_L1D: hit 974863 / 974863 (100.00%), insn 148636000
- hotspot / 1024_2_2___d -PERFECT_L1D: hit 695640 / 695640 (100.00%), insn 372923416
- hotspot / 1024_2_2___d -PAPER_LAT-PERFECT_L1D: hit 695640 / 695640 (100.00%), insn 372923416
- hotspot / 512_2_2___da -PERFECT_L1D: hit 172720 / 172720 (100.00%), insn 93180768
- hotspot / 512_2_2___da -PAPER_LAT-PERFECT_L1D: hit 172720 / 172720 (100.00%), insn 93180768
- lud / _i___data_51 -PERFECT_L1D: hit 1048576 / 1048576 (100.00%), insn 191267104
- lud / _i___data_51 -PAPER_LAT-PERFECT_L1D: hit 1048576 / 1048576 (100.00%), insn 191267104
- lud / _s_256__v -PERFECT_L1D: hit 131072 / 131072 (100.00%), insn 23802000
- lud / _s_256__v -PAPER_LAT-PERFECT_L1D: hit 131072 / 131072 (100.00%), insn 23802000
- nw / 2048_10 -PERFECT_L1D: hit 1081344 / 1081344 (100.00%), insn 146463808
- nw / 2048_10 -PAPER_LAT-PERFECT_L1D: hit 1081344 / 1081344 (100.00%), insn 146463808
- histo / _i___data_de -PERFECT_L1D: hit 15146551 / 15146551 (100.00%), insn 1000003367
- histo / _i___data_de -PAPER_LAT-PERFECT_L1D: hit 15145676 / 15145676 (100.00%), insn 1000000665
- mri-q / _i___data_sm -PERFECT_L1D: hit 62208 / 62208 (100.00%), insn 1000001280
- mri-q / _i___data_sm -PAPER_LAT-PERFECT_L1D: hit 62208 / 62208 (100.00%), insn 1000005120
- srad_v1 / 100_0_5_502_ -PERFECT_L1D: hit 6126347 / 6126347 (100.00%), insn 1000004752
- srad_v1 / 100_0_5_502_ -PAPER_LAT-PERFECT_L1D: hit 6123255 / 6123255 (100.00%), insn 1000003151

All perfect-L1D jobs carry `accelsim-commit-ac28769_modified_0.0` and `gpgpu-sim_git-commit-7ed7810e_modified_0.0`.

## 7. Reading

- **Confirmed (S6):** hotspot, mri-q and lud have little latency left to hide in our simulator.
  - With every load hitting in L1, hotspot gains +0.1–0.3%, mri-q +1.9–2.2% and lud +3.0–4.3%, against the paper's +13%, +4% and +18%.
  - No prefetcher can reach the paper's gains on these apps here, under either latency setting.
- **Where there is headroom** (backprop +42%, histo +39–44%, srad_v1 +21–24%, nw +17–18% with a perfect L1), both ceilings also remove the loads' traffic.
  - S6 showed backprop and histo are DRAM-queue bound and srad_v1 is miss-queue bound. A real prefetcher still has to move those bytes, so these ceilings are upper bounds that no prefetcher reaches.
  - Snake makes no prediction at all for nw.
- **histo is the only app where a perfect L1 beats perfect memory** (+39.2% vs +29.6%). Its loss is in the L1D itself (miss queue and MSHR reservation failures), which perfect memory keeps.
- **Across apps the ceilings average about the paper's gain** (perfect L1 GMEAN +16.6%, +17.9% with PAPER_LAT; paper +14.1%). But the headroom sits in other apps than the paper's gains: backprop +42% vs the paper's +1%; hotspot +0.3% vs +13%.
- **PAPER_LAT** (the paper's measured L1 / L2 latencies) raises every ceiling slightly (GMEAN +16.8 → +18.4% for perfect memory). It does not change which apps have headroom. Snake stays within ±0.3% in both.
