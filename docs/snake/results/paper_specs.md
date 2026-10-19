# Paper specifications vs our simulations (2026-10-19, no runs)

**Paper:** Mostofi et al., "Snake: A Variable-length Chain-based Prefetching for GPUs", MICRO '23 (`docs/snake.pdf`, identical to `/home/razul/snake/Snake Prefetcher.pdf`). It was read directly with `pdftotext` and the figure pages were rendered.
- **Page numbers** are the printed proceedings pages. Printed page = PDF page + 726: Table 1 is on printed p.736 = PDF p.10.
- **Quotes are verbatim.** Math symbols are rendered as plain text, e.g. "tCCD =1".

**Simulator values are the effective (winning) ones** from the startup option dump of the Tier 1 runs:
- PAPER_V100: `sim_run_baseline/runs/backprop-rodinia-3.1/65536/QV100-SASS-PAPER_V100-1B_INSN/sim.out`
- PAPER_LAT: `…/QV100-SASS-PAPER_V100-1B_INSN-PAPER_LAT/sim.out`
- The option dumps of all 9 workloads are identical apart from the job.
- **File references:**
  - **Q** = `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config` (base QV100, the `QV100` entry of Y:34-35)
  - **T** = `gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config`
  - **Y** = `util/job_launching/configs/define-standard-cfgs.yml`
  - **S** = `gpu-simulator/gpgpu-sim/src/gpgpu-sim/snake_prefetcher.cc` (gpgpu-sim `7ed7810e`); source line numbers are at that commit
- **Override order:** PAPER_V100 = Y:196-198 (`-gpgpu_scheduler gto`, `-gpgpu_clock_domains 1530.0:1530.0:1530.0:850.0`). PAPER_LAT = Y:206-208 (`-gpgpu_l1_latency 23`, `-gpgpu_l2_rop_latency 199`). 1B_INSN = Y:96-97. These lines are appended after Q and T in each job's `gpgpusim.config`, so they win.
- **Measured latencies:** from the `l2_lat` microbenchmark (`sim_run_snake_calib`). Latency = kernel cycles / 32,768 dependent loads (results/S3e.md §3, results/S4.md §0, missq_config_check.md). The calibration runs' effective options are PAPER_V100's: GTO, 1530 MHz.

**Status values:** Match | Differs | Paper ambiguous | Not in paper | Measured vs setting.

## Part 1. Table 1 (baseline GPU), printed p.736

| parameter | paper (verbatim) | page | simulator option(s) | base QV100 (Q) | PAPER_V100 (effective) | PAPER_LAT (effective) | status | notes |
|---|---|---|---|---|---|---|---|---|
| Number of SM | "Number of SM 80" | 736, Tab. 1 | `-gpgpu_n_clusters` (Q:64) × `-gpgpu_n_cores_per_cluster` (Q:65) | 80 × 1 | 80 × 1 | 80 × 1 | Match | |
| Core clock | "Core clock 1530 MHz" | 736, Tab. 1 | `-gpgpu_clock_domains` core:icnt:L2:DRAM (Q:72; Y:198) | 1132.0:1132.0:1132.0:850.0 | 1530.0:1530.0:1530.0:850.0 | 1530.0:1530.0:1530.0:850.0 | Match (core); Not in paper (other 3 domains) | The paper gives only the core clock. PAPER_V100 also raises the interconnect and L2 clocks to 1530 (our choice, keeping the stock QV100 ratio core = icnt = L2). DRAM stays 850 MHz (Q:215 "V100 HBM runs at 850 MHZ"). The base QV100 is calibrated at 1132 MHz (OQ-23). |
| Scheduler | "Scheduler Greedy-then-Oldest" | 736, Tab. 1 | `-gpgpu_scheduler` (Q:134; Y:197) | lrr | gto | gto | Match (PAPER_V100); base Differs | Q:133's comment says "Greedy then oldest scheduler" but Q:134 sets `lrr` (OQ-18). |
| Schedulers per SM | "Number of scheduler per SM 4" | 736, Tab. 1 | `-gpgpu_num_sched_per_core` (Q:132), `-gpgpu_sub_core_model` (Q:114) | 4 (sub-core 1) | 4 (sub-core 1) | 4 (sub-core 1) | Match | Sub-core model: each scheduler owns its warps and units (Q:112-113); the paper does not say. |
| Threads per SM | "Number of threads per SM 2048" | 736, Tab. 1 | `-gpgpu_shader_core_pipeline` (Q:82) | 2048:32 | 2048:32 | 2048:32 | Match | 2048 threads, warp size 32, so 64 warps per SM. |
| Register file per SM | "Register file size per SM 65536" | 736, Tab. 1 | `-gpgpu_shader_registers` (Q:77) | 65536 | 65536 | 65536 | Paper ambiguous (unit) | The paper gives no unit. The option counts 32-bit registers (65536 × 4 B = 256 KB). The numbers are equal. |
| Unified cache: size | "Unified cache 128KB, 256-way, 128B, 4-bank, 28-cycle" | 736, Tab. 1 | `-gpgpu_unified_l1d_size` (Q:149), `-gpgpu_adaptive_cache_config` (Q:147), `-gpgpu_shmem_option` (Q:148) | 128, adaptive 1, options 0,8,16,32,64,96 KB | same | same | Match (unified size); Paper ambiguous (effective L1) | **The effective L1 is 128 KB minus the shared-memory carve-out, per kernel** (table "Effective L1" below): 96-128 KB. The paper does not say how much L1 its kernels got. |
| Unified cache: associativity | "256-way" | 736, Tab. 1 | `-gpgpu_cache:dl1` (Q:152) `S:4:128:64,…` | 4 sets × 64 ways (32 KB base) | 4 sets × 192/224/240/256 ways (per kernel) | same | Match at 128 KB | Adaptive: ways = round(max_assoc × (1 − carve-out / 128 KB)) with max_assoc = 4 × 64 = 256 (`shader.cc:3684-3703`; `get_max_assoc`, `gpu-cache.h:791`). Sets stay 4. At 128 KB: 4 × 256 × 128 B, consistent with the paper's 128 KB / 256-way / 128 B (= 4 sets). |
| Unified cache: line | "128B" | 736, Tab. 1 | `-gpgpu_cache:dl1` (Q:152) | 128 B, 4 × 32 B sectors | same | same | Match | Sector cache (`S:`); the paper does not mention sectors. |
| Unified cache: banks | "4-bank" | 736, Tab. 1 | `-gpgpu_l1_banks` (Q:151) | 4 | 4 | 4 | Match | |
| Unified cache: latency | "28-cycle" | 736, Tab. 1 | `-gpgpu_l1_latency` (Q:154; Y:207) | 20 | 20 → **measured 25.000732421875** | 23 → **measured 28.001922607421875** | Measured vs setting | Measured L1 hit = 819,224 / 32,768 (PAPER_V100) and 917,567 / 32,768 (PAPER_LAT). The simulator adds about 5 cycles to the knob. Table 1's value is taken as a measured latency (VERSION_DIFF). An earlier PAPER_LAT knob of 28 measured 33.002044677734375 (1,081,411 / 32,768). |
| MSHR | "MSHR 512 entries, 8 merge capability" | 736, Tab. 1 | `-gpgpu_cache:dl1` MSHR field (Q:152) `A:512:8` | 512, merge 8 | same | same | Match | Associative MSHR. The paper does not say which cache; this is the L1D's. |
| L1 instruction cache | "L1 instruction cache 128KB, 16-way, 128B" | 736, Tab. 1 | `-gpgpu_cache:il1` (Q:172) `N:64:128:16` | 64 sets × 128 B × 16 = 128 KB | same | same | Match | |
| Constant cache | "Constant cache 64KB, 8-way, 64B-line" | 736, Tab. 1 | `-gpgpu_const_cache:l1` (Q:178) `N:128:64:8` | 128 × 64 B × 8 = 64 KB | same | same | Match | |
| L2: size | "L2 cache 96KB per sub-partition, 24-way, 128B-line, 64-bank, 212-cycle" | 736, Tab. 1 | `-gpgpu_cache:dl2` (Q:165) `S:32:128:24`; `-gpgpu_n_mem` (Q:66); `-gpgpu_n_sub_partition_per_mchannel` (Q:67) | 32 × 128 B × 24 = 96 KB per sub-partition; 32 channels × 2 = 64 sub-partitions; total 64 × 96 KB = 6,144 KB | same | same | Match | |
| L2: associativity, line | "24-way, 128B-line" | 736, Tab. 1 | `-gpgpu_cache:dl2` (Q:165) | 24-way, 128 B (sector) | same | same | Match | |
| L2: banks | "64-bank" | 736, Tab. 1 | sub-partitions = `-gpgpu_n_mem` 32 × `-gpgpu_n_sub_partition_per_mchannel` 2 | 64 | 64 | 64 | Paper ambiguous (term) | In the simulator, an L2 "bank" is a memory sub-partition with its own L2 slice; there are 64. DRAM has 16 banks per channel (`nbk=16`, Q:216), a different thing. |
| L2: latency | "212-cycle" | 736, Tab. 1 | `-gpgpu_l2_rop_latency` (Q:193; Y:208), `-dram_latency` (Q:194) | ROP 160 | 160 → **measured L2 hit 172.96905517578125** | 199 → **measured 211.97152709960938** | Measured vs setting | Measured L2 hit (L1-bypassing loads) = 5,667,850 / 32,768 and 6,945,883 / 32,768. The ROP knob is a fixed delay; the rest is pipeline and interconnect. |
| DRAM tCCD | "tCCD =1" in "DRAM parameters (ns)" | 736, Tab. 1 | `-gpgpu_dram_timing_opt` (Q:216-217) `CCD=1` | 1 DRAM cycle | same | same | Paper ambiguous (unit) | **The units differ.** In the simulator every timing is a DRAM-clock-cycle counter: set from the option (e.g. `RCDc = tRCD`, dram.cc:654), decremented once per `dram_t::cycle` (dram.cc:530-540), called only on DRAM clock edges (gpu-sim.cc:2024-2031) at 850 MHz (Q:215 "Timing for 850 MHZ"). 1 cycle = 1/850 MHz = 1.1764705882352942 ns. The paper's numbers equal the config's cycle counts exactly, but are labelled ns. |
| DRAM tRRD | "tRRD =3" | 736 | `RRD=3` | 3 cycles = 3.5294117647058822 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tRCD | "tRCD =12" | 736 | `RCD=12` | 12 cycles = 14.117647058823529 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tRAS | "tRAS =28" | 736 | `RAS=28` | 28 cycles = 32.94117647058824 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tRP | "tRP =12" | 736 | `RP=12` | 12 cycles = 14.117647058823529 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tRC | "tRC =40" | 736 | `RC=40` | 40 cycles = 47.05882352941177 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tCL | "tCL =12" | 736 | `CL=12` | 12 cycles = 14.117647058823529 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tWL | "tWL =2" | 736 | `WL=2` | 2 cycles = 2.3529411764705883 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tCDLR | "tCDLR =3" | 736 | `CDLR=3` | 3 cycles = 3.5294117647058822 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tWR | "tWR =10" | 736 | `WR=10` | 10 cycles = 11.764705882352942 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tCCDL | "tCCDL =2" | 736 | `CCDL=2` | 2 cycles = 2.3529411764705883 ns | same | same | Paper ambiguous (unit) | as tCCD |
| DRAM tRTPL | "tRTPL =3" | 736 | `RTPL=3` | 3 cycles = 3.5294117647058822 ns | same | same | Paper ambiguous (unit) | as tCCD |

### Effective L1 per workload (from the run output)

Each kernel launch prints "GPGPU-Sim: Reconfigure L1 cache to N KB" (shader.cc:3725). Counts are kernel launches in the Tier 1 baseline (1B cap). Shared memory per CTA comes from the trace headers (`-shmem = N`, bytes). Ways = N KB × 1024 / (4 sets × 128 B).

| workload | L1 size (launches) | 4 sets × ways | shared memory per CTA (bytes) |
|---|---|---|---|
| backprop 65536 | 112 KB (1), 128 KB (1) | 224, 256 | 1088, 0 |
| hotspot 512 | 96 KB (1) | 192 | 3072 |
| hotspot 1024 | 96 KB (1) | 192 | 3072 |
| lud 512 | 112 KB (14), 120 KB (80) | 224, 240 | 1024, 2048, 3072 |
| lud 256 | 120 KB (46) | 240 | 1024, 2048, 3072 |
| nw 2048_10 | 120 KB (255) | 240 | 2180 |
| histo | 96 KB (9), 120 KB (10), 128 KB (19) | 192, 240, 256 | 24576, 4096, 0 |
| mri-q | 128 KB (4) | 256 | 0 |
| srad_v1 | 112 KB (10), 120 KB (10), 128 KB (30) | 224, 240, 256 | 4096, 0 |

## Part 2. Other simulation specs in the paper

| item | paper (verbatim) | page | ours | status | notes |
|---|---|---|---|---|---|
| Simulator | "utilizing Accel-Sim [21] v1.2.0" | 736, §4 | accel-sim base `3016c65` (dev, post-v1.2.0) + gpgpu-sim base `6c3cf4ff`, plus the trace-parser fix and Snake; final runs built from accel-sim `2c3a333`, gpgpu-sim `3f635b67` | Differs | VERSION_DIFF: our base was chosen over v1.2.0 (plan decision 2026-10-08). The motivation study also uses v1.2.0 ("we use Accel-Sim v1.2.0", 730, §2). |
| Power model | "we analyze the power consumption of Snake using AccelWattch [19] v1.0" | 736, §4 | AccelWattch present (`Accelwattch_SASS_SIM`, Y:165-168) but off: `-power_simulation_enabled 0` (run dump) | Differs | Not used: +55% simulation time (SPEC §8). Energy is unavailable; Snake's own energy is computed offline (OQ-16). |
| Run length | "Each application is simulated until the end of its execution or when the simulated instruction count reaches 1 billion." | 736, §4 | `-gpgpu_max_insn 1000000000` (Y:96-97). Stops when gpu_tot_sim_insn + gpu_sim_insn ≥ the cap (gpu-sim.cc:1155-1156) | Match (paper ambiguous on what is counted) | gpgpu-sim counts **thread instructions**: `gpu_sim_insn += inst.active_count()` (shader.cc:1982). The paper does not define "instruction"; Accel-Sim's own counter is the likely one. histo, mri-q and srad_v1 reach the cap; the other six finish. |
| Target GPU | "Table 1 modeling the NVIDIA VOLTA V100 architecture"; "the fabrication technology in NVIDIA Volta V100 … die size (815 mm2)"; "scale the extracted area to 12nm" | 736 §4; 739 §5.5 | Accel-Sim `SM7_QV100` config (Volta, 80 SMs; Q) | Match (architecture) | 12 nm and 815 mm² are used only for the paper's area estimate; the simulator has no process parameters. |
| Coverage | "coverage (the number of correctly predicted addresses to the total number of demand addresses)" | 736, §4 | coverage_pred (predicted sectors within 5,000 cycles / demand sectors) and coverage_paper ((timely + late) / demand sectors) | Paper ambiguous | "Addresses" vs our sector granularity, and "correctly predicted" vs issued: SPEC OQ-14. |
| Accuracy | "accuracy (the number of correctly predicted addresses that are timely enough to be used by demand requests to the total number of demand addresses)" | 736, §4 | accuracy_paper = timely / demand sectors | Match (definition); Paper ambiguous (granularity) | OQ-14. |
| Comparison points | "we compare it to the baseline GPU and 9 other prefetching mechanisms. (1) INTRA … (2) INTER … (3) MTA … (4) CTA … (5) Tree … 64KB chunks … (6) s-Snake … (7) Snake-DT … (8) Snake-T … (9) Snake+CTA" | 736, §4 | Implemented: baseline, s-Snake, Snake-DT, Snake-T, plus Snake-ISO (737/739, §5.7 "Isolated-Snake"). Not implemented: INTRA, INTER, MTA, CTA, Tree, Snake+CTA | Differs | OQ-25 (decided 2026-10-08: Snake ablations only). |

## Part 3. Table 2 (benchmarks), printed p.736

The paper gives **no inputs** (Table 2 and §4 list only suites and names, OQ-13). Our traces: tracer version 3, nvbit 1.4, `-binary version = 70` (trace headers). Rodinia traces sit under `hw_run/rodinia-3.1/11.0/`; the "11.0" directory name is the only CUDA-version information, and Parboil's directory (`hw_run/parboil/`) gives none, so their CUDA version is **unknown**.

| suite (ref.) | benchmark (paper) | abbr. (paper) | we run it | our input (`util/job_launching/apps/define-all-apps.yml`) | trace | notes |
|---|---|---|---|---|---|---|
| ISPASS [5] | Coulombic Potential | CP | no | n/a | n/a | No local V100 trace; absent from the public V100 trace set; its build line is commented out under "# Compile Issues" (define-all-apps.yml:857-858). OQ-13: ISPASS deferred (decided 2026-10-01). |
| ISPASS [5] | 3D Laplace Solver | LPS | no | n/a | n/a | Listed in the suite (yml:862) and the only one built in gpu-app-collection, but no V100 trace locally. OQ-13: deferred. |
| ISPASS [5] | LIBOR Monte Carlo | LIB | no | n/a | n/a | Listed (yml:860); no V100 trace locally. OQ-13: deferred. |
| ISPASS [5] | MUMmerGPU | MUM | no | n/a | n/a | Listed (yml:864); no V100 trace locally. OQ-13: deferred. |
| Rodinia [31] | Back Propagation | Backprop | yes | `65536` (yml:544) | v3, nvbit 1.4, sm_70 | Rodinia version not given in the paper; we use 3.1 (OQ-24). |
| Rodinia [31] | HotSpot | Hotspot | yes (2 inputs) | `512 2 2 ./data/temp_512 ./data/power_512 output.out`; `1024 2 2 ./data/temp_1024 ./data/power_1024 output.out` (yml:580, 582) | same | |
| Rodinia [31] | Speckle Reducing Anisotropic Diffusion | Srad | yes (srad_v1) | `100 0.5 502 458` (yml:634) | same | The paper does not say v1 or v2; we use srad_v1 (Rodinia 3.1, OQ-24). |
| Rodinia [31] | LU Decomposition | lud | yes (2 inputs) | `-s 256 -v`; `-i ./data/512.dat` (yml:609, 611) | same | |
| Rodinia [31] | Needleman-Wunsch | nw | yes | `2048 10` (yml:622) | same | |
| Parboil [44] | Histogram | histo | yes | `-i ./data/default/input/img.bin -o ref.bmp -- 20 4` (yml:671) | same | |
| Parboil [44] | mri-q | MRQ | yes | `-i ./data/small/input/32_32_32_dataset.bin -o 32_32_32_dataset.out` (yml:664) | same | The paper abbreviates mri-q as "MRQ". |

## Part 4. Snake's own parameters (§3, §5.4, §5.5, Table 3, Figs. 13-15, 20-23)

"Our default" is the registered default in S (line given). "Paper-literal" is the setting closest to the paper's text where ours differs.

| item | paper (verbatim) | page | our option (S:line) | our default | paper-literal setting | status | OQ |
|---|---|---|---|---|---|---|---|
| Head table rows | "with a total of N rows (N = #warps/2)"; Table 3 "Head 14 bytes per entry, 32 entries … 448 bytes" | 738 §5.5; 739 Tab. 3 | `-snake_head_entries` (S:17) | 32 | 32 | Match | OQ-1 |
| Head table columns | "The Head table consists of five columns (two warp IDs, two base addresses, and one PCld)… for non-greedy schedulers, we use only three columns" | 738 §5.5 | `-snake_head_two_warps` (S:20) | 1 (five columns) | 1 under GTO | Match (columns); Paper ambiguous (organisation) | OQ-1: "indexed by warp IDs" (733, §3.1) vs N = #warps/2 rows. We model a per-warp register plus per-PC rows. |
| Head bytes | "14 bytes per entry, 32 entries … 448 bytes" | 739 Tab. 3 | none (the simulator uses 64-bit fields) | n/a | n/a | Not modelled bitwise | OQ-2 |
| Tail entries | "The Tail table has 10 entries"; Table 3 "Tail 32 bytes per entry, 10 entries … 320 bytes" | 738 §5.5; 739 Tab. 3 | `-snake_tail_entries` (S:24) | 10 | 10 | Match | |
| Tail fields | "Each entry of this table consists of eight key fields: (1) a head PCld … (2) a consecutive PCld … (3) the stride between these two PCs … (4) the train status of the Inter-thread stride, (5) … warpID vector, (6) Intra-warp stride, (7) the train status of the Intra-warp stride, and (8) the Inter-warp stride." | 733 §3.1 | all 8 fields in `tail_entry` (snake_prefetcher.h); plus LRU stamp, removed-warp count and, with `-snake_footprint 1` (S:57), a PC2 lane descriptor (~8 B) | 8 fields | 8 fields | Match (+ bookkeeping) | OQ-2, OQ-9 |
| Promotion threshold | "promotes the discovered chains of strides to prefetch for other warps if they are detected at least for three warps"; "In cases where three warps demonstrate similar inter-thread stride behavior, the T1 status is upgraded to promotion status (i.e., 10)" | 732 §3; 735 §3.4 | `-snake_promote_warps` (S:35) | 3 | 3 | Match | OQ-5 |
| Inter-warp detection | "The Inter-warp stride is detected when a fixed stride occurs between at least three warps executing the same PCld." | 733 §3.1 | `-snake_interwarp_min_warps` (S:44) | 3 | 3 | Match | OQ-6 |
| Intra-warp training | "Upon establishing consistency of intra-warp stride in three distinct warps, Snake proceeds to update the train status … (i.e., T2)." | 735 §3.4 | `-snake_intra_confirm_warps` (S:40) | 3 | 3 | Match | OQ-26 |
| Demotion | "If more than two warps are removed from the warpID vector of an entry, the inter-thread train status is updated to "not-trained" (i.e., '00'), and it returns to the detection step." | 734 §3.2 | `-snake_demote_removed` (S:48) | 2 | 2 | Match (+ vector cleared) | OQ-28 |
| Training time | "The training process takes an average of 3 cycles (up to 10 cycles)."; "the designated 10-cycle period, which represents the maximum required time for training Snake" | 734 §3.2; 735 §3.5 | `-snake_untrained_cycles` (S:184), `-snake_decouple_variant` (S:178) | 10 (variant 1) | 10 | Match (variant 1); variant 0 Differs | OQ-15 |
| Eviction policy | "we first identify the most eligible entries for eviction using the LRU policy. Then, from the selected group of entries, we evict the one with the fewest '1's in its warpID vector." | 733 §3.1 | `-snake_tail_evict` (S:27), `-snake_tail_lru_group` (S:30), `-snake_interwarp_weight` (S:84) | lru_popcount, group 3, weight 3 | lru_popcount, group not given, weight 0 | Paper ambiguous (group size); Differs (weight) | OQ-3, OQ-30 |
| T1/T2 encodings | "'promoted' (i.e., '10')"; "'not-trained' (i.e., '00')"; "its train status (T2) is initiated to '00'". Fig. 15 prints T1 '01' (b) and '11' (c); Fig. 13 prints '11' and '00' | 734 §3.2; 733 §3.1; 735 Fig. 15; 734 Fig. 13 | internal states untrained / observed / promoted / trained | states | states | Paper ambiguous | OQ-5 |
| Chain depth | "The depth of Inter-thread prefetching, i.e., the distance from the current PCld, is controlled by a throttling mechanism" | 734 §3.2 | `-snake_max_chain_depth` (S:67), `-snake_throttle_chain_depth` (S:141) | 3; 1 while throttled | no number | Not in paper (number) | OQ-7 |
| Chain step condition | "Snake revisits the Tail table to locate an entry that possesses the same PC1 value as the current entry's PC2 value, while the corresponding bit in warpID vector is set and the status is trained." | 734 §3.2 | `-snake_promoted_all_warps` (S:79) | 1 (first step also without the bit) | 0 | Differs | OQ-32 |
| Inter-warp targets | "Snake issues prefetching requests for all future warps, as soon as the train status in the Tail table is updated to "promoted"" | 734 §3.2 | `-snake_interwarp_degree` (S:70), `-snake_interwarp_scope` (S:172), `-snake_interwarp_chain_depth` (S:74) | 4, scope 0 (same CTA), chain 0 | unbounded ("all future warps") | Not in paper (bound) | OQ-8, OQ-29 |
| Inter-thread priority | "Snake accords priority to the inter-thread stride over the inter-warp stride due to its higher accuracy." | 735 §3.4 | queue and issue priority (no option) | on | on | Match (+ queue displacement) | OQ-31 |
| Lane check | "we examine the stride between threads and only retain the addresses of the first thread if the stride is equal. Otherwise, we exclude the warp from prefetching." | 735 §3.4 | `-snake_lane_mode` (S:53) | 1 (run filter) | 0 | Differs | OQ-17 |
| Not-in-L1 filter | "We proceed to issue prefetch requests in the case of not locating the prefetch request in the L1 data cache." | 733 §3.2 | `-snake_l1_filter` (S:114) | 1 | 1 | Match | |
| Pipeline latency | "The latency of Snake is two cycles" | 739 §5.5 | `-snake_latency` (S:98) | 2 | 2 | Match | |
| Decoupled split | "the remaining space is allocated between the prefetch space and the standard L1 data cache space … Snake differentiates between prefetch and L1 data cache entries using a flag." | 734 §3.2 | `-snake_decouple_mode` (S:147) | 1 | 1 | Match | OQ-12 |
| 50% until trained | "Snake initially restricts the L1 cache to utilize only up to 50% of the available decoupled space until the prefetcher is trained." | 734 §3.2 | `-snake_untrained_l1_frac` (S:151) | 0.50 | 0.50 | Match | OQ-15 |
| 50-cycle confinement | "In instances of prefetch throttling, we confine the L1 data cache to operate within its designated space for up to 50 cycles (i.e., typically, the data is consumed within 50 cycles.)." | 734 §3.2 | `-snake_throttle_confine_cycles` (S:155) | 50 | 50 | Match | OQ-15 |
| Flag flip | "it is always permitted to transfer data that hits the prefetch space to the L1 data space. This transfer does not involve a physical movement but instead occurs through the alteration of the corresponding flag." | 734 §3.2 | `-snake_decouple_variant` (S:178) | 1 (flip when no prefetched sector of the line is fresh) | 0 (flip on hit) | Differs | OQ-12 |
| 25% free | "In instances where there is no free space left in the unified cache, we free up 25% of the unified cache space following the LRU policy."; footnote 2 "We choose 25% as the hit rate of L1 data cache is up to 75%." | 734 §3.2 | `-snake_free_frac` (S:159) | 0.25 (per set, of evictable lines) | 0.25 (scope unstated) | Match (value); Paper ambiguous (scope) | OQ-12 |
| 80% rule | "If more than 80% of the prefetched data has been transferred to the L1 data cache, we evict older L1 data cache entries … Otherwise, we evict older prefetched data entries to free up space." | 734 §3.2 | `-snake_transfer_frac` (S:162), `-snake_pf_stale_cycles` (S:189) | 0.80; variant 1 frees only prefetched lines unused > 50 cycles | 0.80, any older prefetched line | Differs (variant 1) | OQ-12 |
| Space throttling | "(1) When the unified memory has no free space, our throttling mechanism halts prefetching for a predefined number of cycles (50 cycles, See Section 5.4 for more information)" | 734 §3.3 | `-snake_throttle_space_cycles` (S:120), `-snake_throttle_enable` (S:118) | 50, on | 50 | Match | OQ-10 |
| Bandwidth throttling | "(2) When the bandwidth approaches saturation (i.e., when the measured bandwidth reaches around 70% of the theoretical peak bandwidth), our throttling mechanism halts prefetching until the bandwidth saturation is resolved (i.e., when the measured bandwidth reaches 50% of the theoretical peak bandwidth)." | 734 §3.3 | `-snake_bw_high` (S:123), `-snake_bw_low` (S:126), `-snake_bw_window` (S:129), `-snake_bw_mode` (S:132) | 0.70 / 0.50, window 100 cycles, request slots of the L1D miss-queue drain | 0.70 / 0.50; which bandwidth and window not given | Paper ambiguous | OQ-10 |
| Tail sensitivity | "The coverage of Snake decreases by only 8% when the entry size is limited to 10."; Fig. 20 legend "Unlimited entries / 40 entries / 20 entries / 10 entries" | 738 §5.4, Fig. 20 | sweep `SNAKE_TAIL20/40/INF` | 10; 20, 40, 1024 | 10, 20, 40, unlimited | Match (unlimited = 1024 entries; backprop still evicts) | OQ-3 |
| Eviction sensitivity | "Figure 22 shows the coverage of Snake that considers the number of '1's in its warpID vector (i.e., the entry with the fewest '1's is evicted) without using the LRU mechanism"; Fig. 22 legend "40 Entries / 20 Entries / 10 Entries" | 738 §5.4, Fig. 22 | sweep `SNAKE_EVPOP` × Tail 10/20/40 | lru_popcount | popcount, 10/20/40 | Match | |
| Throttle sensitivity | "turning off Snake for 50 cycles can achieve an acceptable level of accuracy (75%) with only 2% coverage loss."; Fig. 23 x-axis "Clock Cycles" 25, 50, 75, 100, 150, 200, 300 | 738 §5.4, Fig. 23 | sweep `SNAKE_THR25…300` | 50 | 25-300 | Match | Fig. 23's right axis reads "Coverage Loss (%)" while its bars are labelled "Average Coverage": Paper ambiguous. |
| Isolated-Snake buffer | "a variant of Snake that stores prefetched data in a buffer distinct from the unified memory, called Isolated-Snake" | 739 §5.7 | `-snake_isolated_lines` (S:167) | 256 lines (32 KB) | size not given | Not in paper | OQ-22 |
| Prefetch queue, issue rate | not given | n/a | `-snake_queue_size` (S:101), `-snake_issue_width` (S:103) | 32 lines, 1 line/cycle | n/a | Not in paper | OQ-4 |
| MSHR and miss-queue reserve | not given | n/a | `-snake_mshr_reserve` (S:105), `-snake_missq_reserve` (S:109) | 8, 8 | n/a | Not in paper | OQ-11 |

## Part 5. Cost and power methodology (§5.5, printed p.739)

| item | paper (verbatim) | page | ours | status |
|---|---|---|---|---|
| Area tool and node | "we model the proposed tables in CACTI V7.0 [48] for 22nm technology and scale the extracted area to 12nm (i.e., the fabrication technology in NVIDIA Volta V100 [36])" | 739 §5.5 | not reproduced | Not reproduced |
| Area result | "Snake imposes less than 1% area overhead over the NVIDIA Volta V100 [36]) die size (815 mm2)." | 739 §5.5 | not reproduced | Not reproduced |
| Synthesis | "we synthesize the Hardware Description Language (HDL) model of the tables in Snake for NanGate 28nm open cell library using Synopsys Design Compiler, scaled to 12nm. We then import the calculated numbers into AccelWattch [19]" | 739 §5.5 | not reproduced; AccelWattch off | Not reproduced |
| Energy per access, static power | "Snake consumes 6.4 pJ per access and 6 mW of static power" | 739 §5.5 | `-snake_energy_pj_per_access 6.4` (S:195), `-snake_static_mw 6.0` (S:198); applied offline: accesses × 6.4 pJ + 6 mW × 80 SMs × simulated time (`util/snake/collect_results.py`) | Match (values); our use is offline (OQ-16) |
| Latency implementation | "We implemented the prefetcher using Verilog HDL, synthesized it in the Synopsys Design Compiler [1], and performed placement and routing in Cadence SOC Encounter [2]." | 739 §5.5 | not reproduced; we take the stated 2 cycles | Not reproduced |

## Part 6. Specs the paper does not give, which affect results

| item | our value (effective) | source |
|---|---|---|
| L1D miss queue: size | 16 entries | `-gpgpu_cache:dl1` miss-queue field `16:0` (Q:152) |
| L1D miss queue: drain | at most 1 request per core cycle (when the interconnect has buffer space) | `baseline_cache::cycle` (gpu-cache.cc:1403-1411) |
| L1D write ratio, L1 bypass | `-gpgpu_l1_cache_write_ratio 25` (Q:153); `-gpgpu_gmem_skip_L1D 0` (Q:155) | run dump |
| L2 MSHR, miss queue | 192 entries, merge 4; miss queue 32 | `-gpgpu_cache:dl2 …,A:192:4,32:0,32` (Q:165) |
| Interconnect | local crossbar `-network_mode 2` (Q:185); in/out buffers 512 / 512 packets (Q:186-187); 2 subnets (Q:188); flit 40 B (Q:189); cluster ejection buffer 32 (Q:157) | run dump |
| Memory partitions, channels | 32 channels (`-gpgpu_n_mem`, Q:66) × 2 sub-partitions (Q:67) = 64 L2 slices; 1 chip per channel (Q:202); HBM dual bus interface (Q:220) | run dump |
| DRAM organisation | nbk = 16 banks, nbkgrp = 4 (Q:216-217); bus width 16 B (Q:203); burst length 2 (Q:204); data/command frequency ratio 2 (Q:205); FR-FCFS (`-gpgpu_dram_scheduler 1`, Q:197), queue 64 (Q:198), return queue 192 (Q:199); `-dram_latency 100` (Q:194); partition queues 64:64:64:64 (Q:167) | run dump |
| DRAM peak bandwidth | **derived, not printed**: atom = BL × busW × chips = 2 × 16 B × 1 = 32 B (gpu-sim.h:295-296); one atom per BL / data_command_freq_ratio = 1 DRAM cycle (dram.cc:574, 582); 32 B × 850 MHz × 32 channels = **870.4 GB/s** | derived from the options above |
| Shared memory | `-gpgpu_shmem_size 98304` (96 KB, Q:159), `-gpgpu_shmem_per_block 65536` (Q:161); carve-out options 0, 8, 16, 32, 64, 96 KB (Q:148) | run dump; effective L1 per workload in Part 1 |
| Max CTAs per SM | 32 (`-gpgpu_shader_cta`, Q:83) | run dump |
| Max warps per SM | 64 (2048 threads / warp 32, Q:82) | derived |
| Warps per scheduler | 16 (64 warps / 4 schedulers, sub-core model, Q:114, Q:132) | derived |
| Kernel launch latency | `-gpgpu_kernel_launch_latency 5000` (Q:51) | Q |
| Execution latencies | trace.config T:1-19 (e.g. `-trace_opcode_latency_initiation_int 2,2`, T:1) | T |
| Trace versions | accel-sim tracer 3, nvbit 1.4, binary version 70 (sm_70), from the trace headers; Rodinia traces under `hw_run/rodinia-3.1/11.0/` (CUDA 11.0 by directory name only); Parboil CUDA version **unknown** | trace headers |
| Simulator build toolchain | CUDA 11.8 (`nvcc --version`: cuda_11.8.r11.8); builds the simulator only, not the traces | local |
| Benchmark inputs | Part 3 | `define-all-apps.yml` |

## Items marked Differs or Paper ambiguous, and how they could affect results

**Differs**
- **Simulator version** (v1.2.0 vs our dev-era base): VERSION_DIFF lists the model changes (iSLIP, L1 streaming in release, …). They can move the memory-system bottlenecks that decide Snake's gain.
- **Power model unused:** energy (Fig. 19) cannot be compared.
- **Comparison points:** six of the nine are not implemented, so the paper's relative claims against MTA and CTA cannot be checked.
- **Base QV100 core clock 1132 MHz and LRR:** overridden by PAPER_V100 (1530, GTO) in every final run. Raising the interconnect and L2 clocks with the core clock is our choice. It shortens L2 latency in ns relative to DRAM, which stays at 850 MHz.
- **L1 / L2 latency settings** (20 / 160 vs 28 / 212): measured 25.0 / 173.0 with PAPER_V100. PAPER_LAT reproduces 28.0 / 212.0; Snake's GMEAN IPC gain is −0.3% (PAPER_V100) and −0.2% (PAPER_LAT) (final_tier1.md), so the latency setting does not change the result.
- **Eviction weight 3, promoted strides for all warps, lane run filter, decoupling variant 1:** our deviations, each with a paper-literal switch (OQ-30, OQ-32, OQ-17, OQ-12). They change coverage and timeliness; their effect is measured in the S2-S4b stages and Tier 2.

**Paper ambiguous**
- **DRAM timings "(ns)":** the values equal the config's DRAM-cycle counts at 850 MHz. Taken as ns they would be 15% shorter than ours (1 cycle = 1.176 ns), which would make DRAM slightly faster.
- **Register file unit (65536):** equal to our 65536 32-bit registers; no effect if the paper means registers.
- **Effective L1 size:** depends on each kernel's shared memory (96-128 KB here). The paper's "128KB" may be the unified size only; L1 capacity drives the hit rate (Fig. 25).
- **"64-bank" L2:** we read it as 64 sub-partitions (6 MB total). Another reading would change L2 capacity or parallelism.
- **Run length:** "instruction count" is not defined. If the paper counted warp instructions, 1B would cover 32× more work; three of our workloads are cut by the cap.
- **Coverage / accuracy granularity** (addresses vs sectors) and "correctly predicted" vs issued (OQ-14): coverage_pred vs coverage_paper differ by up to 78.7 pp (mri-q 88.0 vs 9.3, final_tier1.md).
- **Head organisation, T1/T2 codes, LRU group size, 25%-free scope, bandwidth definition** (OQ-1, -5, -3, -12, -10): each is a choice we made; their sensitivities are partly in Tier 2.
- **Fig. 23 axis labels:** "Coverage Loss (%)" vs "Average Coverage". The 2% loss at 50 cycles is stated in the text, so it is not affected.

## Corrections made to SPEC.md

1. **Per SM (§5):** "64K registers" became "65536 (Table 1 gives no unit; our config counts 32-bit registers)".
2. **Other caches (§5):** the L1I line size (128B) was missing.
3. **DRAM (§5):** SPEC copied the "(ns)" label. The values are DRAM clock cycles at 850 MHz in the simulator (1 cycle = 1.176 ns); note added.
4. **Benchmarks (§5):** Table 2 abbreviates mri-q as "MRQ"; added.

Checked and correct: the Table 1 values; the motivation numbers (30% reservation fails of L1D accesses, Fig. 3, 730; 33% L1-L2 bandwidth, Fig. 4, 730; 55% of stalls, Fig. 5, 730); "Snake outperforms Snake-DT and Snake-T by 13% and 7%" (737, §5.2); the 45% / 79% / 84% L1 hit rates (739, §5.7); and every §, page, figure and table citation in SPEC §1-§6 sampled above.
