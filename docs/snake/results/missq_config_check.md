# Miss-queue and memory-path configuration vs the paper (2026-10-14)

No runs. Sources:
- configs: `configs/tested-cfgs/SM7_QV100/gpgpusim.config` at ours (`6c3cf4ff`), dev `6b244a5d` and release `971dd69e`, plus `PAPER_V100` (`-gpgpu_scheduler gto`, `-gpgpu_clock_domains 1530:1530:1530:850`);
- the effective values from the option dump of the cached `bin_ref_v2` outputs (`sim_run_snake_refcache/bin_ref_v2/runs`, QV100-SASS-PAPER_V100-100M_INSN);
- source at `6c3cf4ff` (line numbers below; the Snake branch adds hooks, which shifts some lines).
- Tables in §3 come from `util/snake/missq_check.py sim_run_snake_refcache/bin_ref_v2/runs`.

The paper gives no miss-queue size or drain rate. It only says reservation fails are "primarily due to miss queue congestion" (§2).

## 1. Configuration

**Cache strings.** Format (`gpu-cache.h:578`): `type:nset:line:assoc, replacement:write:alloc:write_alloc:set_index, mshr_type:mshr_entries:mshr_merge, miss_queue:result_fifo, data_port_width`.

- **dl1, ours = dev:** `S:4:128:64,L:T:m:L:L,A:512:8,16:0,32`
  - sector cache; 4 sets × 128 B × 64 ways = 32 KB base
  - LRU, write-through, allocate on miss, lazy fetch-on-read write allocation, linear set index
  - associative MSHR with 512 entries and merge 8
  - miss queue 16, result FIFO 0, 32 B data port
  - With `-gpgpu_unified_l1d_size 128` and `-gpgpu_adaptive_cache_config 1`, the ways scale to 4 × 64 = 256 (128 KB), minus each kernel's shared-memory carve-out (`shader.cc:3671-3690`). Our kernels run at 96–128 KB: hotspot 96, backprop 112, lud and nw 120, histo 96–128, srad_v1 112–128, mri-q 128 (from the outputs' "Reconfigure L1 cache" lines).
- **dl1, release:** `S:1:128:256,L:L:s:N:L,A:256:8,16:0,32`
  - 1 set × 256 ways (32 KB base); carved out up to 1,024 ways (128 KB, fully associative)
  - local write-back / global write-through, streaming (made allocate-on-fill), no write allocation
  - the streaming cache overrides the MSHRs to 4,096 entries (lines × 4 sectors) and merge 64 (release `gpu-cache.h:562-566`)
  - miss queue 16
- **dl2, all three:** `S:32:128:24,L:B:m:L:P,A:192:4,32:0,32`
  - per memory sub-partition: 32 sets × 128 B × 24 ways = 96 KB; 64 sub-partitions (32 channels × 2), 6 MB in total
  - LRU, write-back, allocate on miss, lazy fetch-on-read, IPOLY hashed set index
  - MSHR 192 entries, merge 4; miss queue 32; 32 B port

| setting | ours (6c3cf4ff + PAPER_V100) | dev 6b244a5d | release 971dd69e | paper Table 1 | verdict (ours) |
|---|---|---|---|---|---|
| L1D size / assoc | 128 KB max, 4 sets × 256 ways (96–128 KB per kernel) | same | 128 KB max, 1 set × 1,024 ways (streaming) | 128 KB, 256-way | **match** (size depends on the carve-out) |
| L1D line | 128 B, 4 × 32 B sectors | same | same | 128 B | match |
| L1D banks (`-gpgpu_l1_banks`) | 4 | 4 | 4 | 4 banks | match |
| L1D latency (`-gpgpu_l1_latency`) | 20 | 20 | 20 | 28 cycles | **differs** (see §4) |
| L1D MSHR entries / merge | 512 / 8 | 512 / 8 | 4,096 / 64 (streaming override) | 512 / 8 | match (release differs) |
| L1D miss queue | 16 | 16 | 16 | — | not in paper |
| L1D allocation | on miss | on miss | on fill (streaming) | — | not in paper |
| L2 per sub-partition | 96 KB, 24-way, 128 B | same | same | 96 KB, 24-way, 128 B | match |
| L2 banks | 64 sub-partitions (`-gpgpu_n_mem 32` × 2) | same | same | 64 banks | match |
| L2 latency (`-gpgpu_l2_rop_latency`) | 160 | 160 | 160 | 212 cycles | **differs** (see §4) |
| L2 MSHR / miss queue | 192 / merge 4 / 32 | same | same | — | not in paper |
| `-dram_latency` | 100 | 100 | 100 | — | not in paper |
| DRAM timing | nbk=16 CCD=1 RRD=3 RCD=12 RAS=28 RP=12 RC=40 CL=12 WL=2 CDLR=3 WR=10 CCDL=2 RTPL=3 | same | same | same values | match |
| `-gpgpu_dram_partition_queues` (icnt→L2 : L2→DRAM : DRAM→L2 : L2→icnt) | 64:64:64:64 | same | same | — | not in paper |
| FR-FCFS queue / DRAM return queue | 64 / 192 | same | same | — | not in paper |
| `-network_mode` | 2 (local crossbar, iSLIP) | 2 | 2 | — | not in paper |
| `-icnt_in_buffer_limit` / `-icnt_out_buffer_limit` | 512 / 512 packets | same | same | — | not in paper |
| `-icnt_flit_size` | 40 (unused by the local crossbar) | 40 | 40 | — | not in paper |
| `config_volta_islip.icnt` (mode 1 only) | input / ejection / VC buffers 256, flit 40 | same | same | — | not used |
| cluster ejection / LD/ST response buffers | 32 / 2 | 32 / 2 | 32 / 2 | — | not in paper |
| core clock | 1530 MHz (base file 1132) | 1447 | 1447 | 1530 MHz | match (via PAPER_V100) |
| scheduler | gto (base file lrr) | lrr | gto | GTO | match (via PAPER_V100) |

## 2. Code (base `6c3cf4ff`)

- **Drain: at most one request per cycle per L1D.** `baseline_cache::cycle()` (`gpu-cache.cc:1215-1222`) looks only at `m_miss_queue.front()`, and sends it if `!m_memport->full(...)`. It is called once per core cycle from `ldst_unit::cycle` (`shader.cc:2910`).
- **What blocks the drain.**
  - `shader_memory_interface::full` (`shader.h:2741-2743`) calls `simt_core_cluster::icnt_injection_buffer_full` (`shader.cc:4594-4598`): reads use `READ_PACKET_SIZE`, writes their size, then `!icnt_has_buffer(cluster, size)`.
  - With `-network_mode 2`, `icnt_has_buffer` is `LocalInterconnect_has_buffer` (`icnt_wrapper.cc:104-106, 184`) → `LocalInterconnect::HasBuffer` (`local_interconnect.cc:385-394`). That **ignores the size** and calls `Has_Buffer_In(sm, 1)` on the request network.
  - `Has_Buffer_In` is true while `in_buffers[sm].size() + 1 <= 512` (`local_interconnect.cc:99-104`).
  - The crossbar (`iSLIP_Advance`, `local_interconnect.cc:193-236`) moves at most one packet per output (`Has_Buffer_Out(dest, 1)`) and per input (`input_nodes.erase`) per icnt cycle.
  - In our outputs the input buffer was never full (`Req_Network_in_buffer_full_per_cycle` = 0.0000 on all 9 workloads). The drain limit is therefore the one-per-cycle rule, not the interconnect.
- **Banks share one miss queue.** Each L1D (`baseline_cache`) has a single `std::list<mem_fetch*> m_miss_queue` (`gpu-cache.h:1396`).
  - The 4 banks are only the L1 latency pipelines (`l1_latency_queue[bank][stage]`). `process_memory_access_queue_l1cache` (`shader.cc:2063-2106`) places at most one access per bank per cycle.
  - `L1_latency_queue_cycle` (`shader.cc:2122-2213`) gives each bank's head its tag access every cycle.
  - So up to 4 misses per cycle can enter the one queue, which drains 1 per cycle.
- **Entry tests.**
  - A read miss fails when `miss_queue_full(1)`, i.e. size + 1 ≥ 16 (`rd_miss_base`, `gpu-cache.cc:1843-1851`; `miss_queue_full` at `gpu-cache.h:1434-1436`). The access becomes RESERVATION_FAIL, counted as MISS_QUEUE_FULL.
  - `send_read_request` also needs size < 16 (`gpu-cache.cc:1375`).
  - Writes test `miss_queue_full(0)` / `(1)` / `(2)` by policy (write hit and miss handlers, `gpu-cache.cc:1455-1840`).
- **Rejected demand: retried every cycle, blocking its bank.**
  - On RESERVATION_FAIL, `L1_latency_queue_cycle` leaves the request at `l1_latency_queue[bank][0]` (`shader.cc:2183-2185`). It is accessed again next cycle, and every retry counts as another RESERVATION_FAIL.
  - Later stages of that bank shift only into empty slots (`shader.cc:2207-2212`), so its pipeline fills up behind the head.
  - When its entry stage (`[l1_latency-1]`) is occupied, new accesses to that bank get BK_CONF (`shader.cc:2081-2097`). The instruction stays in the LD/ST dispatch register (`COAL_STALL`/`BK_CONF`, `shader.cc:2323-2328`), which stalls issue to the memory pipeline.

## 3. Motivation figures from the bin_ref_v2 baselines

Definitions are in the `missq_check.py` docstring:
- **Fig. 4 peak:** the local crossbar moves ≤ 1 packet per input and per output each cycle, and the 64 L2 sub-partitions bound both directions, so 64 packets per cycle per subnet. A reply packet carries a 32 B sector, i.e. 2 KB/cycle (3.1 TB/s at 1530 MHz).
- **Request-slot utilisation:** request packets / (cycles × 80 SMs), since each L1D sends at most 1 per cycle. It includes the few L1I/L1C requests.
- **Fig. 5:** not in the outputs as defined. The proxy is the scheduler's no-issue cycles, from the "Warp Occupancy Distribution":
  - W0_Scoreboard: every valid warp blocked on a register dependency (mostly loads);
  - Stall: a warp was ready but the pipeline was stalled, e.g. the LD/ST unit blocked by retries;
  - W0_Idle: no valid instruction.

### Fig. 3: reservation-fail rate and causes

| workload | L1D accesses | RESERVATION_FAIL | rate | paper | of fails: MISS_QUEUE_FULL / MSHR entry / MSHR merge / line alloc | fails from reads / writes |
|---|---|---|---|---|---|---|
| backprop / 65536 | 526658 | 141535 | 0.27 | 0.29 | 100% / 0% / 0% / 0% | 75253 / 66282 |
| hotspot / 1024_2_2___data_temp_102 | 271555 | 20730 | 0.08 | 0.57 | 100% / 0% / 0% / 0% | 20275 / 455 |
| hotspot / 512_2_2___data_temp_512_ | 234131 | 17891 | 0.08 | 0.57 | 100% / 0% / 0% / 0% | 17562 / 329 |
| lud / _i___data_512_dat | 778523 | 43497 | 0.06 | 0.20 | 100% / 0% / 0% / 0% | 42855 / 642 |
| lud / _s_256__v | 216855 | 38183 | 0.18 | 0.20 | 100% / 0% / 0% / 0% | 38177 / 6 |
| nw / 2048_10 | 1253404 | 0 | 0.00 | 0.03 | - / - / - / - | 0 / 0 |
| histo / _i___data_default_input_ | 1907758 | 149403 | 0.08 | 0.16 | 100% / 0% / 0% / 0% | 112252 / 37151 |
| mri-q / _i___data_small_input_32 | 50116 | 28484 | 0.57 | 0.80 | 100% / 0% / 0% / 0% | 28484 / 0 |
| srad_v1 / 100_0_5_502_458 | 793086 | 171227 | 0.22 | 0.71 | 71% / 0% / 29% / 0% | 134382 / 36845 |

### Fig. 4: L1<->L2 interconnect utilisation (% of 64 packets/cycle) and request-slot utilisation (% of 1 request/SM/cycle)

| workload | icnt cycles | request pkts | reply pkts | request util | reply util | paper (Fig. 4) | request-slot util | request in-buffer full per cycle |
|---|---|---|---|---|---|---|---|---|
| backprop / 65536 | 25995 | 336405 | 331388 | 20.2% | 19.9% | 44% | 16.2% | 0.0000 |
| hotspot / 1024_2_2___data_temp_102 | 19451 | 249835 | 244415 | 20.1% | 19.6% | 32% | 16.1% | 0.0000 |
| hotspot / 512_2_2___data_temp_512_ | 19686 | 215888 | 215888 | 17.1% | 17.1% | 32% | 13.7% | 0.0000 |
| lud / _i___data_512_dat | 247887 | 701115 | 698840 | 4.4% | 4.4% | 22% | 3.5% | 0.0000 |
| lud / _s_256__v | 501526 | 178032 | 178032 | 0.6% | 0.6% | 22% | 0.4% | 0.0000 |
| nw / 2048_10 | 1515147 | 1242171 | 1242171 | 1.3% | 1.3% | 24% | 1.0% | 0.0000 |
| histo / _i___data_default_input_ | 141169 | 1760111 | 1759964 | 19.5% | 19.5% | 45% | 15.6% | 0.0000 |
| mri-q / _i___data_small_input_32 | 30503 | 21632 | 21632 | 1.1% | 1.1% | 20% | 0.9% | 0.0000 |
| srad_v1 / 100_0_5_502_458 | 60676 | 433742 | 431102 | 11.2% | 11.1% | 32% | 8.9% | 0.0000 |

### Fig. 5 (proxy): no-issue scheduler-cycles in which all warps wait on dependencies

| workload | Stall | W0_Idle | W0_Scoreboard | Scoreboard share | (Scoreboard + Stall) share | paper (Fig. 5) |
|---|---|---|---|---|---|---|
| backprop / 65536 | 3086256 | 496 | 9017 | 0.00 | 1.00 | 0.76 |
| hotspot / 1024_2_2___data_temp_102 | 1209682 | 400 | 47547 | 0.04 | 1.00 | 0.52 |
| hotspot / 512_2_2___data_temp_512_ | 1092005 | 78051 | 214924 | 0.16 | 0.94 | 0.52 |
| lud / _i___data_512_dat | 4301422 | 3240649 | 1181403 | 0.14 | 0.63 | 0.26 |
| lud / _s_256__v | 738315 | 2807682 | 1805747 | 0.34 | 0.48 | 0.26 |
| nw / 2048_10 | 383720 | 121088387 | 41323285 | 0.25 | 0.26 | 0.71 |
| histo / _i___data_default_input_ | 789931 | 404761 | 8785761 | 0.88 | 0.96 | 0.47 |
| mri-q / _i___data_small_input_32 | 1721964 | 5338 | 1480958 | 0.46 | 1.00 | 0.96 |
| srad_v1 / 100_0_5_502_458 | 2348828 | 461911 | 650737 | 0.19 | 0.87 | 0.32 |

### Per app (mean over inputs) and GMEAN over the 7 apps

| app | Fig. 3 ours / paper | Fig. 4 reply util ours / paper | request-slot util | Fig. 5 proxy (Scoreboard; +Stall) ours / paper |
|---|---|---|---|---|
| backprop | 0.27 / 0.29 | 19.9% / 44% | 16.2% | 0.00; 1.00 / 0.76 |
| hotspot | 0.08 / 0.57 | 18.4% / 32% | 14.9% | 0.10; 0.97 / 0.52 |
| srad_v1 | 0.22 / 0.71 | 11.1% / 32% | 8.9% | 0.19; 0.87 / 0.32 |
| lud | 0.12 / 0.20 | 2.5% / 22% | 2.0% | 0.24; 0.55 / 0.26 |
| nw | 0.00 / 0.03 | 1.3% / 24% | 1.0% | 0.25; 0.26 / 0.71 |
| histo | 0.08 / 0.16 | 19.5% / 45% | 15.6% | 0.88; 0.96 / 0.47 |
| mri-q | 0.57 / 0.80 | 1.1% / 20% | 0.9% | 0.46; 1.00 / 0.96 |
| **GMEAN** (Fig. 3 without nw's 0; Fig. 4 and slot: mean) | 0.17 / 0.26 | 10.5% / 33% | 8.5% | 0.14; 0.73 / 0.55 |

## 4. Conclusions

- **Memory-path behaviour: qualitatively plausible, quantitatively milder than the paper.**
  - The paper's claim holds: MISS_QUEUE_FULL causes 100% of reservation fails in 6 workloads and 71% in srad_v1, where the rest are MSHR merge fails. There are no MSHR-entry or line-allocation fails.
  - The fail rate matches for backprop (0.27 vs 0.29) and lud 256 (0.18 vs 0.20). It is lower elsewhere: hotspot 0.08 vs 0.57, srad_v1 0.22 vs 0.71, mri-q 0.57 vs 0.80, GMEAN 0.17 (without nw's 0) vs 0.26.
  - Interconnect utilisation is lower everywhere: average 10.5% vs 33% by our peak definition, and the request network's input buffers never fill.
  - The Fig. 5 proxy is not comparable. Our blocked-memory-pipeline cycles show up as "Stall", not as scoreboard waits: Scoreboard alone gives GMEAN 0.14, Scoreboard + Stall 0.73, against the paper's 0.55.
  - Possible reasons, not verified: the paper's run length (1B instructions or to completion, vs our 100M cap), its inputs (OQ-13) and its simulator version (VERSION_DIFF). The bottleneck itself (a 16-entry queue drained at 1 per cycle and fed by 4 banks) has the same mechanism.
- **Settings that differ from Table 1, and how they could affect Snake:**
  - **L1 latency 20 vs 28.** `-gpgpu_l1_latency` is the depth of the L1 bank pipeline before the tag access. Table 1's 28 is probably the measured load-to-use latency of a V100 L1 hit, which our pipeline plus core stages may also reach. The load-to-use latency is not in our outputs (unavailable). A shorter L1 makes demand misses reach the miss queue sooner, so a prefetch has less head start: lateness, which S3c found to be the main loss, could be slightly worse.
  - **L2 latency 160 vs 212.** `-gpgpu_l2_rop_latency` is only the fixed ROP delay. Table 1's 212 is probably the L2 hit latency measured from the SM, which in ours adds the interconnect and queues. L2 hit latency is not in the outputs (unavailable; `averagemflatency`, over all requests including DRAM, is 240–353 cycles, mri-q 734).
    - If our L2 round trip is shorter than the paper's, prefetches are less needed, and Snake's IPC gain would be lower.
    - A longer real latency would let late prefetches cover more of the demand's wait.
  - **L1 size per kernel 96–128 KB** (carve-out) vs 128 KB: less room for prefetched lines in hotspot (96 KB); relevant to decoupling (S4).
  - The release config differs more, and would change Snake's issue path: 4,096 MSHRs with merge 64 and a streaming, allocate-on-fill L1D, i.e. no line-allocation or MSHR limits. The paper's 512 / 8 matches ours and dev, not the release.
- **Miss-queue size: no paper basis for a different value.** The paper gives no size or drain rate, and our MSHR, bank and line settings match Table 1. **Keep the Accel-Sim default of 16** (same in all three versions).
