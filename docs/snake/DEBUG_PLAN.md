# Snake debug and fix plan, A to Z

## Status (one line per stage; updated as each stage closes)

| Stage | Status | Results |
|---|---|---|
| 0 Set-up | done 2026-10-29: tag `snake-pre-debug` in both repos (accel-sim `e261d63`, gpgpu-sim `19e3d746`; binary build `4b51495` / `19e3d746` frozen in `/home/razul/snake/bin_pre_debug`); folders created; reference numbers recorded | results/debug/S0_reference.md |
| 1 Environment parity (H1) | in progress: 1.1 and 1.2 done 2026-10-29 (no runs); 1.3 bug-compatible decoder (`-trace_paperera_decode`, accel-sim `bc03386`, default off) running, 27 runs in `sim_run_snake_debug/s1s2` from 19:41; gates DBG1 passed | results/debug/S1_environment.md |
| 2 Upper bounds (oracles) | in progress: O-TIME and O-PRED built (gpgpu-sim `37846351`, flags default off); 18 runs in the same batch as 1.3 | |
| 3 Prefetch lifecycle trace | in progress: log built (`-snake_lc_stride`, gpgpu-sim `04bbd427`, default off; token `SNAKE_LC16`); logging on vs off identical on the 11 synthetic runs; 9 runs (`sim_run_snake_debug/s3`, frozen) start after the Stage 1.3/2 batch; analysis `util/snake/debug/s3_lifecycle.py` | |
| 4 What the programs allow | not started | |
| 5 Implementation audit | not started | |
| 6 Fixes | waits for approval | |
| 7 Write-up | not started | |

The plan below is the request as given (2026-10-29).

---

Prompt for Claude Code. Save a copy as `docs/snake/DEBUG_PLAN.md` and keep a status line per stage in it.

## Standing rules (unchanged)

- One Claude Code session at a time in these repos. Do not switch branches without asking.
- Clean trees (`_modified_0.0` build tags) before every launch. Never rebuild while jobs are queued unless each job pins its own `accel-sim.out` and `libcudart.so`.
- Every new behaviour sits behind a run-time flag that defaults to off. With all new flags off, Snake-off output stays byte-identical to the current build on the 6 check workloads. New stat lines are the only allowed difference, and the check must exclude them explicitly.
- After any change to Snake code: Fig. 15 test, Snake-off byte-identical check, prefetch-outcome invariant (issued = timely + late + evicted unused + unused at end).
- Primary result: whole-run IPC GMEAN, PAPER_V100, 1B instructions. Best-kernel numbers are diagnostics only and are never headlined.
- Update README, SNAKE_PLAN and SPEC (OQ changes) at every stage. Do not commit the paper PDF. Avoid runs not listed here.
- Report after each stage with a short TLDR and wait for approval before Stage 6.

## What we know (Tier 1 Snake, PAPER_V100, 1B, means over our 7 apps)

| Step | Snake paper | Ours |
|---|---|---|
| Baseline L1 hit rate | 45% (Fig. 25, its 11 apps) | 18.7% |
| L1 hit rate with Snake | 79% | 25.2% |
| Demand sectors predicted | 76% | 63% |
| Prefetches issued per demand sector | not reported | 0.26 |
| Demand served by a used prefetch (timely + late) | not reported | 22% |
| Demand served by a timely prefetch | 69% | 6.1% |
| Whole-run IPC gain | +14.1% | −0.29% |

Perfect-L1 ceilings (PAPER_V100, 1B): backprop +41.7, histo +39.2, srad_v1 +20.6, nw +16.6, lud 512 +3.6, mri-q +2.2, hotspot 512 +0.3. The paper reports hotspot +13, lud +18 and mri-q +4, all above our ceilings. Where our headroom is large, our timely accuracy is small: backprop 0.2%, histo 10.8%, srad_v1 3.5%, nw 0%. Where we are most timely (hotspot, 26%), there is no headroom.

Paper text that matters here:
- §2, Figs 3 to 5: in its memory-bound apps, reservation fails are about 30% of L1 accesses, L1-to-L2 bandwidth use about 33%, and about 55% of stalls are cycles where all warps wait on memory.
- §3.2: "when a PC_ld is encountered for the first time, Snake issues prefetching requests for all future warps, as soon as the train status … is promoted", and "for warps whose corresponding bits are set in the warpID vector, Snake issues prefetching requests based on the Inter-thread stride".
- §5.1: credits timeliness to the inter-thread chains ("prefetches consecutive PC_ld requests, reducing the chance of late prefetching").
- §5.2: histo and srad gain because Snake removes bursty misses and reservation fails.
- §6: concedes that "warps within a CTA are scheduled close in time and can not hide memory access latency" (citing CTA-aware prefetching).
- Table 3 and the latency paragraph: Snake is a 2-cycle pipeline overlapped with the issue of memory requests.
- §4: Accel-Sim v1.2.0.

## Hypotheses and the test that decides each

| ID | Hypothesis | Decided by |
|---|---|---|
| H1 | Our baseline is not at the paper's operating point (less memory-bound, lower hit rate) | Stage 1 |
| H1a | The trace decoder in the Accel-Sim release the paper names reads the 2020 traces differently (we measured backprop's hit rate at 36.2% with it against 13.4% fixed). This holds only if the paper used the public trace set. | 1.3 |
| H1b | Config differences from the v1.2.0 QV100 config or the paper's Table 1 | 1.2 |
| H2 | Most predictions never become prefetches (63% predicted, 26% issued) | 3a |
| H3 | Chains cannot be timely here because a warp issues its loads back-to-back | 4a |
| H4 | Inter-warp targets inside a CTA cannot be timely (the paper concedes this) | 4b |
| H5 | Snake triggers too late in the pipeline (at L1 accept or after reservation-fail retries, not at issue) | 5.1, 5.2 |
| H6 | The prefetch path adds delay (queue wait, miss-queue reserve, a mis-scaled throttle meter) | 3b, 5.3, 5.4 |
| H7 | A demand that merges into an in-flight prefetch waits longer than a plain miss (bug) | 3c, 5.10 |
| H8 | Unused floods and pollution (known on histo's prescan) | 3a, existing |
| H9 | Table capacity or the lane filter caps coverage (histo 34% vs the paper's 97%) | 5.11, 5.12 |

## Stage 0: set-up (no runs)

1. Confirm the branch and a clean tree. Tag the current build as `snake-pre-debug`.
2. Create `docs/snake/results/debug/` and `util/snake/debug/`.
3. Record the reference numbers (Tier 1 Snake and SK_R0_D0, PAPER_V100, 1B) from `Snake_all_versions_2026-10-28.xlsx`. Reuse its Block 3 metric definitions.

## Stage 1: environment parity (H1)

1.1 **From existing baseline runs (no new runs)**, per app at PAPER_V100, 1B:
- L1 hit rate.
- Reservation-fail share: RESERVATION_FAIL / all L1 accesses (hit + miss + reserved + fail), the paper's Fig. 3 definition.
- L1-to-L2 interconnect bandwidth used vs theoretical (Fig. 4).
- Share of stall cycles where all warps wait on memory (Fig. 5). If no existing stat gives this, add a counter-only stat.

Put these next to the paper's values: its averages (30%, 33%, 55%, 45%) and its per-app values read off Figs 3, 4, 5 and 25 (±3 points).

1.2 **Config diff.** Compare our PAPER_V100 against the Accel-Sim v1.2.0 QV100 config and the paper's Table 1:
- L1: 128 KB, 256-way, 128 B lines, 4 banks, 28 cycles.
- MSHR: 512 entries, 8 merges.
- L2: 96 KB per sub-partition, 24-way, 212 cycles.
- DRAM timings, interconnect and adaptive-cache settings.

List every difference and say whether it plausibly changes how memory-bound an app is.

1.3 **Bug-compatible decoder.** Add a flag that reproduces the paper-era trace decoding we fixed (default off). Run baseline, Tier 1 Snake and perfect-L1 at PAPER_V100, 1B on the 9 workloads (27 runs). Per app, report: L1 hit rate, reservation-fail share, perfect-L1 ceiling, and Snake coverage, accuracy and IPC. Show three columns: paper, bug-compatible, corrected.

**Gate G1.** If the bug-compatible runs move the baseline hit rates and Snake's numbers clearly toward the paper, the main gap is the environment. Write that up neutrally ("the decoder in the release the paper names reads these traces differently; with it, our Snake gets X"). Our primary results stay on the corrected decoder. Continue to Stage 2 either way.

## Stage 2: upper bounds for Snake in our environment

Add two oracle flags (default off):
- **O-TIME:** every prefetch Snake actually issues is installed in L1 the moment it is issued. The request still goes to memory, so traffic is counted, but its fill is discarded.
- **O-PRED:** every prediction Snake makes, whether issued or dropped for any reason, is installed the same way.

Run Tier 1 with each flag at PAPER_V100, 1B on the 9 workloads (18 runs). Report the whole-run GMEAN and per-app gains next to Tier 1 and perfect-L1.

**Gate G2.**
- Timeliness fixes are worth pursuing only if O-TIME beats Tier 1 by at least 3 GMEAN points, or by at least 10 points on some app.
- Issue-loss fixes are worth pursuing only if O-PRED beats O-TIME by at least 3 GMEAN points.
- If both fail, Snake's own predictions cannot produce large gains in this environment. Run Stages 3 to 5 only to rule out bugs, skip the extensions in Stage 6, and write the conclusion.

## Stage 3: prefetch lifecycle trace (instrumentation only, no behaviour change)

Log sampled records: a few SMs, or every 16th prediction. Keep the total logs to a few GB.

For each prediction, log:
- kernel and SM;
- trigger warp, CTA and PC;
- source: inter-thread at depth k, intra-warp, or inter-warp at distance d;
- target warp, CTA, PC and sector;
- cycles when it was predicted, enqueued, dropped (with the reason) or sent to L1, allocated or merged in the MSHR, and filled;
- L2 hit or miss;
- the first demand to the sector: its cycle, its warp and the outcome (timely, late merge, miss after eviction, or never used).

For each demand miss, log its issue cycle, its fill cycle and whether it merged into a prefetch.

Run Tier 1 at PAPER_V100, 1B on the 9 workloads with logging on (9 runs). Put the analysis scripts in `util/snake/debug/`.

- **3a Funnel**, per app and per source: predicted, then dropped by reason (already in L1, already in flight, queue full, throttled, reserve, filter, duplicate), then issued, filled, and finally timely, late, evicted unused or unused at the end. This must explain how 63% predicted becomes 26% issued.
- **3b Lead time**, per source, as distributions: available lead (demand cycle − predicted cycle), queue wait (sent − predicted) and memory latency (filled − sent).
- **3c Late-merge latency:** latency of a demand that merged into an in-flight prefetch vs a plain miss in the same kernel. A merged demand must never wait longer than a plain miss. If it does, that is a bug (H7).

## Stage 4: what the programs allow (from the Stage 3 logs and existing runs)

- **4a Load spacing in a warp:** cycles between consecutive global loads of the same warp (PC_i to PC_i+1), as a distribution. Report the share of pairs whose gap is at least the median miss latency. If most pairs are back-to-back, chain prefetches cannot be timely by construction (H3).
- **4b Inter-warp spacing:** for each load PC, the cycles between warp w and warp w+d reaching it. Cover d from 1 to (warps per CTA − 1), and across CTAs. This repeats the CTA-aware paper's Fig. 1 on our apps and bounds the lead any inter-warp prefetch can have (H4).
- **4c Loop iteration time** for loads with intra-warp strides.
- **4d Per-kernel ceilings:** perfect-L1 gain per kernel from the existing D1 runs (`kernel_gap.py`) for histo, srad_v1, backprop and nw, with Snake's Stage 3 funnel for the kernels that hold the headroom.

## Stage 5: implementation audit (bug hunt)

For each item: read the code, write a unit test or a micro-trace, and state pass or fail with evidence in `results/debug/S5_audit.md`.

1. **Trigger point:** where Snake sees a load (issue, operand collection or L1 accept). The paper says Snake overlaps with the issue of memory requests.
2. **Reservation-fail retries:** does a retried load re-train, re-trigger or duplicate prefetches?
3. **Prefetch queue:** drain rate, priority vs demand, the reserve, starvation time.
4. **Throttle meter:** the share of cycles throttled, and the meter's units against the 70%-on and 50%-off thresholds of peak bandwidth.
5. **Decoupling:** the 50% L1 limit until trained and the 50-cycle confinement must not lower the demand hit rate on lines Snake never touched.
6. **Inter-warp targets:** the target is resident, in the same CTA and has not yet executed the PC. Count prefetches aimed at warps that have already passed the PC.
7. **Chain walk:** which warp's address is the base for each step, the depth limit and upkeep of the warpID vector. List exactly which (warp, PC) targets one trigger generates, and compare with §3.2 and Fig. 15.
8. **In-flight and duplicate filters:** a prediction dropped because demand is already in flight must be counted as "born late", not silently lost.
9. **Fill and insert:** a prefetched sector is visible to demand (tag and sector mask). A demand miss on a prefetched sector that is still resident must be 0.
10. **Late-merge wake-up:** a demand merged into a prefetch's MSHR gets its data at that fill, with no extra round trip and no lost wake-up. Cross-check with 3c.
11. **Head and Tail capacity:** eviction and thrash rates per app. Explain histo's 34% coverage against the paper's 97%.
12. **Lane and shape filter (OQ-17):** the share of warps rejected per app, and what the paper-literal reading would do.

For every bug found, apply the fix, then pass the three checks in the standing rules. Then run a 100M gate on histo, srad_v1 and backprop, followed by 1B on the 9 workloads.

## Stage 6: fixes (only what Gates G1 and G2 allow; wait for approval)

In order:
1. **Faithful bug fixes** from Stage 5. The result becomes "Tier 1 Snake (fixed)" and replaces Tier 1 as the baseline.
2. **Faithful re-readings.** Each must still reproduce Fig. 15; record the OQ change.
   - Trigger at issue.
   - Prefetch for all remaining warps of the CTA once promoted (§3.2, "all future warps"), not just the next 4.
   - Inter-thread prefetches for every warp set in the warpID vector, if our current reading is narrower.
3. **The paper's own combination, Snake+CTA:** per-CTA base addresses as in CTA-aware prefetching (the CAPS PDF is in the project), so inter-warp targets can reach later CTAs with correct addresses.
4. **Our extensions,** labelled as extensions, not Snake:
   - lead-time-aware distance, which picks d so that the expected lead from 4b is at least the measured miss latency;
   - prefetch-aware warp priority, as in CAPS PAS.

**Acceptance:** first a 100M gate on histo, srad_v1 and backprop; then 1B on the 9 workloads. Accept only if the whole-run GMEAN improves and no app loses more than 0.5%.

## Stage 7: write-up

- Classify the gap with numbers into environment, interpretation (OQ), implementation bug and inherent (what O-PRED says Snake's predictions can buy).
- Add the new versions to the all-versions xlsx (Blocks 1 and 2), and update `results/debug/*.md`, README, SNAKE_PLAN and SPEC.
- One-paragraph conclusion. If the gates fail, freeze Tier 1 Snake (fixed) and the SnakeCharmer-configured Snake as baselines and move on to SCAP.

## Run budget before Stage 6

| Stage | Runs | Cap |
|---|---|---|
| 1.1, 1.2 | 0 | existing |
| 1.3 bug-compatible | 27 | 1B |
| 2 oracles | 18 | 1B |
| 3 lifecycle trace | 9 | 1B, sampled logs |
| 4 | 0 | Stage 3 logs and existing |
| 5 | unit tests; at most 3 runs at 100M per bug found | 100M |

## Questions for the Snake authors (if still unanswered)

1. The exact Accel-Sim commit and trace set. Were they the public 2020 traces or traces of your own?
2. The inputs for each app.
3. Does accuracy count a demand to an address that was predicted but already in L1?
4. How far does "all future warps" reach: the CTA, the SM, or a fixed count?
5. Do inter-thread prefetches target only the triggering warp, or every warp in the warpID vector?
6. The per-app baseline L1 hit rates behind Fig. 25.
