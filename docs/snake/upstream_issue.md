# Draft upstream issue (not posted)

Target: accel-sim/accel-sim-framework issues. Drafted 2026-10-13 on branch `Snake`; the fix is accel-sim commit `fe8f37e`. **Not posted.**

---

## Traces from before tracer commit 756ed33 (tracer v3) are mis-decoded for predicated-off lanes

### Summary
Before commit `756ed33` ("Resolve wrong address generation issue in tracer", 2020-11-06), `tracer_tool.cu` printed one mask but compressed the addresses over another:
- The printed mask was `active_mask & predicate_mask`: `fprintf(resultsFile, "%08x ", ma->active_mask & ma->predicate_mask);`.
- The addresses were compressed over `std::bitset<32> mask(ma->active_mask);`.

`gpu-simulator/trace-parser/trace_parser.cc` decodes the addresses with the printed mask. Any memory instruction with active but predicated-off lanes is therefore decoded wrongly:
- **base+delta (mode 2):** the line has one delta per *active* lane after the first. The parser reads one per *printed* lane, so a printed lane that follows a predicated-off lane gets the partial delta sum of other lanes. That is usually a wild address.
- **base+stride (mode 1):** the base belongs to the first *active* lane. The parser gives it to the first *printed* lane, and every printed lane after a gap in the printed mask decodes to 0.
- **list (mode 0):** the line lists one address per active lane; the parser assigns them to the printed lanes in order.

`756ed33` fixed the tracer, but traces written before it are still read by the current parser. On `dev` (`0db0445`, Accel-Sim 2.0) the parser reads one delta fewer per line, but it still assumes the deltas follow the printed mask. The v1.2.0 release (`ed6d38b`) has both the pre-fix tracer (`tracer_tool.cu:549`) and the same parser.

### Affected traces
Every trace written by a tracer older than `756ed33`. They report `-accelsim tracer version = 3` (or older); the version was not bumped by the fix.

We found it in the Rodinia 3.1 and Parboil V100 traces we use. Their headers say `-nvbit version = 1.4`, `-accelsim tracer version = 3`, and the files are dated 2020-08-09; we believe they are the public V100 trace set.

### Minimal reproduction
backprop-rodinia-3.1 (input 65536), `kernel-1.traceg`, PC 0x90 (`input_node[ty] = input_cuda[index_in]` under `if (tx == 0)`; 16×16 blocks, so lanes 0 and 16 are printed):

```
0090 00010001 1 R12 LDG.E.SYS 1 R8 4 2 0x7f0641700004 733341581233 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -733341581229 733341581229 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

- The line has 31 deltas (lanes 1..31; the warp was fully active) for a printed mask of 2 lanes.
- **Correct:** lane 16 = base + Σ deltas[1..16] = base + (733341581233 − 733341581229) = `0x7f0641700008`. That is 4 bytes after lane 0, as expected for consecutive `ty`.
- **Current parser:** lane 16 = base + 733341581233 = `0x7fb100007fb5`.

A stand-alone check that needs no simulation:

```cpp
#include <bitset>
#include <cstdio>
#include "trace_parser.h"   // gpu-simulator/trace-parser
int main() {
  inst_trace_t t;
  t.parse_from_string("0090 00010001 1 R12 LDG.E.SYS 1 R8 4 2 0x7f0641700004 "
      "733341581233 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -733341581229 733341581229 "
      "0 0 0 0 0 0 0 0 0 0 0 0 0 0", 3, 0);
  printf("lane 16 = 0x%llx\n", (unsigned long long)t.memadd_info->addrs[16]);
  // prints 0x7fb100007fb5; expected 0x7f0641700008
}
```

`g++ -std=c++17 -Igpu-simulator/trace-parser t.cc gpu-simulator/trace-parser/trace_parser.cc -lz`

### Extent (V100 traces, all inputs and kernels)
Counted by `util/snake/scan_legacy_traces.py`, on the Snake branch of RazulHaris/accel-sim-framework. "Harmless" lines carry extra deltas but decode correctly: either the mask is empty, or lane 0 is the only printed lane.

| app | memory lines (global) | base+delta, 31 deltas, partial mask | base+delta harmless | base+stride with a mask gap (all shared memory here) |
|---|---|---|---|---|
| backprop | 1,048,583 (425,991) | 172,032 (16.4%) | 0 | 0 |
| hotspot | 1,452,024 (202,960) | 5,314 | 0 | 114,168 (7.9%) |
| srad_v1 | 49,597,440 | 4,000 | 200 | 0 |
| lud | 2,706,808 | 0 | 144 | 0 |
| nw | 3,686,400 | 0 | 16,384 | 0 |
| parboil histo | 6,565,000 | 0 | 0 | 0 |
| parboil mri-q | 21,792 | 0 | 0 | 0 |

In backprop, every execution of PC 0x90 reads a wild global address for lane 16. That adds an L1 miss and a DRAM access to a random line per warp. The effect on cycles, L1 and DRAM counts of a 100M-instruction run is in the table below.

Simulated with the clean base `3016c65` / `6c3cf4ff`, without and with the fix (QV100, GTO, 1530 MHz, 100M instructions):

| workload (100M insn, PAPER_V100) | cycles | L1D read hits | L1D read misses | DRAM reads | lines re-decoded |
|---|---|---|---|---|---|
| backprop 65536 | 26,170 → 25,995 (−0.67%) | 28,310 → 6,830 (−75.9%) | 49,798 → 44,172 (−11.3%) | 150,193 → 142,210 (−5.3%) | 172,032 |
| hotspot 512 | 19,697 → 19,686 (−0.06%) | 888 → 322 (−63.7%) | 88,430 → 85,256 (−3.6%) | 67,612 → 65,536 (−3.1%) | 1,673 |
| hotspot 1024 | 19,458 → 19,451 (−0.04%) | 1,414 → 761 (−46.2%) | 103,174 → 102,283 (−0.9%) | 79,422 → 78,173 (−1.6%) | 1,004 |
| srad_v1 | 60,676 (0) | 130,839 (0) | 54,659 (0) | 28,984 (0) | 40 |
| nw 2048_10 | 1,515,147 (0) | 11,233 (0) | 460,910 (0) | 519,049 (0) | 0 |
| parboil histo | 141,169 (0) | 991,495 (0) | 147,213 (0) | 172,871 (0) | 0 |

### Suggested fix
- **base+delta:** for tracer version ≤ 3 (which has no immediate field), a line with a partial printed mask and exactly 31 deltas comes from a fully active warp. Decode the deltas over all 32 lanes and keep only the printed lanes. This is exact, and every other line is parsed as before. Our patch (`fe8f37e`, 37 lines in `trace_parser.cc`) does this and prints the number of re-decoded lines once per run.
- **Not fixable from the trace alone:** base+stride and list lines, and base+delta lines with fewer than 31 deltas, would need the active mask, which these traces do not record. Regenerating the public traces with a post-`756ed33` tracer would fix all formats. Alternatively, the trace header could be checked and a warning printed for version ≤ 3 traces.
