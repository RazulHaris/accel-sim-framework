#!/usr/bin/env python3
"""Normalize an accel-sim output for the Snake off-regression.

    regress_norm.py <sim.out> [--strip-snake-block]

Removes only:
  * volatile lines: wall-clock time, simulation rate, slowdown,
    build strings and binary/library paths;
  * the -snake_* lines that gpgpu-sim's option dump gains from the new options
    (OptionParser::Print, src/option_parser.cc:422);
  * with --strip-snake-block, Snake's own stats block (a blank line,
    "Snake prefetcher:", snake_* lines), which only exists with Snake on.
and sorts every run of consecutive "  mf: uid=" lines: memory_sub_partition::
print (l2cache.cc:629) lists in-flight requests from a std::set<mem_fetch *>,
i.e. in heap-address order, which differs between binaries.  Sorting keeps
the set of requests compared while ignoring that order.
In "MSHR: tag=" lines (mshr_table::display, gpu-cache.cc:632-636) the
mem_fetch pointer printed with %p is replaced by <mf>; the rest of the line
is kept.  Everything else is passed through unchanged.
"""
import re
import sys

VOLATILE = re.compile(
    r"^-snake_|gpgpu_simulation_time|gpgpu_simulation_rate|gpu_total_sim_rate|"
    r"gpgpu_silicon_slowdown|Accel-Sim \[build|GPGPU-Sim.*build |"
    r"accelsim-commit-|gpgpu-sim_git-commit-|"
    r"/accel-sim\.out|/libcudart"
)
MSHR_PTR = re.compile(r" : 0x[0-9a-f]+ :")


def main():
    path = sys.argv[1]
    strip_block = "--strip-snake-block" in sys.argv[2:]
    with open(path, errors="replace") as f:
        lines = f.read().split("\n")

    out, mf_run = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if strip_block and line == "" and i + 1 < len(lines) \
                and lines[i + 1] == "Snake prefetcher:":
            i += 2
            while i < len(lines) and lines[i].startswith("snake_"):
                i += 1
            continue
        if line.startswith("  mf: uid="):
            mf_run.append(line)
            i += 1
            continue
        if mf_run:
            out.extend(sorted(mf_run))
            mf_run = []
        if line.startswith("MSHR: tag="):
            line = MSHR_PTR.sub(" : <mf> :", line)
        if not VOLATILE.search(line):
            out.append(line)
        i += 1
    out.extend(sorted(mf_run))
    try:
        sys.stdout.write("\n".join(out))
    except BrokenPipeError:  # reader (diff | head) stopped early
        pass


if __name__ == "__main__":
    main()
