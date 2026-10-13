#!/usr/bin/env python3
"""Scan accel-sim traces for addresses written by the pre-756ed33 tracer.

    scan_legacy_traces.py [-j N] <app trace dir> [...]

Before tracer commit 756ed33 (2020-11-06) the tracer compressed a warp's
addresses over the active mask but printed active & predicate as the line's
mask.  Lines of memory instructions with predicated-off lanes are then
mis-decoded by gpu-simulator/trace-parser (see docs/snake/upstream_issue.md).
Per app (all inputs, all kernel-*.traceg files under the directory) this
counts the memory-instruction lines and, per address format:

  base_delta  fixed      partial mask and 31 deltas: decoded exactly by the
                         parser fix (trace_parser.cc, legacy_base_delta_*)
  base_delta  harmless   extra deltas, but every printed lane decodes right:
                         empty mask, or lane 0 is the only printed lane
  base_delta  wrong      any other token count != popcount(mask) - 1
  list_all    mismatch   address count != popcount(mask)
  base_stride gap        the printed mask has a gap: lanes after it decode to
                         0 and the base belongs to the first active lane,
                         which may precede the first printed one (global /
                         other spaces counted separately)

A base+stride line whose predicated-off lanes precede all printed lanes
decodes with a shifted base and cannot be detected from the trace.  Only
tracer version <= 3 is checked (later versions add an immediate field).
"""
import glob
import os
import sys
from multiprocessing import Pool

KEYS = ("mem", "global", "list_all", "list_bad", "stride", "stride_gap_global",
        "stride_gap_other", "delta", "delta_fixed", "delta_harmless",
        "delta_wrong", "files_v_gt3")


def scan_file(path):
    c = dict.fromkeys(KEYS, 0)
    with open(path, errors="replace") as f:
        for line in f:
            if line[0] in "-#\n":
                if line.startswith("-accelsim tracer version"):
                    if int(line.split("=")[1]) > 3:
                        c["files_v_gt3"] += 1
                        return c
                continue
            t = line.split()
            if len(t) < 6:
                continue
            try:
                nd = int(t[2])
                i = 4 + nd
                i += 1 + int(t[i])
                width = int(t[i])
            except (ValueError, IndexError):
                continue
            if width == 0:
                continue
            mode = int(t[i + 1])
            rest = len(t) - (i + 2)
            p = bin(int(t[1], 16)).count("1")
            c["mem"] += 1
            m = int(t[1], 16)
            glob_op = t[3 + nd].startswith(("LDG", "STG", "ATOMG", "RED"))
            c["global"] += glob_op
            if mode == 0:
                c["list_all"] += 1
                c["list_bad"] += rest != p
            elif mode == 1:
                c["stride"] += 1
                if "0" in bin(m)[2:].strip("0"):
                    c["stride_gap_global" if glob_op else "stride_gap_other"] += 1
            else:
                c["delta"] += 1
                n = rest - 1
                if n != p - 1:
                    if n == 31 and p < 32:
                        c["delta_fixed"] += 1
                    elif p == 0 or m == 1:
                        c["delta_harmless"] += 1
                    else:
                        c["delta_wrong"] += 1
    return c


def main():
    args = sys.argv[1:]
    jobs = 16
    if args[:1] == ["-j"]:
        jobs, args = int(args[1]), args[2:]
    files = {a: sorted(glob.glob(os.path.join(a, "**", "kernel-*.traceg"),
                                 recursive=True)) for a in args}
    with Pool(jobs) as pool:
        res = dict(zip([(a, f) for a in args for f in files[a]],
                       pool.map(scan_file, [f for a in args for f in files[a]],
                                chunksize=1)))
    print("| app | kernel files | memory lines (global) | base+delta fixed "
          "| base+delta harmless / wrong | list_all mismatch "
          "| base+stride gap: global / other | still mis-decoded after fix |")
    print("|---|---|---|---|---|---|---|---|")
    for a in args:
        t = dict.fromkeys(KEYS, 0)
        for f in files[a]:
            for k, v in res[(a, f)].items():
                t[k] += v
        left = (t["delta_wrong"] + t["list_bad"] + t["stride_gap_global"] +
                t["stride_gap_other"])
        print("| %s | %d%s | %d (%d) | %d (%.2f%%) | %d / %d | %d | %d / %d | %d (%.2f%%) |" % (
            os.path.basename(a.rstrip("/")), len(files[a]),
            " (%d v>3 skipped)" % t["files_v_gt3"] if t["files_v_gt3"] else "",
            t["mem"], t["global"], t["delta_fixed"],
            100.0 * t["delta_fixed"] / max(t["mem"], 1), t["delta_harmless"],
            t["delta_wrong"], t["list_bad"], t["stride_gap_global"],
            t["stride_gap_other"], left, 100.0 * left / max(t["mem"], 1)))


if __name__ == "__main__":
    main()
