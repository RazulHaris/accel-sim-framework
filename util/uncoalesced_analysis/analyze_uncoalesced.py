#!/usr/bin/env python3
"""
Find repeating address deltas/"chains" across the uncoalesced-load dumps
produced by gpgpu-sim's ldst_unit::dump_uncoalesced_load_addresses
(see gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc).

Each dump is a CSV with one row per lane that actually issued a transaction
for an uncoalesced global load:

    cycle,sm_id,warp_id,uid,pc,num_coalesced_transactions,thread,address

This script looks for two kinds of repeating structure:

1. Lane signature ("chain within one load"): for every dynamic uncoalesced
   load (one uid), sort its lanes and compute each lane's address offset
   from the group's minimum address. If the same (lanes -> offsets) pattern
   keeps showing up for a given load PC, that load has a deterministic,
   structural cause (e.g. a fixed non-unit stride or an array-of-structs
   layout) rather than data-dependent/random divergence.

2. Temporal stride ("chain across time"): for a fixed (sm, warp, thread, pc)
   tuple, addresses seen across different dynamic instances of that same
   load (different loop iterations / uids) form a sequence. The delta
   between consecutive values shows the per-thread stride across
   iterations of the same static load.

For both, it also reports the most common single-step address deltas and
their GCD, which is often the easiest way to spot "always off by N bytes".

Memory: logs are processed one row at a time and only small aggregate
counters are kept in memory (never the raw rows), so file size does not
bound how much RAM is needed -- multi-GB dumps are fine. This relies on
two properties of how the dump is written: all lanes of one dynamic load
(one uid) are written together in one contiguous block in thread order,
and rows for a given (sm, warp, thread, pc) appear in the file in
non-decreasing cycle order. (Verified on a 20.96 GB / 460M-row real dump:
peak RSS ~640MB, ~17.7 minutes.)

Usage:
    ./analyze_uncoalesced.py sim_run_12.8/
    ./analyze_uncoalesced.py bench1/uncoalesced_loads.log bench2/uncoalesced_loads.log
    ./analyze_uncoalesced.py sim_run_dir/ --top 10 --label-parts 2 --out-dir results/
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from collections import Counter, defaultdict


def find_logs(inputs, pattern):
    """Return a de-duplicated list of (path, root) pairs. `root` is the
    directory the file was discovered under, used later to build a short,
    human-readable label for each log (e.g. the benchmark name)."""
    found = []
    for inp in inputs:
        if os.path.isfile(inp):
            found.append((inp, os.path.dirname(inp) or "."))
        elif os.path.isdir(inp):
            for p in glob.glob(os.path.join(inp, "**", pattern), recursive=True):
                found.append((p, inp))
        else:
            # Treat as a glob pattern.
            for p in glob.glob(inp, recursive=True):
                found.append((p, os.path.dirname(p) or "."))

    seen = set()
    deduped = []
    for path, root in found:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            deduped.append((path, root))
    return sorted(deduped, key=lambda pr: pr[0])


def benchmark_label(path, root, label_parts):
    """Derive a short label identifying which benchmark run a log came
    from, using the directory path between `root` and the log file."""
    rel = os.path.relpath(os.path.dirname(path) or ".", root)
    if rel in (".", ""):
        return os.path.basename(path)
    if label_parts <= 0:
        return rel
    comps = rel.split(os.sep)
    return os.sep.join(comps[:label_parts])


class Aggregator:
    """Holds only the small, summarized counters that survive across an
    entire (possibly huge) log file. Never holds raw per-row data."""

    def __init__(self):
        self.sig_counts = Counter()                    # (benchmark, pc, lanes, offsets) -> count
        self.lane_delta_counts = defaultdict(Counter)   # (benchmark, pc) -> Counter[delta]
        self.temporal_delta_counts = defaultdict(Counter)  # (benchmark, pc) -> Counter[delta]
        self.row_count = 0

    def finalize_group(self, benchmark, pc, group):
        """`group` is the list of (thread, address) for one dynamic
        uncoalesced load (one uid), already in thread order because the
        dump writes lanes in increasing thread order."""
        if not group:
            return
        addrs = [a for _t, a in group]
        lanes = tuple(t for t, _a in group)
        base = min(addrs)
        offsets = tuple(a - base for a in addrs)
        deltas = tuple(b - a for a, b in zip(addrs, addrs[1:]))
        self.sig_counts[(benchmark, pc, lanes, offsets)] += 1
        counter = self.lane_delta_counts[(benchmark, pc)]
        for d in deltas:
            counter[d] += 1


def process_file(path, label, agg, progress_every=5_000_000):
    """Single streaming pass over one log file. Keeps only:
      - the lane buffer for the uid currently being read (<=32 rows), and
      - a `last_seen` map from (sm,warp,thread,pc) -> last address, used to
        compute temporal deltas without buffering history.
    Both are bounded by the number of distinct lanes/instructions the
    simulator can produce, not by the number of rows in the file.
    """
    current_uid = None
    current_pc = None
    current_group = []
    last_seen = {}  # (sm_id, warp_id, thread, pc) -> last address
    start = time.time()

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        expected = ["cycle", "sm_id", "warp_id", "uid", "pc",
                    "num_coalesced_transactions", "thread", "address"]
        if header != expected:
            raise SystemExit(f"{path}: unexpected header {header!r}, expected {expected!r}")

        for row in reader:
            if not row:
                continue
            _cycle_s, sm_s, warp_s, uid_s, pc_s, _ntx_s, thread_s, addr_s = row
            sm_id = int(sm_s)
            warp_id = int(warp_s)
            uid = int(uid_s)
            pc = int(pc_s, 16)
            thread = int(thread_s)
            address = int(addr_s, 16)

            if uid != current_uid:
                if current_group:
                    agg.finalize_group(label, current_pc, current_group)
                current_group = []
                current_uid = uid
                current_pc = pc
            current_group.append((thread, address))

            key = (sm_id, warp_id, thread, pc)
            prev = last_seen.get(key)
            if prev is not None:
                agg.temporal_delta_counts[(label, pc)][address - prev] += 1
            last_seen[key] = address

            agg.row_count += 1
            if progress_every and agg.row_count % progress_every == 0:
                elapsed = time.time() - start
                print(f"  ... {label}: {agg.row_count:,} rows processed "
                      f"({elapsed:.0f}s elapsed)", file=sys.stderr)

    if current_group:
        agg.finalize_group(label, current_pc, current_group)


def gcd_of(values):
    g = 0
    for v in values:
        g = math.gcd(g, abs(v))
    return g


def print_top_signatures(sig_counts, top_n):
    print("\n=== Most common lane/offset chains per (benchmark, pc) ===")
    grouped = defaultdict(list)
    for (benchmark, pc, lanes, offsets), count in sig_counts.items():
        grouped[(benchmark, pc)].append((count, lanes, offsets))

    for benchmark, pc in sorted(grouped):
        entries = sorted(grouped[(benchmark, pc)], key=lambda e: -e[0])
        total = sum(e[0] for e in entries)
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} uncoalesced load instances, "
              f"{len(entries)} distinct chain(s))")
        for count, lanes, offsets in entries[:top_n]:
            pct = 100.0 * count / total
            print(f"    {count:6d}x ({pct:5.1f}%)  lanes={lanes}  "
                  f"offsets(bytes)={offsets}")


def print_top_deltas(delta_counts, title, top_n):
    print(f"\n=== {title} ===")
    for benchmark, pc in sorted(delta_counts):
        counts = delta_counts[(benchmark, pc)]
        total = sum(counts.values())
        if total == 0:
            continue
        g = gcd_of(counts.keys())
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} deltas observed, gcd={g} bytes)")
        for delta, count in counts.most_common(top_n):
            pct = 100.0 * count / total
            print(f"    delta={delta:+d} bytes  {count:6d}x ({pct:5.1f}%)")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="log files, directories to search recursively, or glob patterns")
    ap.add_argument("--pattern", default="uncoalesced_loads.log",
                    help="filename to look for when a path is a directory "
                         "(default: uncoalesced_loads.log)")
    ap.add_argument("--top", type=int, default=5,
                    help="how many top entries to print per (benchmark, pc) group")
    ap.add_argument("--label-parts", type=int, default=1,
                    help="number of leading path components (relative to the "
                         "search root) to use as the benchmark label "
                         "(default: 1, e.g. 'pathfinder-rodinia-3.1')")
    ap.add_argument("--out-dir", default=None,
                    help="if set, also write full (non-truncated) summary "
                         "tables as CSV files into this directory")
    ap.add_argument("--max-size-gb", type=float, default=None,
                    help="skip log files larger than this size in GB instead "
                         "of processing them (default: no limit -- files are "
                         "streamed so size does not affect memory usage)")
    args = ap.parse_args()

    logs = find_logs(args.paths, args.pattern)
    if not logs:
        raise SystemExit(f"No log files found matching '{args.pattern}' under {args.paths}")

    sized = [(path, root, os.path.getsize(path)) for path, root in logs]
    skipped = []
    if args.max_size_gb is not None:
        limit = args.max_size_gb * (1024 ** 3)
        kept = [(p, r, s) for p, r, s in sized if s <= limit]
        skipped = [(p, r, s) for p, r, s in sized if s > limit]
        sized = kept

    print(f"Found {len(sized) + len(skipped)} log file(s):")
    for path, _root, size in sized:
        print(f"  {size / (1024**3):6.2f} GB  {path}")
    if skipped:
        print(f"\nSkipping {len(skipped)} file(s) larger than {args.max_size_gb} GB "
              f"(--max-size-gb):")
        for path, _root, size in skipped:
            print(f"  {size / (1024**3):6.2f} GB  {path}")

    if not sized:
        raise SystemExit("No log files left to process.")

    agg = Aggregator()
    for path, root, size in sized:
        label = benchmark_label(path, root, args.label_parts)
        print(f"\nProcessing [{label}] ({size / (1024**3):.2f} GB): {path}")
        t0 = time.time()
        process_file(path, label, agg)
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nTotal rows processed: {agg.row_count:,}")
    if agg.row_count == 0:
        raise SystemExit("No rows parsed from the given log file(s).")

    print_top_signatures(agg.sig_counts, args.top)
    print_top_deltas(agg.lane_delta_counts, top_n=args.top,
                     title="Most common address delta BETWEEN LANES within one uncoalesced load")
    print_top_deltas(agg.temporal_delta_counts, top_n=args.top,
                     title="Most common address delta for the SAME thread/PC ACROSS iterations")

    if args.out_dir:
        write_csv(
            os.path.join(args.out_dir, "lane_signatures.csv"),
            ["benchmark", "pc", "lanes", "offsets", "count"],
            [(b, hex(pc), lanes, offsets, count)
             for (b, pc, lanes, offsets), count in agg.sig_counts.items()])
        write_csv(
            os.path.join(args.out_dir, "lane_deltas.csv"),
            ["benchmark", "pc", "delta", "count"],
            [(b, hex(pc), delta, count)
             for (b, pc), counts in agg.lane_delta_counts.items()
             for delta, count in counts.items()])
        write_csv(
            os.path.join(args.out_dir, "temporal_deltas.csv"),
            ["benchmark", "pc", "delta", "count"],
            [(b, hex(pc), delta, count)
             for (b, pc), counts in agg.temporal_delta_counts.items()
             for delta, count in counts.items()])
        print(f"\nWrote summary CSVs to {args.out_dir}/")


if __name__ == "__main__":
    main()
