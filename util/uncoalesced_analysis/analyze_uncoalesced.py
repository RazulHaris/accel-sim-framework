#!/usr/bin/env python3
"""
Find repeating cache-line stride structure across the uncoalesced-load dumps
produced by gpgpu-sim's ldst_unit::dump_uncoalesced_load_addresses
(see gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc).

Each dump is a CSV with one row per lane that actually issued a transaction
for an uncoalesced global load:

    cycle,sm_id,warp_id,uid,pc,num_coalesced_transactions,thread,address

One `uid` is one *instance*: one dynamic execution of one static load PC by
one warp. Every analysis below works at cache-line granularity (--line-size,
default 128B), because that is what actually determines how many transactions
an uncoalesced load costs.

The two questions this answers:

1. IS THE STRIDE WARP-INVARIANT?  For each instance, take the distinct cache
   lines it touches, sort them, and take the deltas between consecutive lines
   -- the instance's "line-stride signature". A signature of (128,128,128)
   with a 128B line means the warp walks neighbouring lines; (256,256,256)
   means it skips every other line; an irregular tuple means data-dependent
   scatter. The signature is then counted per PC together with *how many
   distinct warps and SMs produced it*. If one signature covers nearly all
   instances and nearly all warps of a PC, the uncoalescing is structural
   (layout/stride in the source) rather than data-dependent, and it will
   respond to a layout or access-pattern fix.
   Complementing this, adjacent-warp base deltas (base address of warp w+1
   minus that of warp w at the same iteration of the same PC) show how the
   warps of a CTA are spaced relative to each other.

2. IS THERE TEMPORAL STRUCTURE ACROSS ITERATIONS?  For a fixed
   (sm, warp, pc), consecutive instances form the iterations of a loop. For
   each consecutive pair this records (a) the delta between their base
   addresses -- the per-iteration stride of the whole warp -- (b) how the two
   line *sets* relate (identical / rigidly shifted by k lines / partially
   overlapping / disjoint), and (c) n-grams of consecutive base deltas, which
   expose repeating cycles such as (+128, +128, -3968) from a row-major walk
   that wraps. A bounded LRU also measures true reuse: how often a line
   touched by a PC is touched again by the same PC, and how many instances
   later.

The older per-lane offset signature and per-thread temporal delta reports are
still produced, since they are cheap and useful for eyeballing a single load.

Memory: logs are processed one row at a time and only small aggregate
counters are kept (never the raw rows), so file size does not bound RAM --
multi-GB dumps are fine. This relies on two properties of the dump: all lanes
of one uid are written together in one contiguous block in thread order, and
rows for a given (sm, warp, thread, pc) appear in non-decreasing cycle order.
Per-warp state is bounded by the number of hardware warp slots
(SMs x warps/SM), signature/n-gram tables are capped by --max-sigs, and the
reuse tracker by --reuse-capacity.

Usage:
    ./analyze_uncoalesced.py sim_run_12.8/
    ./analyze_uncoalesced.py bench1/uncoalesced_loads.log bench2/uncoalesced_loads.log
    ./analyze_uncoalesced.py sim_run_dir/ --line-size 32 --top 10 --out-dir results/
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict

# Placeholder key used when a per-PC table hits its --max-sigs cap. Keeping a
# single overflow bucket means the counts still add up to the true instance
# total even when a PC has a pathological number of distinct patterns.
OTHER = "<other>"


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


def capped_key(table, key, cap):
    """Return `key` if `table` already tracks it or still has room, else the
    OTHER overflow bucket. Keeps every per-PC table bounded by `cap`."""
    if key in table or len(table) < cap:
        return key
    return OTHER


class Aggregator:
    """Holds only the small, summarized counters that survive across an
    entire (possibly huge) log file. Never holds raw per-row data."""

    def __init__(self, line_size, max_sigs, gram):
        self.line_size = line_size
        self.max_sigs = max_sigs
        self.gram = gram
        self.row_count = 0

        # --- per-PC totals -------------------------------------------------
        self.instances = Counter()                      # (bench, pc) -> instances
        self.lines_total = Counter()                    # (bench, pc) -> sum of distinct lines
        self.tx_total = Counter()                       # (bench, pc) -> sum of transactions
        self.nlines_hist = defaultdict(Counter)         # (bench, pc) -> Counter[distinct lines]
        self.pc_warps = defaultdict(set)                # (bench, pc) -> {(sm, warp)}
        self.pc_sms = defaultdict(set)                  # (bench, pc) -> {sm}

        # --- 1. stride within an instance, and its warp invariance ---------
        self.line_sig_counts = defaultdict(Counter)     # (bench, pc) -> Counter[sig]
        self.line_sig_warps = defaultdict(lambda: defaultdict(set))  # (bench,pc) -> sig -> {(sm,warp)}
        self.uniform_stride = defaultdict(Counter)      # (bench, pc) -> Counter[stride|label]
        self.adjacent_warp_delta = defaultdict(Counter)  # (bench, pc) -> Counter[base delta]

        # --- 2. structure across iterations of the same (sm, warp, pc) -----
        self.iter_base_delta = defaultdict(Counter)     # (bench, pc) -> Counter[base delta]
        self.set_relation = defaultdict(Counter)        # (bench, pc) -> Counter[relation]
        self.iter_overlap = Counter()                   # (bench, pc) -> pairs sharing >=1 line
        self.iter_pairs = Counter()                     # (bench, pc) -> consecutive pairs
        self.gram_counts = defaultdict(Counter)         # (bench, pc) -> Counter[delta n-gram]
        self.reuse_hits = Counter()                     # (bench, pc) -> lines seen again
        self.reuse_misses = Counter()                   # (bench, pc) -> lines seen first time
        self.reuse_dist = defaultdict(Counter)          # (bench, pc) -> Counter[log2 bucket]

        # --- legacy per-lane views ----------------------------------------
        self.sig_counts = Counter()                     # (bench, pc, lanes, offsets) -> count
        self.lane_delta_counts = defaultdict(Counter)   # (bench, pc) -> Counter[delta]
        self.temporal_delta_counts = defaultdict(Counter)  # (bench, pc) -> Counter[delta]

    # ------------------------------------------------------------------ #

    def finalize_group(self, label, pc, sm_id, warp_id, ntx, group, st):
        """Fold one instance (one uid) into the aggregates.

        `group` is the list of (thread, address) for that dynamic load,
        already in thread order because the dump writes lanes in increasing
        thread order. `st` is the per-file transient state."""
        if not group:
            return
        key = (label, pc)
        addrs = [a for _t, a in group]
        line_size = self.line_size

        # ---- legacy per-lane signature (offsets from the group minimum) ---
        lanes = tuple(t for t, _a in group)
        base_addr = min(addrs)
        offsets = tuple(a - base_addr for a in addrs)
        self.sig_counts[(label, pc, lanes, offsets)] += 1
        lane_counter = self.lane_delta_counts[key]
        for a, b in zip(addrs, addrs[1:]):
            lane_counter[b - a] += 1

        # ---- cache-line view of this instance -----------------------------
        lines = sorted({a // line_size for a in addrs})
        base_line = lines[0]
        deltas = [(b - a) * line_size for a, b in zip(lines, lines[1:])]

        self.instances[key] += 1
        self.lines_total[key] += len(lines)
        self.tx_total[key] += ntx
        self.nlines_hist[key][len(lines)] += 1
        self.pc_warps[key].add((sm_id, warp_id))
        self.pc_sms[key].add(sm_id)

        # 1a. Stride signature of this instance, tracked per distinct warp so
        #     we can tell "one warp does this" from "every warp does this".
        sig = tuple(deltas)
        sig_counter = self.line_sig_counts[key]
        sig_key = capped_key(sig_counter, sig, self.max_sigs)
        sig_counter[sig_key] += 1
        self.line_sig_warps[key][sig_key].add((sm_id, warp_id))

        if not deltas:
            self.uniform_stride[key]["single-line"] += 1
        elif len(set(deltas)) == 1:
            self.uniform_stride[key][deltas[0]] += 1
        else:
            self.uniform_stride[key]["irregular"] += 1

        # 1b. Spacing between neighbouring warps at the same iteration of the
        #     same PC. `st.warp_bases[(pc, sm)]` keeps only the latest
        #     (iteration, base line) per warp, so this is bounded by the
        #     number of hardware warp slots.
        iter_idx = st.iter_idx.get((sm_id, warp_id, pc), 0)
        peers = st.warp_bases[(pc, sm_id)]
        prev_peer = peers.get(warp_id - 1)
        if prev_peer is not None and prev_peer[0] == iter_idx:
            self.adjacent_warp_delta[key][(base_line - prev_peer[1]) * line_size] += 1
        next_peer = peers.get(warp_id + 1)
        if next_peer is not None and next_peer[0] == iter_idx:
            self.adjacent_warp_delta[key][(next_peer[1] - base_line) * line_size] += 1
        peers[warp_id] = (iter_idx, base_line)

        # 2. Relationship to the previous instance of this same (sm, warp, pc)
        #    -- i.e. to the previous loop iteration of this static load.
        wkey = (sm_id, warp_id, pc)
        prev = st.last_instance.get(wkey)
        line_set = frozenset(lines)
        if prev is not None:
            prev_base, prev_lines, prev_sorted = prev
            delta = (base_line - prev_base) * line_size
            self.iter_base_delta[key][delta] += 1
            relation, shared = self._relate(prev_sorted, lines, prev_lines, line_set)
            self.set_relation[key][relation] += 1
            self.iter_pairs[key] += 1
            if shared:
                self.iter_overlap[key] += 1

            # Rolling n-gram of base deltas exposes repeating cycles that a
            # single-delta histogram flattens out.
            hist = st.delta_hist.get(wkey)
            hist = (hist + (delta,))[-self.gram:] if hist else (delta,)
            st.delta_hist[wkey] = hist
            if len(hist) == self.gram:
                grams = self.gram_counts[key]
                grams[capped_key(grams, hist, self.max_sigs)] += 1
        st.last_instance[wkey] = (base_line, line_set, lines)
        st.iter_idx[wkey] = iter_idx + 1

        # 2c. True reuse of a cache line by the same PC, measured in
        #     instances-of-this-PC and capped by an LRU so it stays bounded.
        seen = st.reuse_lru[key]
        now = self.instances[key]
        for ln in lines:
            when = seen.pop(ln, None)
            if when is None:
                self.reuse_misses[key] += 1
            else:
                self.reuse_hits[key] += 1
                dist = now - when
                self.reuse_dist[key][dist.bit_length() - 1] += 1
            seen[ln] = now
        while len(seen) > st.reuse_capacity:
            seen.popitem(last=False)

    @staticmethod
    def _relate(prev_sorted, cur_sorted, prev_set, cur_set):
        """Classify how this iteration's set of cache lines relates to the
        previous iteration's: an exact repeat, the same shape moved by a
        constant number of lines, an overlap (partial reuse), or no reuse.
        Returns (relation, lines shared with the previous iteration) -- a
        rigid shift can still share no lines, so the two are reported apart."""
        shared = len(cur_set & prev_set)
        if cur_set == prev_set:
            return "identical", shared
        if len(cur_sorted) == len(prev_sorted):
            shift = cur_sorted[0] - prev_sorted[0]
            if all(c - p == shift for p, c in zip(prev_sorted, cur_sorted)):
                return ("shifted", shift), shared
        if shared == 0:
            return "disjoint", shared
        return ("overlap", shared), shared


class FileState:
    """Transient, per-file state. Every map here is keyed by hardware
    identifiers (SM, warp slot, PC), so it is bounded by the machine
    configuration rather than by the size of the log."""

    def __init__(self, reuse_capacity):
        self.last_instance = {}   # (sm, warp, pc) -> (base line, line set, sorted lines)
        self.iter_idx = {}        # (sm, warp, pc) -> instances seen so far
        self.delta_hist = {}      # (sm, warp, pc) -> last n base deltas
        self.warp_bases = defaultdict(dict)   # (pc, sm) -> warp -> (iteration, base line)
        self.last_seen = {}       # (sm, warp, thread, pc) -> last address
        self.reuse_lru = defaultdict(OrderedDict)  # (bench, pc) -> line -> instance index
        self.reuse_capacity = reuse_capacity


def process_file(path, label, agg, reuse_capacity, progress_every=5_000_000):
    """Single streaming pass over one log file, folding each uid's block of
    lanes into `agg` as soon as the block ends."""
    st = FileState(reuse_capacity)
    current_uid = None
    current = None  # (pc, sm_id, warp_id, ntx)
    current_group = []
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
            _cycle_s, sm_s, warp_s, uid_s, pc_s, ntx_s, thread_s, addr_s = row
            sm_id = int(sm_s)
            warp_id = int(warp_s)
            uid = int(uid_s)
            pc = int(pc_s, 16)
            thread = int(thread_s)
            address = int(addr_s, 16)

            if uid != current_uid:
                if current_group:
                    agg.finalize_group(label, *current, current_group, st)
                current_group = []
                current_uid = uid
                current = (pc, sm_id, warp_id, int(ntx_s))
            current_group.append((thread, address))

            key = (sm_id, warp_id, thread, pc)
            prev = st.last_seen.get(key)
            if prev is not None:
                agg.temporal_delta_counts[(label, pc)][address - prev] += 1
            st.last_seen[key] = address

            agg.row_count += 1
            if progress_every and agg.row_count % progress_every == 0:
                elapsed = time.time() - start
                print(f"  ... {label}: {agg.row_count:,} rows processed "
                      f"({elapsed:.0f}s elapsed)", file=sys.stderr)

    if current_group:
        agg.finalize_group(label, *current, current_group, st)


def gcd_of(values):
    g = 0
    for v in values:
        if isinstance(v, int):
            g = math.gcd(g, abs(v))
    return g


def fmt_sig(sig, line_size):
    """Render a stride signature compactly, e.g. (128,128,128) -> 3x+128B."""
    if sig == OTHER:
        return OTHER
    if not sig:
        return "single line"
    parts = []
    for d in sig:
        if parts and parts[-1][0] == d:
            parts[-1][1] += 1
        else:
            parts.append([d, 1])
    body = ", ".join(f"{n}x{d:+d}B" if n > 1 else f"{d:+d}B" for d, n in parts)
    if len(set(sig)) == 1 and sig[0] == line_size:
        body += "  [adjacent lines]"
    return body


def fmt_relation(rel):
    if isinstance(rel, tuple):
        kind, val = rel
        if kind == "shifted":
            return f"shifted by {val:+d} lines"
        return f"partial overlap ({val} lines reused)"
    return rel


def pct(count, total):
    return 100.0 * count / total if total else 0.0


# Below this share, the "most common" entry is just noise from a scattered/
# data-dependent distribution (many near-unique values) -- printing it as if
# it were dominant is misleading, so headline lines fall back to a distinct-
# value count instead.
DOMINANCE_FLOOR = 0.05


def print_headline(agg, top_n):
    """Per-PC summary: size of the problem, then the two headline answers --
    is the stride the same for every warp, and does it repeat over time."""
    line_size = agg.line_size
    print(f"\n=== Uncoalesced load summary (cache line = {line_size} B) ===")
    for key in sorted(agg.instances):
        benchmark, pc = key
        n = agg.instances[key]
        warps = len(agg.pc_warps[key])
        sms = len(agg.pc_sms[key])
        print(f"\n[{benchmark}] pc=0x{pc:x}")
        print(f"    {n:,} uncoalesced instances from {warps} warp slot(s) on {sms} SM(s); "
              f"mean {agg.lines_total[key] / n:.2f} distinct lines and "
              f"{agg.tx_total[key] / n:.2f} transactions per instance")

        strides = agg.uniform_stride[key]
        top = strides.most_common(1)[0] if strides else None
        if top and top[1] >= DOMINANCE_FLOOR * n:
            label = (f"uniform {top[0]:+d} B" if isinstance(top[0], int) else top[0])
            note = ""
            if isinstance(top[0], int):
                if abs(top[0]) == line_size:
                    note = "  (neighbouring lines)"
                elif top[0] % line_size == 0:
                    note = f"  (every {abs(top[0]) // line_size} lines)"
            print(f"    dominant intra-instance line stride: {label} "
                  f"in {pct(top[1], n):.1f}% of instances{note}")
        elif strides:
            print(f"    no dominant intra-instance line stride "
                  f"({len(strides)} distinct value(s) across {n} instances -- "
                  f"scattered/data-dependent)")

        sigs = agg.line_sig_counts[key]
        if sigs:
            sig, count = sigs.most_common(1)[0]
            sig_warps = len(agg.line_sig_warps[key][sig])
            verdict = ("warp-invariant / structural"
                       if sig_warps >= 0.9 * warps and count >= 0.5 * n
                       else "warp-specific or data-dependent")
            print(f"    dominant stride signature seen in {pct(count, n):.1f}% of "
                  f"instances and {sig_warps}/{warps} warps -> {verdict}")

        pairs = agg.iter_pairs[key]
        if pairs:
            rels = agg.set_relation[key]
            rel, rel_count = rels.most_common(1)[0]
            overlap_line = (f"    iteration-to-iteration: "
                             f"{pct(agg.iter_overlap[key], pairs):.1f}% of "
                             f"consecutive instances share a line; ")
            if rel_count >= DOMINANCE_FLOOR * pairs:
                overlap_line += (f"dominant relation is {fmt_relation(rel)} "
                                  f"({pct(rel_count, pairs):.1f}%)")
            else:
                overlap_line += (f"no dominant relation ({len(rels)} distinct "
                                  f"kind(s) across {pairs} transitions)")
            print(overlap_line)

        hits = agg.reuse_hits[key]
        seen = hits + agg.reuse_misses[key]
        if seen:
            print(f"    line reuse by this PC: {pct(hits, seen):.1f}% of line "
                  f"touches hit a line this PC already touched")


def print_stride_signatures(agg, top_n):
    print("\n=== 1. Intra-instance line strides and how far they hold across warps ===")
    line_size = agg.line_size
    for key in sorted(agg.line_sig_counts):
        benchmark, pc = key
        counts = agg.line_sig_counts[key]
        total = agg.instances[key]
        warps = len(agg.pc_warps[key])
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} instances, {len(counts)} distinct "
              f"signature(s), {warps} warp slot(s))")
        for sig, count in counts.most_common(top_n):
            sig_warps = len(agg.line_sig_warps[key][sig])
            sig_sms = len({sm for sm, _w in agg.line_sig_warps[key][sig]})
            print(f"    {count:8d}x ({pct(count, total):5.1f}%)  "
                  f"warps={sig_warps:4d}/{warps:<4d} sms={sig_sms:3d}  "
                  f"stride={fmt_sig(sig, line_size)}")


def print_adjacent_warp(agg, top_n):
    print("\n=== 1b. Base-address delta between adjacent warps at the same iteration ===")
    for key in sorted(agg.adjacent_warp_delta):
        benchmark, pc = key
        counts = agg.adjacent_warp_delta[key]
        total = sum(counts.values())
        if not total:
            continue
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} adjacent-warp pairs, "
              f"gcd={gcd_of(counts.keys())} bytes)")
        for delta, count in counts.most_common(top_n):
            print(f"    warp(w+1) - warp(w) = {delta:+d} bytes  "
                  f"{count:8d}x ({pct(count, total):5.1f}%)")


def print_temporal(agg, top_n):
    print("\n=== 2. Structure across iterations of the same (sm, warp, pc) ===")
    for key in sorted(agg.iter_base_delta):
        benchmark, pc = key
        deltas = agg.iter_base_delta[key]
        total = sum(deltas.values())
        if not total:
            continue
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} iteration transitions, "
              f"gcd={gcd_of(deltas.keys())} bytes)")
        print("  base-address stride per iteration:")
        for delta, count in deltas.most_common(top_n):
            print(f"    {delta:+d} bytes  {count:8d}x ({pct(count, total):5.1f}%)")

        rels = agg.set_relation[key]
        rel_total = sum(rels.values())
        if rel_total:
            print(f"  how consecutive iterations' line sets relate "
                  f"({pct(agg.iter_overlap[key], rel_total):.1f}% share >=1 line):")
            for rel, count in rels.most_common(top_n):
                print(f"    {fmt_relation(rel):<34s} {count:8d}x "
                      f"({pct(count, rel_total):5.1f}%)")

        grams = agg.gram_counts[key]
        gram_total = sum(grams.values())
        if gram_total:
            print(f"  repeating {agg.gram}-step delta cycles:")
            for gram, count in grams.most_common(top_n):
                text = (OTHER if gram == OTHER
                        else " -> ".join(f"{d:+d}" for d in gram))
                print(f"    {text:<34s} {count:8d}x ({pct(count, gram_total):5.1f}%)")

        hits = agg.reuse_hits[key]
        seen = hits + agg.reuse_misses[key]
        if seen:
            print(f"  line reuse: {hits:,}/{seen:,} touches ({pct(hits, seen):.1f}%) "
                  f"revisit a line this PC already touched")
            dist = agg.reuse_dist[key]
            dist_total = sum(dist.values())
            for bucket, count in sorted(dist.items())[:top_n]:
                lo = 1 << bucket
                print(f"    reuse distance {lo}-{2 * lo - 1} instances  "
                      f"{count:8d}x ({pct(count, dist_total):5.1f}%)")


def print_top_signatures(sig_counts, top_n):
    print("\n=== Appendix: most common lane/offset chains per (benchmark, pc) ===")
    grouped = defaultdict(list)
    for (benchmark, pc, lanes, offsets), count in sig_counts.items():
        grouped[(benchmark, pc)].append((count, lanes, offsets))

    for benchmark, pc in sorted(grouped):
        entries = sorted(grouped[(benchmark, pc)], key=lambda e: -e[0])
        total = sum(e[0] for e in entries)
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} uncoalesced load instances, "
              f"{len(entries)} distinct chain(s))")
        for count, lanes, offsets in entries[:top_n]:
            print(f"    {count:6d}x ({pct(count, total):5.1f}%)  lanes={lanes}  "
                  f"offsets(bytes)={offsets}")


def print_top_deltas(delta_counts, title, top_n):
    print(f"\n=== Appendix: {title} ===")
    for benchmark, pc in sorted(delta_counts):
        counts = delta_counts[(benchmark, pc)]
        total = sum(counts.values())
        if total == 0:
            continue
        print(f"\n[{benchmark}] pc=0x{pc:x}  ({total} deltas observed, "
              f"gcd={gcd_of(counts.keys())} bytes)")
        for delta, count in counts.most_common(top_n):
            print(f"    delta={delta:+d} bytes  {count:6d}x ({pct(count, total):5.1f}%)")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_all_csvs(agg, out_dir):
    write_csv(
        os.path.join(out_dir, "pc_summary.csv"),
        ["benchmark", "pc", "instances", "warp_slots", "sms",
         "mean_lines_per_instance", "mean_transactions_per_instance"],
        [(b, hex(pc), n, len(agg.pc_warps[(b, pc)]), len(agg.pc_sms[(b, pc)]),
          agg.lines_total[(b, pc)] / n, agg.tx_total[(b, pc)] / n)
         for (b, pc), n in agg.instances.items()])
    write_csv(
        os.path.join(out_dir, "line_stride_signatures.csv"),
        ["benchmark", "pc", "stride_signature_bytes", "instances",
         "distinct_warps", "distinct_sms", "pct_of_instances"],
        [(b, hex(pc), sig, count, len(agg.line_sig_warps[(b, pc)][sig]),
          len({sm for sm, _w in agg.line_sig_warps[(b, pc)][sig]}),
          pct(count, agg.instances[(b, pc)]))
         for (b, pc), counts in agg.line_sig_counts.items()
         for sig, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "uniform_strides.csv"),
        ["benchmark", "pc", "stride", "instances"],
        [(b, hex(pc), stride, count)
         for (b, pc), counts in agg.uniform_stride.items()
         for stride, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "adjacent_warp_base_deltas.csv"),
        ["benchmark", "pc", "delta_bytes", "count"],
        [(b, hex(pc), delta, count)
         for (b, pc), counts in agg.adjacent_warp_delta.items()
         for delta, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "iteration_base_deltas.csv"),
        ["benchmark", "pc", "delta_bytes", "count"],
        [(b, hex(pc), delta, count)
         for (b, pc), counts in agg.iter_base_delta.items()
         for delta, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "iteration_set_relation.csv"),
        ["benchmark", "pc", "relation", "count", "pairs_sharing_a_line",
         "total_pairs"],
        [(b, hex(pc), fmt_relation(rel), count, agg.iter_overlap[(b, pc)],
          agg.iter_pairs[(b, pc)])
         for (b, pc), counts in agg.set_relation.items()
         for rel, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "delta_grams.csv"),
        ["benchmark", "pc", "delta_cycle_bytes", "count"],
        [(b, hex(pc), gram, count)
         for (b, pc), counts in agg.gram_counts.items()
         for gram, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "line_reuse.csv"),
        ["benchmark", "pc", "line_touches", "reuse_hits", "reuse_pct"],
        [(b, hex(pc), hits + agg.reuse_misses[(b, pc)], hits,
          pct(hits, hits + agg.reuse_misses[(b, pc)]))
         for (b, pc), hits in agg.reuse_hits.items()])
    write_csv(
        os.path.join(out_dir, "lane_signatures.csv"),
        ["benchmark", "pc", "lanes", "offsets", "count"],
        [(b, hex(pc), lanes, offsets, count)
         for (b, pc, lanes, offsets), count in agg.sig_counts.items()])
    write_csv(
        os.path.join(out_dir, "lane_deltas.csv"),
        ["benchmark", "pc", "delta", "count"],
        [(b, hex(pc), delta, count)
         for (b, pc), counts in agg.lane_delta_counts.items()
         for delta, count in counts.items()])
    write_csv(
        os.path.join(out_dir, "temporal_deltas.csv"),
        ["benchmark", "pc", "delta", "count"],
        [(b, hex(pc), delta, count)
         for (b, pc), counts in agg.temporal_delta_counts.items()
         for delta, count in counts.items()])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="log files, directories to search recursively, or glob patterns")
    ap.add_argument("--pattern", default="uncoalesced_loads.log",
                    help="filename to look for when a path is a directory "
                         "(default: uncoalesced_loads.log)")
    ap.add_argument("--line-size", type=int, default=128,
                    help="cache line size in bytes used to group addresses "
                         "(default: 128; use 32 to analyse at sector granularity)")
    ap.add_argument("--top", type=int, default=5,
                    help="how many top entries to print per (benchmark, pc) group")
    ap.add_argument("--gram", type=int, default=3,
                    help="length of the repeating base-delta cycles to look for "
                         "across iterations (default: 3)")
    ap.add_argument("--max-sigs", type=int, default=4096,
                    help="cap on distinct stride signatures / delta cycles tracked "
                         "per PC; the rest are folded into one '<other>' bucket "
                         "(default: 4096)")
    ap.add_argument("--reuse-capacity", type=int, default=1 << 16,
                    help="how many recently touched cache lines to remember per PC "
                         "when measuring reuse (default: 65536)")
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

    if args.line_size <= 0 or args.line_size & (args.line_size - 1):
        raise SystemExit("--line-size must be a positive power of two")
    if args.gram < 1:
        raise SystemExit("--gram must be at least 1")

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

    agg = Aggregator(args.line_size, args.max_sigs, args.gram)
    for path, root, size in sized:
        label = benchmark_label(path, root, args.label_parts)
        print(f"\nProcessing [{label}] ({size / (1024**3):.2f} GB): {path}")
        t0 = time.time()
        process_file(path, label, agg, args.reuse_capacity)
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nTotal rows processed: {agg.row_count:,}")
    if agg.row_count == 0:
        raise SystemExit("No rows parsed from the given log file(s).")

    print_headline(agg, args.top)
    print_stride_signatures(agg, args.top)
    print_adjacent_warp(agg, args.top)
    print_temporal(agg, args.top)
    print_top_signatures(agg.sig_counts, args.top)
    print_top_deltas(agg.lane_delta_counts, top_n=args.top,
                     title="address delta BETWEEN LANES within one uncoalesced load")
    print_top_deltas(agg.temporal_delta_counts, top_n=args.top,
                     title="address delta for the SAME thread/PC ACROSS iterations")

    if args.out_dir:
        write_all_csvs(agg, args.out_dir)
        print(f"\nWrote summary CSVs to {args.out_dir}/")


if __name__ == "__main__":
    main()
