#!/usr/bin/env python3
"""Debug plan Stage 3: prefetch lifecycle analyses 3a-3c
(docs/snake/results/debug/S3_lifecycle.md).

    s3_lifecycle.py [<runs dir> [<config>]]

Reads snake_lc.txt.gz (-snake_lc_stride, gpgpu-sim 04bbd427) of every job of
<config> under <runs dir> (default: the Stage 3 runs, Tier 1 Snake with the
log on SMs 0, 16, 32, 48 and 64) and its sim.out.  Event lines (one per
line, cycles are GPU core cycles):
  K sid cyc kuid                       an SM starts a kernel launch
  P sid cyc kuid w gw wpc pc mode depth dist tw tgw tpc line mask res newmask
                                       a predicted line reached the queue:
                                       res n new entry, j joined a queued line
                                       (newmask: sectors not already queued),
                                       f dropped, queue full
  D sid cyc line mask why              queued line dropped: x displaced by an
                                       own-warp line, a accuracy filter
  O sid cyc sector res enq             the L1D's verdict on an offered sector
                                       (snake_issue_result; enq: the queue
                                       entry's creation)
  F sid cyc sector issue late l2       prefetch fill (late: a demand merged;
                                       l2: L2 access status, 0 = hit)
  E sid cyc sector why                 unused prefetched sector lost: e
                                       evicted, s invalid at refill, k L1D
                                       invalidated at a kernel end
  U sid cyc sector wid class pcyc mode src   first demand of a predicted
                                       sector within the window (class:
                                       snake_funnel_class; pcyc: the latest
                                       prediction; mode: its source; src: the
                                       first prediction's source)
  X sid cyc sector class pcyc mode src no demand within the window
  T sid cyc on                         throttling starts (1) or ends (0)
  M sid cyc uid sector wid kind        demand read to the MSHRs: n new
                                       request, p merged into an in-flight
                                       prefetch, d merged into a demand
  R sid cyc uid                        its data left the L1D for writeback
  L sid cyc w gw pc addr ok            a load Snake observed (ok: it passed
                                       the lane filter); Stage 4 uses these
"""
import collections
import glob
import gzip
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CFG = "QV100-SASS-PAPER_V100-1B_INSN-SNAKE-SNAKE_FP1-SNAKE_LC16"
RUNS = os.path.join(ROOT, "sim_run_snake_debug", "s3", "runs")
MODES = ["inter-thread", "intra-warp", "promotion", "inter-warp", "inter-warp chain"]
CLASSES = ["queued", "drop_qfull", "drop_mq", "drop_res", "dup_l1", "dup_mshr", "late", "evicted", "timely",
           "issued_other", "wrong_issued", "wrong_not_issued"]
RES = ["issued", "dup_l1", "dup_mshr", "drop_line", "drop_mshr", "drop_mq", "drop_mq_reserve", "pq_queued",
       "drop_pq_full"]


def popcount(x):
    return bin(x).count("1")


def pct(v, q):
    """q-th percentile of a sorted list (nearest rank)."""
    if not v:
        return None
    return v[min(len(v) - 1, int(q / 100.0 * len(v)))]


class Run:
    def __init__(self):
        self.cls = collections.Counter()      # (src, class) of each prediction record
        self.cls_k = collections.Counter()    # (kernel uid, class), for Stage 4d
        self.queued_thr = collections.Counter()  # src -> queued at demand while throttled
        self.pred = collections.Counter()     # (mode, res) -> P events
        self.pred_sec = collections.Counter()  # (mode, res) -> sectors named
        self.pred_new = collections.Counter()  # mode -> sectors newly queued
        self.drop = collections.Counter()     # why -> sectors
        self.offer = collections.Counter()    # (mode, res) -> offered sectors
        self.fill = collections.Counter()     # (mode, late, l2 hit) -> fills
        self.lost = collections.Counter()     # why -> unused sectors lost
        self.lead = collections.defaultdict(list)    # mode -> demand - latest prediction
        self.qwait = collections.defaultdict(list)   # mode -> offered - queue entry created
        self.mlat = collections.defaultdict(list)    # mode -> filled - issued
        self.dlat = collections.defaultdict(list)    # (kernel, kind) -> demand data back - access
        self.events = 0


def parse(path):
    r = Run()
    kern, thr = {}, {}
    qmode, imode, pend = {}, {}, {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split()
            if not f:
                continue
            r.events += 1
            t, sid, cyc = f[0], int(f[1]), int(f[2])
            if t == "M":
                pend[int(f[3])] = (cyc, f[6], kern.get(sid, -1))
            elif t == "R":
                m = pend.pop(int(f[3]), None)
                if m:
                    r.dlat[(m[2], m[1])].append(cyc - m[0])
            elif t == "P":
                mode, line_a, mask, res, new = int(f[8]), int(f[14]), int(f[15]), f[16], int(f[17])
                r.pred[(mode, res)] += 1
                r.pred_sec[(mode, res)] += popcount(mask)
                r.pred_new[mode] += popcount(new)
                for b in range(8):
                    if (new | (mask if res == "j" and mode <= 2 else 0)) >> b & 1:
                        qmode[line_a + 32 * b] = mode
            elif t == "O":
                sec, res, enq = int(f[3]), int(f[4]), int(f[5])
                mode = qmode.pop(sec, -1)
                r.offer[(mode, res)] += 1
                r.qwait[mode].append(cyc - enq)
                if res == 0:
                    imode[sec] = mode
            elif t == "F":
                sec, issue, late, l2 = int(f[3]), int(f[4]), int(f[5]), int(f[6])
                mode = imode.pop(sec, -1)
                r.fill[(mode, late, l2 == 0)] += 1
                r.mlat[mode].append(cyc - issue)
            elif t == "U":
                c, pcyc, mode, src = int(f[5]), int(f[6]), int(f[7]), int(f[8])
                r.cls[(src, c)] += 1
                r.cls_k[(kern.get(sid, -1), c)] += 1
                if c == 0 and thr.get(sid):
                    r.queued_thr[src] += 1
                r.lead[mode].append(cyc - pcyc)
            elif t == "X":
                r.cls[(int(f[7]), int(f[4]))] += 1
                r.cls_k[(kern.get(sid, -1), int(f[4]))] += 1
            elif t == "D":
                line_a, mask = int(f[3]), int(f[4])
                r.drop[f[5]] += popcount(mask)
                for b in range(8):
                    if mask >> b & 1:
                        qmode.pop(line_a + 32 * b, None)
            elif t == "E":
                r.lost[f[4]] += 1
            elif t == "T":
                thr[sid] = f[3] == "1"
            elif t == "K":
                kern[sid] = int(f[3])
    for v in list(r.lead.values()) + list(r.qwait.values()) + list(r.mlat.values()) + list(r.dlat.values()):
        v.sort()
    return r


def whole(sim):
    """Whole-run counters of the job (all 80 SMs) for context."""
    t = open(sim, errors="replace").read() if os.path.exists(sim) else ""
    g = lambda k: (lambda m: int(m[-1]) if m else None)(re.findall(r"^snake_%s = (\d+)" % k, t, re.M))
    dem = g("demand_sector_accesses")
    if not dem:
        return None
    return dict(dem=dem, pred=g("pred_sectors") / dem, cov=100.0 * g("demand_pred_matched") / dem,
                issued=g("pf_issued") / dem, deliv=100.0 * (g("pf_useful_timely") + g("pf_late")) / dem,
                timely=100.0 * g("pf_useful_timely") / dem)


def kernel_names(sim):
    """kernel uid -> name, from the launch lines of sim.out."""
    out = {}
    t = open(sim, errors="replace").read() if os.path.exists(sim) else ""
    for m in re.finditer(r"launching kernel name: (\S+) uid: (\d+)", t):
        out[int(m.group(2))] = m.group(1)
    return out


def fmt(x, d=1):
    return "n/a" if x is None else "%.*f" % (d, x)


def funnel_table(r):
    """3a per source: the share of prediction records in each end state."""
    srcs = sorted({s for s, _ in r.cls})
    cols = [("records", None), ("queue full", [1]), ("queued at demand, throttled", "qt"),
            ("queued at demand, not throttled", "qn"), ("in L1", [4]), ("in flight", [5]),
            ("no line / MSHR / miss-queue slot", [2, 3]), ("timely", [8]), ("late", [6]),
            ("evicted before demand", [7]), ("issued, other", [9]), ("no demand: issued", [10]),
            ("no demand: not issued", [11])]
    out = ["| source (first prediction) | " + " | ".join(c for c, _ in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for s in srcs + ["all"]:
        ss = srcs if s == "all" else [s]
        n = sum(r.cls[(x, c)] for x in ss for c in range(12))
        if not n:
            continue
        thr = sum(r.queued_thr[x] for x in ss)
        q = sum(r.cls[(x, 0)] for x in ss)
        cells = ["%d" % n]
        for _, k in cols[1:]:
            v = thr if k == "qt" else q - thr if k == "qn" else sum(r.cls[(x, c)] for x in ss for c in k)
            cells.append("%.1f" % (100.0 * v / n))
        out.append("| %s | %s |" % ("**all**" if s == "all" else MODES[s], " | ".join(cells)))
    return out


def issue_table(r):
    """3a sector level: predicted -> queued -> offered (by verdict) -> filled."""
    modes = sorted({m for m, _ in r.pred} | {m for m, _ in r.offer if m >= 0})
    out = ["| source (queue entry) | sectors predicted | already queued | dropped at queue (full) | newly queued | "
           "offered | issued | in L1 | in flight | no line | MSHR reserve | miss queue (incl. reserve) | filled | "
           "late merge at fill | L2 hit of fills |", "|---|" + "---|" * 14]
    for m in modes + ["all"]:
        ms = modes if m == "all" else [m]
        p = sum(v for (x, _), v in r.pred_sec.items() if x in ms)
        new = sum(r.pred_new[x] for x in ms)
        full = sum(v for (x, res), v in r.pred_sec.items() if x in ms and res == "f")
        off = lambda rs: sum(v for (x, res), v in r.offer.items() if (x in ms or (m == "all" and x < 0)) and res in rs)
        o = off(range(9))
        fl = sum(v for (x, _, _), v in r.fill.items() if x in ms or (m == "all" and x < 0))
        late = sum(v for (x, lt, _), v in r.fill.items() if (x in ms or (m == "all" and x < 0)) and lt)
        hit = sum(v for (x, _, h), v in r.fill.items() if (x in ms or (m == "all" and x < 0)) and h)
        out.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            "**all**" if m == "all" else MODES[m], p, p - new - full, full, new, o, off([0]), off([1]), off([2]),
            off([3]),
            off([4]), off([5, 6]), fl, fmt(100.0 * late / fl if fl else None) + "%",
            fmt(100.0 * hit / fl if fl else None) + "%"))
    return out


def dist_cells(v):
    return [fmt(pct(v, q), 0) for q in (10, 25, 50, 75, 90)] + ["%d" % len(v)]


def lead_table(r):
    """3b per source: available lead, queue wait and memory latency."""
    out = ["| source | measure | p10 | p25 | p50 | p75 | p90 | n |", "|---|---|---|---|---|---|---|---|"]
    for m in sorted(set(r.lead) | set(r.qwait) | set(r.mlat)):
        nm = MODES[m] if m >= 0 else "unknown"
        for lbl, d in (("available lead (demand - latest prediction)", r.lead),
                       ("queue wait (offered - queue entry created)", r.qwait),
                       ("memory latency (filled - issued)", r.mlat)):
            if d.get(m):
                out.append("| %s | %s | %s |" % (nm, lbl, " | ".join(dist_cells(d[m]))))
    return out


def merge_table(r, names, top=6):
    """3c per kernel: latency of a demand read from its L1D access to its data
    leaving the L1D, by kind (new request, merged into a prefetch, merged into
    another demand)."""
    kinds = {"n": "new miss", "p": "merged into prefetch", "d": "merged into demand"}
    per_k = collections.Counter()
    for (k, kind), v in r.dlat.items():
        per_k[k] += len(v)
    out = ["| kernel | kind | n | p10 | p50 | p90 | mean | max | share above the new misses' p90 |",
           "|---|---|---|---|---|---|---|---|---|"]
    flags = []
    for k, _ in per_k.most_common(top):
        base = r.dlat.get((k, "n"), [])
        b90 = pct(base, 90)
        for kind in "npd":
            v = r.dlat.get((k, kind), [])
            if not v:
                continue
            above = sum(1 for x in v if b90 is not None and x > b90) / len(v) if kind != "n" else None
            out.append("| %s | %s | %d | %s | %s | %s | %.0f | %d | %s |" % (
                "%d %s" % (k, names.get(k, "?")[:28]), kinds[kind], len(v), fmt(pct(v, 10), 0), fmt(pct(v, 50), 0),
                fmt(pct(v, 90), 0), sum(v) / len(v), v[-1], "–" if above is None else "%.1f%%" % (100 * above)))
            if kind == "p" and base and pct(v, 50) > pct(base, 50):
                flags.append((k, pct(v, 50), pct(base, 50)))
    return out, flags


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else RUNS
    cfg = sys.argv[2] if len(sys.argv) > 2 else CFG
    jobs = sorted(glob.glob(os.path.join(runs, "*", "*", cfg, "snake_lc.txt.gz")))
    out = ["# Debug plan Stage 3: prefetch lifecycle (generated by `util/snake/debug/s3_lifecycle.py`)\n",
           "Runs: `%s`, config `%s`. The log covers SMs 0, 16, 32, 48 and 64; the whole-run line is all 80 SMs.\n"
           % (os.path.relpath(runs, ROOT), cfg)]
    summary = []
    for lg in jobs:
        d = os.path.dirname(lg)
        app, args = d.split(os.sep)[-3:-1]
        label = app.replace("-rodinia-3.1", "").replace("parboil-", "") + " " + args[:12]
        r = parse(lg)
        w = whole(os.path.join(d, "sim.out"))
        names = kernel_names(os.path.join(d, "sim.out"))
        out.append("## %s\n" % label)
        out.append("Log events: %d. " % r.events + (
            "Whole run: predicted sectors per demand sector %.2f, coverage_pred %.1f%%, issued per demand %.2f, "
            "timely + late %.1f%%, timely %.1f%%.\n" % (w["pred"], w["cov"], w["issued"], w["deliv"], w["timely"])
            if w else "Whole run: n/a.\n"))
        out.append("### 3a Funnel: end state of each prediction record (one per predicted sector, OQ-14), % of "
                   "records closed by a demand (U) or by the window (X); records still open at the end are not "
                   "counted\n")
        out += funnel_table(r)
        out.append("\n### 3a Funnel, sector level (counts)\n")
        out += issue_table(r)
        out.append("\nUnused prefetched sectors lost: evicted %d, invalid at refill %d, at a kernel end %d. "
                   "Dropped from the queue: displaced %d, accuracy filter %d sectors.\n" % (
                       r.lost["e"], r.lost["s"], r.lost["k"], r.drop["x"], r.drop["a"]))
        out.append("### 3b Lead time (cycles)\n")
        out += lead_table(r)
        out.append("\n### 3c Demand read latency, access to data back (cycles), top kernels by reads\n")
        t, flags = merge_table(r, names)
        out += t
        out.append("\n3c check (a demand merged into a prefetch must not wait longer than a new miss): %s\n" % (
            "**FLAG** " + "; ".join("kernel %d median %d vs %d" % f for f in flags) if flags else "pass (medians)"))
        summary.append((label, w, r, flags))
    print("\n".join(out))


if __name__ == "__main__":
    main()
