#!/usr/bin/env python3
"""Debug plan Stage 4: what the programs allow (docs/snake/results/debug/S4_spacing.md).

    s4_spacing.py [<runs dir> [<config>]]

From the Stage 3 lifecycle logs (s3_lifecycle.py; SMs 0, 16, 32, 48, 64) and
existing runs.  The yardstick is the run's median demand miss latency: a new
L1D miss (M kind n) to its data leaving the L1D (R), same log.
  4a load spacing in a warp: cycles between consecutive observed loads of a
     warp (L events, any PCs), and the share of gaps >= the median miss
     latency.  If most pairs are back to back, a chain prefetch triggered by
     PC_i cannot be timely for PC_i+1 (H3).
  4b inter-warp spacing: for a load PC, the cycles between warp w and warp
     w+d reaching it (their n-th executions, n < 4), d = 1..63 global warp ids,
     same CTA or not (warps per CTA from the P events).  This bounds the lead
     an inter-warp prefetch can have (H4; the CTA-aware paper's Fig. 1).
  4c loop iteration time: cycles between consecutive executions of one PC by
     one warp, for (warp, PC) sequences with a constant non-zero address
     stride over at least 3 executions (intra-warp strides).
  4d per-kernel perfect-L1 ceilings from the existing D1 runs (PAPER_V100,
     1B: sim_run_snake_gate/D1_pl1d vs sim_run_baseline; launches summed per
     kernel, the capped last launch excluded) for histo, srad_v1, backprop
     and nw, with each kernel's Snake funnel from the Stage 3 log.
"""
import collections
import glob
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from s3_lifecycle import CFG, RUNS, ROOT, CLASSES, pct, fmt, parse, kernel_names  # noqa: E402
from per_kernel import launches, deltas  # noqa: E402

BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63)]
P1 = "QV100-SASS-PAPER_V100-1B_INSN"
D4 = ["histo", "srad_v1", "backprop", "nw"]


def load_events(path):
    """L events per (sid, kernel, global warp) in order; M/R latencies; the
    warps per CTA of each kernel (from P events)."""
    kern, pend = {}, {}
    loads = collections.defaultdict(list)  # (sid, kuid, gw) -> [(cyc, pc, addr, ok)]
    miss = []
    wpc = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            t = line[0]
            if t == "L":
                f = line.split()
                sid = int(f[1])
                loads[(sid, kern.get(sid, -1), int(f[4]))].append((int(f[2]), int(f[5]), int(f[6]), f[7] == "1"))
            elif t == "M":
                f = line.split()
                if f[6] == "n":
                    pend[int(f[3])] = int(f[2])
            elif t == "R":
                f = line.split()
                c = pend.pop(int(f[3]), None)
                if c is not None:
                    miss.append(int(f[2]) - c)
            elif t == "K":
                f = line.split()
                kern[int(f[1])] = int(f[3])
            elif t == "P":
                f = line.split()
                wpc[int(f[3])] = int(f[6])
    miss.sort()
    return loads, miss, wpc


def dist_row(v, bar):
    v.sort()
    share = 100.0 * sum(1 for x in v if x >= bar) / len(v) if v and bar else None
    return [fmt(pct(v, q), 0) for q in (10, 25, 50, 75, 90)] + ["%d" % len(v), fmt(share)]


def spacing(loads, miss, wpc, names):
    bar = pct(miss, 50)
    out = ["Median demand miss latency (new L1D miss to data back, these SMs): **%s** cycles (n = %d).\n"
           % (fmt(bar, 0), len(miss))]
    # 4a
    gaps_k = collections.defaultdict(list)
    for (sid, k, gw), ev in loads.items():
        for a, b in zip(ev, ev[1:]):
            gaps_k[k].append(b[0] - a[0])
    out += ["### 4a Load spacing in a warp (consecutive observed loads, cycles)\n",
            "| kernel | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |",
            "|---|---|---|---|---|---|---|---|"]
    allg = [x for v in gaps_k.values() for x in v]
    for k in sorted(gaps_k, key=lambda k: -len(gaps_k[k]))[:6]:
        out.append("| %d %s | %s |" % (k, names.get(k, "?")[:30], " | ".join(dist_row(gaps_k[k], bar))))
    out.append("| **all kernels** | %s |" % " | ".join(dist_row(allg, bar)))
    # 4b
    by_pc = collections.defaultdict(dict)  # (sid, k, pc) -> gw -> [cycles]
    for (sid, k, gw), ev in loads.items():
        for cyc, pc, _, _ in ev:
            lst = by_pc[(sid, k, pc)].setdefault(gw, [])
            if len(lst) < 4:
                lst.append(cyc)
    gap_b = collections.defaultdict(list)  # (bucket, same CTA) -> gaps
    for (sid, k, pc), warps in by_pc.items():
        w = wpc.get(k)
        for gw, ts in warps.items():
            for lo, hi in BUCKETS:
                for d in range(lo, hi + 1):
                    t2 = warps.get(gw + d)
                    if not t2:
                        continue
                    same = None if not w else (gw // w == (gw + d) // w)
                    for n in range(min(len(ts), len(t2))):
                        gap_b[((lo, hi), same)].append(t2[n] - ts[n])
    out += ["\n### 4b Inter-warp spacing: warp w+d reaches a load PC this many cycles after warp w\n",
            "| d (global warp ids) | same CTA | p10 | p25 | p50 | p75 | p90 | pairs | gap >= median miss latency (%) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for (lo, hi) in BUCKETS:
        for same in (True, False, None):
            v = gap_b.get(((lo, hi), same))
            if v:
                out.append("| %s | %s | %s |" % ("%d" % lo if lo == hi else "%d-%d" % (lo, hi),
                                                {True: "yes", False: "no", None: "unknown"}[same],
                                                " | ".join(dist_row(v, bar))))
    # 4c
    it = []
    for (sid, k, gw), ev in loads.items():
        seq = collections.defaultdict(list)
        for cyc, pc, addr, ok in ev:
            if ok:
                seq[pc].append((cyc, addr))
        for pc, s in seq.items():
            if len(s) < 3:
                continue
            strides = [b[1] - a[1] for a, b in zip(s, s[1:])]
            if strides[0] != 0 and all(x == strides[0] for x in strides):
                it += [b[0] - a[0] for a, b in zip(s, s[1:])]
    out += ["\n### 4c Loop iteration time, loads with a constant intra-warp stride (cycles)\n",
            "| p10 | p25 | p50 | p75 | p90 | iterations | iteration >= median miss latency (%) |",
            "|---|---|---|---|---|---|---|"]
    out.append("| %s |" % " | ".join(dist_row(it, bar)) if it else "| n/a | | | | | 0 | |")
    return out


def ceilings(app_dir, args, r, names):
    """4d: perfect-L1 gain per kernel (D1 runs) with the kernel's Snake funnel."""
    b = os.path.join(ROOT, "sim_run_baseline", "runs", app_dir, args, P1, "sim.out")
    p = os.path.join(ROOT, "sim_run_snake_gate", "D1_pl1d", "runs", app_dir, args, P1 + "-PERFECT_L1D", "sim.out")
    if not (os.path.exists(b) and os.path.exists(p)):
        return ["4d: D1 runs missing."]
    (lb, cb), (lp, cp) = launches(b), launches(p)
    db, dp = deltas(lb), deltas(lp)
    n = min(len(db), len(dp)) - (1 if cb or cp else 0)
    agg = collections.OrderedDict()
    tot = sum(x[2] for x in db[:n])
    for (uid, name, cyc, ins, _), (_, _, pcyc, pins, _) in zip(db[:n], dp[:n]):
        a = agg.setdefault(name, [0, 0, 0, 0, []])
        a[0] += cyc; a[1] += ins; a[2] += pcyc; a[3] += pins; a[4].append(uid)
    out = ["| kernel | launches | share of baseline cycles (%) | perfect-L1 gain (%) | records (log) | timely | late | "
           "issued, unused or other | not issued | no demand |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, (cyc, ins, pcyc, pins, uids) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if cyc < 0.01 * tot:
            continue
        gain = ((pins / pcyc) / (ins / cyc) - 1) * 100 if cyc and pcyc and ins else None
        cls = collections.Counter()
        for u in uids:
            for c in range(12):
                cls[c] += r.cls_k[(u, c)]
        nrec = sum(cls.values())
        sh = lambda cs: fmt(100.0 * sum(cls[c] for c in cs) / nrec) if nrec else "n/a"
        out.append("| %s | %d | %.1f | %s | %d | %s | %s | %s | %s | %s |" % (
            name[:34], len(uids), 100.0 * cyc / tot, fmt(gain), nrec, sh([8]), sh([6]), sh([7, 9]),
            sh([0, 1, 2, 3, 4, 5]), sh([10, 11])))
    return out


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else RUNS
    cfg = sys.argv[2] if len(sys.argv) > 2 else CFG
    out = ["# Debug plan Stage 4: what the programs allow (generated by `util/snake/debug/s4_spacing.py`)\n",
           "Runs: `%s`, config `%s` (Tier 1 Snake, PAPER_V100, 1B; log on SMs 0, 16, 32, 48, 64).\n"
           % (os.path.relpath(runs, ROOT), cfg)]
    for lg in sorted(glob.glob(os.path.join(runs, "*", "*", cfg, "snake_lc.txt.gz"))):
        d = os.path.dirname(lg)
        app_dir, args = d.split(os.sep)[-3:-1]
        app = app_dir.replace("-rodinia-3.1", "").replace("parboil-", "")
        names = kernel_names(os.path.join(d, "sim.out"))
        loads, miss, wpc = load_events(lg)
        out.append("## %s %s\n" % (app, args[:12]))
        out += spacing(loads, miss, wpc, names)
        if app in D4:
            out.append("\n### 4d Per-kernel perfect-L1 ceiling (D1) and Snake's funnel (Stage 3 log, % of records)\n")
            out += ceilings(app_dir, args, parse(lg), names)
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
