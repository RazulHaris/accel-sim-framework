#!/usr/bin/env python3
"""Debug plan Stage 5: the data behind the implementation audit
(docs/snake/results/debug/S5_audit.md; the code reading is in that file).

    s5_audit.py [<runs dir> [<config>]]

From the Stage 3 lifecycle logs (SMs 0, 16, 32, 48, 64; event format in
s3_lifecycle.py) and the same runs' whole-run counters:
  item 3  prefetch queue: queue wait, drops for the reserves, port waits
  item 4  throttle: share of Snake cycles throttled, by trigger; meter windows
  item 6  inter-warp targets (P events of modes 3 and 4) against the target
          warp's own loads (L events): ahead (it has not yet executed the
          target PC this iteration), already passed it, never executes it
          after the prediction (finished or another path), other CTA
  item 8  offered sectors the L1D called duplicates (DUP_L1 = valid or
          reserved, DUP_MSHR): valid in L1, or in flight for a demand or a
          prefetch; predictions made while a demand miss for the sector
          was already in flight (born late)
  item 9  demand misses (M kind n) on a sector holding an unused prefetch
          (filled, not yet lost or used): must be 0
  item 10 demand reads with no data back by the end of the log, by kind
  item 11 Head and Tail churn (whole-run counters)
  item 12 lane filter: loads excluded and lane-run shapes (whole run)
  metric  SnakeCharmer: loads merged into an in-flight prefetch (M kind p)
          against snake_sc_ld_late (first merge per prefetch) and
          snake_sc_ld_miss (every MISS return, merges included)
"""
import bisect
import collections
import glob
import gzip
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s3_lifecycle import CFG, RUNS, ROOT, pct, fmt  # noqa: E402


def stats(sim):
    t = open(sim, errors="replace").read() if os.path.exists(sim) else ""
    return {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}


def audit(path):
    kern = {}
    loads = collections.defaultdict(list)      # (sid, k, gw, pc) -> cycles
    iw = []                                    # (sid, k, cyc, gw, wpc, pc, tgw, tpc, mode)
    open_miss = collections.defaultdict(int)   # (sid, sector) -> demand misses in flight
    miss_uid = {}                              # uid -> (sid, sector, kind)
    open_pf = set()                            # (sid, sector) issued, not filled
    resident = set()                           # (sid, sector) filled unused prefetch
    filled_at = {}
    dup = collections.Counter()
    born_late = born = 0
    bad9 = []
    p_merges = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split()
            t, sid, cyc = f[0], int(f[1]), int(f[2])
            if t == "L":
                loads[(sid, kern.get(sid, -1), int(f[4]), int(f[5]))].append(cyc)
            elif t == "P":
                mode, new, line_a = int(f[8]), int(f[17]), int(f[14])
                if mode >= 3:
                    iw.append((sid, kern.get(sid, -1), cyc, int(f[5]), int(f[6]), int(f[7]), int(f[12]),
                               int(f[13]), mode))
                for b in range(8):
                    if new >> b & 1:
                        born += 1
                        born_late += open_miss[(sid, line_a + 32 * b)] > 0
            elif t == "M":
                key = (sid, int(f[4]))
                if f[6] == "n":
                    if key in resident:
                        bad9.append((cyc, sid, int(f[4])))
                    open_miss[key] += 1
                elif f[6] == "p":
                    p_merges += 1
                miss_uid[int(f[3])] = (key, f[6])
            elif t == "R":
                m = miss_uid.pop(int(f[3]), None)
                if m and m[1] == "n":
                    open_miss[m[0]] -= 1
            elif t == "O":
                key, res = (sid, int(f[3])), int(f[4])
                if res == 0:
                    open_pf.add(key)
                elif res in (1, 2):
                    if open_miss[key] > 0:
                        dup[(res, "demand in flight")] += 1
                    elif key in open_pf:
                        dup[(res, "prefetch in flight")] += 1
                    else:
                        dup[(res, "valid in L1")] += 1
            elif t == "F":
                key = (sid, int(f[3]))
                open_pf.discard(key)
                if f[5] == "0":
                    resident.add(key)
                    filled_at[key] = cyc
            elif t == "E":
                resident.discard((sid, int(f[3])))
            elif t == "U":
                if f[5] in ("8", "7"):  # timely use, or the record saw the eviction
                    resident.discard((sid, int(f[3])))
            elif t == "K":
                kern[sid] = int(f[3])
    for v in loads.values():
        v.sort()
    # item 6
    iwc = collections.Counter()
    for sid, k, cyc, gw, wpc, pc, tgw, tpc, mode in iw:
        if wpc and gw // wpc != tgw // wpc:
            iwc["other CTA"] += 1
            continue
        trig = loads.get((sid, k, gw, pc), [])
        tgt = loads.get((sid, k, tgw, tpc), [])
        n_trig = bisect.bisect_right(trig, cyc)
        n_tgt = bisect.bisect_left(tgt, cyc)
        if n_tgt >= n_trig and n_trig > 0:
            iwc["already passed"] += 1
        elif len(tgt) <= n_tgt:
            iwc["never executes it later"] += 1
        else:
            iwc["ahead"] += 1
    lost = collections.Counter(kind for _, kind in miss_uid.values())
    return dict(iw=iwc, dup=dup, born=born, born_late=born_late, bad9=bad9, lost=lost, p_merges=p_merges)


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else RUNS
    cfg = sys.argv[2] if len(sys.argv) > 2 else CFG
    rows = []
    for lg in sorted(glob.glob(os.path.join(runs, "*", "*", cfg, "snake_lc.txt.gz"))):
        d = os.path.dirname(lg)
        app_dir, args = d.split(os.sep)[-3:-1]
        label = app_dir.replace("-rodinia-3.1", "").replace("parboil-", "") + " " + args[:10]
        rows.append((label, audit(lg), stats(os.path.join(d, "sim.out"))))
    out = ["# Debug plan Stage 5: audit data (generated by `util/snake/debug/s5_audit.py`)\n",
           "Runs: `%s`, config `%s`. Log items: SMs 0, 16, 32, 48 and 64; counter items: all SMs.\n"
           % (os.path.relpath(runs, ROOT), cfg)]
    pc_ = lambda a, b: fmt(100.0 * a / b if b else None)
    out += ["## Item 3: prefetch queue (whole run)\n",
            "| workload | offered lines waited for the port (per issued sector) | drops: MSHR reserve | "
            "drops: miss-queue reserve | drops: miss queue full | queue full (sectors) | displaced |",
            "|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        iss = s.get("pf_issued", 0)
        out.append("| %s | %s | %d | %d | %d | %d | %d |" % (
            lbl, fmt(s.get("pf_wait_port", 0) / iss if iss else None, 2), s.get("pf_drop_mshr", 0),
            s.get("pf_drop_mq_reserve", 0), s.get("pf_drop_miss_queue", 0) - s.get("pf_drop_mq_reserve", 0),
            s.get("pf_drop_queue_full", 0), s.get("pf_queue_displaced", 0)))
    out += ["\n## Item 4: throttling (whole run)\n",
            "| workload | throttled (% of Snake cycles) | by space | by bandwidth | space events | bandwidth events | "
            "meter windows >= 70% (%) |", "|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        ticks = s.get("hook_tick", 0)
        win = s.get("bw_windows", 0)
        out.append("| %s | %s | %s | %s | %d | %d | %s |" % (
            lbl, pc_(s.get("throttled_cycles", 0), ticks), pc_(s.get("throttled_cycles_space", 0), ticks),
            pc_(s.get("throttled_cycles_bw", 0), ticks), s.get("throttle_space_events", 0),
            s.get("throttle_bw_events", 0),
            pc_(sum(s.get("bw_util_hist_%d" % k, 0) for k in (70, 80, 90, 100)), win)))
    out += ["\n## Item 6: inter-warp targets (log)\n",
            "| workload | inter-warp predictions | ahead (%) | already passed the PC (%) | never execute it later (%) | "
            "other CTA (%) |", "|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        n = sum(a["iw"].values())
        out.append("| %s | %d | %s | %s | %s | %s |" % (lbl, n, pc_(a["iw"]["ahead"], n),
                                                       pc_(a["iw"]["already passed"], n),
                                                       pc_(a["iw"]["never executes it later"], n),
                                                       pc_(a["iw"]["other CTA"], n)))
    out += ["\n## Item 8: duplicates at offer and predictions born late (log)\n",
            "| workload | DUP_L1: valid in L1 | DUP_L1: demand in flight | DUP_L1: prefetch in flight | "
            "DUP_MSHR: demand | DUP_MSHR: prefetch | DUP_MSHR: other | sectors newly queued | of them, a demand "
            "miss already in flight (%) |", "|---|---|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        d = a["dup"]
        out.append("| %s | %d | %d | %d | %d | %d | %d | %d | %s |" % (
            lbl, d[(1, "valid in L1")], d[(1, "demand in flight")], d[(1, "prefetch in flight")],
            d[(2, "demand in flight")], d[(2, "prefetch in flight")], d[(2, "valid in L1")], a["born"],
            pc_(a["born_late"], a["born"])))
    out += ["\n## Items 9 and 10: misses on resident prefetches; reads never answered (log)\n",
            "| workload | new misses on a sector holding an unused prefetch (must be 0) | first cases (cycle, SM, sector) | "
            "reads with no data back at the end: new / merged into prefetch / merged into demand |",
            "|---|---|---|---|"]
    for lbl, a, s in rows:
        out.append("| %s | %d | %s | %d / %d / %d |" % (lbl, len(a["bad9"]), "; ".join(
            "%d %d %d" % x for x in a["bad9"][:3]) or "–", a["lost"]["n"], a["lost"]["p"], a["lost"]["d"]))
    out += ["\n## Item 11: Head and Tail churn (whole run)\n",
            "| workload | Head row evictions per update (%) | Tail evictions per allocation (%) | Tail hit rate (%) | "
            "T1 promotions | T1 demotions | warps removed |", "|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        out.append("| %s | %s | %s | %s | %d | %d | %d |" % (
            lbl, pc_(s.get("head_row_evictions", 0), s.get("head_updates", 0)),
            pc_(s.get("tail_evictions", 0), s.get("tail_allocs", 0)),
            pc_(s.get("tail_hits", 0), s.get("tail_lookups", 0)), s.get("t1_promotions", 0),
            s.get("t1_demotions", 0), s.get("warps_removed", 0)))
    out += ["\n## Item 12: lane filter (whole run, lane mode 1)\n",
            "| workload | loads observed | excluded (fit no run length, %) | runs of 32 (%) | 16 (%) | 8 (%) | 4 (%) |",
            "|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        n = s.get("loads_observed", 0)
        out.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            lbl, n, pc_(s.get("warp_excluded_lane_runs", 0), n), pc_(s.get("lane_runs_32", 0), n),
            pc_(s.get("lane_runs_16", 0), n), pc_(s.get("lane_runs_8", 0), n), pc_(s.get("lane_runs_4", 0), n)))
    out += ["\n## Metric: SnakeCharmer late loads (whole run and log)\n",
            "| workload | sc_ld_ontime | sc_ld_late (first merge per prefetch) | sc_ld_miss (all MISS returns) | "
            "cov_on as reported | cov_on, late not counted twice | log: loads merged into a prefetch |",
            "|---|---|---|---|---|---|---|"]
    for lbl, a, s in rows:
        on, late, miss = s.get("sc_ld_ontime", 0), s.get("sc_ld_late", 0), s.get("sc_ld_miss", 0)
        out.append("| %s | %d | %d | %d | %s | %s | %d |" % (
            lbl, on, late, miss, pc_(on, on + late + miss), pc_(on, on + miss), a["p_merges"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
