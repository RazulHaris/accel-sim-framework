#!/usr/bin/env python3
"""Paper motivation figures (Figs. 3-5) from baseline outputs.

    missq_check.py <reference runs dir> [<config>]

Reads <runs>/<app>/<args>/<config>/sim.out (default config
QV100-SASS-PAPER_V100-100M_INSN, the cached bin_ref_v2 baselines) and prints
the tables of docs/snake/results/missq_config_check.md §3.  Every counter is
the last value printed (cumulative over the run).

  Fig. 3  reservation-fail rate = RESERVATION_FAIL / (HIT + HIT_RESERVED +
          MISS + SECTOR_MISS + RESERVATION_FAIL), L1D, all access types
          (MSHR_HIT is excluded: it double-counts a merge).  Every retry of
          a failed access counts again, as in the simulator's own stats.
          Fail causes: Total_core_cache_fail_stats_breakdown.
  Fig. 4  L1<->L2 utilisation = interconnect packets / (icnt cycles * 64),
          per subnet: the local crossbar (-network_mode 2) moves at most one
          packet per input and per output each cycle, and 64 L2
          sub-partitions bound both directions.  Request-slot utilisation =
          request packets / (cycles * 80 SMs): an SM's L1D miss queue sends
          at most one request per cycle.
  Fig. 5  proxy: W0_Scoreboard / (W0_Idle + W0_Scoreboard + Stall), the
          scheduler-cycles with no issue in which every valid warp waited on
          a register dependency (mostly loads); also (W0_Scoreboard + Stall)
          / the same total, counting pipeline stalls (e.g. a blocked LD/ST
          unit) as memory.
"""
import glob
import math
import os
import re
import sys

PAPER3 = {"backprop": 0.29, "hotspot": 0.57, "srad_v1": 0.71, "lud": 0.20,
          "nw": 0.03, "histo": 0.16, "mri-q": 0.80}
PAPER4 = {"backprop": 44, "hotspot": 32, "srad_v1": 32, "lud": 22, "nw": 24,
          "histo": 45, "mri-q": 20}
PAPER5 = {"backprop": 0.76, "hotspot": 0.52, "srad_v1": 0.32, "lud": 0.26,
          "nw": 0.71, "histo": 0.47, "mri-q": 0.96}
STATUS = ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS", "RESERVATION_FAIL"]
CAUSES = ["MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL", "MSHR_MERGE_ENRTY_FAIL",
          "LINE_ALLOC_FAIL"]


def app_of(d):
    return (d.replace("-rodinia-3.1", "").replace("parboil-", ""))


def last(t, pat, cast=int):
    m = re.findall(pat, t)
    return cast(m[-1]) if m else None


def gmean(xs):
    xs = [x for x in xs if x is not None]
    xs = [x for x in xs if x > 0]  # a zero (nw's Fig. 3 rate) is left out
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def collect(runs, cfg="QV100-SASS-PAPER_V100-100M_INSN"):
    """One dict per workload (the rows of the tables below)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "*", "*", cfg, "sim.out"))):
        app_dir, args = p.split(os.sep)[-4:-2]
        t = open(p, errors="replace").read()
        st = {}
        for k, s, v in re.findall(r"Total_core_cache_stats_breakdown\[(\w+)\]\[(\w+)\] = (\d+)", t):
            st[(k, s)] = int(v)  # later prints overwrite earlier ones
        fl = {}
        for k, c, v in re.findall(r"Total_core_cache_fail_stats_breakdown\[(\w+)\]\[(\w+)\] = (\d+)", t):
            fl[(k, c)] = int(v)
        types = {k for k, _ in st}
        acc = sum(st.get((k, s), 0) for k in types for s in STATUS)
        rf = sum(st.get((k, "RESERVATION_FAIL"), 0) for k in types)
        causes = {c: sum(v for (k, cc), v in fl.items() if cc == c) for c in CAUSES}
        occ = re.findall(r"Stall:(\d+)\tW0_Idle:(\d+)\tW0_Scoreboard:(\d+)", t)
        stall, idle, sb = map(int, occ[-1]) if occ else (0, 0, 0)
        r = dict(app=app_of(app_dir), args=args[:24], acc=acc, rf=rf, causes=causes,
                 req=last(t, r"Req_Network_injected_packets_num = (\d+)"),
                 rep=last(t, r"Reply_Network_injected_packets_num = (\d+)"),
                 cyc=last(t, r"Req_Network_cycles = (\d+)"),
                 gcyc=last(t, r"gpu_tot_sim_cycle = (\d+)"),
                 inbuf=last(t, r"Req_Network_in_buffer_full_per_cycle =\s*([\d.]+)", float),
                 stall=stall, idle=idle, sb=sb,
                 rd_rf=st.get(("GLOBAL_ACC_R", "RESERVATION_FAIL"), 0),
                 wr_rf=st.get(("GLOBAL_ACC_W", "RESERVATION_FAIL"), 0),
                 mflat=last(t, r"averagemflatency = (\d+)"),
                 l1kb=re.findall(r"Reconfigure L1 cache to (\d+)KB", t))
        r["fig3"] = rf / acc if acc else None
        r["fig4_req"] = 100.0 * r["req"] / (r["cyc"] * 64) if r["cyc"] else None
        r["fig4_rep"] = 100.0 * r["rep"] / (r["cyc"] * 64) if r["cyc"] else None
        r["slot"] = 100.0 * r["req"] / (r["cyc"] * 80) if r["cyc"] else None
        tot = stall + idle + sb
        r["fig5"] = sb / tot if tot else None
        r["fig5b"] = (sb + stall) / tot if tot else None
        rows.append(r)
    return rows


def main():
    runs = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else "QV100-SASS-PAPER_V100-100M_INSN"
    rows = collect(runs, cfg)

    print("### Fig. 3: reservation-fail rate and causes\n")
    print("| workload | L1D accesses | RESERVATION_FAIL | rate | paper | of fails: MISS_QUEUE_FULL / MSHR entry / MSHR merge / line alloc | fails from reads / writes |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        c = r["causes"]
        tc = sum(c.values())
        print("| %s / %s | %d | %d | %.2f | %.2f | %s | %d / %d |" % (
            r["app"], r["args"], r["acc"], r["rf"], r["fig3"], PAPER3[r["app"]],
            " / ".join("%.0f%%" % (100.0 * c[k] / tc) if tc else "-" for k in CAUSES),
            r["rd_rf"], r["wr_rf"]))
    print("\n### Fig. 4: L1<->L2 interconnect utilisation (% of 64 packets/cycle) and request-slot utilisation (% of 1 request/SM/cycle)\n")
    print("| workload | icnt cycles | request pkts | reply pkts | request util | reply util | paper (Fig. 4) | request-slot util | request in-buffer full per cycle |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s / %s | %d | %d | %d | %.1f%% | %.1f%% | %d%% | %.1f%% | %.4f |" % (
            r["app"], r["args"], r["cyc"], r["req"], r["rep"], r["fig4_req"],
            r["fig4_rep"], PAPER4[r["app"]], r["slot"], r["inbuf"]))
    print("\n### Fig. 5 (proxy): no-issue scheduler-cycles in which all warps wait on dependencies\n")
    print("| workload | Stall | W0_Idle | W0_Scoreboard | Scoreboard share | (Scoreboard + Stall) share | paper (Fig. 5) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s / %s | %d | %d | %d | %.2f | %.2f | %.2f |" % (
            r["app"], r["args"], r["stall"], r["idle"], r["sb"], r["fig5"],
            r["fig5b"], PAPER5[r["app"]]))
    # Per app: mean over its inputs (hotspot, lud have two), then GMEAN over apps.
    print("\n### Per app (mean over inputs) and GMEAN over the 7 apps\n")
    print("| app | Fig. 3 ours / paper | Fig. 4 reply util ours / paper | request-slot util | Fig. 5 proxy (Scoreboard; +Stall) ours / paper |")
    print("|---|---|---|---|---|")
    apps = []
    for a in PAPER3:
        rs = [r for r in rows if r["app"] == a]
        if not rs:
            continue
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        apps.append((a, m("fig3"), m("fig4_rep"), m("slot"), m("fig5"), m("fig5b")))
        print("| %s | %.2f / %.2f | %.1f%% / %d%% | %.1f%% | %.2f; %.2f / %.2f |" % (
            a, m("fig3"), PAPER3[a], m("fig4_rep"), PAPER4[a], m("slot"), m("fig5"),
            m("fig5b"), PAPER5[a]))
    g3, g4, gs, g5, g5b = (gmean([x[i] for x in apps]) for i in (1, 2, 3, 4, 5))
    avg4 = sum(x[2] for x in apps) / len(apps)
    fmt = lambda v, f: f % v if v is not None else "n/a"
    print("| **GMEAN** (Fig. 3 without nw's 0; Fig. 4 and slot: mean) | %s / 0.26 | %.1f%% / 33%% | %.1f%% | %s; %s / 0.55 |" % (
        fmt(g3, "%.2f"), avg4, sum(x[3] for x in apps) / len(apps), fmt(g5, "%.2f"),
        fmt(g5b, "%.2f")))


if __name__ == "__main__":
    main()
