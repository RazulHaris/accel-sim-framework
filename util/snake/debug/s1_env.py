#!/usr/bin/env python3
"""Debug plan Stage 1.1 / 1.3: environment parity (docs/snake/results/debug/S1_environment.md).

    PYTHONPATH=<dir with openpyxl> python3 util/snake/debug/s1_env.py

Per workload at PAPER_V100, 1B: our baseline's operating point and Tier 1
Snake's results with the corrected trace decoder (existing runs) and with the
bug-compatible one (-trace_paperera_decode 1, Stage 1.3 runs in
sim_run_snake_debug/s1s2), next to the paper. Each cell is
paper / bug-compatible / corrected.
  L1 hit        GLOBAL_ACC_R HIT / TOTAL_ACCESS
  Fig. 3        RESERVATION_FAIL / (HIT + HIT_RESERVED + MISS + SECTOR_MISS +
                RESERVATION_FAIL), L1D, all access types (missq_check.py)
  Fig. 4        reply-network packets / (interconnect cycles x 64) (missq_check.py)
  Fig. 5        proxy: W0_Scoreboard / (Stall + W0_Idle + W0_Scoreboard); and,
                where the build has it, W0_Scoreboard_memory / the same total
                (no warp issued and every scoreboard-blocked warp waits on a
                memory register)
  Snake         coverage_pred, delivered (timely + late) / demand, timely /
                demand, L1 hit, IPC gain vs the same decoder's baseline
Paper values: Figs. 3-5 per app from missq_check.py (read off the bars,
±3 points); Fig. 25 per app measured from the bars' pixels of page 13
(calibrated on the axis; the measured averages 45.1 / 78.6 / 83.7 match the
text's 45 / 79 / 84); coverage, accuracy and IPC gain from SPEC §5.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
from missq_check import collect, PAPER3, PAPER4, PAPER5  # noqa: E402
import make_report_xlsx as R  # noqa: E402

P1 = "QV100-SASS-PAPER_V100-1B_INSN"
S = "-SNAKE-SNAKE_FP1"
# Fig. 25 (L1D hit rate, %), measured from the bars: baseline, Snake, Isolated-Snake
FIG25 = {"backprop": (62.4, 67.1, 72.5), "hotspot": (1.0, 84.1, 91.9), "srad_v1": (56.9, 85.4, 93.6),
         "lud": (30.8, 78.6, 84.7), "nw": (1.0, 7.8, 11.5), "histo": (59.3, 94.2, 95.6),
         "mri-q": (7.8, 60.3, 69.5)}
# the paper's stated averages over all its apps (as quoted in DEBUG_PLAN Stage 1.1; Fig. 25 text for Snake)
STATED = {"fig3": 30, "fig4": 33, "fig5": 55, "fig5mem": 55, "hit": 45, "hit_s": 79}
SOURCES = {  # label: (baseline runs, cfg), (Snake runs, cfg), (perfect-L1 runs, cfg)
    "bug-compatible": (("sim_run_snake_debug/s1s2/runs", P1 + "-PAPERERA"),
                       ("sim_run_snake_debug/s1s2/runs", P1 + S + "-PAPERERA"),
                       ("sim_run_snake_debug/s1s2/runs", P1 + "-PERFECT_L1D-PAPERERA")),
    "corrected": (("sim_run_baseline/runs", P1), ("sim_run_snake/runs", P1 + S),
                  ("sim_run_snake_gate/D1_pl1d/runs", P1 + "-PERFECT_L1D")),
}
# corrected decoder at 100M with the new Fig. 5 counter: the DBG1 off-regression
DBG1 = ("sim_run_snake_regress/DBG1/off", "QV100-SASS-PAPER_V100-100M_INSN")
OPS = [("hit", "L1 hit", lambda ap: FIG25[ap][0]),
       ("fig3", "res. fails (Fig. 3)", lambda ap: 100 * PAPER3[ap]),
       ("fig4", "L1-L2 reply BW (Fig. 4)", lambda ap: PAPER4[ap]),
       ("fig5", "all warps on scoreboard (Fig. 5 proxy)", lambda ap: 100 * PAPER5[ap]),
       ("fig5mem", "all blocked warps on memory (Fig. 5)", lambda ap: 100 * PAPER5[ap]),
       ("ceil", "perfect-L1 ceiling", lambda ap: None)]
SNK = [("cov", "predicted coverage", lambda ap: PV[ap][0]),
       ("deliv", "timely + late / demand", lambda ap: None),
       ("acc", "timely / demand (paper's accuracy)", lambda ap: PV[ap][1]),
       ("hit_s", "L1 hit with Snake", lambda ap: FIG25[ap][1]),
       ("gain", "IPC gain", lambda ap: PV[ap][2])]
SIGNED = ("ceil", "gain")
PV = R.paper_values()[0]  # SPEC §5: app -> (coverage, accuracy, IPC gain %, energy)


def rd(p):
    return open(p, errors="replace").read() if os.path.exists(p) else None


def last(t, pat, conv=float):
    m = re.findall(pat, t, re.M)
    return conv(m[-1]) if m else None


def whole(path):
    """Whole-run stats of one finished job (None if missing or unfinished)."""
    t = rd(path)
    if not t or "exit detected" not in t:
        return None
    r = {"ipc": last(t, r"^gpu_tot_ipc =\s*([\d.]+)")}
    b = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"
    h, a = last(t, b % "HIT", int), last(t, b % "TOTAL_ACCESS", int)
    r["hit"] = 100.0 * h / a if a else None
    g = lambda k: last(t, r"^snake_%s = (\d+)" % k, int)
    dem = g("demand_sector_accesses")
    if dem:
        r["cov"] = 100.0 * g("demand_pred_matched") / dem
        r["deliv"] = 100.0 * (g("pf_useful_timely") + g("pf_late")) / dem
        r["acc"] = 100.0 * g("pf_useful_timely") / dem
    occ = re.findall(r"Stall:(\d+)\tW0_Idle:(\d+)\tW0_Scoreboard:(\d+)", t)
    mem = last(t, r"^W0_Scoreboard_memory:(\d+)", int)
    if occ and mem is not None:
        tot = sum(map(int, occ[-1]))
        r["fig5mem"] = 100.0 * mem / tot if tot else None
    return r


def sim(src, a, g):
    return os.path.join(ROOT, src[0], a, g, src[1], "sim.out")


def gather(base, snake, pl1):
    mq = {(r["app"], r["args"]): r for r in collect(os.path.join(ROOT, base[0]), base[1])}
    out = {}
    for a, g in R.workloads():
        b, s, p = whole(sim(base, a, g)), whole(sim(snake, a, g)), whole(sim(pl1, a, g))
        m = mq.get((R.app_of(a), g[:24]))
        out[(a, g)] = dict(
            hit=b and b["hit"], fig3=m and m["fig3"] is not None and 100 * m["fig3"],
            fig4=m and m["fig4_rep"], fig5=m and m["fig5"] is not None and 100 * m["fig5"],
            fig5mem=b and b.get("fig5mem"),
            ceil=(p["ipc"] / b["ipc"] - 1) * 100 if p and b else None,
            cov=s and s.get("cov"), deliv=s and s.get("deliv"), acc=s and s.get("acc"),
            hit_s=s and s["hit"], gain=(s["ipc"] / b["ipc"] - 1) * 100 if s and b else None)
    return out


def f(x, k):
    if x is None or x is False:
        return "n/a"
    return ("%+.1f" if k in SIGNED else "%.1f") % x


def app_mean(per_wl):
    """Arithmetic mean over the 7 apps (an app's inputs averaged first); None unless all 7."""
    per = {}
    for (a, _), v in per_wl.items():
        if v is not None and v is not False:
            per.setdefault(R.app_of(a), []).append(v)
    vals = [sum(v) / len(v) for v in per.values()]
    return sum(vals) / len(vals) if len(vals) == len(R.APPS) else None


def table(metrics, data):
    WL = R.workloads()
    out = ["| workload | " + " | ".join(lbl for _, lbl, _ in metrics) + " |",
           "|---|" + "---|" * len(metrics)]
    for a, g in WL:
        ap = R.app_of(a)
        cells = []
        for k, _, pf in metrics:
            pv = pf(ap)
            cells.append("%s / %s / %s" % ("–" if pv is None else ("%+.0f" if k in SIGNED else "%.0f") % pv,
                                           f(data["bug-compatible"][(a, g)][k], k),
                                           f(data["corrected"][(a, g)][k], k)))
        out.append("| %s | %s |" % (R.wl_label(a, g), " | ".join(cells)))
    cells = []
    for k, _, pf in metrics:
        pv = [pf(ap) for ap in R.APPS]
        pm = sum(pv) / len(pv) if None not in pv else None
        cells.append("%s / %s / %s" % ("–" if pm is None else f(pm, k),
                                       f(app_mean({w: d[k] for w, d in data["bug-compatible"].items()}), k),
                                       f(app_mean({w: d[k] for w, d in data["corrected"].items()}), k)))
    out.append("| **mean, 7 apps** | %s |" % " | ".join(cells))
    out.append("| paper's stated average (all its apps) | %s |" % " | ".join(
        "%d" % STATED[k] if k in STATED else "–" for k, _, _ in metrics))
    return out


def main():
    data = {src: gather(*SOURCES[src]) for src in SOURCES}
    out = ["# Debug plan Stage 1: environment parity (PAPER_V100, 1B)\n",
           "Generated by `util/snake/debug/s1_env.py`. Each cell is **paper / bug-compatible / corrected**, in %. "
           "Bug-compatible = `-trace_paperera_decode 1` (the v1.2.0 trace decoding the paper most likely used; Stage 1.3 runs in "
           "`sim_run_snake_debug/s1s2`). Corrected = our existing baseline, Tier 1 Snake (`-SNAKE-SNAKE_FP1`) and "
           "perfect-L1 runs. Paper per-app values are read off Figs. 3, 4, 5 and 25 (±3 points); coverage, accuracy "
           "and IPC gain per app from SPEC §5. 'n/a' = not measured; '–' = the paper gives no per-app value.\n",
           "### Baseline operating point\n"]
    out += table(OPS, data)
    out += ["", "### Tier 1 Snake (IPC gain against the same decoder's baseline)\n"]
    out += table(SNK, data)
    rows = {(r["app"], r["args"]): r for r in collect(os.path.join(ROOT, DBG1[0]), DBG1[1])}
    out += ["", "### Fig. 5 counter, corrected decoder, 100M (DBG1 off-regression, `%s`)\n" % os.path.dirname(DBG1[0]),
            "| workload | all warps on scoreboard (proxy) | all blocked warps on memory | paper Fig. 5 |",
            "|---|---|---|---|"]
    for a, g in R.workloads():
        r = rows.get((R.app_of(a), g[:24]))
        w = whole(sim(DBG1, a, g))
        if r is None or w is None:
            continue
        out.append("| %s | %.1f | %s | %.0f |" % (R.wl_label(a, g), 100 * r["fig5"], f(w.get("fig5mem"), ""),
                                                 100 * PAPER5[R.app_of(a)]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
