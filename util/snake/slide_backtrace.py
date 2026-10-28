#!/usr/bin/env python3
"""Backtrace of the MTP slide's IPC numbers (docs/snake/results/slide_ipc_backtrace.md).

    slide_backtrace.py

Slide text: "Snake predicts 63% of addresses (76%), but only 6% arrive in
time, so IPC is flat (1.7% vs 14.9%)".  No simulations: every value is
recomputed from existing outputs, for each candidate definition, and the
candidates that reproduce the slide's numbers are flagged.

Candidates (our 7 apps, 9 workloads; Snake = exact footprint):
  measurement  whole run | best kernel (K1: largest IPC gain among kernels
               with >= 1% of the baseline cycles, complete launches) | best
               kernel over all seven Tier 1 Snake variants (K1 §2)
  conditions   PAPER_V100 1B (Tier 1) | PAPER_V100 100M | base QV100 100M
               (the 100M runs keep the launch cut by the cap, kernel_gap.py)
  averaging    arithmetic or geometric (IPC), over the 7 apps (an app's
               inputs first) or over the 9 workloads
  paper        SPEC §5 values per app, averaged the same ways; the paper's
               own 11-app average
Metrics: coverage_pred, accuracy_paper (timely / demand sectors) and IPC
gain, as in final_tier1.md and K1_best_kernel.md.
Then the slide's method applied to SK_R0_D0 and the fix_today configs
(PAPER_V100, 1B), per workload side by side with Tier 1 Snake.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel_gap import Pair, kname, workloads, P1, P, Q, S as S_SUF, SK as SK_SUF  # noqa: E402
from kernel_diag import per_kernel  # noqa: E402
from sk_sc_report import short  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T1 = P1 + S_SUF
SK = P1 + SK_SUF
X = SK + "-SNAKE_ACCF-SNAKE_ACCFB10K"
# Tier 1 Snake variants (K1 §2): (label, run dir, config, baseline config)
VARIANTS = [("exact", "sim_run_snake", T1, P1), ("line", "sim_run_snake", P1 + "-SNAKE", P1),
            ("Snake-DT", "sim_run_snake_dt", P1 + "-SNAKE_DT-SNAKE_FP1", P1),
            ("Snake-T", "sim_run_snake_t", P1 + "-SNAKE_T-SNAKE_FP1", P1),
            ("s-Snake", "sim_run_ssnake", P1 + "-S_SNAKE-SNAKE_FP1", P1),
            ("Snake-ISO", "sim_run_snake", P1 + "-SNAKE_ISO-SNAKE_FP1", P1),
            ("PAPER_LAT", "sim_run_snake", P1 + "-PAPER_LAT-SNAKE-SNAKE_FP1", P1 + "-PAPER_LAT")]
# fix_today and SK configs at Tier 1 conditions
OTHERS = [("SK_R0_D0", "sim_run_snake_skm/v100p_1b", SK), ("A", "sim_run_snake_fix/p2", T1 + "-SNAKE_R2"),
          ("B", "sim_run_snake_fix/p2", SK + "-SNAKE_R2"), ("C", "sim_run_snake_fix/p2", SK + "-SNAKE_R2-SNAKE_PQARB1"),
          ("D", "sim_run_snake_fix/p3", T1 + "-SNAKE_ACCF"), ("E", "sim_run_snake_fix/p3", SK + "-SNAKE_ACCF"),
          ("F", "sim_run_snake_fix/p3b", SK + "-SNAKE_R2-SNAKE_ACCF"), ("E10", "sim_run_snake_fix/p3b", X),
          ("G8", "sim_run_snake_fix/p4", X + "-SNAKE_IWD8"), ("G16", "sim_run_snake_fix/p4", X + "-SNAKE_IWD16"),
          ("G32", "sim_run_snake_fix/p4", X + "-SNAKE_IWD32")]
SLIDE = {"IPC ours": 1.7, "IPC paper": 14.9, "coverage": 63, "coverage ()": 76, "in time": 6}


def sim(d, w, c):
    return os.path.join(ROOT, d, "runs", w[0], w[1], c, "sim.out")


def amean(xs):
    return sum(xs) / len(xs)


def gmean(xs):
    return (math.exp(sum(math.log1p(x / 100) for x in xs) / len(xs)) - 1) * 100


def avg(per, how, ipc):
    """per: {(app, args): value}; how: '7 apps' or '9 workloads'."""
    f = gmean if ipc == "geo" else amean
    if how == "9 workloads":
        return f(list(per.values()))
    apps = {}
    for (a, _), v in per.items():
        apps.setdefault(a, []).append(v)
    return f([f(v) for v in apps.values()])


def whole(bp, sp):
    """coverage_pred, accuracy_paper, IPC gain of a whole run (last stats block)."""
    import re
    t = open(sp, errors="replace").read()
    g = lambda k: int(re.findall(r"^snake_%s = (\d+)" % k, t, re.M)[-1])
    ipc = lambda p: float(re.findall(r"gpu_tot_ipc =\s*([\d.]+)", open(p, errors="replace").read())[-1])
    dem = g("demand_sector_accesses")
    return {"cov": 100.0 * g("demand_pred_matched") / dem, "acc": 100.0 * g("pf_useful_timely") / dem,
            "ipc": (ipc(sp) / ipc(bp) - 1) * 100}


def kernel(bp, sp, cut):
    """The best kernel's name and its coverage_pred, accuracy_paper, IPC gain."""
    pr = Pair(bp, sp, cut=cut)
    nm, g = pr.best()
    d = per_kernel(sp) if cut else None
    if d is None or nm not in d:  # 100M runs keep the cut launch: sum their launch deltas
        idx = [i for i in pr.idx if pr.ds[i][1] == nm]
        tot = {}
        for i in idx:
            for k, v in pr.ds[i][4].items():
                tot[k] = tot.get(k, 0) + v
        dem = tot.get("demand_sector_accesses", 0)
        return nm, {"cov": 100.0 * tot.get("demand_pred_matched", 0) / dem if dem else 0.0,
                    "acc": 100.0 * tot.get("pf_useful_timely", 0) / dem if dem else 0.0, "ipc": g}
    k = d[nm]
    dem = k.get("snake_demand_sector_accesses", 0)
    return nm, {"cov": 100.0 * k.get("snake_demand_pred_matched", 0) / dem if dem else 0.0,
                "acc": 100.0 * k.get("snake_pf_useful_timely", 0) / dem if dem else 0.0, "ipc": g}


def paper():
    import make_report_xlsx as R
    p, rows, _ = R.paper_values()
    app = {"backprop-rodinia-3.1": "backprop", "hotspot-rodinia-3.1": "hotspot", "lud-rodinia-3.1": "lud",
           "nw-rodinia-3.1": "nw", "parboil-histo": "histo", "parboil-mri-q": "mri-q",
           "srad_v1-rodinia-3.1": "srad_v1"}
    return p, app, {nm: v for nm, v, _ in rows}


def flag(v, target, tol):
    return " **← slide %s**" % target if abs(v - target) <= tol else ""


def main():
    WL = workloads()
    base = lambda w, c=P1: sim("sim_run_baseline", w, c)
    pv, app, prow = paper()
    out = []
    cand = {}  # (measurement, conditions) -> {metric: {w: v}}
    # whole run and best kernel, three conditions
    cond = [("PAPER_V100, 1B (Tier 1)", "sim_run_baseline", P1, "sim_run_snake", T1, True),
            ("PAPER_V100, 100M", "sim_run_snake_refcache/bin_ref_v2", P, "sim_run_snake_skc/v100p_7apps", P + S_SUF, False),
            ("base QV100, 100M", "sim_run_snake_skc/qv100_7apps", Q, "sim_run_snake_skc/qv100_7apps", Q + S_SUF, False)]
    names = {}
    for cl, bd, bc, sd, sc, cut in cond:
        for meas in ("whole run", "best kernel"):
            per = {"cov": {}, "acc": {}, "ipc": {}}
            for w in WL:
                bp, sp = sim(bd, w, bc), sim(sd, w, sc)
                if meas == "whole run":
                    r = whole(bp, sp)
                else:
                    nm, r = kernel(bp, sp, cut)
                    names[(cl, w)] = kname(nm)
                for m in per:
                    per[m][w] = r[m]
            cand[(meas, cl)] = per
    # best kernel over all Tier 1 variants (K1 §2)
    per = {"cov": {}, "acc": {}, "ipc": {}}
    vbest = {}
    for w in WL:
        best = None
        for lab, d, c, bc in VARIANTS:
            nm, r = kernel(base(w, bc), sim(d, w, c), True)
            if best is None or r["ipc"] > best[2]["ipc"]:
                best = (lab, kname(nm), r)
        vbest[w] = best
        for m in per:
            per[m][w] = best[2][m]
    cand[("best kernel over 7 variants", "PAPER_V100, 1B (Tier 1)")] = per
    # paper per workload (an app's value for each of its inputs)
    ppw = {"cov": {w: pv[app[w[0]]][0] for w in WL}, "acc": {w: pv[app[w[0]]][1] for w in WL},
           "ipc": {w: pv[app[w[0]]][2] for w in WL}}

    out.append("## Candidates (Tier 1 Snake, exact footprint; our 9 workloads)\n")
    out.append("Values in %. IPC: arithmetic / geometric mean of gains. Flags: within ±0.05 of the slide's 1.7 / "
               "14.9, or rounding to its 63 / 76 / 6.\n")
    out.append("| measurement | conditions | averaging | coverage_pred | timely (accuracy_paper) | IPC gain, arithmetic | "
               "IPC gain, geometric |")
    out.append("|---|---|---|---|---|---|---|")
    for (meas, cl), per in cand.items():
        for how in ("7 apps", "9 workloads"):
            cov, acc = avg(per["cov"], how, "a"), avg(per["acc"], how, "a")
            ia, ig = avg(per["ipc"], how, "a"), avg(per["ipc"], how, "geo")
            out.append("| %s | %s | %s | %.1f%s | %.1f%s | %+.2f%s | %+.2f%s |" % (
                meas, cl, how, cov, flag(round(cov), 63, 0) or flag(round(cov), 76, 0), acc, flag(round(acc), 6, 0),
                ia, flag(ia, 1.7, 0.05), ig, flag(ig, 1.7, 0.05)))
    for how in ("7 apps", "9 workloads"):
        cov, acc = avg(ppw["cov"], how, "a"), avg(ppw["acc"], how, "a")
        ia, ig = avg(ppw["ipc"], how, "a"), avg(ppw["ipc"], how, "geo")
        out.append("| **paper** (SPEC §5, Figs. 16–18) | paper | %s | %.1f%s | %.1f | %+.2f%s | %+.2f%s |" % (
            how, cov, flag(round(cov), 76, 0), acc, ia, flag(ia, 14.9, 0.05), ig, flag(ig, 14.9, 0.05)))
    a11 = prow.get("Avg / GMEAN")
    p11 = [v for nm, v in prow.items() if nm != "Avg / GMEAN"]
    out.append("| **paper**, its own 11 apps | paper | 11 apps | %.1f (table: %.0f) | %.1f (table: %.0f) | %+.2f | %+.2f "
               "(table: %+.0f) |" % (amean([v[0] for v in p11]), a11[0], amean([v[1] for v in p11]), a11[1],
                                     amean([v[2] for v in p11]), gmean([v[2] for v in p11]), a11[2]))

    # the slide's method, per workload, for Tier 1 Snake and the SK / fix_today configs
    out.append("\n## Slide method per workload: best kernel, PAPER_V100, 1B (normalized IPC of the best kernel)\n")
    cfgs = [("Tier 1 Snake", "sim_run_snake", T1)] + OTHERS
    rows = {}
    for lab, d, c in cfgs:
        rows[lab] = {}
        for w in WL:
            nm, r = kernel(base(w), sim(d, w, c), True)
            rows[lab][w] = (kname(nm), r)
    out.append("| workload | paper (app) | " + " | ".join(l for l, _, _ in cfgs) + " |")
    out.append("|---|---|" + "---|" * len(cfgs))
    for w in WL:
        out.append("| %s | %.2f | %s |" % (short(*w), 1 + ppw["ipc"][w] / 100, " | ".join(
            "%.4f `%s`" % (1 + rows[l][w][1]["ipc"] / 100, rows[l][w][0]) for l, _, _ in cfgs)))
    out.append("| **arithmetic mean of gains, 9 workloads** | %+.2f%% | %s |" % (
        amean(list(ppw["ipc"].values())), " | ".join("%+.2f%%" % amean([rows[l][w][1]["ipc"] for w in WL]) for l, _, _ in cfgs)))
    out.append("\n## Best kernel's coverage_pred / timely accuracy, PAPER_V100, 1B\n")
    out.append("| workload | paper | " + " | ".join(l for l, _, _ in cfgs[:2]) + " |")
    out.append("|---|---|---|---|")
    for w in WL:
        out.append("| %s | %.0f / %.0f | %s |" % (short(*w), ppw["cov"][w], ppw["acc"][w], " | ".join(
            "%.1f / %.1f" % (rows[l][w][1]["cov"], rows[l][w][1]["acc"]) for l, _, _ in cfgs[:2])))
    out.append("| arithmetic mean, 9 workloads | %.1f / %.1f | %s |" % (
        amean(list(ppw["cov"].values())), amean(list(ppw["acc"].values())), " | ".join(
            "%.1f / %.1f" % (amean([rows[l][w][1]["cov"] for w in WL]), amean([rows[l][w][1]["acc"] for w in WL]))
            for l, _, _ in cfgs[:2])))
    out.append("\n## Best kernel over the seven Tier 1 variants (candidate c)\n")
    out.append("| workload | variant | kernel | IPC gain |")
    out.append("|---|---|---|---|")
    for w in WL:
        lab, nm, r = vbest[w]
        out.append("| %s | %s | `%s` | %+.2f%% |" % (short(*w), lab, nm, r["ipc"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
