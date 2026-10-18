#!/usr/bin/env python3
"""Per-kernel Snake results vs the paper's per-app values (K1).

    per_kernel.py [--min-share 1.0] [--dir <Tier 1 run dir>] [--base <suffix>]
                  [--cfg <Snake config suffix>] [--detail]

Hypothesis tested: the paper's per-benchmark numbers (SPEC §5, Figs. 16-18)
come from each benchmark's best-performing kernel.  For every Tier 1
workload this pairs the baseline run (sim_run_baseline, QV100-SASS-
PAPER_V100-1B_INSN) with a Snake run (sim_run_snake or another Tier 1 run
dir; default exact footprint) kernel by kernel:
  - gpgpu-sim prints a stats block at the end of every kernel launch:
    gpu_sim_cycle / gpu_sim_insn / gpu_ipc for that launch, and cumulative
    L1D (Total_core_cache_stats_breakdown) and snake_* counters, so a
    launch's counters are differences of consecutive blocks
  - the last launch of a run stopped by the 1B cap is partial and excluded
  - per kernel (function name): launches summed (IPC = insn / cycles)
  - IPC gain = Snake IPC / baseline IPC - 1; coverage_pred, coverage_paper,
    accuracy_paper as in SPEC OQ-14; L1 hit = GLOBAL_ACC_R HIT / TOTAL_ACCESS
  - "best performing kernel" = the largest IPC gain among kernels holding
    at least --min-share % of the workload's baseline cycles (default 1)
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_paper import PAPER, app_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
B = "QV100-SASS-PAPER_V100-1B_INSN"
SNK = ["demand_sector_accesses", "demand_pred_matched", "pf_useful_timely", "pf_late",
       "pf_issued"]
CB = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"


def launches(path):
    """[(uid, name, cycles, insn, cum)] per kernel launch, cum = cumulative
    counters at its end; and whether the run stopped at the instruction cap."""
    t = open(path, errors="replace").read()
    capped = "break due to reaching the maximum" in t
    out = []
    parts = re.split(r"(?m)^kernel_name = ", t)[1:]
    for p in parts:
        g = lambda pat, c=int: (lambda m: c(m[0]) if m else 0)(re.findall(pat, p, re.M))
        cum = {"hit": g(CB % "HIT"), "acc": g(CB % "TOTAL_ACCESS")}
        for k in SNK:
            cum[k] = g(r"^snake_%s = (\d+)" % k)
        out.append((g(r"^kernel_launch_uid = (\d+)"), p.split()[0], g(r"^gpu_sim_cycle = (\d+)"),
                    g(r"^gpu_sim_insn = (\d+)"), cum))
    return out, capped


def deltas(ls):
    prev = None
    res = []
    for uid, name, cyc, ins, cum in ls:
        d = {k: cum[k] - (prev[k] if prev else 0) for k in cum}
        res.append((uid, name, cyc, ins, d))
        prev = cum
    return res


def metrics(base, snk):
    """base, snk: summed dicts with cyc, ins and counter deltas."""
    ipc_b = base["ins"] / base["cyc"] if base["cyc"] else 0
    ipc_s = snk["ins"] / snk["cyc"] if snk["cyc"] else 0
    d = snk["demand_sector_accesses"]
    p = lambda x: 100.0 * x / d if d else None
    return {"gain": (ipc_s / ipc_b - 1) * 100 if ipc_b else None,
            "cov_pred": p(snk["demand_pred_matched"]),
            "cov_paper": p(snk["pf_useful_timely"] + snk["pf_late"]),
            "acc": p(snk["pf_useful_timely"]),
            "hit_b": 100.0 * base["hit"] / base["acc"] if base["acc"] else None,
            "hit_s": 100.0 * snk["hit"] / snk["acc"] if snk["acc"] else None}


def add(a, cyc, ins, d):
    a["cyc"] = a.get("cyc", 0) + cyc
    a["ins"] = a.get("ins", 0) + ins
    for k, v in d.items():
        a[k] = a.get(k, 0) + v


def f(x, fmt="%.1f"):
    return "-" if x is None else fmt % x


def main():
    argv = sys.argv[1:]
    share, cfg, detail, sdir, bsuf = 1.0, "-SNAKE-SNAKE_FP1", False, "sim_run_snake", ""
    while argv:
        if argv[0] == "--min-share":
            share, argv = float(argv[1]), argv[2:]
        elif argv[0] == "--cfg":
            cfg, argv = argv[1], argv[2:]
        elif argv[0] == "--dir":
            sdir, argv = argv[1], argv[2:]
        elif argv[0] == "--base":  # e.g. -PAPER_LAT: baseline B-PAPER_LAT, Snake B-PAPER_LAT<cfg>
            bsuf, argv = argv[1], argv[2:]
        elif argv[0] == "--detail":
            detail, argv = True, argv[1:]
        else:
            sys.exit(__doc__)
    best_rows = []
    for bp in sorted(glob.glob(os.path.join(ROOT, "sim_run_baseline", "runs", "*", "*", B + bsuf, "sim.out"))):
        app, args = bp.split(os.sep)[-4:-2]
        sp = os.path.join(ROOT, sdir, "runs", app, args, B + bsuf + cfg, "sim.out")
        if not os.path.exists(sp):
            continue
        lb, cb = launches(bp)
        ls, cs = launches(sp)
        db, ds = deltas(lb), deltas(ls)
        n = min(len(db), len(ds))
        if cb or cs:
            n -= 1  # the last launch stopped at the cap
        assert all(db[i][0] == ds[i][0] and db[i][1] == ds[i][1] for i in range(n)), app
        tot_cyc = sum(db[i][2] for i in range(n))
        per = {}
        per_launch = []
        for i in range(n):
            uid, name, cyc_b, ins_b, d_b = db[i]
            _, _, cyc_s, ins_s, d_s = ds[i]
            e = per.setdefault(name, [{}, {}, 0])
            add(e[0], cyc_b, ins_b, d_b)
            add(e[1], cyc_s, ins_s, d_s)
            e[2] += 1
            lb_, ls_ = {}, {}
            add(lb_, cyc_b, ins_b, d_b)
            add(ls_, cyc_s, ins_s, d_s)
            per_launch.append((uid, name, metrics(lb_, ls_), 100.0 * cyc_b / tot_cyc))
        whole_b, whole_s = {}, {}
        for e in per.values():
            add(whole_b, e[0]["cyc"], e[0]["ins"], {k: v for k, v in e[0].items() if k not in ("cyc", "ins")})
            add(whole_s, e[1]["cyc"], e[1]["ins"], {k: v for k, v in e[1].items() if k not in ("cyc", "ins")})
        rows = []
        for name, (sb, ss, cnt) in per.items():
            m = metrics(sb, ss)
            rows.append((name, cnt, 100.0 * sb["cyc"] / tot_cyc, m))
        rows.sort(key=lambda r: -r[2])
        a = app_of(app)
        if detail:
            print("### %s / %s (%d launches compared%s)\n" % (a, args[:24], n, ", last launch at the 1B cap excluded" if cb or cs else ""))
            print("| kernel | launches | share of baseline cycles | IPC gain | coverage_pred | coverage_paper | accuracy_paper | L1 hit base -> Snake |")
            print("|---|---|---|---|---|---|---|---|")
            for name, cnt, sh, m in rows:
                print("| `%s` | %d | %.1f%% | %s | %s | %s | %s | %s -> %s |" % (
                    name[:48], cnt, sh, f(m["gain"], "%+.2f%%"), f(m["cov_pred"]), f(m["cov_paper"]),
                    f(m["acc"]), f(m["hit_b"]), f(m["hit_s"])))
            mw = metrics(whole_b, whole_s)
            print("| **whole run (compared launches)** | %d | 100%% | %s | %s | %s | %s | %s -> %s |\n" % (
                n, f(mw["gain"], "%+.2f%%"), f(mw["cov_pred"]), f(mw["cov_paper"]), f(mw["acc"]),
                f(mw["hit_b"]), f(mw["hit_s"])))
        elig = [r for r in rows if r[2] >= share and r[3]["gain"] is not None]
        bk = max(elig, key=lambda r: r[3]["gain"]) if elig else None
        bl = max([x for x in per_launch if x[3] >= share / 10 and x[2]["gain"] is not None],
                 key=lambda x: x[2]["gain"], default=None)
        best_rows.append((a, args, bk, bl, metrics(whole_b, whole_s)))

    print("## Best-performing kernel per workload vs the paper (Snake%s)\n" % cfg.replace("-SNAKE-SNAKE_FP1", " exact footprint"))
    print("Best kernel = the kernel function (all its launches) with the largest IPC gain among those with >= %.1f%% of the baseline cycles. Paper values: SPEC §5 (Figs. 16-18, ±3 pp).\n" % share)
    print("| workload | best kernel (share of cycles) | IPC gain: paper / best kernel / whole run | coverage: paper / best kernel pred, paper-def | accuracy: paper / best kernel | L1 hit, best kernel: base -> Snake |")
    print("|---|---|---|---|---|---|")
    for a, args, bk, bl, mw in best_rows:
        p = PAPER[a]
        if not bk:
            print("| %s / %s | - | %+d%% / - / %s | | | |" % (a, args[:12], p[2], f(mw["gain"], "%+.2f%%")))
            continue
        name, cnt, sh, m = bk
        print("| %s / %s | `%s` (%.1f%%) | %+d%% / %s / %s | %d / %s, %s | %d / %s | %s -> %s |" % (
            a, args[:12], name[:34], sh, p[2], f(m["gain"], "%+.2f%%"), f(mw["gain"], "%+.2f%%"),
            p[0], f(m["cov_pred"]), f(m["cov_paper"]), p[1], f(m["acc"]), f(m["hit_b"]), f(m["hit_s"])))
    print("\n### Best single launch (>= %.2f%% of the cycles)\n" % (share / 10))
    print("| workload | launch (kernel) | share | IPC gain | coverage pred / paper-def | accuracy |")
    print("|---|---|---|---|---|---|")
    for a, args, bk, bl, mw in best_rows:
        if bl:
            uid, name, m, sh = bl
            print("| %s / %s | %d (`%s`) | %.2f%% | %s | %s / %s | %s |" % (
                a, args[:12], uid, name[:30], sh, f(m["gain"], "%+.2f%%"), f(m["cov_pred"]),
                f(m["cov_paper"]), f(m["acc"])))


if __name__ == "__main__":
    main()
