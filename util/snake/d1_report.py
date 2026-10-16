#!/usr/bin/env python3
"""D1 ceilings vs Tier 1 (docs/snake/results/D1_ceilings.md).

    d1_report.py <D1_pmem runs> <D1_pl1d runs> <S6_ceiling runs>

Tier 1 run dirs (sim_run_baseline, sim_run_snake) are read from the
accel-sim root.  One table per config (PAPER_V100, PAPER_LAT), one row per
workload: baseline IPC, perfect-memory gain, perfect-L1D gain, Snake (exact
footprint) gain, the paper's Snake gain.  Gains = IPC / baseline IPC - 1.
GMEAN rows: over the apps (an app's inputs first), as in compare_paper.py.
Then: the S6 perfect-memory runs vs D1's (confirmed if the IPCs are equal),
and a sanity check that perfect-L1D runs hit on every GLOBAL_ACC_R access.
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_paper import PAPER, app_of  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def stats(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    if "exit detected" not in t:
        return None
    last = lambda pat, c=float: (lambda m: c(m[-1]) if m else None)(re.findall(pat, t, re.M))
    b = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"
    return {"ipc": last(r"^gpu_tot_ipc =\s*([\d.]+)"),
            "insn": last(r"^gpu_tot_sim_insn = (\d+)", int),
            "hit": last(b % "HIT", int) or 0, "acc": last(b % "TOTAL_ACCESS", int) or 0,
            "tags": " ".join(sorted(set(re.findall(
                r"(?:accelsim-commit|gpgpu-sim_git-commit)-[0-9a-f]+_modified_[0-9.]+", t))))}


def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def pct(r):
    return "%+.1f%%" % ((r - 1) * 100) if r else "-"


def main():
    pmem, pl1d, s6 = sys.argv[1:4]
    B = "QV100-SASS-PAPER_V100-1B_INSN"
    wl = sorted({tuple(p.split(os.sep)[-4:-2]) for p in glob.glob(
        os.path.join(ROOT, "sim_run_baseline", "runs", "*", "*", B, "sim.out"))})
    for label, cfg in (("PAPER_V100 (QV100, GTO, 1530 MHz)", B),
                       ("PAPER_LAT (PAPER_V100 + L1 23 / L2 ROP 199: measured L1 28, L2 212 cycles)",
                        B + "-PAPER_LAT")):
        print("## %s\n" % label)
        print("| workload | baseline IPC | perfect memory gain | perfect L1 gain | Snake gain | paper's Snake gain |")
        print("|---|---|---|---|---|---|")
        per = {}
        for a, g in wl:
            j = lambda d, c: os.path.join(d, a, g, c, "sim.out")
            b = stats(j(os.path.join(ROOT, "sim_run_baseline", "runs"), cfg))
            pm = stats(j(pmem, cfg + "-PERFECT_MEM"))
            pl = stats(j(pl1d, cfg + "-PERFECT_L1D"))
            sn = stats(j(os.path.join(ROOT, "sim_run_snake", "runs"), cfg + "-SNAKE-SNAKE_FP1"))
            r = lambda x: x["ipc"] / b["ipc"] if x and b and b["ipc"] else None
            per.setdefault(app_of(a), []).append((r(pm), r(pl), r(sn)))
            print("| %s / %s | %s | %s | %s | %s | %+d%% |" % (
                app_of(a), g[:20], "%.1f" % b["ipc"] if b else "-",
                pct(r(pm)), pct(r(pl)), pct(r(sn)), PAPER[app_of(a)][2]))
        col = lambda i: gmean([gmean([v[i] for v in vs]) for vs in per.values()])
        print("| **GMEAN** (%d apps) | | %s | %s | %s | %+.1f%% |\n" % (
            len(per), pct(col(0)), pct(col(1)), pct(col(2)),
            (gmean([1 + PAPER[a][2] / 100.0 for a in per]) - 1) * 100))

    print("## S6 perfect-memory runs vs D1 (PAPER_V100)\n")
    print("| workload | S6 IPC | D1 IPC | same |")
    print("|---|---|---|---|")
    for a, g in wl:
        x = stats(os.path.join(s6, a, g, B + "-PERFECT_MEM", "sim.out"))
        y = stats(os.path.join(pmem, a, g, B + "-PERFECT_MEM", "sim.out"))
        print("| %s / %s | %s | %s | %s |" % (
            app_of(a), g[:20], "%.4f" % x["ipc"] if x else "not run",
            "%.4f" % y["ipc"] if y else "-",
            "yes" if x and y and x["ipc"] == y["ipc"] and x["insn"] == y["insn"] else ("-" if not x else "**no**")))

    print("\n## Sanity: perfect-L1D runs (GLOBAL_ACC_R hits / accesses) and build tags\n")
    for a, g in wl:
        for c in (B, B + "-PAPER_LAT"):
            for d, suf in ((pl1d, "-PERFECT_L1D"), (pmem, "-PERFECT_MEM")):
                s = stats(os.path.join(d, a, g, c + suf, "sim.out"))
                if s and suf == "-PERFECT_L1D":
                    print("- %s / %s %s: hit %d / %d (%.2f%%), insn %d, %s" % (
                        app_of(a), g[:12], (c + suf)[len(B):], s["hit"], s["acc"],
                        100.0 * s["hit"] / max(s["acc"], 1), s["insn"], s["tags"]))


if __name__ == "__main__":
    main()
