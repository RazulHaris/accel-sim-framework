#!/usr/bin/env python3
"""Snake final evaluation vs the paper (docs/snake/SPEC.md §5), S5.

    compare_paper.py [--insn 1B_INSN] <run dir> [<run dir> ...]

Looks up each config below in the given run dirs (first match wins; jobs are
<run dir>/runs/<app>/<args>/<config>/sim.out, parsed by collect_results.py)
and prints markdown tables. A missing job shows as "-".
  1. Per app: coverage (paper vs our coverage_pred and coverage_paper),
     accuracy (paper vs accuracy_paper), IPC gain and energy, Snake exact
     footprint vs the baseline; plus the same averages for Snake with the line
     footprint and for PAPER_LAT.
  2. Demand L1 hit rate (Fig. 25, §5.7): baseline, Snake, Snake-ISO.
  3. Ablations (§5.2): IPC gain of Snake, Snake-T, Snake-DT, s-Snake.
  4. Fig. 20 (Tail size), Fig. 22 (eviction policy x Tail size), Fig. 23
     (space-throttle cycles): coverage and accuracy averages.
  5. Tier 2: each sensitivity config vs Snake exact, and the 1132 MHz pair.
Averages: per app first (hotspot and lud have two inputs), then the
arithmetic mean over apps for rates and the geometric mean for IPC and energy
ratios.  The paper's averages are recomputed over the same 7 apps.
Energy (Fig. 19) needs AccelWattch output (collect_results.py); without it
only Snake's own energy (OQ-16) is shown and the normalized energy is
"unavailable".
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_results import parse_job  # noqa: E402

# SPEC §5 (bar charts, ±3 pp): coverage, accuracy, IPC gain %, energy norm.
PAPER = {"backprop": (83, 68, 1, 0.92), "hotspot": (88, 85, 13, 0.86),
         "srad_v1": (87, 83, 29, 0.71), "lud": (91, 82, 18, 0.82),
         "nw": (12, 10, 5, 0.95), "histo": (97, 96, 33, 0.64),
         "mri-q": (72, 60, 4, 0.96)}
APPS = ["backprop", "hotspot", "srad_v1", "lud", "nw", "histo", "mri-q"]


def app_of(dirname):
    return dirname.replace("-rodinia-3.1", "").replace("parboil-", "")


class Runs:
    def __init__(self, dirs):
        self.dirs, self.cache = dirs, {}
        self.workloads = sorted({(a, g) for d in dirs
                                 for a in _ls(os.path.join(d, "runs"))
                                 for g in _ls(os.path.join(d, "runs", a))
                                 if app_of(a) in PAPER})

    def get(self, app, args, cfg):
        k = (app, args, cfg)
        if k not in self.cache:
            self.cache[k] = None
            for d in self.dirs:
                r = parse_job(os.path.join(d, "runs", app, args, cfg))
                if r and r["done"]:
                    self.cache[k] = r
                    break
        return self.cache[k]


def _ls(p):
    return sorted(os.listdir(p)) if os.path.isdir(p) else []


def rates(r):
    d = r.get("demand_sector_accesses") or 0
    if not d:
        return None, None, None
    return (100.0 * r["demand_pred_matched"] / d,
            100.0 * (r["pf_useful_timely"] + r["pf_late"]) / d,
            100.0 * r["pf_useful_timely"] / d)


def gain(r, b):
    return r["ipc"] / b["ipc"] if r and b and r["ipc"] and b["ipc"] else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def f(x, fmt="%.1f"):
    return "-" if x is None else fmt % x


def main():
    argv = sys.argv[1:]
    insn = "1B_INSN"
    if argv[:1] == ["--insn"]:
        insn, argv = argv[1], argv[2:]
    runs = Runs(argv)
    B = "QV100-SASS-PAPER_V100-" + insn
    S = B + "-SNAKE-SNAKE_FP1"
    C = {"base": B, "snake": S, "line": B + "-SNAKE", "iso": B + "-SNAKE_ISO-SNAKE_FP1",
         "dt": B + "-SNAKE_DT-SNAKE_FP1", "t": B + "-SNAKE_T-SNAKE_FP1",
         "ssnake": B + "-S_SNAKE-SNAKE_FP1", "lat_base": B + "-PAPER_LAT",
         "lat_snake": B + "-PAPER_LAT-SNAKE-SNAKE_FP1",
         "clk_base": "QV100-SASS-PAPER_V100_1132-" + insn,
         "clk_snake": "QV100-SASS-PAPER_V100_1132-" + insn + "-SNAKE-SNAKE_FP1"}
    g = runs.get

    # ---- 1. per app
    print("## 1. Ours vs the paper, per app (Snake, exact footprint, vs the baseline)\n")
    print("| app / input | paper coverage | coverage_pred | coverage_paper | paper accuracy | accuracy_paper | paper IPC gain | IPC gain | paper energy (norm.) | energy (norm.) | Snake's own energy (mJ) |")
    print("|---" * 11 + "|")
    agg = {}
    for a, gg in runs.workloads:
        app = app_of(a)
        b, s = g(a, gg, C["base"]), g(a, gg, C["snake"])
        cp, cv, ac = rates(s) if s else (None, None, None)
        ig = gain(s, b)
        en = None
        if s and b and s["gpu_energy_j"] is not None and b["gpu_energy_j"]:
            en = (s["gpu_energy_j"] + s["snake_energy_j"]) / b["gpu_energy_j"]
        agg.setdefault(app, []).append((cp, cv, ac, ig, en))
        p = PAPER[app]
        print("| %s / %s | %d | %s | %s | %d | %s | %+d%% | %s | %.2f | %s | %s |" % (
            app, gg[:24], p[0], f(cp), f(cv), p[1], f(ac), p[2],
            f((ig - 1) * 100 if ig else None, "%+.1f%%"), p[3],
            f(en, "%.2f") if en is not None else ("unavailable" if s else "-"),
            f(s["snake_energy_j"] * 1e3 if s and s["snake_energy_j"] is not None else None, "%.3f")))

    def avg_row(label, data, paper=True):
        # only apps with a run, so the paper's average covers the same apps
        data = {a: v for a, v in data.items() if any(x is not None for t in v for x in t)}
        cols = list(zip(*[tuple(mean([v[i] for v in data[app]]) if i < 3 else
                                gmean([v[i] for v in data[app]]) for i in range(5))
                          for app in data])) if data else [[]] * 5
        pc = [PAPER[a] for a in data]
        print("| **%s** (%d apps) | %s | %s | %s | %s | %s | %s | %s | %s | %s | |" % (
            label, len(data),
            f(mean([p[0] for p in pc])) if paper else "", f(mean(cols[0])), f(mean(cols[1])),
            f(mean([p[1] for p in pc])) if paper else "", f(mean(cols[2])),
            f((gmean([1 + p[2] / 100.0 for p in pc]) - 1) * 100, "%+.1f%%") if paper else "",
            f((gmean(cols[3]) - 1) * 100 if gmean(cols[3]) else None, "%+.1f%%"),
            f(gmean([p[3] for p in pc]), "%.2f") if paper else "",
            f(gmean(cols[4]), "%.2f") if gmean(cols[4]) else "unavailable"))
    if agg:
        avg_row("Average", agg)
    for label, sk, bk in (("Snake, line footprint", "line", "base"),
                          ("PAPER_LAT: Snake vs PAPER_LAT baseline", "lat_snake", "lat_base")):
        d = {}
        for a, gg in runs.workloads:
            b, s = g(a, gg, C[bk]), g(a, gg, C[sk])
            cp, cv, ac = rates(s) if s else (None, None, None)
            d.setdefault(app_of(a), []).append((cp, cv, ac, gain(s, b), None))
        avg_row(label, d, paper=False)

    # ---- 2. L1 hit rate
    print("\n## 2. Demand L1 hit rate (Fig. 25, §5.7; paper averages: baseline 45, Snake 79, Isolated-Snake 84)\n")
    print("| app / input | baseline | Snake | Snake-ISO | Snake-DT |")
    print("|---" * 5 + "|")
    cols = {k: {} for k in ("base", "snake", "iso", "dt")}
    for a, gg in runs.workloads:
        vals = []
        for k in cols:
            r = g(a, gg, C[k])
            v = r["l1_hit_rate"] if r else None
            cols[k].setdefault(app_of(a), []).append(v)
            vals.append(f(v))
        print("| %s / %s | %s |" % (app_of(a), gg[:24], " | ".join(vals)))
    print("| **Average** (apps) | %s |" % " | ".join(
        f(mean([mean(v) for v in cols[k].values()])) for k in cols))

    # ---- 3. ablations
    print("\n## 3. Ablations: IPC gain vs the baseline (§5.2: Snake beats Snake-DT by 13% and Snake-T by 7%)\n")
    abl = ["snake", "t", "dt", "ssnake", "iso"]
    names = {"snake": "Snake", "t": "Snake-T", "dt": "Snake-DT", "ssnake": "s-Snake", "iso": "Snake-ISO"}
    print("| app / input | " + " | ".join(names[k] for k in abl) + " |")
    print("|---" * (len(abl) + 1) + "|")
    gains = {k: {} for k in abl}
    for a, gg in runs.workloads:
        b = g(a, gg, C["base"])
        row = []
        for k in abl:
            x = gain(g(a, gg, C[k]), b)
            gains[k].setdefault(app_of(a), []).append(x)
            row.append(f((x - 1) * 100 if x else None, "%+.1f%%"))
        print("| %s / %s | %s |" % (app_of(a), gg[:24], " | ".join(row)))
    gm = {k: gmean([gmean(v) for v in gains[k].values()]) for k in abl}
    print("| **GMEAN** (apps) | %s |" % " | ".join(
        f((gm[k] - 1) * 100 if gm[k] else None, "%+.1f%%") for k in abl))
    if gm["snake"] and gm["dt"] and gm["t"]:
        print("\nSnake vs Snake-DT: %+.1f%%; vs Snake-T: %+.1f%% (paper +13%%, +7%%)." % (
            (gm["snake"] / gm["dt"] - 1) * 100, (gm["snake"] / gm["t"] - 1) * 100))

    # ---- 4. sweeps
    def sweep(title, items):
        print("\n## %s\n" % title)
        print("| config | coverage_pred | coverage_paper | accuracy_paper | IPC gain (GMEAN) | demand L1 hit | apps |")
        print("|---" * 7 + "|")
        for label, cfg, bcfg in items:
            per = {}
            for a, gg in runs.workloads:
                r, b = g(a, gg, cfg), g(a, gg, C[bcfg] if bcfg in C else bcfg)
                if not r:
                    continue
                per.setdefault(app_of(a), []).append(rates(r) + (gain(r, b), r["l1_hit_rate"]))
            m = [mean([mean([x[i] for x in v]) for v in per.values()]) for i in (0, 1, 2, 4)]
            gi = gmean([gmean([x[3] for x in v]) for v in per.values()])
            print("| %s | %s | %s | %s | %s | %s | %d |" % (
                label, f(m[0]), f(m[1]), f(m[2]),
                f((gi - 1) * 100 if gi else None, "%+.1f%%"), f(m[3]), len(per)))
    sweep("4a. Fig. 20: Tail entries (paper: Tail 10 costs 8% coverage vs unlimited)",
          [("Tail 10 (default)", S, "base"), ("Tail 20", S + "-SNAKE_TAIL20", "base"),
           ("Tail 40", S + "-SNAKE_TAIL40", "base"), ("Tail unlimited (1024)", S + "-SNAKE_TAILINF", "base")])
    sweep("4b. Fig. 22: Tail eviction policy x Tail entries",
          [("lru_popcount, 10", S, "base"), ("lru_popcount, 20", S + "-SNAKE_TAIL20", "base"),
           ("lru_popcount, 40", S + "-SNAKE_TAIL40", "base"), ("popcount, 10", S + "-SNAKE_EVPOP", "base"),
           ("popcount, 20", S + "-SNAKE_EVPOP-SNAKE_TAIL20", "base"),
           ("popcount, 40", S + "-SNAKE_EVPOP-SNAKE_TAIL40", "base")])
    sweep("4c. Fig. 23: space-throttle halt (paper: 50 cycles gives 75% accuracy at 2% coverage loss)",
          [("%d cycles%s" % (n, " (default)" if n == 50 else ""),
            S if n == 50 else S + "-SNAKE_THR%d" % n, "base") for n in (25, 50, 75, 100, 150, 200, 300)])
    # ---- 5. tier 2
    sweep("5. Tier 2: sensitivity to our interpretations (one change from Snake exact)",
          [("Snake exact (reference)", S, "base")] +
          [(lab, S + "-" + tok, "base") for lab, tok in (
              ("interwarp_degree 1", "SNAKE_IWD1"), ("interwarp_degree 2", "SNAKE_IWD2"),
              ("interwarp_degree 8", "SNAKE_IWD8"), ("missq_reserve 4", "SNAKE_MQR4"),
              ("pf_stale_cycles 200", "SNAKE_STALE200"), ("pf_stale_cycles 1000", "SNAKE_STALE1000"),
              ("bandwidth in bytes", "SNAKE_BWBYTES"), ("interwarp_chain_depth 3", "SNAKE_IWC3"),
              ("decouple_variant 0", "SNAKE_DV0"), ("queue_size 64", "SNAKE_Q64"),
              ("issue_width 2", "SNAKE_ISSUE2"))] +
          [("1132 MHz: Snake vs 1132 MHz baseline", C["clk_snake"], "clk_base")])


if __name__ == "__main__":
    main()
