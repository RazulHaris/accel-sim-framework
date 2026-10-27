#!/usr/bin/env python3
"""Best Snake (SK_R0_D0) vs Tier 1's Snake: whole run, best kernel, and the
window / setup / config decomposition (docs/snake/results/SK_match_qv100.md).

    kernel_gap.py

No simulations; existing runs only (paths below, relative to the accel-sim
root).  Our 7 apps, 9 workloads:
  T1         Tier 1 Snake (exact footprint), PAPER_V100, 1B: sim_run_snake vs
             sim_run_baseline
  SK 1B      SK_R0_D0, PAPER_V100, 1B: sim_run_snake_skm/v100p_1b vs
             sim_run_baseline (Tier 1's baseline)
  P100       Snake, PAPER_V100, 100M: sim_run_snake_skc/v100p_7apps vs
             sim_run_snake_refcache/bin_ref_v2
  Q100       Snake, base QV100, 100M: sim_run_snake_skc/qv100_7apps
  SK Q100    SK_R0_D0, base QV100, 100M: sim_run_snake_skm/qv100 vs
             sim_run_snake_skc/qv100_7apps
  "win"      a 1B run restricted to the launches that start within its first
             100M instructions (complete launches; at 100M the last one is
             cut by the cap)
Whole-run gain = gpu_tot_ipc / baseline - 1.  Per kernel (as per_kernel.py,
K1): launches pair in order, a kernel = all launches of a function, IPC =
instructions / cycles; best kernel = largest gain among kernels with >= 1% of
the baseline cycles.  At 1B the capped last launch is excluded (K1); at 100M
it is kept and marked '*' (it is the only launch of backprop and hotspot 1024
and holds mri-q's and srad_v1's main kernels).  Best-kernel numbers are
diagnostic only.  GMEAN over the 7 apps (an app's inputs combined first,
geometric), as in final_tier1.md and K1; "9 wl" = over the 9 workloads.
Reference: SK_R0_D0's 14 workloads on base QV100 at 100M (SK match), with and
without particlefilter_naive.  A missing or unfinished run gives "n/a".
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_kernel import launches, deltas  # noqa: E402
from sk_sc_report import short, SEVEN  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P1, P, Q = "QV100-SASS-PAPER_V100-1B_INSN", "QV100-SASS-PAPER_V100-100M_INSN", "QV100-SASS-100M_INSN"
S = "-SNAKE-SNAKE_FP1"
SK = S + "-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0"
WIN = 100000000
NAIVE = "particlefilter_naive-rodinia-3.1"
# step -> (baseline runs, baseline config, prefetch runs, prefetch config, window only)
STEPS = {
    "T1": ("sim_run_baseline/runs", P1, "sim_run_snake/runs", P1 + S, False),
    "T1 win": ("sim_run_baseline/runs", P1, "sim_run_snake/runs", P1 + S, True),
    "P100": ("sim_run_snake_refcache/bin_ref_v2/runs", P, "sim_run_snake_skc/v100p_7apps/runs", P + S, False),
    "Q100": ("sim_run_snake_skc/qv100_7apps/runs", Q, "sim_run_snake_skc/qv100_7apps/runs", Q + S, False),
    "SK 1B": ("sim_run_baseline/runs", P1, "sim_run_snake_skm/v100p_1b/runs", P1 + SK, False),
    "SK 1B win": ("sim_run_baseline/runs", P1, "sim_run_snake_skm/v100p_1b/runs", P1 + SK, True),
    "SK Q100": ("sim_run_snake_skc/qv100_7apps/runs", Q, "sim_run_snake_skm/qv100/runs", Q + SK, False),
}
LABEL = {
    "T1": "Tier 1 Snake: PAPER_V100, 1B",
    "T1 win": "Tier 1 Snake, launches started in the first 100M",
    "P100": "Snake: PAPER_V100, 100M",
    "Q100": "Snake: base QV100, 100M",
    "SK 1B": "SK_R0_D0: PAPER_V100, 1B",
    "SK 1B win": "SK_R0_D0 1B, launches started in the first 100M",
    "SK Q100": "SK_R0_D0: base QV100, 100M",
}


def kname(m):
    g = re.match(r"_Z(\d+)", m)
    return m[g.end():g.end() + int(g.group(1))] if g else m


def done(p):
    return os.path.exists(p) and "exit detected" in open(p, errors="replace").read()


def tot_ipc(p):
    m = re.findall(r"gpu_tot_ipc =\s*([\d.]+)", open(p, errors="replace").read())
    return float(m[-1])


def gm(xs):
    xs = [x for x in xs if x is not None]
    return (math.exp(sum(math.log1p(x / 100) for x in xs) / len(xs)) - 1) * 100 if xs else None


def gm_apps(per_wl):
    """GMEAN over apps, an app's inputs combined first; None if any is missing."""
    if not per_wl or any(v is None for v in per_wl.values()):
        return None
    apps = {}
    for (a, _), v in per_wl.items():
        apps.setdefault(a, []).append(v)
    return gm([gm(v) for v in apps.values()])


def f(x, d=2):
    return "n/a" if x is None else "%+.*f%%" % (d, x)


class Pair:
    """A baseline / prefetch run pair of one workload (sim.out paths).  win:
    only the launches that start within the first 100M instructions; cut: drop
    the launch cut by the instruction cap (1B, K1) instead of keeping it."""

    def __init__(self, bp, sp, win=False, cut=False):
        self.bp, self.sp, self.win = bp, sp, win
        self.ok = done(bp) and done(sp)
        if not self.ok:
            return
        (lb, cb), (ls, cs) = launches(bp), launches(sp)
        self.db, self.ds = deltas(lb), deltas(ls)
        n = min(len(self.db), len(self.ds))
        assert all(self.db[i][1] == self.ds[i][1] for i in range(n)), (bp, sp)
        capped = cb or cs or len(self.db) != len(self.ds)
        if win:
            ins, k = 0, 0
            while k < n and ins < WIN:
                ins += self.db[k][3]
                k += 1
            self.idx, self.part = list(range(k)), None
        else:
            self.idx = list(range(n - (1 if capped and cut else 0)))
            self.part = n - 1 if capped and not cut else None

    @classmethod
    def step(cls, st, a, g):
        bd, bc, sd, sc, win = STEPS[st]
        return cls(os.path.join(ROOT, bd, a, g, bc, "sim.out"), os.path.join(ROOT, sd, a, g, sc, "sim.out"),
                   win=win, cut=bc == P1)

    def whole(self):
        if not self.ok:
            return None
        if self.win:
            sum_ = lambda d, j: sum(d[i][j] for i in self.idx)
            return ((sum_(self.ds, 3) / sum_(self.ds, 2)) / (sum_(self.db, 3) / sum_(self.db, 2)) - 1) * 100
        return (tot_ipc(self.sp) / tot_ipc(self.bp) - 1) * 100

    def kernel(self, name):
        """(gain, launches, partial, share of baseline cycles) of one kernel."""
        if not self.ok:
            return None
        ix = [i for i in self.idx if self.db[i][1] == name]
        cb, cs = sum(self.db[i][2] for i in ix), sum(self.ds[i][2] for i in ix)
        if not ix or not cb or not cs:
            return None
        ib, is_ = sum(self.db[i][3] for i in ix), sum(self.ds[i][3] for i in ix)
        tot = sum(self.db[i][2] for i in self.idx)
        return ((is_ / cs) / (ib / cb) - 1) * 100, len(ix), self.part in ix, 100.0 * cb / tot

    def best(self):
        if not self.ok:
            return None, None
        c = [(k[0], k[3], nm) for nm in {self.db[i][1] for i in self.idx}
             for k in [self.kernel(nm)] if k and k[3] >= 1.0]
        return (max(c)[2], max(c)[0]) if c else (None, None)  # ties: the larger share


def workloads():
    return sorted({tuple(p.split(os.sep)[-4:-2]) for p in glob.glob(
        os.path.join(ROOT, "sim_run_baseline", "runs", "*", "*", P1, "sim.out"))})


def main():
    WL = workloads()
    pairs = {(st, w): Pair.step(st, *w) for st in STEPS for w in WL}
    whole = {st: {w: pairs[(st, w)].whole() for w in WL} for st in STEPS}
    bestk = {st: {w: pairs[(st, w)].best() for w in WL} for st in STEPS}
    bkg = {st: {w: bestk[st][w][1] for w in WL} for st in STEPS}
    name = lambda w: short(*w)

    print("## 1. Whole run (primary): SK_R0_D0 vs Tier 1's Snake, PAPER_V100, 1B\n")
    print("IPC gain vs Tier 1's no-prefetch run (gpu_tot_ipc). particlefilter_naive is not a Tier 1 workload, "
          "so the split with / without it applies only to the 14-workload reference (§4).\n")
    print("| workload | Tier 1 Snake | SK_R0_D0 | difference |")
    print("|---|---|---|---|")
    for w in WL:
        a, b = whole["T1"][w], whole["SK 1B"][w]
        print("| %s | %s | %s | %s |" % (name(w), f(a), f(b), "n/a" if None in (a, b) else "%+.2f pp" % (b - a)))
    for lab, fn in (("**GMEAN** (7 apps)", gm_apps), ("GMEAN (9 workloads)", lambda d: gm(list(d.values()))
                                                      if None not in d.values() else None)):
        a, b = fn(whole["T1"]), fn(whole["SK 1B"])
        print("| %s | %s | %s | %s |" % (lab, f(a), f(b), "n/a" if None in (a, b) else "%+.2f pp" % (b - a)))

    print("\n## 2. Best kernel (diagnostic only): SK_R0_D0 vs Tier 1's Snake, PAPER_V100, 1B\n")
    print("| workload | Tier 1 Snake: best kernel (share of cycles) | gain | SK_R0_D0: best kernel (share) | gain |")
    print("|---|---|---|---|---|")
    for w in WL:
        cells = []
        for st in ("T1", "SK 1B"):
            nm, g = bestk[st][w]
            k = pairs[(st, w)].kernel(nm) if nm else None
            cells += ["`%s` (%.1f%%)" % (kname(nm), k[3]) if k else "n/a", f(g)]
        print("| %s | %s |" % (name(w), " | ".join(cells)))
    print("| **GMEAN** (7 apps) | | %s | | %s |" % (f(gm_apps(bkg["T1"])), f(gm_apps(bkg["SK 1B"]))))

    print("\n## 3. Window, setup and config\n")
    print("GMEAN over the 7 apps at each step (best kernel chosen per step).\n")
    print("| step | whole run | best kernel (diagnostic) |")
    print("|---|---|---|")
    G = {st: (gm_apps(whole[st]), gm_apps(bkg[st])) for st in STEPS}
    for st in STEPS:
        print("| %s | %s | %s |" % (LABEL[st], f(G[st][0]), f(G[st][1])))
    same = [w for w in WL if pairs[("T1 win", w)].ok and pairs[("P100", w)].ok and all(
        pairs[("T1 win", w)].db[i][2:4] == pairs[("P100", w)].db[i][2:4]
        and pairs[("T1 win", w)].ds[i][2:4] == pairs[("P100", w)].ds[i][2:4]
        for i in pairs[("T1 win", w)].idx[:-1])]
    print("\nCheck: inside the first 100M, every complete launch (cycles, instructions) of Tier 1's baseline and "
          "Snake runs equals that of the 100M runs on %d of %d workloads (the last launch is cut at 100M)."
          % (len(same), len(WL)))
    print("\nEffects, in points (whole run / best kernel):\n")
    d = lambda x, y, i: "n/a" if G[x][i] is None or G[y][i] is None else "%+.2f" % (G[y][i] - G[x][i])
    for lab, x, y in (("window, Tier 1 Snake (T1 -> T1 win)", "T1", "T1 win"),
                      ("window vs 100M runs (T1 win -> P100; complete launches identical)", "T1 win", "P100"),
                      ("setup, Snake (P100 -> Q100)", "P100", "Q100"),
                      ("config at base QV100, 100M (Q100 -> SK Q100)", "Q100", "SK Q100"),
                      ("config at Tier 1 conditions (T1 -> SK 1B)", "T1", "SK 1B"),
                      ("window, SK_R0_D0 (SK 1B -> SK 1B win)", "SK 1B", "SK 1B win"),
                      ("setup, SK_R0_D0 (SK 1B win -> SK Q100)", "SK 1B win", "SK Q100")):
        print("- %s: %s / %s" % (lab, d(x, y, 0), d(x, y, 1)))

    print("\n### Per kernel (diagnostic only)\n")
    print("Kernel IPC gain of each workload's best kernels at every step; '*' = includes the launch cut by the "
          "100M cap; launches = at 1B / in the first 100M.\n")
    cols = list(STEPS)
    print("| workload | kernel (best in) | launches | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for w in WL:
        names = []
        for st in ("T1", "SK 1B", "SK Q100"):
            nm = bestk[st][w][0]
            if nm and nm not in names:
                names.append(nm)
        for nm in names:
            where = ", ".join(st for st in ("T1", "SK 1B", "SK Q100") if bestk[st][w][0] == nm)
            vals = []
            for st in cols:
                k = pairs[(st, w)].kernel(nm)
                vals.append("n/a" if k is None else "%+.2f%s" % (k[0], "*" if k[2] else ""))
            n1, nw = pairs[("T1", w)].kernel(nm), pairs[("T1 win", w)].kernel(nm)
            print("| %s | `%s` (%s) | %s/%s | %s |" % (name(w), kname(nm), where, n1[1] if n1 else 0,
                                                       nw[1] if nw else 0, " | ".join(vals)))

    # ---- reference: the 14 SK match workloads (base QV100, 100M)
    print("\n## 4. Reference: SK_R0_D0 on its 14 workloads (base QV100, 100M)\n")
    wh, bk = {}, {}
    for sp in glob.glob(os.path.join(ROOT, "sim_run_snake_skm/qv100/runs/*/*", Q + SK, "sim.out")):
        a, g = sp.split(os.sep)[-4:-2]
        bp = glob.glob(os.path.join(ROOT, "sim_run_snake_skc/*/runs", a, g, Q, "sim.out"))
        if not bp or not done(sp) or not done(bp[0]):
            wh[(a, g)] = bk[(a, g)] = None
            continue
        wh[(a, g)] = (tot_ipc(sp) / tot_ipc(bp[0]) - 1) * 100
        pr = Pair(bp[0], sp)
        bk[(a, g)] = pr.best()[1]
    print("| GMEAN over workloads | n | whole run | best kernel (diagnostic) |")
    print("|---|---|---|---|")
    for lab, keep in (("all", lambda w: True), ("without particlefilter_naive", lambda w: w[0] != NAIVE),
                      ("our 7 apps", lambda w: w[0] in SEVEN), ("SnakeCharmer apps", lambda w: w[0] not in SEVEN)):
        ws = [w for w in wh if keep(w)]
        print("| %s | %d | %s | %s |" % (lab, len(ws), f(gm([wh[w] for w in ws])), f(gm([bk[w] for w in ws]))))


if __name__ == "__main__":
    main()
