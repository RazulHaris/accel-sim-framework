#!/usr/bin/env python3
"""fix_today tables (docs/snake/results/fix_today.md): every config at Tier 1
conditions (PAPER_V100, 1B, our 7 apps) against Tier 1's no-prefetch runs.

    fix_report.py [<label> ...]

Labels: the CONFIGS below (default: all whose 9 runs exist).  Per config:
  - whole-run IPC gain per workload (gpu_tot_ipc / baseline - 1); 7-app
    GMEAN (an app's inputs combined first), lowest workload; success criteria
    (a) histo >= 0, (b) no workload below -1%, (c) GMEAN above Tier 1 Snake's,
    (d) timely share of used prefetches above Tier 1 Snake's
  - timely share = snake_pf_useful_timely / (timely + snake_pf_late): pooled
    over the 9 workloads and the mean over workloads (snake_sc_* is missing
    in Tier 1's build, so this is the measure all configs share)
  - per kernel for histo and srad_v1 (kernels with >= 1% of the baseline
    cycles; complete launches): IPC gain, prefetches issued per demand
    sector, unused share of issued, DRAM reads vs no prefetching, R2's
    prefetch-queue wait (mean, log2 buckets) and arbitration wins
  - the perfect-L1 ceiling (D1, sim_run_snake_gate/D1_pl1d)
Every run must have finished with the outcome and funnel invariants OK and
no warp woken by a prefetch; a failed check is listed and makes the exit
status 1.
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel_diag import per_kernel, log2_stats, kname  # noqa: E402
from sk_sc_report import short  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P1 = "QV100-SASS-PAPER_V100-1B_INSN"
S = P1 + "-SNAKE-SNAKE_FP1"
SK = S + "-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0"
FIX = "sim_run_snake_fix"
# label -> (run dir, config, description)
CONFIGS = [
    ("Tier 1 Snake", "sim_run_snake", S, "Snake, exact footprint (Tier 1)"),
    ("SK_R0_D0", "sim_run_snake_skm/v100p_1b", SK, "SK settings, same-CTA, per-warp Head, reserve 0, decoupling off"),
    ("A", FIX + "/p2", S + "-SNAKE_R2", "Tier 1 Snake + R2 (arb 0, mshr 0)"),
    ("B", FIX + "/p2", SK + "-SNAKE_R2", "SK_R0_D0 + R2 (arb 0, mshr 0)"),
    ("C", FIX + "/p2", SK + "-SNAKE_R2-SNAKE_PQARB1", "SK_R0_D0 + R2 (arb 1, mshr 0)"),
    ("D", FIX + "/p3", S + "-SNAKE_ACCF", "Tier 1 Snake + accuracy filter (OQ-39, 60%)"),
    ("E", FIX + "/p3", SK + "-SNAKE_ACCF", "SK_R0_D0 + accuracy filter (OQ-39, 60%)"),
]
BASE = ("sim_run_baseline", P1)
CEIL = ("sim_run_snake_gate/D1_pl1d", P1 + "-PERFECT_L1D")
T1_GMEAN_LABEL = "Tier 1 Snake"


def rd(p):
    return open(p, errors="replace").read()


def last(t, pat, conv=float):
    m = re.findall(pat, t, re.M)
    return conv(m[-1]) if m else None


def workloads():
    return sorted({tuple(p.split(os.sep)[-4:-2]) for p in glob.glob(
        os.path.join(ROOT, BASE[0], "runs", "*", "*", BASE[1], "sim.out"))})


def path(rdir, cfg, w):
    return os.path.join(ROOT, rdir, "runs", w[0], w[1], cfg, "sim.out")


def run(p, snake=True):
    """Whole-run numbers of one job, or None if missing / unfinished."""
    if not os.path.exists(p):
        return None
    t = rd(p)
    if "exit detected" not in t:
        return None
    r = {"ipc": last(t, r"^gpu_tot_ipc =\s*([\d.]+)"), "ok": True}
    if snake:
        for k in ("pf_useful_timely", "pf_late", "pf_issued", "pf_woken_warps", "demand_sector_accesses"):
            r[k] = last(t, r"^snake_%s = (\d+)" % k, int) or 0
        inv = all(last(t, r"^snake_%s = (\w+)" % k, str) == "OK"
                  for k in ("pf_invariant", "funnel_invariant", "funnel_src_invariant"))
        r["ok"] = inv and r["pf_woken_warps"] == 0
        tags = sorted(set(re.findall(r"(?:accelsim-commit|gpgpu-sim_git-commit)-[0-9a-f]+_modified_[0-9.]+", t[:200000])))
        r["tags"] = " ".join(tags)
        r["ok"] &= len(tags) == 2 and all("_modified_0.0" in x for x in tags)
    return r


def gm(xs):
    xs = [x for x in xs if x is not None]
    return (math.exp(sum(math.log1p(x / 100) for x in xs) / len(xs)) - 1) * 100 if xs else None


def gm_apps(per):
    if not per or any(v is None for v in per.values()):
        return None
    apps = {}
    for (a, _), v in per.items():
        apps.setdefault(a, []).append(v)
    return gm([gm(v) for v in apps.values()])


def f(x, d=2, sign=True):
    return "n/a" if x is None else (("%+.*f%%" if sign else "%.*f%%") % (d, x))


def main():
    WL = workloads()
    want = sys.argv[1:]
    cfgs = [c for c in CONFIGS if not want or c[0] in want]
    base = {w: run(path(BASE[0], BASE[1], w), snake=False) for w in WL}
    ceil = {w: run(path(CEIL[0], CEIL[1], w), snake=False) for w in WL}
    res, bad = {}, []
    for lab, rdir, cfg, _ in cfgs:
        rs = {w: run(path(rdir, cfg, w)) for w in WL}
        if all(r is None for r in rs.values()):
            continue
        for w, r in rs.items():
            if r is None:
                bad.append("%s %s: missing or unfinished" % (lab, short(*w)))
            elif not r["ok"]:
                bad.append("%s %s: failed check (%s)" % (lab, short(*w), r.get("tags")))
        res[lab] = rs
    gain = {lab: {w: ((r["ipc"] / base[w]["ipc"] - 1) * 100 if r and base[w] else None) for w, r in rs.items()}
            for lab, rs in res.items()}
    cg = {w: (ceil[w]["ipc"] / base[w]["ipc"] - 1) * 100 if ceil[w] and base[w] else None for w in WL}

    def tshare(rs):
        t = sum(r["pf_useful_timely"] for r in rs.values() if r)
        lt = sum(r["pf_late"] for r in rs.values() if r)
        per = [r["pf_useful_timely"] / (r["pf_useful_timely"] + r["pf_late"]) * 100
               for r in rs.values() if r and r["pf_useful_timely"] + r["pf_late"]]
        return (100.0 * t / (t + lt) if t + lt else None), (sum(per) / len(per) if per else None)

    labs = list(res)
    print("## Whole run (primary): IPC gain vs no prefetching, PAPER_V100, 1B\n")
    print("| workload | " + " | ".join(labs) + " | perfect-L1 ceiling |")
    print("|---|" + "---|" * (len(labs) + 1))
    for w in WL:
        print("| %s | %s | %s |" % (short(*w), " | ".join(f(gain[l][w]) for l in labs), f(cg[w], 1)))
    G = {l: gm_apps(gain[l]) for l in labs}
    print("| **GMEAN** (7 apps) | %s | %s |" % (" | ".join("**%s**" % f(G[l]) for l in labs), f(gm_apps(cg), 1)))
    print("\n## Success criteria\n")
    t1 = G.get(T1_GMEAN_LABEL)
    t1ts = tshare(res[T1_GMEAN_LABEL])[0] if T1_GMEAN_LABEL in res else None
    print("| config | (a) histo ≥ 0 | (b) lowest workload (≥ −1%) | (c) GMEAN > Tier 1 Snake | "
          "(d) timely share of used: pooled / mean over workloads | all four |")
    print("|---|---|---|---|---|---|")
    hw = [w for w in WL if w[0] == "parboil-histo"][0]
    for l in labs:
        h = gain[l][hw]
        lo_w = min((w for w in WL if gain[l][w] is not None), key=lambda w: gain[l][w], default=None)
        lo = gain[l][lo_w] if lo_w else None
        pooled, mean = tshare(res[l])
        ca = h is not None and h >= 0
        cb = lo is not None and lo >= -1.0
        cc = G[l] is not None and t1 is not None and G[l] > t1
        cd = pooled is not None and t1ts is not None and pooled > t1ts
        print("| %s | %s %s | %s (%s) %s | %s %s | %s / %s %s | %s |" % (
            l, f(h), "yes" if ca else "no", f(lo), short(*lo_w) if lo_w else "-", "yes" if cb else "no",
            f(G[l]), "yes" if cc else ("-" if l == T1_GMEAN_LABEL else "no"),
            f(pooled, 1, False), f(mean, 1, False), "yes" if cd else ("-" if l == T1_GMEAN_LABEL else "no"),
            "**yes**" if ca and cb and cc and cd else "no"))

    print("\n## Per kernel: histo and srad_v1 (complete launches, kernels with ≥ 1% of the baseline cycles)\n")
    print("Columns per config: IPC gain / prefetches issued per demand sector / unused share of issued / DRAM "
          "reads vs no prefetching / R2 prefetch-queue wait mean (cycles) / arbitration wins demand:prefetch.\n")
    for app in ("parboil-histo", "srad_v1-rodinia-3.1"):
        w = [x for x in WL if x[0] == app][0]
        bk = per_kernel(path(BASE[0], BASE[1], w))
        tot = sum(v["_cyc"] for v in bk.values())
        ks = [k for k in sorted(bk, key=lambda k: -bk[k]["_cyc"]) if bk[k]["_cyc"] >= 0.01 * tot]
        print("| %s kernel (share) | config | IPC gain | issued / demand | unused % | DRAM × | pq wait | wins d:p |"
              % short(*w))
        print("|---|---|---|---|---|---|---|---|")
        cache = {}
        for lab, rdir, cfg, _ in cfgs:
            if lab in res and res[lab][w]:
                cache[lab] = per_kernel(path(rdir, cfg, w))
        for k in ks:
            b = bk[k]
            for lab in labs:
                d = cache.get(lab, {}).get(k)
                if d is None:
                    continue
                g = ((d["_ins"] / d["_cyc"]) / (b["_ins"] / b["_cyc"]) - 1) * 100
                iss, dem = d.get("snake_pf_issued", 0), d.get("snake_demand_sector_accesses", 0)
                un = iss - d.get("snake_pf_useful_timely", 0) - d.get("snake_pf_late", 0) - d.get("snake_pf_evicted_unused", 0)
                pw = log2_stats(d.get("hist_pq_wait"))[0]
                wins = "%d:%d" % (d.get("snake_pq_win_demand", 0), d.get("snake_pq_win_prefetch", 0)) \
                    if d.get("snake_pq_sent", 0) else "-"
                print("| `%s` (%.1f%%) | %s | %s | %s | %s | %s | %s | %s |" % (
                    kname(k), 100.0 * b["_cyc"] / tot, lab, f(g), "%.2f" % (iss / dem) if dem else "-",
                    "%.0f" % (100.0 * un / iss) if iss else "-",
                    "%.2f" % (d["dram_rd"] / b["dram_rd"]) if b["dram_rd"] else "-",
                    "%.1f" % pw if pw is not None else "-", wins))
        print()
    print("## Configs\n")
    for lab, rdir, cfg, desc in cfgs:
        if lab in res:
            tags = sorted({r["tags"] for r in res[lab].values() if r})
            print("- **%s**: %s. `%s` in `%s`; build %s" % (lab, desc, cfg, rdir, "; ".join(tags)))
    if bad:
        print("\nCHECKS FAILED:")
        for b_ in bad:
            print("- " + b_)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
