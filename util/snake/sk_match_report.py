#!/usr/bin/env python3
"""SK match, Step 1 tables (docs/snake/results/SK_match_qv100.md).

    sk_match_report.py <skm runs> <skc root>

Base QV100 (LRR, 1132 MHz), 100M, 14 workloads (our 7 apps and SnakeCharmer's
b+tree, particlefilter float/naive, heartwall, streamcluster; no bfs).
<skm runs> holds the Step 1 configs, all SK_SC with same-CTA inter-warp
targets and SnakeCharmer's per-warp Head (SNAKE_HEAD64):
  SK        miss-queue reserve 8, decoupling mode 1
  SK_R0     reserve 0
  SK_R0_D0  reserve 0, decoupling off
<skc root> (SK_SC stage, same build) gives the no-prefetch baselines, our
Snake, the earlier SK_SC (cross-CTA targets, two-warp Head) and SK_SC_CTA
(same-CTA targets, two-warp Head; hotspot and srad_v1 only).
Metrics as in sk_sc_report.py (SPEC OQ-36), plus demand (GLOBAL_ACC_R)
MISS_QUEUE_FULL reservation fails vs the baseline.
Exit status 1 if a run did not finish or broke an invariant.
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4_report import parse  # noqa: E402
from sk_sc_report import metrics, short, build, pct, f1, fs, SEVEN, INPUT_DIFFERS  # noqa: E402

Q = "QV100-SASS-100M_INSN"
B = Q + "-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64"
CONFIGS = [  # label, source ("skm" or "skc"), config
    ("Snake", "skc", Q + "-SNAKE-SNAKE_FP1"),
    ("SK_SC (old)", "skc", Q + "-SNAKE-SNAKE_FP1-SK_SC"),
    ("SK_SC_CTA (old)", "skc", Q + "-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA"),
    ("SK", "skm", B),
    ("SK_R0", "skm", B + "-SNAKE_MQR0"),
    ("SK_R0_D0", "skm", B + "-SNAKE_MQR0-SNAKE_DM0"),
]
MAIN = ("Snake", "SK_SC (old)", "SK", "SK_R0", "SK_R0_D0")
SC_SK = {"on": 48.3, "early": 28.7, "late": 23.0, "acc": 23.1}  # A100, §II-B, Fig. 12
MQF = re.compile(r"Total_core_cache_fail_stats_breakdown\[GLOBAL_ACC_R\]"
                 r"\[MISS_QUEUE_FULL\] = (\d+)")


def mqf(path):
    m = MQF.findall(open(path, errors="replace").read())
    return int(m[-1]) if m else 0


def gmean(xs):
    xs = [x for x in xs if x == x]
    return (math.exp(sum(math.log1p(x / 100) for x in xs) / len(xs)) - 1) * 100 \
        if xs else float("nan")


def amean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    skm, skc = sys.argv[1], sys.argv[2]
    found = {}  # (app, args, label) -> path
    for p in glob.glob(os.path.join(skc, "*", "runs", "*", "*", "*", "sim.out")) + \
            glob.glob(os.path.join(skm, "*", "*", "*", "sim.out")):
        app, args, cfg = p.split(os.sep)[-4:-1]
        if app.startswith("bfs"):
            continue
        src = "skm" if p.startswith(os.path.abspath(skm)) else "skc"
        if cfg == Q and src == "skc":
            found[(app, args, "base")] = p
        for lab, s, c in CONFIGS:
            if cfg == c and src == s:
                found[(app, args, lab)] = p

    bad, builds, runs = 0, {}, {}
    for k, p in sorted(found.items()):
        s = parse(p)
        s["_mqf"] = mqf(p)
        ok = s["_done"] and (k[2] == "base" or (
            s.get("_pf_invariant") == "OK" and s.get("_funnel_invariant") == "OK"
            and s.get("_funnel_src_invariant") == "OK"
            and s.get("pf_woken_warps", 0) == 0))
        if not ok:
            print("CHECK FAILED: %s %s" % (" ".join(k), p))
            bad += 1
        builds.setdefault(build(p), []).append(k[2])
        runs[k] = s
    print("## Builds\n")
    for b, labs in sorted(builds.items()):
        print("- %s: %d runs (%s)" % (b, len(labs), ", ".join(sorted(set(labs)))))

    wl = sorted({(k[0], k[1]) for k in runs if k[2] == "SK"},
                key=lambda w: (w[0] not in SEVEN, w[0], w[1]))
    rows = {}  # label -> list of (set, values)
    print("\n## Per workload\n")
    print("| workload | config | IPC vs no prefetch | coverage_sc on-time | "
          "coverage_sc +late | accuracy_sc | on time | too early | too late | "
          "pf-induced evictions | demand MISS_QUEUE_FULL vs base |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for app, args in wl:
        base = runs.get((app, args, "base"))
        name = short(app, args) + (" †" if app.startswith(INPUT_DIFFERS) else "")
        grp = "7 apps" if app in SEVEN else "SnakeCharmer apps"
        print("| %s | base | | | | | | | | | %d |" % (name, base["_mqf"]))
        for lab, _, _ in CONFIGS:
            s = runs.get((app, args, lab))
            if s is None:
                continue
            m = metrics(s)
            sp = pct(s["_ipc"], base["_ipc"]) - 100
            dq = pct(s["_mqf"], base["_mqf"]) - 100 if base["_mqf"] else float("nan")
            v = dict(m, ipc=sp, mqf=s["_mqf"], mqf_base=base["_mqf"])
            rows.setdefault(lab, []).append((grp, v))
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, lab, fs(sp), f1(m["cov_on"]), f1(m["cov_late"]), f1(m["acc"]),
                f1(m["t_on"]), f1(m["t_early"]), f1(m["t_late"]), f1(m["pfev"]),
                "%d (%s)" % (s["_mqf"], fs(dq)) if dq == dq else "%d" % s["_mqf"]))

    print("\n## Means\n")
    print("Arithmetic over workloads (IPC: geometric mean of speedups; demand "
          "MISS_QUEUE_FULL: summed over the set vs the baselines' sum). "
          "Distance = |on time − 48.3| + |too early − 28.7| + |too late − 23.0| "
          "+ |accuracy − 23.1| (SnakeCharmer's SK, A100), in points.\n")
    print("| set | config | n | IPC vs no prefetch | coverage_sc on-time | "
          "coverage_sc +late | accuracy_sc | on time | too early | too late | "
          "pf-induced evictions | demand MISS_QUEUE_FULL vs base | distance to SK |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for grp in ("SnakeCharmer apps", "7 apps", "all"):
        for lab in MAIN:
            vs = [v for g, v in rows.get(lab, []) if grp == "all" or g == grp]
            if not vs:
                continue
            mean = lambda k: amean([v[k] for v in vs])
            q = sum(v["mqf"] for v in vs)
            qb = sum(v["mqf_base"] for v in vs)
            dist = abs(mean("t_on") - SC_SK["on"]) + abs(mean("t_early") - SC_SK["early"]) \
                + abs(mean("t_late") - SC_SK["late"]) + abs(mean("acc") - SC_SK["acc"])
            print("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                grp, lab, len(vs), fs(gmean([v["ipc"] for v in vs])), f1(mean("cov_on")),
                f1(mean("cov_late")), f1(mean("acc")), f1(mean("t_on")),
                f1(mean("t_early")), f1(mean("t_late")), f1(mean("pfev")),
                fs(pct(q, qb) - 100), f1(dist)))
    print("| SnakeCharmer | SK (A100, 14 apps) | 14 | | 11.5 | | 23.1 | 48.3 | 28.7 "
          "| 23.0 | | | 0.0 |")

    print("\n## Inter-warp training: per-warp Head (SK) vs two-warp Head\n")
    print("`interwarp_trained` counts training events (re-training after a Tail "
          "eviction counts again); `interwarp_trained_pcs` counts distinct "
          "(kernel launch, PC) pairs; `pred_inter_warp` the inter-warp predictions.\n")
    print("| workload | Head | config | interwarp_trained | interwarp_trained_pcs | "
          "pred_inter_warp | pf_issued_inter_warp |")
    print("|---|---|---|---|---|---|---|")
    for app, args in wl:
        name = short(app, args)
        for lab, head in (("SK_SC (old)", "two-warp"), ("SK_SC_CTA (old)", "two-warp"),
                          ("SK", "per-warp (64)")):
            s = runs.get((app, args, lab))
            if s is None:
                continue
            print("| %s | %s | %s | %d | %d | %d | %d |" % (
                name, head, lab, s.get("interwarp_trained", 0),
                s.get("interwarp_trained_pcs", 0), s.get("pred_inter_warp", 0),
                s.get("pf_issued_inter_warp", 0)))
    print("\n† input differs from SnakeCharmer Table II.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
