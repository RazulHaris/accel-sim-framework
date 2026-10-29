#!/usr/bin/env python3
"""SK_SC tables (docs/snake/results/SK_SC.md): Snake as SnakeCharmer's SK
baseline vs our Snake, with SnakeCharmer's metrics (SPEC OQ-36).

    sk_sc_report.py <skc root> <PAPER_V100 baseline runs>

<skc root> holds run dirs (sim_run_snake_skc/<set>/runs/<app>/<args>/<cfg>);
the PAPER_V100 no-prefetch runs of the 7 suite apps come from the cached
bin_ref_v2 outputs (<PAPER_V100 baseline runs>), every other baseline from
<skc root>.  Configs, per GPU setup (100M instructions):
  baseline   <prefix>
  Snake      <prefix>-SNAKE-SNAKE_FP1
  SK_SC      <prefix>-SNAKE-SNAKE_FP1-SK_SC
  SK_SC_CTA  <prefix>-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA
with prefix QV100-SASS-PAPER_V100-100M_INSN (PAPER_V100: GTO, 1530 MHz) or
QV100-SASS-100M_INSN (base QV100: LRR, 1132 MHz, SnakeCharmer's V100).

Metrics (32 B sectors, demand loads):
  coverage_sc   H / (H + M); H = on-time (+ late) uses, M = loads whose access
                returned MISS: new requests and loads merged into an MSHR
                entry, late uses included (debug plan S5: snake_sc_ld_miss
                already holds the late loads, so they are not added again)
  accuracy_sc   (on-time uses + fills a load had merged into) / prefetch fills
  timeliness    on-time / too-early / too-late among used prefetches; too
                early = first load to a sector evicted unused (same kernel)
  pf-induced    demand-filled victim sectors evicted by a prefetch allocation
                / all victim sectors
Exit status 1 if a run did not finish, broke an invariant, or carries
another build than the rest of its set.
"""
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4_report import parse  # noqa: E402

SETUPS = [("PAPER_V100", "QV100-SASS-PAPER_V100-100M_INSN"),
          ("QV100 (LRR, 1132 MHz)", "QV100-SASS-100M_INSN")]
VARIANTS = [("Snake", "-SNAKE-SNAKE_FP1"), ("SK_SC", "-SNAKE-SNAKE_FP1-SK_SC"),
            ("SK_SC_CTA", "-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA")]
SEVEN = ("backprop-rodinia-3.1", "hotspot-rodinia-3.1", "srad_v1-rodinia-3.1",
         "lud-rodinia-3.1", "nw-rodinia-3.1", "parboil-histo", "parboil-mri-q")
INPUT_DIFFERS = ("heartwall", "streamcluster")
# util/snake/scan_legacy_traces.py: global memory lines the tracer wrote
# wrongly that the parser fix cannot repair (SK_SC.md §1)
MISDECODED = {"bfs-rodinia-3.1": "24.8%", "streamcluster-rodinia-2.0-ft": "2.3%"}
TAG = re.compile(r"(accelsim-commit-[0-9a-f]+_modified_[0-9.]+)|"
                 r"(gpgpu-sim_git-commit-[0-9a-f]+_modified_[0-9.]+)")


def short(app, args):
    a = app.replace("-rodinia-3.1", "").replace("-rodinia-2.0-ft", "") \
           .replace("parboil-", "")
    if a == "hotspot":
        a += " " + args.split("_")[0]
    elif a == "lud":
        a += " 512" if "512" in args else " 256"
    return a


def build(path):
    t = open(path, errors="replace").read()
    return " ".join(sorted(set(m.group(0) for m in TAG.finditer(t))))


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def metrics(s):
    on, late, early = s["sc_ld_ontime"], s["sc_ld_late"], s["sc_ld_early"]
    miss, fills = s["sc_ld_miss"], s["sc_pf_fills"]
    used = on + late + early
    # fills a demand had merged into (pf_filled counts only the others); a
    # late prefetch still in flight at the end has not filled yet
    late_fills = fills - s["pf_filled"]
    return {
        # sc_ld_miss counts every MISS return, merges into an in-flight
        # prefetch (the late loads) included (debug plan S5 audit)
        "cov_on": pct(on, on + miss),
        "cov_late": pct(on + late, on + miss),
        "acc": pct(on + late_fills, fills),
        "t_on": pct(on, used), "t_early": pct(early, used),
        "t_late": pct(late, used),
        "pfev": pct(s["sc_victim_demand_by_pf"], s["sc_victim_sectors"]),
        "store_use": s["pf_useful_timely"] - on,  # first uses by non-loads
    }


def f1(x):
    return "-" if x != x else "%.1f" % x


def fs(x):
    return "-" if x != x else "%+.2f%%" % x


def main():
    root, refruns = sys.argv[1], sys.argv[2]
    runs = {}  # (setup, app, args, variant) -> (parsed, path)
    for p in glob.glob(os.path.join(root, "*", "runs", "*", "*", "*", "sim.out")):
        app, args, cfg = p.split(os.sep)[-4:-1]
        for sname, pre in SETUPS:
            if cfg == pre:
                runs[(sname, app, args, "base")] = p
            for vname, suf in VARIANTS:
                if cfg == pre + suf:
                    runs[(sname, app, args, vname)] = p
    for p in glob.glob(os.path.join(refruns, "*", "*", SETUPS[0][1], "sim.out")):
        app, args = p.split(os.sep)[-4:-2]
        runs.setdefault((SETUPS[0][0], app, args, "base"), p)

    bad = 0
    builds = {}
    for k, p in sorted(runs.items()):
        s = parse(p)
        ok = s["_done"] and (k[3] == "base" or (
            s.get("_pf_invariant") == "OK" and s.get("_funnel_invariant") == "OK"
            and s.get("_funnel_src_invariant") == "OK"
            and s.get("pf_woken_warps", 0) == 0))
        if not ok:
            print("CHECK FAILED: %s %s" % (" ".join(k), p))
            bad += 1
        builds.setdefault((k[3] == "base" and refruns in p, build(p)), []).append(p)
        runs[k] = s
    print("## Builds\n")
    for (cached, b), ps in sorted(builds.items()):
        print("- %s%s: %d runs" % (b, " (cached baseline)" if cached else "", len(ps)))
    if len(builds) > 2:
        print("\nMORE THAN ONE BUILD PER ROLE")
        bad += 1

    for sname, _ in SETUPS:
        wl = sorted({(k[1], k[2]) for k in runs if k[0] == sname and k[3] != "base"},
                    key=lambda w: (w[0] not in SEVEN, w[0], w[1]))
        if not wl:
            continue
        print("\n## %s\n" % sname)
        print("| workload | config | IPC vs no prefetch | coverage_sc on-time | "
              "coverage_sc +late | accuracy_sc | on time | too early | too late "
              "| pf-induced evictions |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        groups = {}
        for app, args in wl:
            base = runs.get((sname, app, args, "base"))
            name = short(app, args)
            if app.startswith(INPUT_DIFFERS):
                name += " †"
            if app in MISDECODED:
                name += " ‡"
            grp = "7 apps" if app in SEVEN else "SnakeCharmer apps"
            for vname, _ in VARIANTS:
                s = runs.get((sname, app, args, vname))
                if s is None:
                    continue
                m = metrics(s)
                sp = pct(s["_ipc"], base["_ipc"]) - 100 if base else float("nan")
                groups.setdefault((grp, vname), []).append((sp, m))
                groups.setdefault(("all", vname), []).append((sp, m))
                print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    name, vname, fs(sp), f1(m["cov_on"]), f1(m["cov_late"]),
                    f1(m["acc"]), f1(m["t_on"]), f1(m["t_early"]), f1(m["t_late"]),
                    f1(m["pfev"])))
                if m["store_use"]:
                    print("|  | (first uses by stores: %d) |" % m["store_use"])
        print("\nMeans (arithmetic over workloads; IPC: geometric mean of speedups):\n")
        print("| set | config | n | IPC vs no prefetch | coverage_sc on-time | "
              "coverage_sc +late | accuracy_sc | on time | too early | too late | "
              "pf-induced evictions |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for (grp, vname), rows in sorted(groups.items()):
            n = len(rows)
            sps = [r[0] for r in rows if r[0] == r[0]]
            g = (math.exp(sum(math.log1p(x / 100) for x in sps) / len(sps)) - 1) * 100 \
                if sps else float("nan")
            mean = lambda k: sum(r[1][k] for r in rows if r[1][k] == r[1][k]) / \
                max(1, sum(1 for r in rows if r[1][k] == r[1][k]))
            print("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                grp, vname, n, fs(g), f1(mean("cov_on")), f1(mean("cov_late")),
                f1(mean("acc")), f1(mean("t_on")), f1(mean("t_early")),
                f1(mean("t_late")), f1(mean("pfev"))))
    print("\n† input differs from SnakeCharmer Table II.")
    print("‡ trace partly mis-decoded after the parser fix (global memory "
          "lines): %s." % ", ".join("%s %s" % (a.split("-")[0], v)
                                     for a, v in sorted(MISDECODED.items())))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
