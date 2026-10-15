#!/usr/bin/env python3
"""Snake S4b tables (docs/snake/results/S4.md, S4b sections).

    s4b_report.py <S4b runs> <S4b diag runs> <S4 runs> <Snake-DT runs>
                  <reference runs> [<reference config>]

Runs, by config (exact footprint, 100M):
  S4b: SNAKE-SNAKE_FP1 and SNAKE_T-SNAKE_FP1 (variant 1, the default)
  S4b diag: SNAKE-SNAKE_FP1-SNAKE_DV0 (Snake, variant 0, with the S4b
       counters)
  S4:  SNAKE_ISO-SNAKE_FP1, SNAKE_T-SNAKE_FP1 (Snake-T variant 0) and
       SNAKE-SNAKE_FP1 (Snake variant 0, S4 build: identity check)
  Snake-DT: SNAKE_DT-SNAKE_FP1 (S3e)
Demand L1 hit rate = GLOBAL_ACC_R HIT / TOTAL_ACCESS of the L1D.

Exit status 1 if a run did not finish or broke an invariant.  The S4b target
(variant 1 Snake within 5 points of Snake-ISO and no more than 2% below
Snake-DT on every app) is reported, not enforced.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4_report import parse, pct, hit, short  # noqa: E402

FP = "QV100-SASS-PAPER_V100-100M_INSN-"
WHERE = [  # label, run-dir index, config suffix
    ("S0", 1, "SNAKE-SNAKE_FP1-SNAKE_DV0"), ("S1", 0, "SNAKE-SNAKE_FP1"),
    ("T1", 0, "SNAKE_T-SNAKE_FP1"), ("ISO", 2, "SNAKE_ISO-SNAKE_FP1"),
    ("T0", 2, "SNAKE_T-SNAKE_FP1"), ("S0_S4", 2, "SNAKE-SNAKE_FP1"),
    ("DT", 3, "SNAKE_DT-SNAKE_FP1")]
NAMES = {"S0": "Snake v0", "S1": "Snake v1", "T0": "Snake-T v0", "T1": "Snake-T v1",
         "ISO": "Snake-ISO", "DT": "Snake-DT"}


def dist(path, name):
    m = re.findall(r"^snake_dist_%s n=(\d+) mean=\S+ p10=\d+ p50=(\d+) p90=\d+ "
                   r"p99=\d+ max=(\d+)" % name, open(path, errors="replace").read(), re.M)
    return tuple(int(x) for x in m[-1]) if m else None


def main():
    dirs = sys.argv[1:5]
    ref = sys.argv[5]
    rcfg = sys.argv[6] if len(sys.argv) > 6 else "QV100-SASS-PAPER_V100-100M_INSN"
    apps, paths = {}, {}
    for lab, d, suf in WHERE:
        for p in glob.glob(os.path.join(dirs[d], "*", "*", FP + suf, "sim.out")):
            app, args = p.split(os.sep)[-4:-2]
            apps.setdefault((app, args), {})[lab] = parse(p)
            paths[(app, args, lab)] = p
    keys = sorted(k for k in apps if "S1" in apps[k] or "S0" in apps[k])
    bad = 0
    for k in keys:
        rp = os.path.join(ref, k[0], k[1], rcfg, "sim.out")
        apps[k]["_ref"] = parse(rp) if os.path.exists(rp) else None
        for lab in ("S0", "S1", "T1"):
            s = apps[k].get(lab)
            if s and not (s["_done"] and s.get("pf_woken_warps", 1) == 0
                          and s["_pf_invariant"] == "OK" and s["_funnel_invariant"] == "OK"
                          and s["_funnel_src_invariant"] == "OK"):
                bad += 1
                print("**FAIL** %s %s" % (short(k[0]), lab))

    print("## Step 1. The untrained cap, variant 0 (Snake, S4b counters)\n")
    print("Per SM and kernel launch (an SM starting a launch re-arms the cap, OQ-15). "
          "Cycles are Snake cycles of that SM. (a) = cap cycles of SM-launches that later "
          "promote, (b) = of SM-launches that never promote (incl. those still unpromoted "
          "at the 100M cap).\n")
    print("| app | kernel launches | SM-launches | launch -> first promotion, p50 / max (n) | SM-launches never promoting | cap-active cycles (% of Snake cycles) | (a) waiting for a promotion | (b) never promoting | switches back to a launch already seen |")
    print("|---" * 9 + "|")
    for k in keys:
        s = apps[k].get("S0")
        if not s:
            continue
        p = paths[(k[0], k[1], "S0")]
        launches = len(set(re.findall(r"kernel_launch_uid = (\d+)", open(p, errors="replace").read())))
        d = dist(p, "launch_to_promotion")
        a, b = s.get("untrained_cycles_promoted", 0), s.get("untrained_cycles_never", 0)
        cap = s.get("l1_cap_cycles_untrained", 0)
        print("| %s | %d | %d | %s | %d | %d (%s%%) | %s%% | %s%% | %d |" % (
            short(k[0]), launches, s.get("hook_kernel_switch", 0),
            "%d / %d (%d)" % (d[1], d[2], d[0]) if d and d[0] else "-",
            s.get("sm_launches_no_promo", 0), cap, pct(cap, s.get("hook_tick", 0)),
            pct(a, a + b), pct(b, a + b), s.get("kernel_switch_revisit", 0)))

    print("\n## Variant 0 identity: S4b Snake v0 vs S4 Snake\n")
    print("| app | IPC S4 / S4b | pf_issued S4 / S4b | issued, timely, late, hits and IPC equal |")
    print("|---" * 4 + "|")
    for k in keys:
        a, b = apps[k].get("S0_S4"), apps[k].get("S0")
        if not a or not b:
            continue
        ca = [a.get(x) for x in ("pf_issued", "pf_useful_timely", "pf_late", "_hit")] + [a["_ipc"]]
        cb = [b.get(x) for x in ("pf_issued", "pf_useful_timely", "pf_late", "_hit")] + [b["_ipc"]]
        print("| %s | %s / %s | %d / %d | %s |" % (short(k[0]), a["_ipc"], b["_ipc"],
              a.get("pf_issued", 0), b.get("pf_issued", 0), "yes" if ca == cb else "**no**"))

    print("\n## Demand L1 hit rate and IPC change vs the baseline\n")
    order = ["DT", "ISO", "S0", "S1", "T0", "T1"]
    print("| app | baseline | " + " | ".join(NAMES[c] for c in order) +
          " | target: Snake v1 within 5 pts of ISO | and >= -2% vs DT |")
    print("|---" * (len(order) + 4) + "|")
    miss = 0
    for k in keys:
        a, r = apps[k], apps[k]["_ref"]
        cells = []
        for c in order:
            s = a.get(c)
            if not s:
                cells.append("-")
                continue
            di = 100.0 * (s["_ipc"] / r["_ipc"] - 1) if r and r["_ipc"] else 0.0
            cells.append("%.1f (%+.2f%%)" % (hit(s), di))
        s1, iso, dt = a.get("S1"), a.get("ISO"), a.get("DT")
        t1 = abs(hit(s1) - hit(iso)) <= 5 if s1 and iso else None
        t2 = (hit(s1) / hit(dt) - 1) * 100 >= -2 if s1 and dt and hit(dt) else (True if s1 and dt else None)
        miss += (t1 is False) + (t2 is False)
        print("| %s | %s | %s | %s | %s |" % (
            short(k[0]), "%.1f" % hit(r) if r else "-", " | ".join(cells),
            "-" if t1 is None else ("yes (%+.1f)" % (hit(s1) - hit(iso)) if t1 else "**no** (%+.1f)" % (hit(s1) - hit(iso))),
            "-" if t2 is None else ("yes" if t2 else "**no**") + (" (%+.2f%%)" % ((hit(s1) / hit(dt) - 1) * 100) if hit(dt) else "")))
    print("\nTarget %s on every app (%d miss(es))." % ("met" if not miss else "**missed**", miss))

    print("\n## Decoupling counters, variants 0 and 1\n")
    print("Bulk-free victims in lines (L1 region / stale prefetched / fresh prefetched); "
          "unused prefetched sectors evicted from prefetch-region lines (protected) and from "
          "L1-region lines; flips (deferred, variant 1); cap-active cycles (% of Snake cycles, "
          "untrained / confined). Snake-T v0 is the S4 build, without the S4b counters.\n")
    print("| app | config | bulk frees | victims: L1 / pf stale / pf fresh | evicted unused: protected / L1-region | evicted before use (% of issued) | flips (deferred) | unprotected at flip | cap cycles: untrained / confined |")
    print("|---" * 9 + "|")
    for k in keys:
        for c in ("S0", "S1", "T0", "T1"):
            s = apps[k].get(c)
            if not s:
                continue
            t = s.get("hook_tick", 0)
            new = "bulk_free_lines_l1" in s
            print("| %s | %s | %d | %s | %s | %d (%s%%) | %d (%s) | %d | %s%% / %s%% |" % (
                short(k[0]), NAMES[c], s.get("bulk_frees", 0),
                "%d / %d / %d" % (s["bulk_free_lines_l1"], s["bulk_free_lines_pf_stale"],
                                  s["bulk_free_lines_pf_fresh"]) if new else
                "sides %d L1 / %d pf (%d lines)" % (s.get("bulk_free_l1_side", 0),
                                                     s.get("bulk_free_pf_side", 0),
                                                     s.get("bulk_free_lines", 0)),
                "%d / %d" % (s["evict_pf_protected"], s["evict_pf_unprotected"]) if new else "-",
                s.get("pf_evicted_unused", 0), pct(s.get("pf_evicted_unused", 0), s.get("pf_issued", 0)),
                s.get("region_flips", 0), s.get("flips_deferred", "-") if new else "-",
                s.get("flip_unused_sectors", 0),
                pct(s.get("l1_cap_cycles_untrained", 0), t), pct(s.get("l1_cap_cycles_confine", 0), t)))

    print("\n## Coverage and accuracy, variant 1\n")
    print("| app | config | coverage_pred | coverage_paper | accuracy_paper | issued | throttled |")
    print("|---" * 7 + "|")
    for k in keys:
        for c in ("S1", "T1"):
            s = apps[k].get(c)
            if not s:
                continue
            d = s.get("demand_sector_accesses", 0)
            print("| %s | %s | %s | %s | %s | %d | %s%% |" % (
                short(k[0]), NAMES[c], pct(s.get("demand_pred_matched", 0), d),
                pct(s.get("pf_useful_timely", 0) + s.get("pf_late", 0), d),
                pct(s.get("pf_useful_timely", 0), d), s.get("pf_issued", 0),
                pct(s.get("throttled_cycles", 0), s.get("hook_tick", 0))))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
