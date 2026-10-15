#!/usr/bin/env python3
"""Snake S4 tables (docs/snake/results/S4.md).

    s4_report.py <S4 runs dir> <Snake-DT runs dir> <reference runs dir>
                 [<reference config>]

Configs, by the token before -SNAKE_FP1: SNAKE_DT (from the Snake-DT dir,
the S3e runs), SNAKE_T, SNAKE and SNAKE_ISO (from the S4 dir).  The
baseline is <reference>/<app>/<args>/<reference config>/sim.out (default
QV100-SASS-PAPER_V100-100M_INSN).  Last printed values are used.

Demand L1 hit rate = GLOBAL_ACC_R HIT / TOTAL_ACCESS of the L1D
(Total_core_cache_stats_breakdown; TOTAL_ACCESS excludes reservation fails).

Exit status 1 if a run did not finish or broke an invariant, or if a mode-1
config (Snake, Snake-T) lowers the demand L1 hit rate or the IPC by more than
2% (relative) against Snake-DT on any app (the S4 stop rule).
"""
import glob
import os
import re
import sys

CFGS = [("SNAKE_DT", "Snake-DT"), ("SNAKE_T", "Snake-T"), ("SNAKE", "Snake"),
        ("SNAKE_ISO", "Snake-ISO")]
MODE1 = ("SNAKE", "SNAKE_T")


def parse(path):
    t = open(path, errors="replace").read()
    s = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
    s["_done"] = "exit detected" in t
    last = lambda pat, c=float: (lambda m: c(m[-1]) if m else None)(re.findall(pat, t))
    s["_ipc"] = last(r"gpu_tot_ipc =\s*([\d.]+)")
    b = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"
    s["_hit"] = last(b % "HIT", int) or 0
    s["_acc"] = last(b % "TOTAL_ACCESS", int) or 0
    for k in ("pf_invariant", "funnel_invariant", "funnel_src_invariant"):
        s["_" + k] = last(r"snake_%s = (\w+)" % k, str)
    return s


def pct(a, b, d=1):
    return ("%%.%df" % d) % (100.0 * a / b) if b else "-"


def hit(s):
    return 100.0 * s["_hit"] / s["_acc"] if s and s["_acc"] else None


def short(app):
    return app.replace("-rodinia-3.1", "").replace("parboil-", "")


def main():
    s4, dt, ref = sys.argv[1:4]
    rcfg = sys.argv[4] if len(sys.argv) > 4 else "QV100-SASS-PAPER_V100-100M_INSN"
    apps = {}
    for root, keys in ((dt, ("SNAKE_DT",)), (s4, ("SNAKE_T", "SNAKE", "SNAKE_ISO"))):
        for p in sorted(glob.glob(os.path.join(root, "*", "*", "*", "sim.out"))):
            app, args, cfg = p.split(os.sep)[-4:-1]
            tok = cfg[:-len("-SNAKE_FP1")].rsplit("-", 1)[-1] if cfg.endswith("-SNAKE_FP1") else None
            if tok not in keys:
                continue
            a = apps.setdefault((app, args), {})
            a[tok] = parse(p)
    for (app, args), a in apps.items():
        rp = os.path.join(ref, app, args, rcfg, "sim.out")
        a["_ref"] = parse(rp) if os.path.exists(rp) else None
    keys = sorted(k for k, a in apps.items() if any(c in a for c, _ in CFGS[1:]))
    runs = [(short(k[0]), c, n, apps[k][c], apps[k]["_ref"]) for k in keys
            for c, n in CFGS if c in apps[k]]
    bad = 0

    print("## 1. Coverage, accuracy, IPC and demand L1 hit rate vs the baseline\n")
    print("| app | config | done / invariants | coverage_pred | coverage_paper | accuracy_paper | dIPC vs baseline | demand L1 hit rate, baseline -> config | issued | issued / demand |")
    print("|---" * 10 + "|")
    for app, c, n, s, r in runs:
        d = s.get("demand_sector_accesses", 0)
        ok = (s["_done"] and s.get("pf_woken_warps", 1) == 0 and s["_pf_invariant"] == "OK"
              and s["_funnel_invariant"] == "OK" and s["_funnel_src_invariant"] == "OK")
        bad += not ok
        dipc = 100.0 * (s["_ipc"] / r["_ipc"] - 1) if r and s["_ipc"] and r["_ipc"] else None
        print("| %s | %s | %s | %s | %s | %s | %s | %s -> %s | %d | %s%% |" % (
            app, n, "yes / OK" if ok else "**FAIL**",
            pct(s.get("demand_pred_matched", 0), d),
            pct(s.get("pf_useful_timely", 0) + s.get("pf_late", 0), d),
            pct(s.get("pf_useful_timely", 0), d),
            "%+.2f%%" % dipc if dipc is not None else "n/a",
            "%.1f" % hit(r) if r else "n/a", "%.1f" % hit(s),
            s.get("pf_issued", 0), pct(s.get("pf_issued", 0), d)))

    print("\n## 2. Throttling (cycles as % of Snake cycles, snake_hook_tick; a cycle with both triggers counts in both)\n")
    print("| app | config | throttled | by space | by bandwidth | space events | bandwidth events | meter windows >= 70% | ready lines held (cycles) | loads while throttled | chain steps cut |")
    print("|---" * 11 + "|")
    for app, c, n, s, r in runs:
        if c == "SNAKE_DT":
            continue
        t = s.get("hook_tick", 0)
        win = s.get("bw_windows", 0)
        hi = sum(s.get("bw_util_hist_%d" % k, 0) for k in (70, 80, 90, 100))
        print("| %s | %s | %s%% | %s%% | %s%% | %d | %d | %s%% | %d | %d | %d |" % (
            app, n, pct(s.get("throttled_cycles", 0), t),
            pct(s.get("throttled_cycles_space", 0), t),
            pct(s.get("throttled_cycles_bw", 0), t),
            s.get("throttle_space_events", 0), s.get("throttle_bw_events", 0),
            pct(hi, win), s.get("pf_held_throttled", 0), s.get("loads_throttled", 0),
            s.get("throttle_chain_cut", 0)))

    print("\n## 3. Decoupling\n")
    print("Mode 1 (Snake-T, Snake): region flips (of which on a demand sector miss), unused prefetched sectors that lost protection (on a flipped line / filled into an L1-region line), bulk frees (L1 side / prefetch side) and the lines they freed (unused prefetched sectors among them), demand evictions forced by the L1 cap, L1-cap cycles (until trained / confined while throttled, % of Snake cycles). Evicted before use = `pf_evicted_unused` (% of issued). Mode 2 (Snake-ISO): buffer inserts, hits, evictions, duplicates and failed transfers.\n")
    print("| app | config | region flips (sector miss) | unprotected sectors: flip / L1-line fill | bulk frees (L1 / pf) | lines freed (pf unused sectors) | L1-cap victims | L1-cap cycles: untrained / confined | evicted before use | quota checks / fallbacks |")
    print("|---" * 10 + "|")
    for app, c, n, s, r in runs:
        if c not in MODE1:
            continue
        t = s.get("hook_tick", 0)
        print("| %s | %s | %d (%d) | %d / %d | %d (%d / %d) | %d (%d) | %d | %s%% / %s%% | %d (%s%%) | %d / %d |" % (
            app, n, s.get("region_flips", 0), s.get("region_flips_sector_miss", 0),
            s.get("flip_unused_sectors", 0), s.get("pf_fill_l1_region", 0),
            s.get("bulk_frees", 0), s.get("bulk_free_l1_side", 0), s.get("bulk_free_pf_side", 0),
            s.get("bulk_free_lines", 0), s.get("bulk_free_pf_unused", 0),
            s.get("l1_quota_victims", 0),
            pct(s.get("l1_cap_cycles_untrained", 0), t), pct(s.get("l1_cap_cycles_confine", 0), t),
            s.get("pf_evicted_unused", 0), pct(s.get("pf_evicted_unused", 0), s.get("pf_issued", 0)),
            s.get("quota_checks", 0), s.get("decouple_fallback", 0)))
    print("\n| app | config | buffer inserts | buffer hits | buffer evictions | evicted before use | duplicates in buffer | failed transfers |")
    print("|---" * 8 + "|")
    for app, c, n, s, r in runs:
        if c != "SNAKE_ISO":
            continue
        print("| %s | %s | %d | %d | %d | %d (%s%%) | %d | %d |" % (
            app, n, s.get("iso_inserts", 0), s.get("iso_hits", 0), s.get("iso_evictions", 0),
            s.get("pf_evicted_unused", 0), pct(s.get("pf_evicted_unused", 0), s.get("pf_issued", 0)),
            s.get("iso_dup", 0), s.get("iso_transfer_fail", 0)))

    print("\n## 4. Stop rule: mode 1 vs Snake-DT (relative change; STOP below -2%)\n")
    print("| app | config | demand L1 hit rate, DT -> config | change | IPC change | verdict |")
    print("|---" * 6 + "|")
    stop = 0
    for k in keys:
        a = apps[k]
        base = a.get("SNAKE_DT")
        for c in MODE1:
            s = a.get(c)
            if not s or not base:
                continue
            dh = 100.0 * (hit(s) / hit(base) - 1) if hit(base) else 0.0
            di = 100.0 * (s["_ipc"] / base["_ipc"] - 1) if base["_ipc"] else 0.0
            v = dh < -2 or di < -2
            stop += v
            print("| %s | %s | %.1f -> %.1f | %+.2f%% | %+.2f%% | %s |" % (
                short(k[0]), dict(CFGS)[c], hit(base), hit(s), dh, di,
                "**STOP**" if v else "ok"))

    print("\n## 5. Snake-DT < Snake-T <= Snake (§5.2 trend; reported, not required)\n")
    print("| app | accuracy_paper DT / T / Snake | holds | IPC change vs baseline DT / T / Snake | holds |")
    print("|---" * 5 + "|")
    for k in keys:
        a, r = apps[k], apps[k]["_ref"]
        if not all(c in a for c in ("SNAKE_DT", "SNAKE_T", "SNAKE")):
            continue
        acc = [100.0 * a[c].get("pf_useful_timely", 0) / max(a[c].get("demand_sector_accesses", 0), 1)
               for c in ("SNAKE_DT", "SNAKE_T", "SNAKE")]
        ipc = [100.0 * (a[c]["_ipc"] / r["_ipc"] - 1) if r else 0.0
               for c in ("SNAKE_DT", "SNAKE_T", "SNAKE")]
        print("| %s | %.1f / %.1f / %.1f | %s | %+.2f%% / %+.2f%% / %+.2f%% | %s |" % (
            short(k[0]), acc[0], acc[1], acc[2],
            "yes" if acc[0] < acc[1] <= acc[2] else "no",
            ipc[0], ipc[1], ipc[2], "yes" if ipc[0] < ipc[1] <= ipc[2] else "no"))
    if stop:
        print("\nSTOP: %d mode-1 run(s) below -2%% vs Snake-DT" % stop)
    sys.exit(1 if bad or stop else 0)


if __name__ == "__main__":
    main()
