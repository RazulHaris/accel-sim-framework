#!/usr/bin/env python3
"""Collect Snake final-evaluation results and check every job (S5).

    collect_results.py [-o <csv>] [--require-all] <run dir> [<run dir> ...]

Each <run dir> holds runs/<app>/<args>/<config>/ (from util/snake/s5_runs.sh
or run_apps.sh) and, for a frozen build, bin/ and BUILDS.txt.  Every job with
a sim.out is checked:
  - both build tags are _modified_0.0 (accel-sim and gpgpu-sim), and they are
    the same for every job of a run dir
  - the run finished ("exit detected")
  - with Snake on: no warp woken by a prefetch, and the outcome, funnel and
    per-source funnel invariants are OK
Jobs without a sim.out are counted as not run (a failure with --require-all).
One CSV row per job (-o): identity, tags, cycles, instructions, IPC, demand
L1 hit rate, the Snake counters used by compare_paper.py, and energy:
  - Snake's own energy (OQ-16, offline): accesses x -snake_energy_pj_per_access
    + -snake_static_mw per SM x simulated time.  Accesses = Head updates + Tail
    look-ups + chain/prediction steps (pred_* counters), one table access each.
  - GPU energy from AccelWattch (accelwattch_power_report.log in the job dir,
    last gpu_tot_avg_power x simulated time) when power simulation was on;
    otherwise empty ("unavailable").
Exit status 1 if any check fails.
"""
import csv
import glob
import os
import re
import sys

SNAKE_KEYS = ["demand_sector_accesses", "demand_pred_matched", "pf_issued",
              "pf_useful_timely", "pf_late", "pf_evicted_unused", "head_updates",
              "tail_lookups", "pred_inter_thread", "pred_inter_warp",
              "pred_inter_warp_chain", "pred_intra_warp", "pred_promotion",
              "throttled_cycles", "hook_tick", "t1_promotions"]
PRED = ["pred_inter_thread", "pred_inter_warp", "pred_inter_warp_chain",
        "pred_intra_warp", "pred_promotion"]


def last(pat, text, conv=str):
    m = re.findall(pat, text, re.M)
    return conv(m[-1]) if m else None


def parse_job(job):
    """Stats of one job dir (None if it has no sim.out)."""
    p = os.path.join(job, "sim.out")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    r = {"done": "exit detected" in t}
    r["accelsim_tag"] = last(r"accelsim-commit-[0-9a-f]+_modified_[0-9.]+", t)
    r["gpgpusim_tag"] = last(r"gpgpu-sim_git-commit-[0-9a-f]+_modified_[0-9.]+", t)
    r["cycles"] = last(r"^gpu_tot_sim_cycle = (\d+)", t, int)
    r["insn"] = last(r"^gpu_tot_sim_insn = (\d+)", t, int)
    r["ipc"] = last(r"^gpu_tot_ipc =\s*([\d.]+)", t, float)
    b = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"
    hit, acc = last(b % "HIT", t, int), last(b % "TOTAL_ACCESS", t, int)
    r["l1_hit_rate"] = 100.0 * hit / acc if hit is not None and acc else None
    r["clock_mhz"] = last(r"^-gpgpu_clock_domains\s+([\d.]+):", t, float)
    n_cl = last(r"^-gpgpu_n_clusters\s+(\d+)", t, int) or 0
    n_cpc = last(r"^-gpgpu_n_cores_per_cluster\s+(\d+)", t, int) or 0
    r["n_sm"] = n_cl * n_cpc
    r["snake"] = last(r"^-snake_enable\s+(\d)", t, int) == 1
    r["seconds"] = last(r"gpgpu_simulation_time = .*\((\d+) sec\)", t, int)
    sec = (r["cycles"] / (r["clock_mhz"] * 1e6)) if r["cycles"] and r["clock_mhz"] else None
    r["sim_time_s"] = sec
    if r["snake"]:
        s = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
        for k in SNAKE_KEYS:
            r[k] = s.get(k, 0)
        r["pf_woken_warps"] = s.get("pf_woken_warps")
        for k in ("pf_invariant", "funnel_invariant", "funnel_src_invariant"):
            r[k] = last(r"^snake_%s = (\w+)" % k, t)
        pj = last(r"^-snake_energy_pj_per_access\s+([\d.]+)", t, float) or 0.0
        mw = last(r"^-snake_static_mw\s+([\d.]+)", t, float) or 0.0
        accesses = r["head_updates"] + r["tail_lookups"] + sum(r[k] for k in PRED)
        r["snake_accesses"] = accesses
        r["snake_energy_j"] = (accesses * pj * 1e-12 + mw * 1e-3 * r["n_sm"] * sec
                               if sec is not None else None)
    else:
        r["snake_energy_j"] = 0.0
    pw = os.path.join(job, "accelwattch_power_report.log")
    avg = last(r"gpu_tot_avg_power = ([\d.]+)", open(pw, errors="replace").read(),
               float) if os.path.exists(pw) else None
    r["gpu_energy_j"] = avg * sec if avg is not None and sec is not None else None
    return r


def jobs_of(run_dir):
    for j in sorted(glob.glob(os.path.join(run_dir, "runs", "*", "*", "*"))):
        if os.path.exists(os.path.join(j, "justrun.sh")) or os.path.exists(
                os.path.join(j, "sim.out")):
            app, args, cfg = j.split(os.sep)[-3:]
            yield app, args, cfg, j


def check(r):
    """Problems with one parsed job."""
    bad = []
    tags = (r["accelsim_tag"] or "", r["gpgpusim_tag"] or "")
    if not all(x.endswith("_modified_0.0") or "_modified_0.0_" in x for x in tags) \
            or not all(tags):
        bad.append("build tag not _modified_0.0: %s %s" % tags)
    if not r["done"]:
        bad.append("not finished")
    if r["snake"]:
        if r["pf_woken_warps"] != 0:
            bad.append("pf_woken_warps %s" % r["pf_woken_warps"])
        for k in ("pf_invariant", "funnel_invariant", "funnel_src_invariant"):
            if r[k] != "OK":
                bad.append("%s %s" % (k, r[k]))
    return bad


def main():
    argv = sys.argv[1:]
    out, require = None, False
    while argv and argv[0].startswith("-"):
        if argv[0] == "-o":
            out, argv = argv[1], argv[2:]
        elif argv[0] == "--require-all":
            require, argv = True, argv[1:]
        else:
            sys.exit(__doc__)
    rows, n_bad, n_missing = [], 0, 0
    for d in argv:
        tags_seen = set()
        for app, args, cfg, j in jobs_of(d):
            r = parse_job(j)
            if r is None:
                n_missing += 1
                continue
            bad = check(r)
            tags_seen.add((r["accelsim_tag"], r["gpgpusim_tag"]))
            for b in bad:
                print("FAIL %s/%s/%s/%s: %s" % (os.path.basename(d.rstrip("/")), app, args, cfg, b))
            n_bad += bool(bad)
            row = {"run_dir": os.path.basename(d.rstrip("/")), "app": app, "args": args,
                   "config": cfg, "ok": not bad}
            row.update(r)
            rows.append(row)
        if len(tags_seen) > 1:
            print("FAIL %s: jobs from %d different builds: %s" % (d, len(tags_seen), sorted(tags_seen)))
            n_bad += 1
    if out and rows:
        keys = []
        for row in rows:
            keys += [k for k in row if k not in keys]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print("%d jobs collected, %d failed checks, %d not run yet%s" % (
        len(rows), n_bad, n_missing, (" -> %s" % out) if out and rows else ""))
    sys.exit(1 if n_bad or (require and n_missing) else 0)


if __name__ == "__main__":
    main()
