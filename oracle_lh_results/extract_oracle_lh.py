#!/usr/bin/env python3
"""Rodinia-3.1 results for the latency-hiding oracle prefetcher.

Reads sim_run_oracle_lh/ (baseline, -oracle_prefetcher_uncoalesced and
-oracle_prefetcher_all_miss arms) and writes, next to this script:

  oracle_lh_stats.csv    one row per benchmark/input/config, every stat
  oracle_lh_summary.csv  one row per benchmark/input: speedup, DRAM reads and
                         L1D reservation fails of each oracle arm vs baseline

Log discovery is shared with extract_ipc.py, so an attempt that was killed
(saved as job.log.killed / job.log.oom*) is never picked up.

    ./oracle_lh_results/extract_oracle_lh.py
"""
import csv, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from extract_ipc import find_log  # noqa: E402

RUN_DIR = os.path.join(ROOT, "sim_run_oracle_lh")
ARMS = [("baseline", "A100-SASS"),
        ("uncoal", "A100-SASS-ORACLE_PREF"),
        ("allmiss", "A100-SASS-ORACLE_ALL_MISS")]
DONE_MARKER = "GPGPU-Sim: *** exit detected ***"

# Every stat is cumulative over the run, so the last value printed wins.
STATS = {
    "cycles": (r"gpu_tot_sim_cycle\s*=\s*(\d+)", int),
    "insn": (r"gpu_tot_sim_insn\s*=\s*(\d+)", int),
    "ipc": (r"gpu_tot_ipc\s*=\s*([\d.]+)", float),
    "dram_reads": (r"total dram reads\s*=\s*(\d+)", int),
    "l1d_accesses": (r"L1D_total_cache_accesses\s*=\s*(\d+)", int),
    "l1d_misses": (r"L1D_total_cache_misses\s*=\s*(\d+)", int),
    "l1d_res_fails": (r"L1D_total_cache_reservation_fails\s*=\s*(\d+)", int),
    "total_warp_loads": (r"oracle_pref_total_warp_loads\s*=\s*(\d+)", int),
    "uncoal_warp_loads": (r"oracle_pref_uncoalesced_warp_loads\s*=\s*(\d+)", int),
    "uncoal_misses": (r"oracle_pref_uncoalesced_misses\s*=\s*(\d+)", int),
    "coal_misses": (r"oracle_pref_coalesced_misses\s*=\s*(\d+)", int),
    "all_misses": (r"oracle_pref_all_misses\s*=\s*(\d+)", int),
    "mshr_merge_served": (r"oracle_pref_mshr_merge_served\s*=\s*(\d+)", int),
    "total_served": (r"oracle_pref_total_served\s*=\s*(\d+)", int),
}
STATS = {k: (re.compile(r"^\s*" + p), c) for k, (p, c) in STATS.items()}


def parse_log(path):
    out = {"complete": False}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if DONE_MARKER in line:
                out["complete"] = True
                continue
            for key, (pattern, cast) in STATS.items():
                m = pattern.match(line)
                if m:
                    out[key] = cast(m.group(1))
                    break
    return out


def collect():
    results = {}
    for bench in sorted(os.listdir(RUN_DIR)):
        bench_dir = os.path.join(RUN_DIR, bench)
        if not os.path.isdir(bench_dir) or bench == "gpgpu-sim-builds":
            continue
        for args in sorted(os.listdir(bench_dir)):
            for label, config in ARMS:
                config_dir = os.path.join(bench_dir, args, config)
                log = find_log(config_dir) if os.path.isdir(config_dir) else None
                if log:
                    results.setdefault((bench, args), {})[label] = parse_log(log)
    return results


def ratio(new, old):
    return round(new / float(old), 4) if new is not None and old else ""


def pct(new, old):
    return round((new / float(old) - 1) * 100, 2) if new is not None and old else ""


def main():
    results = collect()
    stat_keys = list(STATS)

    with open(os.path.join(HERE, "oracle_lh_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "args", "arm", "complete"] + stat_keys)
        for (bench, args), arms in sorted(results.items()):
            for label, _ in ARMS:
                s = arms.get(label)
                if s is None:
                    continue
                w.writerow([bench, args, label, s["complete"]] +
                           [s.get(k, "") for k in stat_keys])

    speedups = {"uncoal": [], "allmiss": []}
    header = ["benchmark", "args", "base_cycles"]
    for arm in ("uncoal", "allmiss"):
        header += [arm + "_speedup_pct", arm + "_dram_ratio",
                   arm + "_res_fails_ratio"]
    header += ["base_dram_reads", "uncoal_dram_reads", "allmiss_dram_reads",
               "base_res_fails", "uncoal_res_fails", "allmiss_res_fails",
               "note"]
    with open(os.path.join(HERE, "oracle_lh_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (bench, args), arms in sorted(results.items()):
            done = {k: v for k, v in arms.items() if v["complete"]}
            b = done.get("baseline", {})
            row = [bench, args, b.get("cycles", "")]
            for arm in ("uncoal", "allmiss"):
                s = done.get(arm, {})
                sp = ""
                if b.get("cycles") and s.get("cycles"):
                    sp = round((b["cycles"] / float(s["cycles"]) - 1) * 100, 2)
                    speedups[arm].append(b["cycles"] / float(s["cycles"]))
                row += [sp, ratio(s.get("dram_reads"), b.get("dram_reads")),
                        ratio(s.get("l1d_res_fails"), b.get("l1d_res_fails"))]
            row += [done.get(a, {}).get("dram_reads", "")
                    for a in ("baseline", "uncoal", "allmiss")]
            row += [done.get(a, {}).get("l1d_res_fails", "")
                    for a in ("baseline", "uncoal", "allmiss")]
            missing = [a for a, _ in ARMS if a not in done]
            row.append("incomplete: " + ",".join(missing) if missing else "")
            w.writerow(row)
        for arm, v in speedups.items():
            if v:
                g = math.exp(sum(map(math.log, v)) / len(v))
                w.writerow(["GEOMEAN", arm, "", round((g - 1) * 100, 2)])
                print("%s geomean speedup: %+.2f%% over %d runs"
                      % (arm, (g - 1) * 100, len(v)))
    print("wrote", os.path.join(HERE, "oracle_lh_stats.csv"))
    print("wrote", os.path.join(HERE, "oracle_lh_summary.csv"))


if __name__ == "__main__":
    main()
