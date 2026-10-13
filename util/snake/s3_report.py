#!/usr/bin/env python3
"""Snake S3 gate table: prefetch outcomes vs the bin_ref_v2 baseline.

    s3_report.py <gate runs dir> <reference runs dir> [<reference config>]

For every <gate>/<app>/<args>/<config>/sim.out the baseline is
<reference>/<app>/<args>/<reference config>/sim.out (default config
QV100-SASS-PAPER_V100-100M_INSN).  Columns (docs/snake/SNAKE_PLAN.md §7):
  done        the run reached "exit detected" (no deadlock)
  woken       snake_pf_woken_warps (must be 0)
  invariant   snake_pf_invariant (issued == timely + late + evicted + unused)
  cov / acc   coverage_paper = (timely + late) / demand sectors,
              accuracy_paper = timely / demand sectors
  cov_pred    demand sectors named by an earlier prediction (OQ-14)
  acc_std     (timely + late) / issued
  dIPC        IPC change vs the baseline (gpu_tot_ipc)
  L1 hit      GLOBAL_ACC_R HIT / TOTAL_ACCESS, Snake run (baseline)
  issued / dropped   issued sectors; dropped for resources (line / MSHR /
              miss queue) and for a full prefetch queue
  mqfull      demand MISS_QUEUE_FULL failures (GLOBAL_ACC_R), baseline -> Snake
  resfail     snake_pf_caused_resfail
  ev_iw       Tail evictions of entries with a trained inter-warp stride
              (average warp bits of those entries)
Exit status 1 if any run did not finish, woke a warp or broke the invariant.
"""
import glob
import os
import re
import sys


def last(text, pattern, cast=float):
    m = re.findall(pattern, text)
    return cast(m[-1]) if m else None


def stats(path):
    t = open(path, errors="replace").read()
    s = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
    s["_done"] = "exit detected" in t
    s["_ipc"] = last(t, r"gpu_tot_ipc =\s*([\d.]+)")
    s["_hit"] = last(t, r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\] = (\d+)", int)
    s["_tot"] = last(t, r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\] = (\d+)", int)
    s["_inv"] = last(t, r"snake_pf_invariant = (\w+)", str)
    s["_mqf"] = last(t, r"Total_core_cache_fail_stats_breakdown\[GLOBAL_ACC_R\]\[MISS_QUEUE_FULL\] = (\d+)", int) or 0
    return s


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


def main():
    gate, ref = sys.argv[1], sys.argv[2]
    rcfg = sys.argv[3] if len(sys.argv) > 3 else "QV100-SASS-PAPER_V100-100M_INSN"
    bad = 0
    print("| app / input | config | done | woken | inv | cov | acc | cov_pred | acc_std "
          "| dIPC | L1 hit (base) | issued | dropped line/MSHR/mq | dropped q-full "
          "| mqfull base -> Snake | resfail | ev_iw (avg bits) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for path in sorted(glob.glob(os.path.join(gate, "*", "*", "*", "sim.out"))):
        app, args, cfg = path.split(os.sep)[-4:-1]
        s = stats(path)
        rp = os.path.join(ref, app, args, rcfg, "sim.out")
        r = stats(rp) if os.path.exists(rp) else None
        tl = s.get("pf_useful_timely", 0) + s.get("pf_late", 0)
        dem = s.get("demand_sector_accesses", 0)
        ok = s["_done"] and s.get("pf_woken_warps", 1) == 0 and s["_inv"] == "OK"
        bad += not ok
        dipc = ("%+.2f%%" % (100.0 * (s["_ipc"] / r["_ipc"] - 1))
                if r and s["_ipc"] and r["_ipc"] else "no baseline")
        ev = s.get("tail_evict_interwarp_trained", 0)
        print("| %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %d | %d/%d/%d | %d | %s -> %d | %d | %d (%s) |" % (
            app, args[:24], cfg.replace("QV100-SASS-PAPER_V100-100M_INSN-", ""),
            "yes" if s["_done"] else "**NO**", s.get("pf_woken_warps", "?"), s["_inv"],
            pct(tl, dem), pct(s.get("pf_useful_timely", 0), dem),
            pct(s.get("demand_pred_matched", 0), dem),
            pct(tl, s.get("pf_issued", 0)), dipc,
            pct(s["_hit"] or 0, s["_tot"] or 0),
            pct(r["_hit"] or 0, r["_tot"] or 0) if r else "n/a",
            s.get("pf_issued", 0), s.get("pf_drop_line", 0), s.get("pf_drop_mshr", 0),
            s.get("pf_drop_miss_queue", 0), s.get("pf_drop_queue_full", 0),
            r["_mqf"] if r else "n/a", s["_mqf"],
            s.get("pf_caused_resfail", 0), ev,
            "%.1f" % (s.get("tail_evict_interwarp_wbits", 0) / ev) if ev else "-"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
