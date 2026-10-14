#!/usr/bin/env python3
"""Snake S3c diagnosis tables (docs/snake/results/S3c_diag.md).

    s3c_diag.py <gate runs dir> <reference runs dir> [<reference config>]

Reads the S3c statistics of every <gate>/<app>/<args>/<config>/sim.out (the
last value printed, i.e. at the end of the run) and prints:
  1. the prediction funnel, as % of predicted sectors (snake_pred_sectors):
     every prediction record (OQ-14) ends in one class (snake_funnel_*)
  2. the funnel per prediction mode
  3. timing medians (snake_dist_*): queue entry -> issue, queue wait, port
     wait, issue -> first demand (late / timely), prediction -> demand
  4. the L1D miss queue: waits of demand and prefetch requests, demands
     queued behind prefetches, head-blocked cycles
  5. promotion: loads at a promoted PC1 by a warp without its bit
  6. a summary per run (coverage, IPC vs the baseline, drops) for comparing
     configs, e.g. -snake_missq_reserve 8 vs 4
The baseline is <reference>/<app>/<args>/<reference config>/sim.out
(default QV100-SASS-PAPER_V100-100M_INSN).
"""
import glob
import os
import re
import sys

MODES = ["inter_thread", "intra_warp", "promotion", "inter_warp", "inter_warp_chain"]
SHORT = {"inter_thread": "it", "intra_warp": "intra", "promotion": "promo",
         "inter_warp": "iw", "inter_warp_chain": "iw-chain"}
CLASSES = ["wrong_issued", "wrong_not_issued", "queued", "drop_qfull", "drop_mq",
           "drop_res", "dup_l1", "dup_mshr", "late", "evicted", "timely",
           "issued_other", "live"]


def parse(path):
    t = open(path, errors="replace").read()
    s = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
    d = {}
    for m in re.finditer(r"^snake_dist_(\w+) n=(\d+) mean=([\d.]+) p10=(\d+) "
                         r"p50=(\d+) p90=(\d+) p99=(\d+) max=(\d+)$", t, re.M):
        d[m.group(1)] = dict(n=int(m.group(2)), mean=float(m.group(3)),
                             p10=int(m.group(4)), p50=int(m.group(5)),
                             p90=int(m.group(6)), p99=int(m.group(7)),
                             max=int(m.group(8)))
    s["_dist"] = d
    s["_done"] = "exit detected" in t
    ipc = re.findall(r"gpu_tot_ipc =\s*([\d.]+)", t)
    s["_ipc"] = float(ipc[-1]) if ipc else None
    cyc = re.findall(r"gpu_tot_sim_cycle = (\d+)", t)
    s["_cycles"] = int(cyc[-1]) if cyc else None
    mq = re.findall(r"Total_core_cache_fail_stats_breakdown\[GLOBAL_ACC_R\]"
                    r"\[MISS_QUEUE_FULL\] = (\d+)", t)
    s["_mqf"] = int(mq[-1]) if mq else 0
    inv = re.findall(r"snake_funnel_invariant = (\w+)", t)
    s["_finv"] = inv[-1] if inv else "?"
    inv = re.findall(r"snake_pf_invariant = (\w+)", t)
    s["_inv"] = inv[-1] if inv else "?"
    return s


def pct(a, b):
    return "%.1f" % (100.0 * a / b) if b else "n/a"


def fcls(s, c, modes=MODES):
    return sum(s.get("funnel_%s_%s" % (m, c), 0) for m in modes)


def name(app, args, cfg):
    return "%s / %s / %s" % (app.replace("-rodinia-3.1", ""), args[:12],
                             cfg.replace("QV100-SASS-PAPER_V100-100M_INSN-", ""))


def dist(s, key, f="p50"):
    v = s["_dist"].get(key)
    return "-" if not v or not v["n"] else str(v[f])


def main():
    gate, ref = sys.argv[1], sys.argv[2]
    rcfg = sys.argv[3] if len(sys.argv) > 3 else "QV100-SASS-PAPER_V100-100M_INSN"
    runs = []
    for path in sorted(glob.glob(os.path.join(gate, "*", "*", "*", "sim.out"))):
        app, args, cfg = path.split(os.sep)[-4:-1]
        rp = os.path.join(ref, app, args, rcfg, "sim.out")
        runs.append((app, args, cfg, parse(path),
                     parse(rp) if os.path.exists(rp) else None))

    print("## 1. Prediction funnel (% of predicted sectors)\n")
    print("`matched` = demand sectors that matched a prediction (coverage_pred "
          "numerator), split by where the prediction ended; `wrong` = no "
          "demand within the window (5000 cycles).\n")
    print("| run | predicted sectors | wrong, issued | wrong, not issued | matched "
          "| queued | drop q-full | drop miss-queue | drop line/MSHR | dup L1 "
          "| dup MSHR | late | evicted | timely | issued, other | live | invariant |")
    print("|---" * 17 + "|")
    for app, args, cfg, s, r in runs:
        n = s.get("pred_sectors", 0)
        matched = sum(fcls(s, c) for c in CLASSES[2:12])
        print("| %s | %d | %s | %s | %s | %s |" % (
            name(app, args, cfg), n, pct(fcls(s, "wrong_issued"), n),
            pct(fcls(s, "wrong_not_issued"), n), pct(matched, n),
            " | ".join(pct(fcls(s, c), n) for c in CLASSES[2:]) + " | " + s["_finv"]))

    print("\n## 2. Funnel per mode (% of that mode's predicted sectors)\n")
    print("A record's mode is that of its latest prediction.\n")
    print("| run | mode | records | wrong | queued | drop q-full | drop mq | drop res "
          "| dup L1 | dup MSHR | late | evicted | timely | other | live |")
    print("|---" * 15 + "|")
    for app, args, cfg, s, r in runs:
        for m in MODES:
            tot = sum(s.get("funnel_%s_%s" % (m, c), 0) for c in CLASSES)
            if not tot:
                continue
            g = lambda c: s.get("funnel_%s_%s" % (m, c), 0)
            print("| %s | %s | %d | %s | %s |" % (
                name(app, args, cfg), SHORT[m], tot,
                pct(g("wrong_issued") + g("wrong_not_issued"), tot),
                " | ".join(pct(g(c), tot) for c in CLASSES[2:])))

    print("\n## 3. Timing (cycles; p50, with p90 in parentheses)\n")
    print("`pred->issue` = queue entry created -> sector issued (includes the "
          "2-cycle latency); `queue wait` = entry ready -> its first sector "
          "offered; `port wait` = first -> this sector offered; `lead` = issue "
          "-> first demand of the sector, by outcome; `pred->demand` = latest "
          "prediction -> the matching demand.\n")
    print("| run | mode | pred->issue | queue wait (all) | port wait (all) "
          "| lead, late | lead, timely | lead, evicted | pred->demand | n issued->demanded |")
    print("|---" * 10 + "|")
    for app, args, cfg, s, r in runs:
        first = True
        for m in MODES:
            if not s["_dist"].get("lead_pred_" + m, {}).get("n") and \
               not s["_dist"].get("pred_to_issue_" + m, {}).get("n"):
                continue
            both = lambda k: "-" if dist(s, k) == "-" else "%s (%s)" % (dist(s, k), dist(s, k, "p90"))
            nd = sum(s["_dist"].get("lead_issue_%s_%s" % (m, o), {}).get("n", 0)
                     for o in ("late", "timely", "evicted"))
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
                name(app, args, cfg), SHORT[m], both("pred_to_issue_" + m),
                both("queue_wait") if first else "", both("port_wait") if first else "",
                both("lead_issue_%s_late" % m), both("lead_issue_%s_timely" % m),
                both("lead_issue_%s_evicted" % m), both("lead_pred_" + m), nd))
            first = False

    print("\n## 4. L1D miss queue\n")
    print("`wait` = push -> sent to the interconnect (mean / p50 / p90); `demand "
          "behind pf` = demand pushes with at least one prefetch ahead (mean "
          "prefetches ahead); `head pf, demand behind` = non-empty cycles whose "
          "head is a prefetch with a demand behind it; `blocked` = non-empty "
          "cycles in which the interconnect input buffer refused the head.\n")
    print("| run | demand wait | prefetch wait | demand pushes | demand behind pf "
          "| head pf, demand behind | blocked (head pf / demand) | sent per non-empty cycle |")
    print("|---" * 8 + "|")
    for app, args, cfg, s, r in runs:
        def w(k):
            v = s["_dist"].get(k)
            return "-" if not v or not v["n"] else "%.1f / %d / %d" % (v["mean"], v["p50"], v["p90"])
        dp = s.get("mq_push_demand", 0)
        ne = s.get("mq_cycles_nonempty", 0)
        print("| %s | %s | %s | %d | %s%% (%.2f) | %s%% | %d / %d | %.3f |" % (
            name(app, args, cfg), w("mq_wait_demand"), w("mq_wait_prefetch"), dp,
            pct(s.get("mq_demand_behind_pf", 0), dp),
            s.get("mq_demand_pf_ahead_sum", 0) / dp if dp else 0,
            pct(s.get("mq_head_pf_demand_behind", 0), ne),
            s.get("mq_head_blocked_prefetch", 0), s.get("mq_head_blocked_demand", 0),
            s.get("mq_sent", 0) / ne if ne else 0))

    print("\n## 5. Promotion (§3.2 ¶4)\n")
    print("| run | t1_promotions | promotion predictions | promotion sectors issued "
          "| loads at a promoted PC1 without the warp's bit (% of observed loads) |")
    print("|---" * 5 + "|")
    for app, args, cfg, s, r in runs:
        print("| %s | %d | %d | %d | %d (%s%%) |" % (
            name(app, args, cfg), s.get("t1_promotions", 0), s.get("pred_promotion", 0),
            s.get("pf_issued_promotion", 0), s.get("promoted_nobit_loads", 0),
            pct(s.get("promoted_nobit_loads", 0), s.get("loads_observed", 0))))

    print("\n## 6. Summary per run\n")
    print("| run | done | inv | cycles | dIPC | coverage_pred | coverage_paper "
          "| accuracy_paper | issued | drop miss-queue (of which reserve) "
          "| drop q-full | demand MISS_QUEUE_FULL base -> Snake |")
    print("|---" * 12 + "|")
    bad = 0
    for app, args, cfg, s, r in runs:
        dem = s.get("demand_sector_accesses", 0)
        ok = s["_done"] and s["_inv"] == "OK" and s["_finv"] == "OK" and \
            s.get("pf_woken_warps", 1) == 0
        bad += not ok
        dipc = ("%+.2f%%" % (100.0 * (s["_ipc"] / r["_ipc"] - 1))
                if r and s["_ipc"] and r["_ipc"] else "n/a")
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %d | %d (%d) | %d | %s -> %d |" % (
            name(app, args, cfg), "yes" if s["_done"] else "**NO**",
            "OK" if ok else "**FAIL**", s["_cycles"], dipc,
            pct(s.get("demand_pred_matched", 0), dem),
            pct(s.get("pf_useful_timely", 0) + s.get("pf_late", 0), dem),
            pct(s.get("pf_useful_timely", 0), dem), s.get("pf_issued", 0),
            s.get("pf_drop_miss_queue", 0), s.get("pf_drop_mq_reserve", 0),
            s.get("pf_drop_queue_full", 0), r["_mqf"] if r else "n/a", s["_mqf"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
