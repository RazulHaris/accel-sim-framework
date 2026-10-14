#!/usr/bin/env python3
"""Snake S3d tables (docs/snake/results/S3d.md).

    s3d_report.py [-p <runs dir>:<config> ...] <gate runs dir> <reference runs dir>
                  [<reference config>]

For every <gate>/<app>/<args>/<config>/sim.out (last values printed):
  1. the funnel by prediction source (snake_funnel_src_*: the source of the
     prediction that created the record), as % of that source's records:
     wrong, late, timely, never issued in time (queued, queue-full,
     miss-queue, line/MSHR drops), duplicate (L1, MSHR), live
  2. each source's share of all wrong and of all timely records
  3. loads without an own inter-thread prediction, and the OQ-32
     verification counts
  4. coverage_pred, coverage_paper, accuracy_paper, IPC change and demand
     MISS_QUEUE_FULL vs the baseline (<reference>/<app>/<args>/<reference
     config>, default QV100-SASS-PAPER_V100-100M_INSN); with -p, the same
     app/input of an earlier stage is shown as `prev` (first match wins).
"""
import glob
import os
import re
import sys

SRC = [("inter_thread", "inter-thread, own warp"), ("promotion", "inter-thread, at promotion"),
       ("intra_warp", "intra-warp"), ("inter_warp", "inter-warp"),
       ("inter_warp_chain", "inter-warp chain (OQ-29)")]
NEVER = ["queued", "drop_qfull", "drop_mq", "drop_res"]
ALL = ["wrong_issued", "wrong_not_issued", "queued", "drop_qfull", "drop_mq", "drop_res",
       "dup_l1", "dup_mshr", "late", "evicted", "timely", "issued_other", "live"]


def parse(path):
    t = open(path, errors="replace").read()
    s = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
    s["_done"] = "exit detected" in t
    f = lambda pat, c=float: (lambda m: c(m[-1]) if m else None)(re.findall(pat, t))
    s["_ipc"] = f(r"gpu_tot_ipc =\s*([\d.]+)")
    s["_cyc"] = f(r"gpu_tot_sim_cycle = (\d+)", int)
    s["_mqf"] = f(r"Total_core_cache_fail_stats_breakdown\[GLOBAL_ACC_R\]\[MISS_QUEUE_FULL\] = (\d+)", int) or 0
    for k in ("pf_invariant", "funnel_invariant", "funnel_src_invariant"):
        s["_" + k] = f(r"snake_%s = (\w+)" % k, str)
    return s


def pct(a, b):
    return "%.1f" % (100.0 * a / b) if b else "-"


def name(app, args, cfg):
    return "%s / %s" % (app.replace("-rodinia-3.1", "").replace("parboil-", ""),
                        "(b) IWC0" if cfg.endswith("SNAKE_IWC0") else "(a) default")


def main():
    argv, prev = sys.argv[1:], []
    while argv and argv[0] == "-p":
        prev.append(argv[1].rsplit(":", 1))
        argv = argv[2:]
    gate, ref = argv[0], argv[1]
    rcfg = argv[2] if len(argv) > 2 else "QV100-SASS-PAPER_V100-100M_INSN"
    runs = []
    for p in sorted(glob.glob(os.path.join(gate, "*", "*", "*", "sim.out"))):
        app, args, cfg = p.split(os.sep)[-4:-1]
        rp = os.path.join(ref, app, args, rcfg, "sim.out")
        pp = [x for x in (os.path.join(d, app, args, c, "sim.out") for d, c in prev)
              if os.path.exists(x)]
        runs.append((name(app, args, cfg), parse(p),
                     parse(rp) if os.path.exists(rp) else None,
                     parse(pp[0]) if pp else None))

    g = lambda s, m, c: s.get("funnel_src_%s_%s" % (m, c), 0)
    print("## 1. Funnel by prediction source (% of that source's records)\n")
    print("| run | source | records (% of all) | wrong | late | timely | never issued in time | dup L1/MSHR | evicted / other | live |")
    print("|---" * 10 + "|")
    for n, s, r, q in runs:
        tot = s.get("pred_sectors", 0)
        for m, label in SRC:
            rec = sum(g(s, m, c) for c in ALL)
            if not rec:
                continue
            print("| %s | %s | %d (%s) | %s | %s | %s | %s | %s | %s | %s |" % (
                n, label, rec, pct(rec, tot),
                pct(g(s, m, "wrong_issued") + g(s, m, "wrong_not_issued"), rec),
                pct(g(s, m, "late"), rec), pct(g(s, m, "timely"), rec),
                pct(sum(g(s, m, c) for c in NEVER), rec),
                pct(g(s, m, "dup_l1") + g(s, m, "dup_mshr"), rec),
                pct(g(s, m, "evicted") + g(s, m, "issued_other"), rec),
                pct(g(s, m, "live"), rec)))

    print("\n## 2. Share of all wrong and all timely records by source (%)\n")
    print("| run | wrong records | " + " | ".join("wrong: " + l for _, l in SRC) +
          " | timely records | " + " | ".join("timely: " + l for _, l in SRC) + " |")
    print("|---" * (2 * len(SRC) + 3) + "|")
    for n, s, r, q in runs:
        w = {m: g(s, m, "wrong_issued") + g(s, m, "wrong_not_issued") for m, _ in SRC}
        t = {m: g(s, m, "timely") for m, _ in SRC}
        W, T = sum(w.values()), sum(t.values())
        print("| %s | %d | %s | %d | %s |" % (
            n, W, " | ".join(pct(w[m], W) for m, _ in SRC), T,
            " | ".join(pct(t[m], T) for m, _ in SRC)))

    print("\n## 3. Loads without an own inter-thread prediction (OQ-32)\n")
    print("| run | loads observed | no own inter-thread prediction | of which at a PC with a promoted entry | loads at a promoted PC1 without the bit | predictions without the bit | verified ok / failed / lost | T1 demotions |")
    print("|---" * 8 + "|")
    for n, s, r, q in runs:
        lo = s.get("loads_observed", 0)
        print("| %s | %d | %d (%s%%) | %d (%s%%) | %d (%s%%) | %d | %d / %d / %d | %d |" % (
            n, lo, s.get("no_it_pred_loads", 0), pct(s.get("no_it_pred_loads", 0), lo),
            s.get("no_it_pred_promoted_pc", 0), pct(s.get("no_it_pred_promoted_pc", 0), lo),
            s.get("promoted_nobit_loads", 0), pct(s.get("promoted_nobit_loads", 0), lo),
            s.get("pred_inter_thread_nobit", 0), s.get("nobit_verify_ok", 0),
            s.get("nobit_verify_fail", 0), s.get("nobit_verify_lost", 0),
            s.get("t1_demotions", 0)))

    print("\n## 4. Summary vs the bin_ref_v2 baseline%s\n" % (" (prev in parentheses)" if prev else ""))
    print("| run | done / invariants | coverage_pred | coverage_paper | accuracy_paper | dIPC | demand MISS_QUEUE_FULL base -> Snake | issued |")
    print("|---" * 8 + "|")
    bad = 0
    for n, s, r, q in runs:
        def cov(x):
            d = x.get("demand_sector_accesses", 0)
            return (100.0 * x.get("demand_pred_matched", 0) / d if d else 0,
                    100.0 * (x.get("pf_useful_timely", 0) + x.get("pf_late", 0)) / d if d else 0,
                    100.0 * x.get("pf_useful_timely", 0) / d if d else 0)
        def dipc(x):
            return 100.0 * (x["_ipc"] / r["_ipc"] - 1) if r and x["_ipc"] and r["_ipc"] else None
        a = cov(s)
        b = cov(q) if q else None
        ok = (s["_done"] and s.get("pf_woken_warps", 1) == 0 and s["_pf_invariant"] == "OK"
              and s["_funnel_invariant"] == "OK" and s["_funnel_src_invariant"] == "OK")
        bad += not ok
        fm = lambda i: "%.1f%s" % (a[i], " (%.1f)" % b[i] if b else "")
        di = dipc(s)
        print("| %s | %s | %s | %s | %s | %s | %s -> %d%s | %d |" % (
            n, "yes / OK" if ok else "**FAIL**", fm(0), fm(1), fm(2),
            ("%+.2f%%" % di if di is not None else "n/a") +
            (" (%+.2f%%)" % dipc(q) if q and dipc(q) is not None else ""),
            r["_mqf"] if r else "n/a", s["_mqf"], " (%d)" % q["_mqf"] if q else "",
            s.get("pf_issued", 0)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
