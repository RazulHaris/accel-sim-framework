#!/usr/bin/env python3
"""App-level comparison of any number of simulation arms against a baseline.

An arm is "label:run_dir:config_subdir".  The FIRST arm is the baseline that
every other arm's percentage change is computed against.  Log discovery and
stat parsing come from extract_ipc.py, so numbers match the v2 results exactly.

Default arms reproduce oracle_all_miss_results/oracle_all_miss_ipc.csv:

    ./extract_arms_ipc.py -o oracle_all_miss_results/oracle_all_miss_ipc.csv

The LRR-scheduler comparison:

    ./extract_arms_ipc.py \
        --arm baseline:sim_run_oracle_v2:A100-SASS \
        --arm allmiss_gto:sim_run_oracle_all_miss:A100-SASS-ORACLE_ALL_MISS \
        --arm allmiss_lrr:sim_run_all_miss_lrr:A100-SASS-RR-ORACLE_ALL_MISS \
        -o oracle_all_miss_lrr_results/oracle_all_miss_lrr_ipc.csv
"""
import argparse, csv, os, sys
from extract_ipc import find_log, parse_log

DEFAULT_ARMS = ["baseline:sim_run_oracle_v2:A100-SASS",
                "pref:sim_run_oracle_v2:A100-SASS-ORACLE_PREF",
                "allmiss:sim_run_oracle_all_miss:A100-SASS-ORACLE_ALL_MISS"]
STATS = ["cycles", "ipc", "insn", "uncoal_misses", "coal_misses",
         "buffer_hits", "buffer_peak"]


def pct(new, old):
    return round((new - old) / float(old) * 100.0, 2) if old else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--arm", action="append", metavar="LABEL:RUN_DIR:CONFIG",
                    help="repeatable; the first is the baseline")
    ap.add_argument("--enumerate-from", default=None,
                    help="run dir to enumerate benchmark/args pairs from "
                         "(default: the last arm's run dir)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    specs = a.arm or DEFAULT_ARMS
    arms = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 3:
            sys.exit("bad --arm %r; want LABEL:RUN_DIR:CONFIG" % s)
        arms.append(tuple(parts))
    base_label = arms[0][0]

    enum_dir = os.path.join(a.root, a.enumerate_from or arms[-1][1])
    pairs = [(b, ar) for b in sorted(os.listdir(enum_dir))
             if os.path.isdir(os.path.join(enum_dir, b)) and b != "gpgpu-sim-builds"
             for ar in sorted(os.listdir(os.path.join(enum_dir, b)))
             if os.path.isdir(os.path.join(enum_dir, b, ar))]

    rows = []
    for b, ar in pairs:
        row = {"benchmark": b, "args": ar}
        st, missing = {}, []
        for label, rd, cfg in arms:
            d = os.path.join(a.root, rd, b, ar, cfg)
            log = find_log(d) if os.path.isdir(d) else None
            s = parse_log(log) if log else {}
            st[label] = s
            if not s.get("complete"):
                missing.append(label)
            for c in STATS:
                row["%s_%s" % (label, c)] = s.get(c, "")
        if missing:
            row["status"] = "INCOMPLETE(%s)" % ",".join(missing)
        else:
            for label, _, _ in arms[1:]:
                row["%s_cycle_change_pct" % label] = pct(st[label]["cycles"], st[base_label]["cycles"])
                row["%s_ipc_change_pct" % label] = pct(st[label]["ipc"], st[base_label]["ipc"])
            row["insn_match"] = "yes" if len({st[l].get("insn") for l, _, _ in arms}) == 1 else "NO"
            row["status"] = "OK"
        rows.append(row)

    cols = (["benchmark", "args"]
            + ["%s_%s" % (l, c) for l, _, _ in arms for c in STATS]
            + ["%s_%s" % (l, k) for l, _, _ in arms[1:]
               for k in ("cycle_change_pct", "ipc_change_pct")]
            + ["insn_match", "status"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r["status"] == "OK"]
    print("wrote %s: %d rows (%d complete)" % (a.out, len(rows), len(ok)))
    for r in rows:
        if r["status"] != "OK":
            print("  %-30s %-32s %s" % (r["benchmark"], r["args"][:32], r["status"]))
    bad = [r for r in ok if r["insn_match"] == "NO"]
    if bad:
        print("WARNING: instruction count differs across arms for: %s"
              % ", ".join(r["benchmark"] for r in bad))
    for l, _, _ in arms:
        print("total cycles %-14s %d" % (l, sum(int(r["%s_cycles" % l]) for r in ok)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
