#!/usr/bin/env python3
"""Collect the oracle uncoalesced prefetcher's L1D cache-size sweep.

Three points, all with the oracle enabled and an identical shared-memory
carveout per kernel -- only the unified L1D+shmem pool differs:

    128KB   sim_run_l1d_sweep/<bench>/<args>/A100-SASS-L1D_128KB-ORACLE_PREF
    192KB   sim_run_oracle_v2/<bench>/<args>/A100-SASS-ORACLE_PREF   (the v2 run)
    256KB   sim_run_l1d_sweep/<bench>/<args>/A100-SASS-L1D_256KB-ORACLE_PREF

Log discovery and stat parsing are reused from extract_ipc.py so the numbers
are extracted exactly the same way as the v2 results.

    ./extract_l1d_sweep.py -o l1d_sweep_results.csv
"""

import argparse, csv, os, sys
from extract_ipc import find_log, parse_log

POINTS = [
    ("128", "sim_run_l1d_sweep",  "A100-SASS-L1D_128KB-ORACLE_PREF"),
    ("192", "sim_run_oracle_v2",  "A100-SASS-ORACLE_PREF"),
    ("256", "sim_run_l1d_sweep",  "A100-SASS-L1D_256KB-ORACLE_PREF"),
]
# stats carried through per point, in column order
STAT_COLS = ["cycles", "ipc", "insn", "uncoal_misses", "coal_misses",
             "buffer_hits", "buffer_peak"]


def pct(new, old):
    return round((new - old) / float(old) * 100.0, 2) if old else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("-o", "--out", default="l1d_sweep_results.csv")
    a = ap.parse_args()

    # enumerate benchmark/args pairs from the sweep directory
    sweep = os.path.join(a.root, "sim_run_l1d_sweep")
    pairs = []
    for bench in sorted(os.listdir(sweep)):
        bd = os.path.join(sweep, bench)
        if not os.path.isdir(bd) or bench == "gpgpu-sim-builds":
            continue
        for args in sorted(os.listdir(bd)):
            if os.path.isdir(os.path.join(bd, args)):
                pairs.append((bench, args))

    rows = []
    for bench, args in pairs:
        row = {"benchmark": bench, "args": args}
        stats, missing = {}, []
        for label, run_dir, cfg in POINTS:
            d = os.path.join(a.root, run_dir, bench, args, cfg)
            log = find_log(d) if os.path.isdir(d) else None
            s = parse_log(log) if log else {}
            stats[label] = s
            if not s.get("complete"):
                missing.append(label)
            for c in STAT_COLS:
                row["l1_%s_%s" % (label, c)] = s.get(c, "")

        if missing:
            row["status"] = "INCOMPLETE(" + ",".join(missing) + ")"
        else:
            ref = stats["192"]
            for label in ("128", "256"):
                row["ipc_%s_vs_192_pct" % label] = pct(stats[label]["ipc"], ref["ipc"])
                row["cycles_%s_vs_192_pct" % label] = pct(stats[label]["cycles"], ref["cycles"])
            insns = {stats[l].get("insn") for l in ("128", "192", "256")}
            row["insn_match"] = "yes" if len(insns) == 1 else "NO"
            row["status"] = "OK"
        rows.append(row)

    cols = (["benchmark", "args"]
            + ["l1_%s_%s" % (l, c) for l, _, _ in POINTS for c in STAT_COLS]
            + ["ipc_128_vs_192_pct", "ipc_256_vs_192_pct",
               "cycles_128_vs_192_pct", "cycles_256_vs_192_pct",
               "insn_match", "status"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok = [r for r in rows if r["status"] == "OK"]
    bad = [r for r in rows if r["status"] != "OK"]
    print("wrote %s: %d rows (%d complete, %d incomplete)"
          % (a.out, len(rows), len(ok), len(bad)))
    for r in bad:
        print("  %-32s %-34s %s" % (r["benchmark"], r["args"][:34], r["status"]))
    mism = [r for r in ok if r["insn_match"] == "NO"]
    if mism:
        print("WARNING: instruction count differs across pools for: %s"
              % ", ".join(r["benchmark"] for r in mism))
    return 0


if __name__ == "__main__":
    sys.exit(main())
