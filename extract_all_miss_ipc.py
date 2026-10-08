#!/usr/bin/env python3
"""App-level comparison of the two oracle variants against the baseline.

    baseline        sim_run_oracle_v2/<b>/<a>/A100-SASS
    oracle_pref     sim_run_oracle_v2/<b>/<a>/A100-SASS-ORACLE_PREF      (uncoalesced misses only)
    oracle_allmiss  sim_run_oracle_all_miss/<b>/<a>/A100-SASS-ORACLE_ALL_MISS  (every global load miss)

All three use the same A100 config and the same traces; log discovery and stat
parsing come from extract_ipc.py so the numbers match the v2 results exactly.
"""
import argparse, csv, os, sys
from extract_ipc import find_log, parse_log

ARMS = [("baseline",  "sim_run_oracle_v2",       "A100-SASS"),
        ("pref",      "sim_run_oracle_v2",       "A100-SASS-ORACLE_PREF"),
        ("allmiss",   "sim_run_oracle_all_miss", "A100-SASS-ORACLE_ALL_MISS")]
STATS = ["cycles", "ipc", "insn", "uncoal_misses", "coal_misses",
         "buffer_hits", "buffer_peak"]


def pct(new, old):
    return round((new - old) / float(old) * 100.0, 2) if old else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("-o", "--out", default="oracle_all_miss_results/oracle_all_miss_ipc.csv")
    a = ap.parse_args()

    base = os.path.join(a.root, "sim_run_oracle_all_miss")
    pairs = [(b, ar) for b in sorted(os.listdir(base))
             if os.path.isdir(os.path.join(base, b)) and b != "gpgpu-sim-builds"
             for ar in sorted(os.listdir(os.path.join(base, b)))
             if os.path.isdir(os.path.join(base, b, ar))]

    rows = []
    for b, ar in pairs:
        row = {"benchmark": b, "args": ar}
        st, missing = {}, []
        for label, rd, cfg in ARMS:
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
            for label in ("pref", "allmiss"):
                row["%s_cycle_change_pct" % label] = pct(st[label]["cycles"], st["baseline"]["cycles"])
                row["%s_ipc_change_pct" % label] = pct(st[label]["ipc"], st["baseline"]["ipc"])
            row["allmiss_vs_pref_ipc_pct"] = pct(st["allmiss"]["ipc"], st["pref"]["ipc"])
            row["insn_match"] = "yes" if len({st[l].get("insn") for l, _, _ in ARMS}) == 1 else "NO"
            row["status"] = "OK"
        rows.append(row)

    cols = (["benchmark", "args"]
            + ["%s_%s" % (l, c) for l, _, _ in ARMS for c in STATS]
            + ["pref_cycle_change_pct", "pref_ipc_change_pct",
               "allmiss_cycle_change_pct", "allmiss_ipc_change_pct",
               "allmiss_vs_pref_ipc_pct", "insn_match", "status"])
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
    if ok:
        for l in ("baseline", "pref", "allmiss"):
            print("total cycles %-9s %d" % (l, sum(int(r["%s_cycles" % l]) for r in ok)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
