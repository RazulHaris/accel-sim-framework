#!/usr/bin/env python3
"""Extract cycle counts and IPC from accel-sim run directories.

Walks <run_dir>/<benchmark>/<args>/<config>/ and pulls the final aggregate
stats out of whatever output file each job left behind.  Run directories
produced by different launchers name that file differently, so all three
conventions are searched:

    job.log                 (run_oracle_rodinia.sh / xargs launcher)
    gpgpu-sim-out_*.txt     (justrun.sh's tee output)
    *.o<jobid>              (slurm / procman)

Primary use - compare the oracle uncoalesced prefetcher against the baseline
arm of the same sweep, optionally cross-checking the baseline against older
sweeps:

    ./extract_ipc.py -r sim_run_oracle \
        --baseline-config A100-SASS \
        --test-config A100-SASS-ORACLE_PREF \
        --ref-run uncoal_final --ref-run sim_run_12.8 \
        -o oracle_rodinia_ipc.csv

Stats are taken from the LAST occurrence in the file: accel-sim prints a
cumulative block after every kernel, so the final one covers the whole app.
A run without the "exit detected" marker is reported with status=INCOMPLETE
and its numbers are left out of the totals, since a partial run's cycle count
is not comparable.
"""

import argparse
import csv
import glob
import os
import re
import sys

DONE_MARKER = "GPGPU-Sim: *** exit detected ***"

# stat name -> (regex, type).  Only the last match in the file is kept.
STATS = {
    "cycles": (re.compile(r"^gpu_tot_sim_cycle\s*=\s*(\d+)"), int),
    "insn": (re.compile(r"^gpu_tot_sim_insn\s*=\s*(\d+)"), int),
    "ipc": (re.compile(r"^gpu_tot_ipc\s*=\s*([\d.]+)"), float),
    "uncoal_warp_loads": (
        re.compile(r"^oracle_pref_uncoalesced_warp_loads\s*=\s*(\d+)"), int),
    "total_warp_loads": (
        re.compile(r"^oracle_pref_total_warp_loads\s*=\s*(\d+)"), int),
    "uncoal_misses": (
        re.compile(r"^oracle_pref_uncoalesced_misses\s*=\s*(\d+)"), int),
    "coal_misses": (
        re.compile(r"^oracle_pref_coalesced_misses\s*=\s*(\d+)"), int),
    "buffer_hits": (
        re.compile(r"^oracle_pref_buffer_hits\s*=\s*(\d+)"), int),
    "buffer_peak": (
        re.compile(r"^oracle_pref_buffer_size_peak\s*=\s*(\d+)"), int),
}


def find_log(config_dir):
    """Return the output file for one job, preferring a completed one."""
    candidates = []
    for pattern in ("job.log", "gpgpu-sim-out_*.txt", "*.o[0-9]*"):
        candidates.extend(glob.glob(os.path.join(config_dir, pattern)))
    if not candidates:
        return None
    # A directory can hold output from several attempts; prefer one that
    # actually ran to completion, then fall back to the largest.
    completed = [c for c in candidates if has_marker(c)]
    pool = completed if completed else candidates
    return max(pool, key=lambda p: os.path.getsize(p))


def has_marker(path):
    try:
        with open(path, "r", errors="replace") as f:
            return DONE_MARKER in f.read()
    except OSError:
        return False


def parse_log(path):
    """Last-value-wins scan for every stat of interest."""
    out = {"complete": False}
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if DONE_MARKER in line:
                    out["complete"] = True
                    continue
                if not line or line[0] not in "go":
                    continue  # cheap filter: every stat starts with g or o
                for key, (pattern, cast) in STATS.items():
                    m = pattern.match(line)
                    if m:
                        out[key] = cast(m.group(1))
                        break
    except OSError as e:
        print("warning: cannot read %s: %s" % (path, e), file=sys.stderr)
    return out


def collect(run_dir, configs):
    """{(benchmark, args): {config: stats}} for one run directory."""
    results = {}
    if not os.path.isdir(run_dir):
        print("warning: no such run directory: %s" % run_dir, file=sys.stderr)
        return results
    for bench in sorted(os.listdir(run_dir)):
        bench_dir = os.path.join(run_dir, bench)
        if not os.path.isdir(bench_dir) or bench == "gpgpu-sim-builds":
            continue
        for args in sorted(os.listdir(bench_dir)):
            args_dir = os.path.join(bench_dir, args)
            if not os.path.isdir(args_dir):
                continue
            for config in configs:
                config_dir = os.path.join(args_dir, config)
                if not os.path.isdir(config_dir):
                    continue
                log = find_log(config_dir)
                if log is None:
                    continue
                results.setdefault((bench, args), {})[config] = parse_log(log)
    return results


def pct(new, old):
    """Percent change of new relative to old."""
    if not old:
        return ""
    return round((new - old) / float(old) * 100.0, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--run-dir", default="sim_run_oracle",
                    help="run directory holding both arms (default: sim_run_oracle)")
    ap.add_argument("--baseline-config", default="A100-SASS")
    ap.add_argument("--test-config", default="A100-SASS-ORACLE_PREF")
    ap.add_argument("--ref-run", action="append", default=[],
                    help="older run directory to cross-check the baseline "
                         "against; repeatable")
    ap.add_argument("-o", "--out", default="oracle_rodinia_ipc.csv")
    opts = ap.parse_args()

    base_cfg, test_cfg = opts.baseline_config, opts.test_config
    main_results = collect(opts.run_dir, [base_cfg, test_cfg])
    refs = {ref: collect(ref, [base_cfg]) for ref in opts.ref_run}

    columns = [
        "benchmark", "args",
        "baseline_cycles", "oracle_cycles", "cycle_change_pct",
        "baseline_ipc", "oracle_ipc", "ipc_change_pct",
        "insn_match",
        "baseline_uncoal_misses", "oracle_uncoal_misses",
        "baseline_coal_misses", "oracle_coal_misses",
        "oracle_buffer_hits", "oracle_buffer_size_peak",
        "uncoal_warp_load_pct",
        "status",
    ]
    for ref in opts.ref_run:
        columns += ["%s_cycles" % ref, "%s_matches_baseline" % ref]

    rows = []
    for (bench, args) in sorted(main_results):
        got = main_results[(bench, args)]
        b, t = got.get(base_cfg, {}), got.get(test_cfg, {})

        status = []
        if not b:
            status.append("NO_BASELINE")
        elif not b.get("complete"):
            status.append("BASELINE_INCOMPLETE")
        if not t:
            status.append("NO_ORACLE")
        elif not t.get("complete"):
            status.append("ORACLE_INCOMPLETE")
        comparable = not status
        if comparable:
            status.append("OK")

        row = {
            "benchmark": bench,
            "args": args,
            "baseline_cycles": b.get("cycles", ""),
            "oracle_cycles": t.get("cycles", ""),
            "baseline_ipc": b.get("ipc", ""),
            "oracle_ipc": t.get("ipc", ""),
            "baseline_uncoal_misses": b.get("uncoal_misses", ""),
            "oracle_uncoal_misses": t.get("uncoal_misses", ""),
            "baseline_coal_misses": b.get("coal_misses", ""),
            "oracle_coal_misses": t.get("coal_misses", ""),
            "oracle_buffer_hits": t.get("buffer_hits", ""),
            "oracle_buffer_size_peak": t.get("buffer_peak", ""),
            "status": "+".join(status),
        }

        if comparable:
            row["cycle_change_pct"] = pct(t["cycles"], b["cycles"])
            row["ipc_change_pct"] = pct(t["ipc"], b["ipc"])
            # Same trace on both arms, so the retired instruction count must
            # match; a mismatch means the two runs are not comparable.
            row["insn_match"] = "yes" if b.get("insn") == t.get("insn") else "NO"
        else:
            row["cycle_change_pct"] = row["ipc_change_pct"] = row["insn_match"] = ""

        total = b.get("total_warp_loads") or t.get("total_warp_loads")
        uncoal = b.get("uncoal_warp_loads")
        if uncoal is None:
            uncoal = t.get("uncoal_warp_loads")
        row["uncoal_warp_load_pct"] = (
            round(uncoal / float(total) * 100.0, 2) if total else "")

        for ref in opts.ref_run:
            r = refs[ref].get((bench, args), {}).get(base_cfg, {})
            rc = r.get("cycles", "") if r.get("complete") else ""
            row["%s_cycles" % ref] = rc
            if rc == "" or not b.get("complete"):
                # nothing to compare against - don't report a mismatch that is
                # really just a missing baseline
                row["%s_matches_baseline" % ref] = ""
            else:
                row["%s_matches_baseline" % ref] = (
                    "yes" if rc == b.get("cycles") else "NO")
        rows.append(row)

    with open(opts.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "OK"]
    print("wrote %s: %d rows (%d comparable)" % (opts.out, len(rows), len(ok)))
    if len(ok) < len(rows):
        for r in rows:
            if r["status"] != "OK":
                print("  %-28s %-40s %s" % (r["benchmark"], r["args"], r["status"]))

    if ok:
        tb = sum(r["baseline_cycles"] for r in ok)
        tt = sum(r["oracle_cycles"] for r in ok)
        print("\ntotal cycles over %d comparable runs: baseline %d, oracle %d "
              "(%+.2f%%)" % (len(ok), tb, tt, (tt - tb) / float(tb) * 100.0))
        faster = [r for r in ok if r["oracle_cycles"] < r["baseline_cycles"]]
        slower = [r for r in ok if r["oracle_cycles"] > r["baseline_cycles"]]
        print("oracle faster on %d, slower on %d, unchanged on %d"
              % (len(faster), len(slower), len(ok) - len(faster) - len(slower)))
        mismatched = [r for r in ok if r["insn_match"] == "NO"]
        if mismatched:
            print("WARNING: instruction count differs between arms for: %s"
                  % ", ".join(r["benchmark"] for r in mismatched))


if __name__ == "__main__":
    main()
