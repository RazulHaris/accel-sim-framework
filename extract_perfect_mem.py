#!/usr/bin/env python3
"""App-level comparison of the ideal-memory bound against the oracle arms.

    baseline        sim_run_oracle_v2/<b>/<a>/A100-SASS
    oracle_pref     sim_run_oracle_v2/<b>/<a>/A100-SASS-ORACLE_PREF
    oracle_allmiss  sim_run_oracle_all_miss/<b>/<a>/A100-SASS-ORACLE_ALL_MISS
    perfect_mem     sim_run_perfect_mem/<b>/<a>/A100-SASS-PERFECT_MEM

All four arms use the same A100-SASS config (GTO, 4 schedulers/core) and the
same traces, so cycle counts are directly comparable.

`perfect_mem` is -gpgpu_perfect_mem 1: the SM's outbound mem_fetch_interface
becomes perfect_memory_interface, so anything that would leave the core comes
straight back through the cluster response FIFO. It is a zero-off-SM-latency
bound, NOT a zero-cost-memory bound -- L1 tag probes, MSHR capacity, the
l1_latency pipeline and shared-memory bank conflicts are all still charged.

`headroom_pct` per oracle arm is the fraction of the baseline->perfect_mem
cycle gap that arm recovers.  It is only emitted when the ideal-memory
headroom is at least --min-headroom of baseline cycles, because below that the
ratio is dominated by simulation noise rather than by the prefetcher.
"""
import argparse, csv, glob, os, re, sys
from extract_ipc import DONE_MARKER

# Logs run to 700MB+ (myocyte). Every stat here is last-match-wins and the final
# cumulative block is a couple of MB, so scan a tail window and only fall back
# to a full read if a required stat is missing from it.
TAIL_BYTES = 32 << 20


def _tail(path, nbytes):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - nbytes)
        f.seek(start)
        buf = f.read()
    text = buf.decode("utf-8", errors="replace")
    # drop a partial first line unless we started at the top of the file
    return text if start == 0 else text[text.find("\n") + 1:], size <= nbytes

ARMS = [("baseline", "sim_run_oracle_v2",       "A100-SASS"),
        ("pref",     "sim_run_oracle_v2",       "A100-SASS-ORACLE_PREF"),
        ("allmiss",  "sim_run_oracle_all_miss", "A100-SASS-ORACLE_ALL_MISS"),
        ("perfect",  "sim_run_perfect_mem",     "A100-SASS-PERFECT_MEM")]

# stat -> (regex, type).  Last match in the file wins: accel-sim reprints a
# cumulative block after every kernel, so the final one covers the whole app.
STATS = [
    ("cycles",        r"^gpu_tot_sim_cycle\s*=\s*(\d+)",                 int),
    ("insn",          r"^gpu_tot_sim_insn\s*=\s*(\d+)",                  int),
    ("ipc",           r"^gpu_tot_ipc\s*=\s*([\d.]+)",                    float),
    ("l1d_acc",       r"^\s*L1D_total_cache_accesses\s*=\s*(\d+)",       int),
    ("l1d_miss",      r"^\s*L1D_total_cache_misses\s*=\s*(\d+)",         int),
    ("l1d_miss_rate", r"^\s*L1D_total_cache_miss_rate\s*=\s*([\d.]+)",   float),
    ("l1d_pend_hits", r"^\s*L1D_total_cache_pending_hits\s*=\s*(\d+)",   int),
    ("l1d_resv_fail", r"^\s*L1D_total_cache_reservation_fails\s*=\s*(\d+)", int),
    ("l2_acc",        r"^\s*L2_total_cache_accesses\s*=\s*(\d+)",        int),
    ("l2_miss",       r"^\s*L2_total_cache_misses\s*=\s*(\d+)",          int),
    ("dram_reads",    r"^total dram reads\s*=\s*(\d+)",                  int),
    ("dram_writes",   r"^total dram writes\s*=\s*(\d+)",                 int),
    ("shmem_bkconf",  r"^gpgpu_n_shmem_bkconflict\s*=\s*(\d+)",          int),
    ("l1_bkconf",     r"^gpgpu_n_l1cache_bkconflict\s*=\s*(\d+)",        int),
    ("issue_stall",   r"^Stall:(\d+)",                                   int),
    ("sb_stall",      r"W0_Scoreboard:(\d+)",                            int),
]
STATS = [(n, re.compile(p), t) for n, p, t in STATS]
COLS = [n for n, _, _ in STATS]


def _scan(text, out):
    for line in text.splitlines():
        if DONE_MARKER in line:
            out["complete"] = True
            continue
        for name, rx, cast in STATS:
            m = rx.search(line)
            if m:
                out[name] = cast(m.group(1))
    return out


def _has_marker(path):
    try:
        text, _ = _tail(path, 1 << 16)
        return DONE_MARKER in text
    except OSError:
        return False


def find_log(config_dir):
    """Output file for one job, preferring one that ran to completion.

    Same contract as extract_ipc.find_log, but the completion check reads a
    64KB tail rather than the whole file.
    """
    cands = []
    for pat in ("job.log", "gpgpu-sim-out_*.txt", "*.o[0-9]*"):
        cands.extend(glob.glob(os.path.join(config_dir, pat)))
    if not cands:
        return None
    done = [c for c in cands if _has_marker(c)]
    return max(done or cands, key=os.path.getsize)


# without these a row is not comparable, so a tail miss forces a full re-read
REQUIRED = ("cycles", "insn", "ipc", "l1d_acc", "dram_reads")


def parse_log(path):
    out = {"complete": False}
    try:
        text, whole = _tail(path, TAIL_BYTES)
        _scan(text, out)
        if whole or (out["complete"] and all(k in out for k in REQUIRED)):
            return out
        # tail window missed something: pay for the full scan
        out = {"complete": False}
        with open(path, "r", errors="replace") as f:
            _scan(f.read(), out)
    except OSError:
        return {"complete": False}
    return out


def pct(new, old):
    return round((new - old) / float(old) * 100.0, 2) if old else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("-o", "--out", default="perfect_mem_results/perfect_mem_ipc.csv")
    ap.add_argument("--min-headroom", type=float, default=0.5,
                    help="min baseline->perfect gap (%% of baseline cycles) to "
                         "report headroom_pct")
    a = ap.parse_args()

    base = os.path.join(a.root, "sim_run_perfect_mem")
    if not os.path.isdir(base):
        sys.exit("no sim_run_perfect_mem/ under %s" % a.root)
    pairs = [(b, ar) for b in sorted(os.listdir(base))
             if os.path.isdir(os.path.join(base, b)) and b != "gpgpu-sim-builds"
             for ar in sorted(os.listdir(os.path.join(base, b)))
             if os.path.isdir(os.path.join(base, b, ar))]

    rows = []
    for b, ar in pairs:
        row, st, missing = {"benchmark": b, "args": ar}, {}, []
        for label, rd, cfg in ARMS:
            d = os.path.join(a.root, rd, b, ar, cfg)
            log = find_log(d) if os.path.isdir(d) else None
            s = parse_log(log) if log else {}
            st[label] = s
            if not s.get("complete"):
                missing.append(label)
            for c in COLS:
                row["%s_%s" % (label, c)] = s.get(c, "")
        if missing:
            row["status"] = "INCOMPLETE(%s)" % ",".join(missing)
        else:
            bc, bi = st["baseline"]["cycles"], st["baseline"]["ipc"]
            for label in ("pref", "allmiss", "perfect"):
                row["%s_cycle_pct" % label] = pct(st[label]["cycles"], bc)
                row["%s_ipc_pct" % label] = pct(st[label]["ipc"], bi)
            gap = bc - st["perfect"]["cycles"]
            for label in ("pref", "allmiss"):
                row["%s_headroom_pct" % label] = (
                    round((bc - st[label]["cycles"]) / float(gap) * 100.0, 1)
                    if gap > 0 and gap / float(bc) * 100.0 >= a.min_headroom else "")
            row["insn_match"] = "yes" if len({st[l]["insn"] for l, _, _ in ARMS}) == 1 else "NO"
            row["status"] = "OK"
        rows.append(row)

    cols = (["benchmark", "args"]
            + ["%s_%s" % (l, c) for l, _, _ in ARMS for c in COLS]
            + ["%s_%s" % (l, s) for l in ("pref", "allmiss", "perfect")
               for s in ("cycle_pct", "ipc_pct")]
            + ["pref_headroom_pct", "allmiss_headroom_pct", "insn_match", "status"])
    out = a.out if os.path.isabs(a.out) else os.path.join(a.root, a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    ok = sum(1 for r in rows if r["status"] == "OK")
    print("wrote %s  (%d/%d complete)" % (out, ok, len(rows)))
    for r in rows:
        if r["status"] != "OK":
            print("  %-34s %s %s" % (r["benchmark"], r["args"][:24], r["status"]))


if __name__ == "__main__":
    main()
