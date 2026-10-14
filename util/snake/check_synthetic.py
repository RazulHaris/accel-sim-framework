#!/usr/bin/env python3
"""Check Snake's training on the synthetic traces (SNAKE_PLAN.md S2 gate).

    check_synthetic.py <runs dir> <trace root> [<config> [<apps>]]

<runs dir>/<app>/NO_ARGS/<config>/sim.out must come from a run with
-snake_enable 1 -snake_debug_trace 1 (any config directory if <config> is
not given; there must then be exactly one); <trace root>/manifest.json is written by
gen_synthetic_traces.py.  <apps> (comma-separated, default all) limits the
check to those snake-synth apps.  Exit status 0 only if every check passes.

T1/T2 are compared by state name (OQ-5).  The figure prints codes for them
that differ from the OQ-5 labels; the mapping is printed, not asserted.

From S3a on (prefetches issued), every run must also finish ("exit
detected"), wake no warp with a prefetch and keep the outcome invariant
(snake_pf_invariant OK).  Trace (d) needs coverage_paper >= 90% with the
exact footprint (-snake_footprint 1; reported for the line footprint), trace
(a) is a saturation test (its demand loads fill the L1 miss queue; the
MISS_QUEUE_FULL failures and the prefetch drops are reported) and trace (b)
must issue almost nothing (< 1% of its demand sectors).  coverage_pred
(OQ-14) is reported next to coverage_paper and accuracy_paper.

From S3b on (inter-warp issue), Fig. 15 (c) is also checked against the
sectors issued for the 9th observe ("PF" debug lines).  With
-snake_interwarp_chain_depth 0 the predictions are exactly the figure's
{2600, 2700, 2800, 3000}; with depth >= 1 (default 3) the inter-warp targets
W2..W4 also follow the trained 520->540 stride 500 (OQ-29), which adds
{3100, 3200, 3300}.
"""
import glob
import json
import os
import re
import sys

PF_RE = re.compile(r"^n=(\d+) mode=(\d+) sector=(\d+) res=(\w+)$")
DBG = re.compile(r"^SNAKE_DBG sm=(\d+) cyc=(\d+) (\S+) ?(.*)$")
KV = re.compile(r"(\w+)=(\S+)")
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def parse(path):
    """Per SM: observe list, predictions per observe, state per observe."""
    sms = {}
    cur_state = None
    for line in open(path, errors="replace"):
        m = DBG.match(line.rstrip("\n"))
        if not m:
            continue
        sm = int(m.group(1))
        d = sms.setdefault(sm, {"obs": [], "preds": {}, "state": {}, "pf": []})
        kind, rest = m.group(3), m.group(4)
        kv = dict(KV.findall(rest))
        if kind == "OBS":
            n = int(kv["n"])
            d["obs"].append((n, int(kv["w"]), int(kv["pc"], 16), int(kv["addr"])))
            d["preds"][n] = []
        elif kind == "PRED" and d["obs"]:
            d["preds"][d["obs"][-1][0]].append(
                (kv["mode"], int(kv["target_w"]), int(kv["addr"])))
        elif kind == "PF":
            pm = PF_RE.match(rest)
            d["pf"].append((int(pm.group(1)), int(pm.group(2)),
                            int(pm.group(3)), pm.group(4)))
        elif kind == "STATE_BEGIN":
            cur_state = {"head": {}, "tail": []}
        elif kind == "S_HEAD":
            cur_state["head"][int(kv["w"])] = (int(kv["pc"], 16), int(kv["addr"]))
        elif kind == "TAIL" and rest.startswith("S_TAIL"):
            cur_state["tail"].append(tail_fields(kv))
        elif kind == "STATE_END":
            d["state"][int(kv["n"])] = cur_state
    return sms


def tail_fields(kv):
    return {
        "pc1": int(kv["pc1"], 16),
        "pc2": None if kv["pc2"] == "-" else int(kv["pc2"], 16),
        "it": int(kv["it"]),
        "wvec": int(kv["wvec"], 16),
        "t1": kv["t1"],
        "iw": None if kv["iw"] == "-" else int(kv["iw"]),
        "intra": None if kv["intra"] == "-" else int(kv["intra"]),
        "t2": kv["t2"],
    }


def stats(path):
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith("snake_"):
            k, _, v = line.partition(" = ")
            try:
                out[k.strip()] = int(v)
            except ValueError:
                out[k.strip()] = v.split()[0]
        elif line.startswith("-snake_footprint"):
            out["footprint"] = int(line.split()[1])
        elif line.startswith("-snake_interwarp_chain_depth"):
            out["iw_chain"] = int(line.split()[1])
        elif "exit detected" in line:
            out["done"] = True
        elif "Total_core_cache_fail_stats_breakdown[GLOBAL_ACC_R][MISS_QUEUE_FULL]" in line:
            out["mq_full"] = int(line.split("=")[1])
    return out


def check_issue(path, st):
    """S3a: no deadlock, no warp woken by a prefetch, invariant."""
    check(st.get("done", False), "run finished (no deadlock)")
    check(st.get("snake_pf_woken_warps") == 0,
          "pf_woken_warps == 0: %s" % st.get("snake_pf_woken_warps"))
    check(st.get("snake_pf_invariant") == "OK",
          "outcome invariant: %s" % st.get("snake_pf_invariant"))
    dem = max(st.get("snake_demand_sector_accesses", 0), 1)
    tl = st.get("snake_pf_useful_timely", 0) + st.get("snake_pf_late", 0)
    print("  info  footprint=%s issued=%s timely=%s late=%s evicted_unused=%s "
          "unused_at_end=%s demand=%s" % (
              st.get("footprint"), st.get("snake_pf_issued"),
              st.get("snake_pf_useful_timely"), st.get("snake_pf_late"),
              st.get("snake_pf_evicted_unused"),
              st.get("snake_pf_unused_at_end_total"), dem))
    print("  info  coverage_paper=%.1f%% accuracy_paper=%.1f%% coverage_pred=%.1f%%; "
          "dropped: line %s, MSHR %s, miss queue %s, prefetch queue full %s; "
          "demand MISS_QUEUE_FULL %s" % (
              100.0 * tl / dem, 100.0 * st.get("snake_pf_useful_timely", 0) / dem,
              100.0 * st.get("snake_demand_pred_matched", 0) / dem,
              st.get("snake_pf_drop_line"), st.get("snake_pf_drop_mshr"),
              st.get("snake_pf_drop_miss_queue"), st.get("snake_pf_drop_queue_full"),
              st.get("mq_full", 0)))
    return tl / dem


def bits(*warps):
    return sum(1 << w for w in warps)


def find_tail(state, **want):
    return [e for e in state["tail"]
            if all(e[k] == v for k, v in want.items())]


def sim_out(runs, app, cfg):
    hits = glob.glob(os.path.join(runs, app, "NO_ARGS", cfg, "sim.out"))
    if len(hits) != 1:
        failures.append("%s: expected one sim.out, found %d" % (app, len(hits)))
        return None
    return hits[0]


def check_fig15(path, p):
    print("== (c) Fig. 15 example: %s" % path)
    st = stats(path)
    check_issue(path, st)
    d = parse(path)
    check(list(d) == [0], "all observes on SM 0 (one CTA)")
    d = d.get(0, {"obs": [], "preds": {}, "state": {}, "pf": []})
    W = p["warp_of"]
    A, B = p["pc_a"], p["pc_b"]
    want = ([(W["W%d" % (k + 1)], A, p["phase_a"][k]) for k in range(4)] +
            [(W["W%d" % (k + 1)], B, p["phase_b"][k]) for k in range(4)] +
            [(W["W1"], A, p["phase_c"])])
    got = [(w, pc, a) for _, w, pc, a in d["obs"]]
    check(got == want, "observe order W1..W4 @520, W1..W4 @540, W1 @520: %s" % got)
    if got != want:
        return

    def preds(ns):
        return sorted({a for n in ns for _, _, a in d["preds"].get(n, [])})

    # Phase (a): after the 4th observe.
    s = d["state"][4]
    check(s["head"] == {W["W%d" % (k + 1)]: (A, p["phase_a"][k]) for k in range(4)},
          "(a) Head: W1..W4 = (520, 1000/1100/1200/1300)")
    check(len(s["tail"]) == 1 and find_tail(s, pc1=A, pc2=None, iw=100, wvec=0),
          "(a) Tail: one entry PC1=520, no PC2, inter-warp 100: %s" % s["tail"])
    check(preds(range(1, 5)) == [1300], "(a) predictions {1300}: %s" % preds(range(1, 5)))

    # Phase (b): after the 8th observe.
    s = d["state"][8]
    check(s["head"] == {W["W%d" % (k + 1)]: (B, p["phase_b"][k]) for k in range(4)},
          "(b) Head: W1..W4 = (540, 1500/1600/1700/1800)")
    e1 = find_tail(s, pc1=A, pc2=B, it=500, iw=100, wvec=bits(1, 2, 3, 4),
                   t1="promoted", intra=None)
    e2 = find_tail(s, pc1=B, pc2=None, iw=100, wvec=0)
    check(len(s["tail"]) == 2 and e1 and e2,
          "(b) Tail: 520->540 stride 500 W{1,2,3,4} iw 100 T1 promoted; "
          "540 (no PC2) iw 100: %s" % s["tail"])
    check(preds(range(5, 9)) == [1800], "(b) predictions {1800}: %s" % preds(range(5, 9)))

    # Phase (c): after the 9th observe.
    s = d["state"][9]
    check(s["head"].get(W["W1"]) == (A, p["phase_c"]), "(c) Head: W1 = (520, 2500)")
    check(all(s["head"].get(W["W%d" % (k + 1)]) == (B, p["phase_b"][k])
              for k in range(1, 4)), "(c) Head: W2..W4 unchanged at 540")
    e1 = find_tail(s, pc1=A, pc2=B, it=500, iw=100, wvec=bits(1, 2, 3, 4),
                   t1="trained", intra=1500, t2="observed")
    e2 = find_tail(s, pc1=B, pc2=A, it=1000, iw=100, wvec=bits(1), t1="observed")
    check(len(s["tail"]) == 2 and e1 and e2,
          "(c) Tail: 520->540 T1 trained, intra-warp 1500 (T2 observed); "
          "540->520 stride 1000 W{1} T1 observed: %s" % s["tail"])
    # S3b: the figure's queue, plus the inter-warp targets' chain (OQ-29).
    chain = st.get("iw_chain")
    want_c = [2600, 2700, 2800, 3000] + ([3100, 3200, 3300] if chain else [])
    check(chain is not None, "-snake_interwarp_chain_depth in the option dump: %s"
          % chain)
    check(preds([9]) == want_c, "(c) predictions %s (interwarp_chain_depth %s): %s"
          % (want_c, chain, preds([9])))
    # Issued sectors of the 9th observe: footprint 0 requests each predicted
    # address's 128 B line, footprint 1 the 32 lanes 4 B apart from it; none
    # of them is in L1 or in flight before.
    if st.get("footprint") == 1:
        want_sec = {(a + 4 * i) // 32 * 32 for a in want_c for i in range(32)}
    else:
        want_sec = {a // 128 * 128 + 32 * k for a in want_c for k in range(4)}
    got_sec = {sec for n, _, sec, res in d["pf"] if n == 9 and res == "issued"}
    check(got_sec == want_sec, "(c) issued sectors (footprint %s) = %s: %s"
          % (st.get("footprint"), sorted(want_sec), sorted(got_sec)))
    modes = sorted({(m, a) for m, _, a in d["preds"].get(9, [])})
    print("  info  (c) prediction modes: %s" % modes)
    print("  note  T1/T2 codes: Fig. 15 prints the promoted entry in (b) as '01',"
          " the trained one in (c) as '11', and the one-warp entry and the new"
          " intra stride in (c) as '00'; OQ-5 labels them promoted (10),"
          " trained (11) and observed (01).  States agree; only the printed"
          " codes differ.")


def check_strided(path, p):
    print("== (a) strided loop: %s" % path)
    d = parse(path)
    st = stats(path)
    P1, P2, P3 = p["pcs"]
    chain = [(P1, P2, p["D1"]), (P2, P3, p["D2"]),
             (P3, P1, p["IT"] - p["D1"] - p["D2"])]
    check_issue(path, st)  # S3: saturation test, no coverage gate
    sms = sorted(sm for sm in d if d[sm]["state"])
    check(len(sms) == p["ctas"], "one SM per CTA trained: SMs %s" % sms)
    for sm in sms:
        s = d[sm]["state"][max(d[sm]["state"])]
        all_w = bits(*range(p["warps"]))
        for pc1, pc2, it in chain:
            e = find_tail(s, pc1=pc1, pc2=pc2, it=it)
            ok = (len(e) == 1 and e[0]["t1"] in ("promoted", "trained") and
                  e[0]["wvec"] == all_w and e[0]["iw"] == p["WS"] and
                  e[0]["intra"] == p["IT"] and e[0]["t2"] == "trained")
            check(ok, "SM %d: 0x%x->0x%x stride %d promoted/trained, all %d warps,"
                  " inter-warp %d, intra-warp %d trained: %s"
                  % (sm, pc1, pc2, it, p["warps"], p["WS"], p["IT"], e))
    check(st.get("snake_t1_promotions", 0) >= 3 * p["ctas"],
          "t1_promotions >= 3 per SM: %s" % st.get("snake_t1_promotions"))
    print("  info  t1_promotions=%s t1_trained=%s interwarp_trained=%s "
          "t2_trained=%s tail_hits/lookups=%s/%s"
          % tuple(st.get("snake_" + k) for k in
                  ("t1_promotions", "t1_trained", "interwarp_trained",
                   "t2_trained", "tail_hits", "tail_lookups")))


def check_strided_dep(path, p):
    print("== (d) strided loop, one memory wait per iteration: %s" % path)
    st = stats(path)
    cov = check_issue(path, st)
    if st.get("footprint") == 1:
        check(cov >= 0.90, "coverage_paper >= 90%% (exact footprint): %.1f%%"
              % (100 * cov))
    else:
        print("  info  coverage_paper with the line footprint: %.1f%%" % (100 * cov))


def check_random(path, p):
    print("== (b) random gather: %s" % path)
    st = stats(path)
    check_issue(path, st)
    dem = st.get("snake_demand_sector_accesses", 0)
    check(st.get("snake_pf_issued", 0) < 0.01 * max(dem, 1),
          "issues almost nothing (< 1%% of %d demand sectors): %s"
          % (dem, st.get("snake_pf_issued")))
    for k in ("t1_promotions", "interwarp_trained", "t2_trained"):
        check(st.get("snake_" + k, -1) == 0, "%s == 0: %s" % (k, st.get("snake_" + k)))
    # Scattered P3 lanes fit neither equal strides (lane mode 0) nor any run
    # length (lane mode 1, OQ-17), so every P3 load is excluded.
    excl = (st.get("snake_warp_excluded_lane_stride", 0) +
            st.get("snake_warp_excluded_lane_runs", 0))
    n_p3 = p["ctas"] * p["warps"] * p["iters"]
    check(excl == n_p3, "all %d scattered-lane loads excluded: lane-stride %s, "
          "lane-runs %s" % (n_p3, st.get("snake_warp_excluded_lane_stride"),
                            st.get("snake_warp_excluded_lane_runs")))
    print("  info  loads_observed=%s tail_lookups=%s tail_allocs=%s "
          "tail_evictions=%s" % tuple(st.get("snake_" + k) for k in
                                      ("loads_observed", "tail_lookups",
                                       "tail_allocs", "tail_evictions")))


def main():
    runs, root = sys.argv[1], sys.argv[2]
    cfg = sys.argv[3] if len(sys.argv) > 3 else "*"
    apps = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    man = json.load(open(os.path.join(root, "manifest.json")))
    for app, fn in (("snake-synth-fig15", check_fig15),
                    ("snake-synth-strided", check_strided),
                    ("snake-synth-strided-dep", check_strided_dep),
                    ("snake-synth-random", check_random)):
        if apps is not None and app not in apps:
            continue
        path = sim_out(runs, app, cfg)
        if path:
            fn(path, man[app])
    print("\n%s: %d failure(s)" % ("PASS" if not failures else "FAIL", len(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
