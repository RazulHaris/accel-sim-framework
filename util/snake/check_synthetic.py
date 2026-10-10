#!/usr/bin/env python3
"""Check Snake's training on the synthetic traces (SNAKE_PLAN.md S2 gate).

    check_synthetic.py <runs dir> <trace root>

<runs dir>/<app>/NO_ARGS/<config>/sim.out must come from a run with
-snake_enable 1 -snake_debug_trace 1; <trace root>/manifest.json is written by
gen_synthetic_traces.py.  Exit status 0 only if every check passes.

T1/T2 are compared by state name (OQ-5).  The figure prints codes for them
that differ from the OQ-5 labels; the mapping is printed, not asserted.
"""
import glob
import json
import os
import re
import sys

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
        d = sms.setdefault(sm, {"obs": [], "preds": {}, "state": {}})
        kind, rest = m.group(3), m.group(4)
        kv = dict(KV.findall(rest))
        if kind == "OBS":
            n = int(kv["n"])
            d["obs"].append((n, int(kv["w"]), int(kv["pc"], 16), int(kv["addr"])))
            d["preds"][n] = []
        elif kind == "PRED" and d["obs"]:
            d["preds"][d["obs"][-1][0]].append(
                (kv["mode"], int(kv["target_w"]), int(kv["addr"])))
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
            out[k.strip()] = int(v)
    return out


def bits(*warps):
    return sum(1 << w for w in warps)


def find_tail(state, **want):
    return [e for e in state["tail"]
            if all(e[k] == v for k, v in want.items())]


def sim_out(runs, app):
    hits = glob.glob(os.path.join(runs, app, "NO_ARGS", "*", "sim.out"))
    if len(hits) != 1:
        failures.append("%s: expected one sim.out, found %d" % (app, len(hits)))
        return None
    return hits[0]


def check_fig15(path, p):
    print("== (c) Fig. 15 example: %s" % path)
    d = parse(path)
    check(list(d) == [0], "all observes on SM 0 (one CTA)")
    d = d.get(0, {"obs": [], "preds": {}, "state": {}})
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
    check(preds([9]) == [2600, 2700, 2800, 3000],
          "(c) predictions {2600, 2700, 2800, 3000}: %s" % preds([9]))
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


def check_random(path, p):
    print("== (b) random gather: %s" % path)
    st = stats(path)
    for k in ("t1_promotions", "interwarp_trained", "t2_trained"):
        check(st.get("snake_" + k, -1) == 0, "%s == 0: %s" % (k, st.get("snake_" + k)))
    check(st.get("snake_warp_excluded_lane_stride", 0) > 0,
          "scattered-lane loads excluded: %s" % st.get("snake_warp_excluded_lane_stride"))
    print("  info  loads_observed=%s tail_lookups=%s tail_allocs=%s "
          "tail_evictions=%s" % tuple(st.get("snake_" + k) for k in
                                      ("loads_observed", "tail_lookups",
                                       "tail_allocs", "tail_evictions")))


def main():
    runs, root = sys.argv[1], sys.argv[2]
    man = json.load(open(os.path.join(root, "manifest.json")))
    for app, fn in (("snake-synth-fig15", check_fig15),
                    ("snake-synth-strided", check_strided),
                    ("snake-synth-random", check_random)):
        path = sim_out(runs, app)
        if path:
            fn(path, man[app])
    print("\n%s: %d failure(s)" % ("PASS" if not failures else "FAIL", len(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
