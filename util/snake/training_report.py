#!/usr/bin/env python3
"""Summarize Snake's training statistics from accel-sim outputs.

    training_report.py <sim.out> [<sim.out> ...]

Prints, per run, the final (cumulative) snake_* training counters: Tail hit
rate, allocations/evictions, T1 promotions/trained/demotions, T2 and
inter-warp training (events and distinct trained PCs), lane exclusions (OQ-17:
unequal lane strides in lane mode 0, no fitting run length in lane mode 1) and
the accepted run lengths, predictions per eligible load,
and the inter-thread chain-depth histogram.
"""
import os
import sys


def stats(path):
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith("snake_"):
            k, _, v = line.partition(" = ")
            out[k.strip()[len("snake_"):]] = int(v)
    return out


def main():
    for path in sys.argv[1:]:
        s = stats(path)
        name = "/".join(path.split(os.sep)[-4:-1])  # app/args/config
        if not s:
            print("%s: no Snake block" % name)
            continue
        loads = s["loads_observed"]
        hist = [s.get("chain_depth_hist_%d" % d, 0) for d in range(9)]
        last = max([d for d, v in enumerate(hist) if v] or [0])
        preds = s["pred_inter_thread"] + s["pred_inter_warp"] + s["pred_intra_warp"] + s["pred_promotion"]
        print("%s" % name)
        excl = s["warp_excluded_lane_stride"] + s.get("warp_excluded_lane_runs", 0)
        print("  eligible loads %d (of %d memory instructions), excluded %.1f%% "
              "(lane-stride %d, lane-runs %d); run length 32/16/8/4: %d/%d/%d/%d; "
              "Head row evictions %d"
              % (loads, s["hook_observe"], 100.0 * excl / max(loads, 1),
                 s["warp_excluded_lane_stride"], s.get("warp_excluded_lane_runs", 0),
                 s.get("lane_runs_32", 0), s.get("lane_runs_16", 0),
                 s.get("lane_runs_8", 0), s.get("lane_runs_4", 0),
                 s.get("head_row_evictions", 0)))
        print("  tail hits/lookups %d/%d = %.1f%%; allocs %d (cond1/2/3 %d/%d/%d, fills %d); evictions %d"
              % (s["tail_hits"], s["tail_lookups"], 100.0 * s["tail_hits"] / max(s["tail_lookups"], 1),
                 s["tail_allocs"], s["tail_alloc_cond1"], s["tail_alloc_cond2"], s["tail_alloc_cond3"],
                 s["tail_fills"], s["tail_evictions"]))
        print("  T1 promotions %d, trained %d, demotions %d (warps removed %d); "
              "T2 observed %d, trained %d; inter-warp trained %d, changed %d"
              % (s["t1_promotions"], s["t1_trained"], s["t1_demotions"], s["warps_removed"],
                 s["t2_observed"], s["t2_trained"], s["interwarp_trained"], s["interwarp_changed"]))
        print("  distinct (kernel, PC) with a trained inter-warp stride: %s"
              % s.get("interwarp_trained_pcs", "n/a"))
        print("  intra-warp detected: consecutive %d, via chain %d, chain not found %d"
              % (s["intra_consecutive"], s["intra_chain"], s["intra_chain_not_found"]))
        print("  predictions (not issued) %d = %.2f per load: inter-thread %d, inter-warp %d, "
              "intra-warp %d, at promotion %d"
              % (preds, preds / max(loads, 1), s["pred_inter_thread"], s["pred_inter_warp"],
                 s["pred_intra_warp"], s["pred_promotion"]))
        print("  chain depth histogram: %s" % " ".join(
            "%d:%.1f%%" % (d, 100.0 * hist[d] / max(sum(hist), 1)) for d in range(last + 1)))
        print("  kernel resets %d" % s["kernel_resets"])


if __name__ == "__main__":
    main()
