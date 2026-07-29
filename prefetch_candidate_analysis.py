#!/usr/bin/env python3
"""
prefetch_candidate_analysis.py

Reads the CSV summary files written by analyze_uncoalesced.py
(lane_signatures.csv, lane_deltas.csv, temporal_deltas.csv) and ranks
(benchmark, pc) sites by how good a prefetch candidate they are, then
produces several plots to visualize the ranking.

WHAT "GOOD PREFETCH CANDIDATE" MEANS HERE
------------------------------------------
For each (benchmark, pc) site we compute two independent predictability
scores from the data analyze_uncoalesced.py already extracted:

1. lane_dominance:
   Within a single uncoalesced load, how often does the SAME
   (lanes -> offset) layout repeat, out of all instances of that load?
   High value => the divergence pattern (which lanes go where) is fixed
   and structural (e.g. always a stride-4 layout), not data dependent.
   This tells you HOW the load splits, which matters for coalescing-
   aware prefetch (e.g. "always issue N sequential lines instead of one").

2. temporal_dominance:
   For a fixed (thread, pc), how often does the SAME address delta repeat
   across successive dynamic instances (loop iterations), out of all
   deltas observed? High value => "address(iter+1) = address(iter) + K"
   holds almost always. This is the classic condition for a stride
   prefetcher (K known, launch load K bytes ahead of the current PC hit).

We combine these into:

    prefetch_score = total_instances * max(lane_dominance, temporal_dominance)

which rewards sites that are BOTH frequent (worth optimizing) and
predictable (safe/cheap to prefetch confidently). Sites with huge
instance counts but low dominance (e.g. bfs/hotspot pointer chasing)
sink to the bottom; sites with perfect dominance but only a handful of
instances also sink, since optimizing them barely moves total runtime.

USAGE
-----
    python3 prefetch_candidate_analysis.py --results-dir results_quick/ --top 20 --out-dir plots/

Expects results-dir to contain:
    lane_signatures.csv   (benchmark, pc, lanes, offsets, count)
    lane_deltas.csv        (benchmark, pc, delta, count)
    temporal_deltas.csv    (benchmark, pc, delta, count)
"""

import argparse
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Loading + aggregation
# --------------------------------------------------------------------------

def load_csvs(results_dir):
    lane_sig_path = os.path.join(results_dir, "lane_signatures.csv")
    lane_delta_path = os.path.join(results_dir, "lane_deltas.csv")
    temporal_delta_path = os.path.join(results_dir, "temporal_deltas.csv")

    for p in (lane_sig_path, lane_delta_path, temporal_delta_path):
        if not os.path.exists(p):
            raise SystemExit(f"Missing expected file: {p}\n"
                              f"Run analyze_uncoalesced.py with --out-dir first.")

    lane_sig = pd.read_csv(lane_sig_path)
    lane_delta = pd.read_csv(lane_delta_path)
    temporal_delta = pd.read_csv(temporal_delta_path)
    return lane_sig, lane_delta, temporal_delta


def dominance_table(df, key_cols=("benchmark", "pc")):
    """
    Given a long-format dataframe with a 'count' column (one row per
    distinct pattern/delta per site), compute per-site:
      total        = sum of all counts at that site
      top_count    = count of the single most common pattern/delta
      dominance    = top_count / total
      top_value    = the pattern/delta value that dominates (for labeling)
    """
    out_rows = []
    value_col = "delta" if "delta" in df.columns else "offsets"
    for key, grp in df.groupby(list(key_cols)):
        total = grp["count"].sum()
        top_row = grp.loc[grp["count"].idxmax()]
        top_count = top_row["count"]
        out_rows.append({
            "benchmark": key[0],
            "pc": key[1],
            "total": total,
            "top_count": top_count,
            "dominance": top_count / total if total else 0.0,
            "top_value": top_row[value_col],
            "n_ditinct": len(grp),
        })
    return pd.DataFrame(out_rows)


def build_master_table(lane_sig, lane_delta, temporal_delta):
    lane_sig_dom = dominance_table(lane_sig).add_prefix("lanesig_")
    lane_sig_dom = lane_sig_dom.rename(columns={"lanesig_benchmark": "benchmark",
                                                 "lanesig_pc": "pc"})

    temporal_dom = dominance_table(temporal_delta).add_prefix("temporal_")
    temporal_dom = temporal_dom.rename(columns={"temporal_benchmark": "benchmark",
                                                  "temporal_pc": "pc"})

    lane_delta_dom = dominance_table(lane_delta).add_prefix("lanedelta_")
    lane_delta_dom = lane_delta_dom.rename(columns={"lanedelta_benchmark": "benchmark",
                                                      "lanedelta_pc": "pc"})

    master = lane_sig_dom.merge(temporal_dom, on=["benchmark", "pc"], how="outer")
    master = master.merge(lane_delta_dom, on=["benchmark", "pc"], how="outer")
    master = master.fillna(0)

    # total instance count: prefer the lane-signature total (one row per
    # dynamic uncoalesced load); fall back to temporal total if missing.
    master["total_instances"] = master[["lanesig_total", "temporal_total"]].max(axis=1)

    master["best_dominance"] = master[["lanesig_dominance", "temporal_dominance"]].max(axis=1)
    master["dominant_mode"] = np.where(
        master["temporal_dominance"] >= master["lanesig_dominance"],
        "temporal-stride", "lane-signature"
    )

    master["prefetch_score"] = master["total_instances"] * master["best_dominance"]
    master["site"] = master["benchmark"] + " @ " + master["pc"].astype(str)

    master = master.sort_values("prefetch_score", ascending=False).reset_index(drop=True)
    return master


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_top_candidates_bar(master, top_n, out_dir):
    top = master.head(top_n).iloc[::-1]  # reverse for horizontal bar order
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * top_n)))
    colors = ["#2b7cd3" if m == "temporal-stride" else "#d3812b"
              for m in top["dominant_mode"]]
    ax.barh(top["site"], top["prefetch_score"], color=colors)
    ax.set_xlabel("Prefetch score  (total_instances × best_dominance)")
    ax.set_title(f"Top {top_n} prefetch candidates")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2b7cd3"),
               plt.Rectangle((0, 0), 1, 1, color="#d3812b")]
    ax.legend(handles, ["dominant: temporal stride (cross-iteration)",
                         "dominant: lane-signature (within-load)"],
              loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "top_prefetch_candidates_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scatter_dominance_vs_volume(master, top_n, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = np.log10(master["total_instances"].clip(lower=1))
    y = master["best_dominance"] * 100
    colors = np.where(master["dominant_mode"] == "temporal-stride", "#2b7cd3", "#d3812b")
    sizes = 15 + 60 * (master["prefetch_score"] / master["prefetch_score"].max())

    ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, edgecolors="none")
    ax.set_xlabel("log10(total instances)  — how often this load site fires")
    ax.set_ylabel("Best dominance %  — how predictable the pattern/stride is")
    ax.set_title("Prefetch candidate landscape\n(top-right = frequent AND predictable = best prefetch targets)")
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.7)
    ax.text(x.min(), 91, "90% dominance", fontsize=8, color="gray")

    # Label the top-N highest-scoring points
    top = master.head(top_n)
    top_x = np.log10(top["total_instances"].clip(lower=1))
    top_y = top["best_dominance"] * 100
    for xi, yi, label in zip(top_x, top_y, top["site"]):
        ax.annotate(label, (xi, yi), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#2b7cd3", markersize=8),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#d3812b", markersize=8)]
    ax.legend(handles, ["dominant: temporal stride", "dominant: lane-signature"],
              loc="lower left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "dominance_vs_volume_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_quadrant_lane_vs_temporal(master, top_n, out_dir):
    """
    Plots lane-signature dominance vs temporal dominance directly, to find
    sites that are predictable in BOTH senses (best overall targets) vs
    sites that are only predictable in one sense.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    x = master["lanesig_dominance"] * 100
    y = master["temporal_dominance"] * 100
    sizes = 15 + 200 * (master["total_instances"] / master["total_instances"].max())

    sc = ax.scatter(x, y, s=sizes, c=np.log10(master["total_instances"].clip(lower=1)),
                     cmap="viridis", alpha=0.7, edgecolors="k", linewidths=0.2)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(total instances)")

    ax.axvline(90, color="gray", linestyle="--", linewidth=0.6)
    ax.axhline(90, color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Lane-signature dominance % (within-load predictability)")
    ax.set_ylabel("Temporal dominance % (cross-iteration stride predictability)")
    ax.set_title("Spatial vs temporal predictability\n"
                  "(top-right quadrant = best on both axes)")

    top = master.head(top_n)
    for xi, yi, label in zip(top["lanesig_dominance"] * 100,
                              top["temporal_dominance"] * 100,
                              top["site"]):
        ax.annotate(label, (xi, yi), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()
    path = os.path.join(out_dir, "lane_vs_temporal_quadrant.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_benchmark_summary(master, out_dir):
    """
    One bar per benchmark: total prefetch_score summed across its PCs,
    to see which kernels/benchmarks would benefit most overall.
    """
    agg = master.groupby("benchmark")["prefetch_score"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(agg))))
    ax.barh(agg.index[::-1], agg.values[::-1], color="#3f9b5c")
    ax.set_xlabel("Summed prefetch score across all PCs in the benchmark")
    ax.set_title("Which benchmarks have the most prefetch upside?")
    plt.tight_layout()
    path = os.path.join(out_dir, "per_benchmark_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True,
                     help="Directory containing lane_signatures.csv, lane_deltas.csv, "
                          "temporal_deltas.csv (the --out-dir you passed to "
                          "analyze_uncoalesced.py)")
    ap.add_argument("--out-dir", default="prefetch_plots",
                     help="Where to write the generated plots and ranked CSV")
    ap.add_argument("--top", type=int, default=20,
                     help="How many top candidates to highlight/plot")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lane_sig, lane_delta, temporal_delta = load_csvs(args.results_dir)
    master = build_master_table(lane_sig, lane_delta, temporal_delta)

    # Write the full ranked table so you can inspect/filter it directly.
    ranked_csv = os.path.join(args.out_dir, "ranked_prefetch_candidates.csv")
    master.to_csv(ranked_csv, index=False)

    print(f"Loaded {len(master)} distinct (benchmark, pc) sites.")
    print(f"\nTop {args.top} prefetch candidates:\n")
    cols = ["site", "total_instances", "dominant_mode", "best_dominance", "prefetch_score"]
    with pd.option_context("display.float_format", "{:0.3f}".format):
        print(master[cols].head(args.top).to_string(index=False))

    p1 = plot_top_candidates_bar(master, args.top, args.out_dir)
    p2 = plot_scatter_dominance_vs_volume(master, args.top, args.out_dir)
    p3 = plot_quadrant_lane_vs_temporal(master, args.top, args.out_dir)
    p4 = plot_per_benchmark_summary(master, args.out_dir)

    print(f"\nWrote ranked table: {ranked_csv}")
    print("Wrote plots:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")


if __name__ == "__main__":
    main()
