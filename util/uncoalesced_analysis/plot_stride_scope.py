#!/usr/bin/env python3
"""
Static matplotlib version of the "stride scope" chart: for each benchmark,
which byte strides between consecutive cache lines repeat across uncoalesced
loads, how much traffic they explain, and how likely they are structural
(warp-invariant within a load) rather than coincidental.

Reads the summary CSVs written by analyze_uncoalesced.py --out-dir (namely
pc_summary.csv and line_stride_signatures.csv) and reduces them the same way
the interactive version does:

  1. For each (benchmark, pc), take its most-instance stride signature,
     skipping the overflow bucket ("<other>") and single-line loads ("()").
  2. Keep it only if it explains >= --dominance-floor percent of that PC's
     uncoalesced instances -- below that the stride is either scattered/
     data-dependent or (occasionally) a stale/garbage per-lane address in the
     raw dump, and plotting it would misrepresent both as "the" stride.
  3. Pool PCs that land on the exact same (benchmark, stride) into one point,
     sized by total instances and colored by the instance-weighted share that
     came from a PC where the stride is dominant (>=50%) AND warp-invariant
     (>=90% of that PC's warp slots) -- i.e. how likely it's a real, fixable
     layout stride rather than an accident of which PCs happen to share it.

Per-PC detail (which exact PCs make up a bubble, their individual instance
counts) is left out of the plot on purpose: with up to 34 PCs sharing one
point, listing them would either overlap or force a legend nobody can read.
That detail is still in line_stride_signatures.csv if you need it -- this
plot answers "where should I look", not "here is every PC".

Configure the CSV_DIR / OUT_BASE / DPI / DOMINANCE_FLOOR constants below, then
just run the script (or the whole file as one Colab cell) -- no command-line
arguments needed. It writes <OUT_BASE>.png and <OUT_BASE>.pdf.
"""

import ast
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# =====================================================
# Publication-quality plot settings (shared house style:
# white background, DejaVu Sans, consistent size hierarchy)
# =====================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 36,
    "axes.labelsize": 30,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 24,
    "figure.titlesize": 38,
})

# Size hierarchy used explicitly below (the chart mixes many text roles that
# rcParams alone can't distinguish -- row labels, subtext, bubble labels,
# colorbar, etc. -- so each is set deliberately, scaled up from the original
# dark-theme version and paired with proportionally larger spacing so
# nothing collides). Every role shares one family (rcParams font.family, the
# same DejaVu Sans the benchmark row names use) and every size except the
# benchmark row name itself is scaled up for legibility.
FONT_TITLE = 38
FONT_ROW_NAME = 22
FONT_ROW_SUB = 20
FONT_TICK = 22
FONT_AXIS_LABEL = 24
FONT_BUBBLE_LABEL = 18
FONT_LEGEND = 20
FONT_LEGEND_TITLE = 22
FONT_COLORBAR_TICK = 19
FONT_COLORBAR_LABEL = 24

# Colors: light/pale -> deep, saturated blue as repeat-confidence increases,
# so high-confidence bubbles are the ones that visually pop on a white page.
RAMP = ["#dbe9f5", "#a9cbe4", "#6fa8d0", "#3480b5", "#0b4f80"]
BOUNDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]

INK = "#1f2d36"          # primary text (title, row names)
INK_SOFT = "#5b6b76"     # secondary text (instance counts, ticks, legend)
INK_FAINT = "#93a1ab"    # tertiary text (bubble stride labels)
GRID = "#e7ebee"         # vertical stride gridlines
ROW_LINE = "#eef1f3"     # horizontal row separators
BUBBLE_EDGE = "#33424b"


def fmt_stride(v):
    if v >= 1024 and v % 1024 == 0:
        return f"{v // 1024}KB"
    return f"{v}B"


def fmt_n(v):
    return f"{v:,}"


def short_bench(name):
    return name.replace("-rodinia-3.1", "").replace("parboil-", "")


def load_data(csv_dir, dominance_floor):
    pc_info = {}
    with open(os.path.join(csv_dir, "pc_summary.csv")) as f:
        for row in csv.DictReader(f):
            pc_info[(row["benchmark"], row["pc"])] = row

    best = {}
    with open(os.path.join(csv_dir, "line_stride_signatures.csv")) as f:
        for row in csv.DictReader(f):
            sig = row["stride_signature_bytes"]
            if sig in ("<other>", "()"):
                continue
            key = (row["benchmark"], row["pc"])
            n = int(row["instances"])
            if key not in best or n > int(best[key]["instances"]):
                best[key] = row

    bench_totals = defaultdict(int)
    for (b, _pc), info in pc_info.items():
        bench_totals[b] += int(info["instances"])

    # Instances from PCs that never produce a real bubble: either every load
    # stays inside one cache line (mean_lines_per_instance == 1 -> signature
    # "()" -> no inter-line stride exists to plot -- e.g. parboil-stencil and
    # parboil-mri-q, where every PC is single-line) or the best signature
    # doesn't clear --dominance-floor (scattered/data-dependent). Both get
    # pooled per benchmark into one gray "no stride" marker in `plot()` so a
    # benchmark with real uncoalesced traffic never renders as empty.
    no_dominant_instances = defaultdict(int)
    pc_points = []
    for key, info in pc_info.items():
        b, pc = key
        total = int(info["instances"])
        row = best.get(key)
        if row is None or float(row["pct_of_instances"]) < dominance_floor:
            no_dominant_instances[b] += total
            continue
        tup = ast.literal_eval(row["stride_signature_bytes"])
        stride = tup[0]
        warp_slots = int(info["warp_slots"])
        distinct_warps = int(row["distinct_warps"])
        structural = (float(row["pct_of_instances"]) >= 50.0 and
                      distinct_warps >= 0.9 * warp_slots)
        pc_points.append({
            "benchmark": b, "stride": stride,
            "dom_instances": int(row["instances"]),
            "structural": structural,
        })

    groups = defaultdict(list)
    for p in pc_points:
        groups[(p["benchmark"], p["stride"])].append(p)

    bubbles = []
    for (b, stride), plist in groups.items():
        total = sum(p["dom_instances"] for p in plist)
        structural = sum(p["dom_instances"] for p in plist if p["structural"])
        bubbles.append({
            "benchmark": b, "stride": stride, "instances": total,
            "pc_count": len(plist), "structural_frac": structural / total,
        })

    benches = sorted(bench_totals, key=lambda b: -bench_totals[b])
    return bubbles, no_dominant_instances, benches, bench_totals


def make_radius_fn(instances, r_min=6.0, r_max=17.0):
    """Target on-screen marker radius in points. Capped well below the row
    pitch (see `plot`) so a bubble plus its stride label never reaches into
    the row above or below it."""
    lo, hi = min(instances), max(instances)
    def radius(n):
        if hi <= lo:
            return (r_min + r_max) / 2
        t = (math.log10(n) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        return r_min + t * (r_max - r_min)
    return radius, lo, hi


def plot(bubbles, no_dominant_instances, benches, bench_totals, out_base, dpi):
    cmap = ListedColormap(RAMP)
    norm = BoundaryNorm(BOUNDS, cmap.N)

    # Pool of instances (mainly single-cache-line loads -- e.g. every PC in
    # parboil-stencil / parboil-mri-q, which have no inter-line stride at
    # all) that would otherwise silently vanish. Rendered as one gray marker
    # per benchmark at a reserved "no stride" row below the real axis range,
    # sized the same way as a real bubble, so a benchmark with real
    # uncoalesced traffic never renders as an empty column.
    sentinel_bubbles = [{"benchmark": b, "instances": n}
                         for b, n in no_dominant_instances.items() if n > 0]

    all_instances = [b["instances"] for b in bubbles] + \
        [s["instances"] for s in sentinel_bubbles]
    radius_fn, min_n, max_n = make_radius_fn(all_instances)
    def marker_size(n):
        # matplotlib scatter `s` is marker area in points**2; (2r)**2 makes
        # the rendered diameter come out to ~2r points, i.e. radius ~ r.
        return (2 * radius_fn(n)) ** 2

    col_of = {b: i for i, b in enumerate(benches)}
    if bubbles:
        strides = [b["stride"] for b in bubbles]
        min_stride, max_stride = min(strides), max(strides)
    else:
        # Nothing cleared the dominance floor anywhere -- fall back to a
        # single decade so the sentinel row still has an axis to sit on.
        min_stride = max_stride = 128
    # Reserved row for "no stride" markers, one octave below the smallest
    # real stride. Real strides are always multiples of --line-size (the
    # smallest possible non-trivial one equals line-size itself), so this
    # value can never collide with an actual data point.
    sentinel_y = min_stride / 2
    is_pow2 = lambda v: v & (v - 1) == 0

    n_cols = len(benches)
    # Landscape layout for slides: benchmarks run left-to-right as columns
    # (busiest first), stride runs bottom-to-top on a log axis. Column pitch
    # only has to clear a max-radius bubble and a diagonal two-line category
    # label -- much less demanding than the old horizontal row-label block --
    # so the figure comes out wide and short instead of tall.
    column_pitch_in = 1.6
    plot_height_in = 7.0
    left_margin_in, right_margin_in = 1.6, 0.5
    top_margin_in, bottom_margin_in = 2.9, 3.2
    fig_w = left_margin_in + n_cols * column_pitch_in + right_margin_in
    fig_h = top_margin_in + plot_height_in + bottom_margin_in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=left_margin_in / fig_w, right=1 - right_margin_in / fig_w,
                         top=1 - top_margin_in / fig_h,
                         bottom=bottom_margin_in / fig_h)

    ax.set_yscale("log", base=2)
    pad = 2 ** 0.7
    ax.set_ylim(sentinel_y / pad, max_stride * pad)
    ax.set_xlim(-0.6, n_cols - 0.4)

    # gridlines at every power-of-two stride in range, plus the sentinel row
    p_lo = math.floor(math.log2(min_stride))
    p_hi = math.ceil(math.log2(max_stride))
    yticks = [2 ** p for p in range(p_lo, p_hi + 1)]
    yticklabels = [fmt_stride(v) for v in yticks]
    if sentinel_bubbles:
        yticks = [sentinel_y] + yticks
        yticklabels = ["no stride"] + yticklabels
        # Dashed separator so the sentinel row reads as "not on this scale"
        # rather than as the smallest real stride.
        ax.axhline(math.sqrt(sentinel_y * min_stride), color=GRID,
                    linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color=INK_SOFT, fontsize=FONT_TICK)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xticks(range(n_cols))
    # Benchmark name + its total instance count, combined into one rotated
    # two-line label per column -- diagonal so 13 categories fit without
    # overlapping, at the same size as the (unchanged) benchmark name font.
    ax.set_xticklabels(
        [f"{short_bench(b)}\n{fmt_n(bench_totals[b])}" for b in benches],
        rotation=45, ha="right", va="top", rotation_mode="anchor",
        color=INK, fontsize=FONT_ROW_NAME)
    ax.tick_params(axis="x", length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
    for c in range(n_cols):
        ax.axvline(c, color=ROW_LINE, linewidth=1.0, zorder=0)

    # Short by necessity: a long label rotated 90 deg into a ~7in-tall axis
    # runs out of vertical room and clips against the figure edge -- the
    # full "...within one uncoalesced load..." wording from the portrait
    # version doesn't fit here.
    ax.set_ylabel("cache-line stride (bytes, log scale)", color=INK_SOFT,
                  fontsize=FONT_AXIS_LABEL, labelpad=16)

    xs = [col_of[b["benchmark"]] for b in bubbles]
    ys = [b["stride"] for b in bubbles]
    sizes = [marker_size(b["instances"]) for b in bubbles]
    colors = [b["structural_frac"] for b in bubbles]

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap=cmap, norm=norm,
                     edgecolors=BUBBLE_EDGE, linewidths=1.3, zorder=3)

    if sentinel_bubbles:
        sxs = [col_of[s["benchmark"]] for s in sentinel_bubbles]
        ssizes = [marker_size(s["instances"]) for s in sentinel_bubbles]
        ax.scatter(sxs, [sentinel_y] * len(sentinel_bubbles), s=ssizes,
                   c="#c3ccd2", edgecolors=BUBBLE_EDGE, linewidths=1.3,
                   zorder=3)

    # Stride value labels: skipped for bubbles that already sit exactly on a
    # gridline (its y-axis tick already states the value precisely) and
    # drawn only for the odd, non-power-of-two strides the gridlines can't
    # label -- keeps every stride value recoverable without crowding a
    # label onto all 22 points.
    for b in bubbles:
        if is_pow2(b["stride"]):
            continue
        c = col_of[b["benchmark"]]
        rad_pt = radius_fn(b["instances"])
        ax.annotate(fmt_stride(b["stride"]), (c, b["stride"]),
                    xytext=(rad_pt + 8, 0), textcoords="offset points",
                    ha="left", va="center", color=INK_FAINT,
                    fontsize=FONT_BUBBLE_LABEL, zorder=3)

    # confidence colorbar, docked below the rotated category labels. Sized in
    # absolute inches (not a fraction of the axes) so its 6 tick labels stay
    # legible regardless of column count -- `shrink` is fraction-of-axes-
    # width, and the axes width (n_cols * column_pitch_in) shrinks with fewer
    # benchmarks, so a fixed shrink squashes the labels together once n_cols
    # drops much below the ~13 this layout was first tuned for.
    ax_width_in = n_cols * column_pitch_in
    colorbar_len_in = min(4.2, 0.85 * ax_width_in)
    colorbar_thickness_in = 0.11
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.03,
                         pad=2.1 / plot_height_in,
                         aspect=colorbar_len_in / colorbar_thickness_in,
                         shrink=colorbar_len_in / ax_width_in,
                         anchor=(0.0, 1.0), ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"],
                             color=INK_SOFT, fontsize=FONT_COLORBAR_TICK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    cbar.set_label("repeat confidence", color=INK_SOFT,
                    fontsize=FONT_COLORBAR_LABEL, labelpad=8)

    # size legend: a small borderless inset with three reference dots, docked
    # to its own row well clear of the title block (fixed inches-from-top,
    # so it can't drift into the title regardless of column count or title
    # length -- title and legend are vertically stacked, not side by side).
    # Width is absolute inches, anchored to the right margin, instead of a
    # fixed fraction of the whole figure -- otherwise it shrinks along with
    # a narrower (fewer-column) figure until the three dots overlap.
    size_legend_w_in = min(3.4, 0.5 * fig_w)
    size_ax_left = (fig_w - right_margin_in - size_legend_w_in) / fig_w
    size_ax = fig.add_axes([size_ax_left, 1 - 2.5 / fig_h,
                             size_legend_w_in / fig_w, 0.5 / fig_h])
    size_ax.set_facecolor("none")
    size_ax.axis("off")
    ref_ns = [10_000, 100_000, 600_000]
    size_ax.scatter(range(len(ref_ns)), [0] * len(ref_ns),
                     s=[marker_size(n) for n in ref_ns],
                     c="#6f8494", edgecolors="none")
    for i, n in enumerate(ref_ns):
        off = radius_fn(n) + 8
        size_ax.annotate(f"{n // 1000}K", (i, 0), xytext=(off, 0),
                          textcoords="offset points", ha="left", va="center",
                          color=INK_SOFT, fontsize=FONT_LEGEND)
    size_ax.set_xlim(-0.6, len(ref_ns) - 0.1)
    size_ax.set_ylim(-1, 1)
    size_ax.text(-0.6, 1.3, "instances", color=INK_SOFT,
                 fontsize=FONT_LEGEND_TITLE, ha="left", va="bottom")

    fig.text(0.06, 1 - 0.75 / fig_h, "Repeated Strides",
              color=INK, fontsize=FONT_TITLE, fontweight="bold", ha="left", va="top")

    fig.savefig(f"{out_base}.png", dpi=dpi, facecolor=fig.get_facecolor())
    fig.savefig(f"{out_base}.pdf", facecolor=fig.get_facecolor())
    print(f"wrote {out_base}.png")
    print(f"wrote {out_base}.pdf")


def main():
    bubbles, no_dominant_instances, benches, bench_totals = load_data(
        CSV_DIR, DOMINANCE_FLOOR)
    if not bubbles and not any(no_dominant_instances.values()):
        raise SystemExit("No uncoalesced-load instances found -- nothing to plot.")
    out_base = os.path.splitext(OUT_BASE)[0]
    plot(bubbles, no_dominant_instances, benches, bench_totals, out_base, DPI)


# =====================================================
# Config -- edit these and just run the script/cell.
# =====================================================
CSV_DIR = "sim_run_parboil_uncoal_dump/uncoal_final/csv"  # dir with pc_summary.csv and line_stride_signatures.csv
OUT_BASE = "sim_run_parboil_uncoal_dump/uncoal_final/stride_scope"  # writes OUT_BASE.png and OUT_BASE.pdf
DPI = 170
DOMINANCE_FLOOR = 5.0  # min %% of a PC's instances its top stride must explain to be plotted

main()