#!/usr/bin/env python3
"""Build the Snake results workbook (docs/snake/results/Snake_results_<date>.xlsx).

    PYTHONPATH=<dir with openpyxl> python3 util/snake/make_report_xlsx.py [<date>]

No simulations: every number is read by this script from existing outputs.
Sources:
  - raw sim.out files of the run dirs (sim_run_baseline, sim_run_snake*,
    sim_run_ssnake, sim_run_snake_gate/{S3c,S3d,S4,S4b,S4b_diag,S6_*,D1_*},
    sim_run_snake_refcache, sim_run_snake_refcmp/paperera_vs_v2,
    sim_run_snake_calib)
  - the Tier 1 config list from util/snake/s5_runs.sh, config tokens from
    util/job_launching/configs/define-standard-cfgs.yml
  - docs/snake/SPEC.md (paper values §5/§6, OQs §7, discrepancies §8),
    SNAKE_PLAN.md (§0 status table, §12 OQs), VERSION_DIFF.md (trace scan)
  - parsers shared with collect_results.py, per_kernel.py, missq_check.py
Headline numbers are cross-checked against the tables in
docs/snake/results/{final_tier1,D1_ceilings,K1_best_kernel,S6_ipc}.md and
final_tier1.csv; every mismatch is listed in the Provenance sheet.
A value that is not available is written "n/a", with the reason in Notes.
"""
import datetime
import glob
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
from collect_results import parse_job  # noqa: E402
from missq_check import collect as mq_collect, PAPER3, PAPER4, PAPER5  # noqa: E402
from per_kernel import launches, deltas, metrics as k_metrics, add as k_add  # noqa: E402

from openpyxl import Workbook, load_workbook  # noqa: E402
from openpyxl.chart import BarChart, Reference  # noqa: E402
from openpyxl.styles import Font, PatternFill, Alignment  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

DOCS = os.path.join(ROOT, "docs", "snake")
B = "QV100-SASS-PAPER_V100-1B_INSN"
B100 = "QV100-SASS-PAPER_V100-100M_INSN"
S = B + "-SNAKE-SNAKE_FP1"
VER, PRE, DIAG, TODO = ("Verified (1B final run)", "Preliminary (100M gate run)",
                        "Diagnostic only", "To check later")
FILL = {VER: "C6EFCE", PRE: "FFEB9C", DIAG: "D9D9D9", TODO: "F8CBAD"}
APPS = ["backprop", "hotspot", "srad_v1", "lud", "nw", "histo", "mri-q"]
MISMATCH = []   # (check, item, ours, doc value, source)
CHECKS = []     # (check, n compared, n mismatched, source)


# ------------------------------------------------------------ helpers
def rd(p):
    return open(p, errors="replace").read()


def last(t, pat, conv=int):
    m = re.findall(pat, t, re.M)
    return conv(m[-1]) if m else None


def app_of(d):
    return d.replace("-rodinia-3.1", "").replace("parboil-", "")


def wl_label(app, args):
    a = app_of(app)
    if a == "hotspot":
        return "hotspot " + args.split("_")[0]
    if a == "lud":
        return "lud 512" if "512" in args else "lud 256"
    return a


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct_of(a, b):
    return 100.0 * a / b if a is not None and b else None


def workloads():
    """(app dir, args) of the 9 Tier 1 workloads, from the baseline runs."""
    return sorted({tuple(p.split(os.sep)[-4:-2]) for p in glob.glob(
        os.path.join(ROOT, "sim_run_baseline", "runs", "*", "*", B, "sim.out"))})


def job(run_dir_runs, app, args, cfg):
    """Rich stats of one job (None if missing or unfinished)."""
    d = os.path.join(run_dir_runs, app, args, cfg)
    r = parse_job(d)
    if r is None or not r["done"]:
        return None
    t = rd(os.path.join(d, "sim.out"))
    r["snk"] = {k: int(v) for k, v in re.findall(r"^snake_(\w+) = (\d+)\s*$", t, re.M)}
    r["mqf"] = last(t, r"Total_core_cache_fail_stats_breakdown\[GLOBAL_ACC_R\]\[MISS_QUEUE_FULL\] = (\d+)")
    r["dram_rd"] = last(t, r"total dram reads\s*=\s*(\d+)")
    r["mflat"] = last(t, r"^averagemflatency = (\d+)")
    occ = re.findall(r"Stall:(\d+)\tW0_Idle:(\d+)\tW0_Scoreboard:(\d+)", t)
    r["occ"] = tuple(map(int, occ[-1])) if occ else None
    r["dists"] = {k: (int(n), int(p50), int(p90)) for k, n, p50, p90 in re.findall(
        r"^snake_dist_(\w+) n=(\d+) mean=\S+ p10=\d+ p50=(\d+) p90=(\d+)", t, re.M)}
    r["redecoded"] = last(t, r"re-decoded = (\d+)")
    r["path"] = d
    return r


def snake_rates(r):
    s = r.get("snk") or {}
    d = s.get("demand_sector_accesses", 0)
    if not d:
        return dict(cov_pred=None, cov_paper=None, acc=None, acc_std=None)
    iss = s.get("pf_issued", 0)
    return dict(cov_pred=pct_of(s.get("demand_pred_matched", 0), d),
                cov_paper=pct_of(s.get("pf_useful_timely", 0) + s.get("pf_late", 0), d),
                acc=pct_of(s.get("pf_useful_timely", 0), d),
                acc_std=pct_of(s.get("pf_useful_timely", 0) + s.get("pf_late", 0), iss) if iss else None)


def funnel(s, prefix="funnel"):
    modes = ["inter_thread", "intra_warp", "promotion", "inter_warp", "inter_warp_chain"]
    cls = ["queued", "drop_qfull", "drop_mq", "drop_res", "dup_l1", "dup_mshr", "late",
           "evicted", "timely", "issued_other", "wrong_issued", "wrong_not_issued", "live"]
    return {c: sum(s.get("%s_%s_%s" % (prefix, m, c), 0) for m in modes) for c in cls}, modes, cls


def gain(r, b):
    return (r["ipc"] / b["ipc"] - 1) * 100 if r and b and r["ipc"] and b["ipc"] else None


def md_tables(path):
    """[(heading, [rows of cells])] for every markdown table in a file."""
    out, head, cur = [], "", None
    for line in rd(path).split("\n"):
        if line.startswith("#"):
            head = line.strip("# ").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (head, [])
                out.append(cur)
            if not all(re.fullmatch(r"-+", c) for c in cells if c):
                cur[1].append(cells)
        else:
            cur = None
    return out


def num(s):
    """First number in a cell, unicode minus allowed."""
    m = re.search(r"[-+−]?\d[\d,]*\.?\d*", s.replace("−", "-"))
    return float(m.group(0).replace(",", "").replace("−", "-")) if m else None


def nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d[\d,]*\.?\d*", s.replace("−", "-"))]


def xcheck(check, item, ours, doc, decimals, source):
    """Compare our value rounded like the doc."""
    if ours is None or doc is None:
        MISMATCH.append((check, item, "n/a" if ours is None else ours, doc, source))
        return False
    if abs(round(ours, decimals) - doc) > 0.5 * 10 ** -decimals + 1e-9:
        MISMATCH.append((check, item, round(ours, decimals + 1), doc, source))
        return False
    return True


# ------------------------------------------------------------ sheet writer
class Col:
    def __init__(self, name, kind="text"):
        self.name, self.kind = name, kind


FMT = {"pct": '0.0"%"', "gain": '+0.0"%";-0.0"%";0.0"%"', "ipc": "#,##0.0", "int": "#,##0",
       "dec2": "0.00", "text": "@", "gain2": '+0.00"%";-0.00"%";0.00"%"'}


def write_sheet(wb, title, desc, cols, rows, extra=None):
    """rows: lists of values (None -> "n/a"); the last two columns must be
    Status and Notes.  Returns (ws, header row index)."""
    ws = wb.create_sheet(title)
    for i, line in enumerate(desc):
        ws.cell(row=i + 1, column=1, value=line).font = Font(italic=True)
    hr = len(desc) + 1
    for j, c in enumerate(cols):
        cell = ws.cell(row=hr, column=j + 1, value=c.name)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=False)
    widths = [len(c.name) for c in cols]
    for i, row in enumerate(rows):
        assert len(row) == len(cols), (title, row)
        for j, (v, c) in enumerate(zip(row, cols)):
            if v is None:
                v = "n/a"
            cell = ws.cell(row=hr + 1 + i, column=j + 1, value=v)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cell.number_format = FMT.get(c.kind, "General")
                if c.kind in ("gain", "gain2"):
                    if v > 1:
                        cell.fill = PatternFill("solid", fgColor="C6EFCE")
                    elif v < -1:
                        cell.fill = PatternFill("solid", fgColor="FFC7CE")
            if c.name == "Status" and v in FILL:
                cell.fill = PatternFill("solid", fgColor=FILL[v])
            shown = ("%.1f" % v) if isinstance(v, float) else str(v)
            widths[j] = max(widths[j], len(shown))
    for j, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(j + 1)].width = min(50, w + 2)
    ws.freeze_panes = ws.cell(row=hr + 1, column=2)
    if extra:
        extra(ws, hr)
    return ws, hr


def statcol(rows_status):
    return rows_status


# ------------------------------------------------------------ paper values
def paper_values():
    """SPEC §5 table: app -> (coverage, accuracy, IPC gain %, energy)."""
    spec = rd(os.path.join(DOCS, "SPEC.md"))
    sec = spec[spec.index("## 5. Evaluation setup"):spec.index("## 6. Hardware cost")]
    names = {"backprop": "backprop", "hotspot": "hotspot", "srad": "srad_v1", "lud": "lud",
             "nw": "nw", "histo": "histo", "mri-q": "mri-q"}
    allrows, ours = [], {}
    for line in sec.split("\n"):
        m = re.match(r"\|\s*\**([\w\-/ ]+?)\**\s*\|\s*\**~?(\d+)%\**\s*\|\s*\**~?(\d+)%\**\s*\|\s*\**~?(\d+)%\**\s*\|\s*\**~?([\d.]+)", line)
        if m:
            nm = m.group(1).strip()
            vals = tuple(float(m.group(i)) for i in (2, 3, 4, 5))
            allrows.append((nm, vals, line))
            if nm.lower() in names:
                ours[names[nm.lower()]] = vals
    return ours, allrows, sec


# ------------------------------------------------------------ data
def tier1_list():
    out = subprocess.run([os.path.join(HERE, "s5_runs.sh"), "tier1", "list"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [tuple(l.split()) for l in out.split("\n") if l.startswith("sim_run")]


LABELS = {"": "baseline", "-PAPER_LAT": "PAPER_LAT baseline", "-SNAKE-SNAKE_FP1": "Snake (exact)",
          "-SNAKE": "Snake (line)", "-SNAKE_ISO-SNAKE_FP1": "Snake-ISO", "-PAPER_LAT-SNAKE-SNAKE_FP1": "PAPER_LAT Snake (exact)",
          "-SNAKE_DT-SNAKE_FP1": "Snake-DT", "-SNAKE_T-SNAKE_FP1": "Snake-T", "-S_SNAKE-SNAKE_FP1": "s-Snake",
          "-SNAKE-SNAKE_FP1-SNAKE_TAIL20": "Tail 20", "-SNAKE-SNAKE_FP1-SNAKE_TAIL40": "Tail 40",
          "-SNAKE-SNAKE_FP1-SNAKE_TAILINF": "Tail 1024 (unlimited)", "-SNAKE-SNAKE_FP1-SNAKE_EVPOP": "popcount, Tail 10",
          "-SNAKE-SNAKE_FP1-SNAKE_EVPOP-SNAKE_TAIL20": "popcount, Tail 20",
          "-SNAKE-SNAKE_FP1-SNAKE_EVPOP-SNAKE_TAIL40": "popcount, Tail 40"}


def cfg_label(cfg):
    suf = cfg[len(B):]
    if suf in LABELS:
        return LABELS[suf]
    m = re.fullmatch(r"-SNAKE-SNAKE_FP1-SNAKE_THR(\d+)", suf)
    return ("space throttle %s cycles" % m.group(1)) if m else suf


def load_tier1():
    runs = {}
    for d, cfg in tier1_list():
        for a, g in workloads():
            runs[(cfg, a, g)] = (d, job(os.path.join(ROOT, d, "runs"), a, g, cfg))
    return runs


# ------------------------------------------------------------ main
def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    out = os.path.join(DOCS, "results", "Snake_results_%s.xlsx" % date)
    paper, paper_rows, spec5 = paper_values()
    WL = workloads()
    T1 = load_tier1()
    base = {(a, g): T1[(B, a, g)][1] for a, g in WL}
    base_lat = {(a, g): T1[(B + "-PAPER_LAT", a, g)][1] for a, g in WL}
    snake = {(a, g): T1[(S, a, g)][1] for a, g in WL}
    d1 = os.path.join(ROOT, "sim_run_snake_gate")
    pm = {(a, g, c): job(os.path.join(d1, "D1_pmem", "runs"), a, g, c + "-PERFECT_MEM")
          for a, g in WL for c in (B, B + "-PAPER_LAT")}
    pl = {(a, g, c): job(os.path.join(d1, "D1_pl1d", "runs"), a, g, c + "-PERFECT_L1D")
          for a, g in WL for c in (B, B + "-PAPER_LAT")}
    spec = rd(os.path.join(DOCS, "SPEC.md"))
    wb = Workbook()
    wb.remove(wb.active)

    # ======================= 1. Summary
    cols = [Col("workload"), Col("paper IPC gain (Fig. 18)", "gain"), Col("Snake IPC gain (whole run, exact)", "gain"),
            Col("perfect-L1 ceiling", "gain"), Col("perfect-memory ceiling", "gain"),
            Col("paper coverage (Fig. 16)", "pct"), Col("predicted coverage (coverage_pred)", "pct"),
            Col("delivered coverage (coverage_paper)", "pct"), Col("paper accuracy (Fig. 17)", "pct"),
            Col("timely accuracy (accuracy_paper)", "pct"), Col("L1 hit baseline", "pct"),
            Col("L1 hit Snake", "pct"), Col("paper L1 hit baseline (Fig. 25, avg)", "pct"),
            Col("paper L1 hit Snake (Fig. 25, avg)", "pct"), Col("Status"), Col("Notes")]
    rows, per_app = [], {}
    for a, g in WL:
        ap = app_of(a)
        b, s = base[(a, g)], snake[(a, g)]
        r = snake_rates(s)
        vals = [gain(s, b), gain(pl[(a, g, B)], b), gain(pm[(a, g, B)], b)]
        rows.append([wl_label(a, g), paper[ap][2], vals[0], vals[1], vals[2], paper[ap][0], r["cov_pred"],
                     r["cov_paper"], paper[ap][1], r["acc"], b["l1_hit_rate"], s["l1_hit_rate"], None, None,
                     VER, "paper values per benchmark (both inputs share them); Fig. 25 gives only the average"])
        per_app.setdefault(ap, []).append((vals, r, b["l1_hit_rate"], s["l1_hit_rate"]))
    gm = lambda i: (gmean([gmean([1 + v[0][i] / 100 for v in vs]) for vs in per_app.values()]) - 1) * 100
    am = lambda f: mean([mean([f(v) for v in vs]) for vs in per_app.values()])
    m = re.search(r"Baseline L1 hit rate (\d+)% → Snake (\d+)% → Isolated-Snake (\d+)%", spec)
    fig25 = tuple(float(x) for x in m.groups()) if m else (None, None, None)
    rows.append(["7-app GMEAN / mean", (gmean([1 + paper[x][2] / 100 for x in APPS]) - 1) * 100, gm(0), gm(1), gm(2),
                 mean([paper[x][0] for x in APPS]), am(lambda v: v[1]["cov_pred"]), am(lambda v: v[1]["cov_paper"]),
                 mean([paper[x][1] for x in APPS]), am(lambda v: v[1]["acc"]), am(lambda v: v[2]), am(lambda v: v[3]),
                 fig25[0], fig25[1], VER,
                 "IPC gains: GMEAN over apps (inputs averaged first); rates: mean over apps. Paper averages over the "
                 "same 7 apps; Fig. 25 over the paper's 11 apps (Isolated-Snake %s%%)" % (fig25[2],)])
    summary_rows = rows
    desc = ["Summary: one row per Tier 1 workload (1B instructions, QV100-SASS-PAPER_V100, Snake = exact footprint, default config) and the 7-app average.",
            "Sources: sim_run_baseline / sim_run_snake (Tier 1), sim_run_snake_gate/D1_pl1d and D1_pmem (ceilings), SPEC.md §5 (paper). Key findings and charts below."]

    def summary_extra(ws, hr):
        n = len(summary_rows)
        r0 = hr + n + 2
        ws.cell(row=r0, column=1, value="Key findings").font = Font(bold=True)
        findings = key_findings()
        for i, (txt, src) in enumerate(findings):
            ws.cell(row=r0 + 1 + i, column=1, value="%d. %s" % (i + 1, txt))
            ws.cell(row=r0 + 1 + i, column=15, value=src)
        cats = Reference(ws, min_col=1, min_row=hr + 1, max_row=hr + n)
        anchor_row = r0 + len(findings) + 3
        for k, (title, cols_, ytitle) in enumerate((
                ("(a) IPC gain: paper vs Snake vs perfect-L1 ceiling", (2, 3, 4), "IPC gain (%)"),
                ("(b) Coverage: paper vs predicted vs delivered", (6, 7, 8), "% of demand sectors"),
                ("(c) Accuracy: paper vs Snake (timely)", (9, 10), "% of demand sectors"))):
            ch = BarChart()
            ch.type, ch.title, ch.y_axis.title = "col", title, ytitle
            ch.height, ch.width = 9, 24
            for c in cols_:
                ch.add_data(Reference(ws, min_col=c, min_row=hr, max_row=hr + n), titles_from_data=True)
            ch.set_categories(cats)
            ws.add_chart(ch, "A%d" % (anchor_row + k * 20))

    def key_findings():
        g = summary_rows[-1]
        f = FIND
        bp_tf = [r for r in f["tf"] if r[0] == "backprop"][0]
        return [
            ("Prediction is close to the paper: predicted coverage %.1f%% vs the paper's %.1f%% (7-app means); hotspot, srad_v1, mri-q %.0f-%.0f%%."
             % (g[6], g[5], min(f["covr"]), max(f["covr"])), "final_tier1.md §3; SPEC §8 K1"),
            ("Timeliness is far below the paper: timely accuracy %.1f%% vs %.1f%%; inter-warp predictions come %d-%d cycles (p50) before use, against average memory latencies of %d-%d cycles."
             % (g[9], g[8], min(f["lead"]), max(f["lead"]), min(f["mflat"]), max(f["mflat"])), "S6_ipc.md §3; SPEC §8"),
            ("IPC: Snake %+.1f%% vs the paper's %+.1f%% (GMEAN); the six ablations span %+.2f%% to %+.2f%%."
             % (g[2], g[1], min(f["abl"]), max(f["abl"])), "final_tier1.md §3; Ablations sheet"),
            ("Headroom ceilings: a perfect L1 gives %+.1f%% and perfect memory %+.1f%% (GMEAN), about the paper's average, but in other apps."
             % (g[3], g[4]), "D1_ceilings.md"),
            ("hotspot and lud cannot reach the paper's gains here: a perfect L1 gives hotspot at most %+.1f%% and lud at most %+.1f%% (best lud kernel %+.1f%%), vs the paper's %+.0f%% / %+.0f%%."
             % (max(f["hot"]), max(f["lud"]), f["lud_k"], paper["hotspot"][2], paper["lud"][2]), "D1_ceilings.md; K1_best_kernel.md"),
            ("backprop and histo are DRAM-queue bound, srad_v1 and mri-q miss-queue/MSHR bound: prefetching moves the same bytes (backprop DRAM reads %s baseline vs %s Snake; average memory latency %d -> %d cycles)."
             % ("{:,}".format(f["bp_dram"][0]), "{:,}".format(f["bp_dram"][1]), f["bp_lat"][0], f["bp_lat"][1]), "S6_ipc.md §2"),
            ("Trace decoding: pre-756ed33 traces mis-decode predicated lanes; fixed in accel-sim fe8f37e (backprop L1D read hits %+.0f%%, cycles %+.2f%%, 100M)."
             % ((bp_tf[5] / bp_tf[4] - 1) * 100, bp_tf[3]), "VERSION_DIFF.md 'Pre-756ed33 traces'; Trace_fix sheet"),
            ("Best-performing kernel (Best_kernel sheet): predicted coverage %.1f%% vs the paper's %.1f%%, timely accuracy %.1f%% vs %.1f%%, speedup %+.1f%% vs %+.1f%% (GMEAN; whole run %+.1f%%). Closest: backprop %+.2f%% (best over variants %+.2f%%) vs %+.0f%%; histo %+.1f%% (%.0f%% of the paper's %+.0f%%)."
             % (f["bk"][8], f["bk"][7], f["bk"][11], f["bk"][10], f["bk"][4], f["bk"][3], f["bk"][5],
                f["bk_rows"]["backprop"][4], f["bk_rows"]["backprop"][14], f["bk_rows"]["backprop"][3],
                f["bk_rows"]["histo"][4], 100 * f["bk_rows"]["histo"][4] / f["bk_rows"]["histo"][3], f["bk_rows"]["histo"][3]),
             "K1_best_kernel.md; Best_kernel sheet"),
            ("Energy (Fig. 19) is unavailable: AccelWattch costs %s simulation time (paired run), above the 30%% limit."
             % (("%+.0f%%" % f["power"]) if f["power"] is not None else "n/a"), "SPEC §8; final_tier1.md §0"),
        ]

    SUMMARY = (desc, cols, rows, summary_extra)

    # ======================= 2. Paper reference
    rows = []
    for nm, vals, line in paper_rows:
        rows.append(["SPEC §5 table: " + nm, "coverage %.0f%% / accuracy %.0f%% / IPC gain %+.0f%% / energy %.2f" % vals,
                     "Figs. 16 / 17 / 18 / 19", vals[0], vals[1], vals[2], vals[3],
                     TODO, "read off bar charts (±3 pp, SPEC §5); exact values pending (see Open_items)"])
    for pat, label, src in (
            (r"reservation fails (\d+)% of L1D accesses \(Fig\. 3\)", "Fig. 3 reservation-fail rate (avg)", "Fig. 3"),
            (r"L1↔L2 bandwidth utilisation (\d+)% \(Fig\. 4\)", "Fig. 4 L1<->L2 bandwidth utilisation (avg)", "Fig. 4"),
            (r"memory stalls (\d+)% of stalls \(Fig\. 5\)", "Fig. 5 memory stalls (share of stalls)", "Fig. 5")):
        mm = re.search(pat, spec)
        rows.append([label, (mm.group(1) + "%") if mm else None, src, None, None, None, None, TODO,
                     "SPEC §5 'Motivation numbers'" + ("" if mm else "; not found in SPEC")])
    for k, v in zip(("Fig. 25 L1 hit rate: baseline", "Fig. 25 L1 hit rate: Snake", "Fig. 25 L1 hit rate: Isolated-Snake"), fig25):
        rows.append([k, "%.0f%%" % v if v is not None else None, "§5.7 / Fig. 25", None, None, None, None, VER,
                     "exact value in the text (SPEC §5 'Other reference points')"])
    for key in ("Simulator", "Per SM", "Unified cache", "MSHR", "Other caches", "L2", "DRAM (ns)", "Run length"):
        mm = re.search(r"- \*\*%s:\*\* (.+)" % re.escape(key), spec5)
        rows.append(["Table 1: " + key, mm.group(1).strip() if mm else None, "Table 1 / §4", None, None, None, None,
                     VER, "SPEC §5" + ("" if mm else "; not found")])
    sec6 = spec[spec.index("## 6. Hardware cost"):spec.index("## 7. Open Questions")]
    for line in sec6.split("\n"):
        mm = re.match(r"- \*\*(\w[\w ]*):\*\* (.+)", line)
        if mm:
            rows.append(["Table 3 / §5.5: " + mm.group(1), mm.group(2), "Table 3, §5.5, Fig. 21", None, None, None, None,
                         VER, "SPEC §6"])
    for line in spec5.split("\n"):
        if re.match(r"  - Snake beats Snake-DT|  - Without decoupling|  - Tail = 10", line):
            rows.append(["Other reference point", line.strip("- ").strip(), "§5.1, §5.2, Fig. 20", None, None, None, None,
                         VER, "SPEC §5 'Other reference points'"])
    write_sheet(wb, "Paper reference",
                ["Paper values used for comparison, as extracted into SPEC.md §5-§6 (Mostofi et al., MICRO 2023). Every row cites its section, figure or table.",
                 "Status here: Verified = exact value from the paper's text or tables; To check later = read off a bar chart (±3 pp), exact value pending."],
                [Col("item"), Col("value"), Col("source"), Col("coverage", "pct"), Col("accuracy", "pct"), Col("IPC gain", "gain"),
                 Col("energy (norm.)", "dec2"), Col("Status"), Col("Notes")], rows)

    # ======================= 3. Tier1_all
    cols = [Col("config"), Col("config name"), Col("run dir"), Col("workload"), Col("cycles", "int"),
            Col("instructions", "int"), Col("IPC", "ipc"), Col("IPC gain vs baseline", "gain"),
            Col("L1 hit rate (GLOBAL_ACC_R)", "pct"), Col("coverage_pred", "pct"), Col("coverage_paper", "pct"),
            Col("accuracy_paper", "pct"), Col("accuracy_std", "pct"), Col("pf generated (sectors)", "int"),
            Col("pf issued", "int"), Col("dup: in L1", "int"), Col("dup: in MSHR", "int"),
            Col("dropped: queue full", "int"), Col("dropped: no line", "int"), Col("dropped: MSHR reserve", "int"),
            Col("dropped: miss queue", "int"), Col("of which miss-queue reserve only", "int"),
            Col("ready line held by throttle (cycles)", "int"), Col("timely", "int"), Col("late", "int"),
            Col("wrong (predicted, no demand)", "int"), Col("evicted unused", "int"),
            Col("demand MISS_QUEUE_FULL", "int"), Col("DRAM reads", "int"), Col("avg memory latency", "int"),
            Col("wall time (s)", "int"), Col("peak RSS (MB)", "int"), Col("accel-sim build"), Col("gpgpu-sim build"),
            Col("Status"), Col("Notes")]
    rows = []
    for (cfg, a, g), (d, r) in sorted(T1.items(), key=lambda x: (x[1][0], x[0][0], x[0][1], x[0][2])):
        if r is None:
            rows.append([cfg_label(cfg), cfg, d, wl_label(a, g)] + [None] * (len(cols) - 7) + [None, TODO, "job missing or unfinished"])
            continue
        bref = base_lat[(a, g)] if "PAPER_LAT" in cfg else base[(a, g)]
        s = r["snk"] if r["snake"] else {}
        rt = snake_rates(r) if r["snake"] else dict(cov_pred=None, cov_paper=None, acc=None, acc_std=None)
        fn = funnel(s)[0] if s else {}
        notes = [] if r["snake"] else ["Snake off: prefetch metrics n/a"]
        if cfg == bref and False:
            pass
        g_ = gain(r, bref) if r is not bref else 0.0
        rows.append([cfg_label(cfg), cfg, d, wl_label(a, g), r["cycles"], r["insn"], r["ipc"], g_, r["l1_hit_rate"],
                     rt["cov_pred"], rt["cov_paper"], rt["acc"], rt["acc_std"],
                     s.get("pf_generated") if s else None, s.get("pf_issued") if s else None,
                     s.get("pf_dup_l1") if s else None, s.get("pf_dup_mshr") if s else None,
                     s.get("pf_drop_queue_full") if s else None, s.get("pf_drop_line") if s else None,
                     s.get("pf_drop_mshr") if s else None, s.get("pf_drop_miss_queue") if s else None,
                     s.get("pf_drop_mq_reserve") if s else None, s.get("pf_held_throttled") if s else None,
                     s.get("pf_useful_timely") if s else None, s.get("pf_late") if s else None,
                     (fn["wrong_issued"] + fn["wrong_not_issued"]) if fn else None,
                     s.get("pf_evicted_unused") if s else None, r["mqf"], r["dram_rd"], r["mflat"],
                     round(r["wall_s"]) if r.get("wall_s") else None,
                     round(r["max_rss_kb"] / 1024) if r.get("max_rss_kb") else None,
                     r["accelsim_tag"], r["gpgpusim_tag"], VER, "; ".join(notes)])
    t1_rows = rows
    write_sheet(wb, "Tier1_all",
                ["Every Tier 1 job: 21 configs x 9 workloads, 1B instructions, QV100-SASS-PAPER_V100 (PAPER_LAT configs vs the PAPER_LAT baseline).",
                 "Source: raw sim.out of the Tier 1 run dirs (util/snake/s5_runs.sh tier1), parsed like collect_results.py; 'wrong' = funnel wrong_issued + wrong_not_issued."],
                cols, rows)

    # ======================= 4. Ablations
    ab = [("Snake (exact)", S, "sim_run_snake"), ("Snake (line)", B + "-SNAKE", "sim_run_snake"),
          ("Snake-DT", B + "-SNAKE_DT-SNAKE_FP1", "sim_run_snake_dt"), ("Snake-T", B + "-SNAKE_T-SNAKE_FP1", "sim_run_snake_t"),
          ("s-Snake", B + "-S_SNAKE-SNAKE_FP1", "sim_run_ssnake"), ("Snake-ISO", B + "-SNAKE_ISO-SNAKE_FP1", "sim_run_snake")]
    cols = [Col("workload")] + [Col("%s IPC gain" % n, "gain2") for n, _, _ in ab] + \
           [Col("Snake vs Snake-DT", "gain2"), Col("Snake vs Snake-T", "gain2"),
            Col("exact: coverage_paper", "pct"), Col("line: coverage_paper", "pct"),
            Col("exact: accuracy", "pct"), Col("line: accuracy", "pct"), Col("paper claim (§5.2)"), Col("Status"), Col("Notes")]
    rows, agg = [], {}
    claim = re.search(r"Snake beats Snake-DT by (\d+)% and Snake-T by (\d+)% IPC \(§5\.2\)", spec)
    claim_txt = ("Snake +%s%% over Snake-DT, +%s%% over Snake-T (§5.2)" % claim.groups()) if claim else None
    for a, g in WL:
        b = base[(a, g)]
        rs = [T1[(c, a, g)][1] for _, c, _ in ab]
        gs = [gain(r, b) for r in rs]
        vdt = (rs[0]["ipc"] / rs[2]["ipc"] - 1) * 100
        vt = (rs[0]["ipc"] / rs[3]["ipc"] - 1) * 100
        e, l = snake_rates(rs[0]), snake_rates(rs[1])
        rows.append([wl_label(a, g)] + gs + [vdt, vt, e["cov_paper"], l["cov_paper"], e["acc"], l["acc"],
                                               claim_txt, VER, ""])
        agg.setdefault(app_of(a), []).append([r["ipc"] / b["ipc"] for r in rs] + [rs[0]["ipc"] / rs[2]["ipc"], rs[0]["ipc"] / rs[3]["ipc"]])
    gmr = [(gmean([gmean([v[i] for v in vs]) for vs in agg.values()]) - 1) * 100 for i in range(8)]
    rows.append(["7-app GMEAN"] + gmr + [None, None, None, None, claim_txt, VER, "rates: see Tier1_all"])
    ABL_GM.extend(gmr[:6])
    write_sheet(wb, "Ablations",
                ["Ablations per workload (1B, PAPER_V100): IPC gain vs the baseline, Snake vs Snake-DT / Snake-T, and line vs exact footprint.",
                 "Source: Tier 1 runs (sim_run_snake, _dt, _t, sim_run_ssnake). Paper ordering claim from SPEC §5 'Other reference points'."],
                cols, rows)

    # ======================= 5. Sweeps
    tail_claim = re.search(r"Tail = 10 entries costs (\d+)% coverage vs\. unlimited \(Fig\. 20\)", spec)
    thr_claim = re.search(r"Sensitivity: 25–300 cycles \(Fig\. 23\); (50 gives .+?)\.", spec)
    sweeps = [("Fig. 20", "Tail 10 (default)", S), ("Fig. 20", "Tail 20", S + "-SNAKE_TAIL20"),
              ("Fig. 20", "Tail 40", S + "-SNAKE_TAIL40"), ("Fig. 20", "Tail 1024 (unlimited)", S + "-SNAKE_TAILINF"),
              ("Fig. 22", "lru_popcount, Tail 10", S), ("Fig. 22", "lru_popcount, Tail 20", S + "-SNAKE_TAIL20"),
              ("Fig. 22", "lru_popcount, Tail 40", S + "-SNAKE_TAIL40"), ("Fig. 22", "popcount, Tail 10", S + "-SNAKE_EVPOP"),
              ("Fig. 22", "popcount, Tail 20", S + "-SNAKE_EVPOP-SNAKE_TAIL20"),
              ("Fig. 22", "popcount, Tail 40", S + "-SNAKE_EVPOP-SNAKE_TAIL40")] + \
             [("Fig. 23", "space throttle %d cycles%s" % (n, " (default)" if n == 50 else ""),
               S if n == 50 else S + "-SNAKE_THR%d" % n) for n in (25, 50, 75, 100, 150, 200, 300)]
    trend = {"Fig. 20": ("Tail 10 costs %s%% coverage vs unlimited (Fig. 20)" % tail_claim.group(1)) if tail_claim else None,
             "Fig. 22": None,
             "Fig. 23": ("25-300 cycles; %s (Fig. 23)" % thr_claim.group(1)) if thr_claim else None}
    cols = [Col("figure"), Col("setting"), Col("workload"), Col("coverage_pred", "pct"), Col("coverage_paper", "pct"),
            Col("accuracy_paper", "pct"), Col("IPC gain", "gain2"), Col("L1 hit", "pct"), Col("Tail evictions", "int"),
            Col("paper trend"), Col("Status"), Col("Notes")]
    rows = []
    for fig, lab, cfg in sweeps:
        per = {}
        for a, g in WL:
            r = T1[(cfg, a, g)][1]
            rt = snake_rates(r)
            note = "unlimited Tail = 1,024 entries; backprop still evicts" if "1024" in lab and app_of(a) == "backprop" else ""
            rows.append([fig, lab, wl_label(a, g), rt["cov_pred"], rt["cov_paper"], rt["acc"], gain(r, base[(a, g)]),
                         r["l1_hit_rate"], r["snk"].get("tail_evictions"), None, VER, note])
            per.setdefault(app_of(a), []).append((rt, r["ipc"] / base[(a, g)]["ipc"], r["l1_hit_rate"]))
        mk = lambda f: mean([mean([f(v) for v in vs]) for vs in per.values()])
        rows.append([fig, lab, "7-app mean (IPC: GMEAN)", mk(lambda v: v[0]["cov_pred"]), mk(lambda v: v[0]["cov_paper"]),
                     mk(lambda v: v[0]["acc"]), (gmean([gmean([v[1] for v in vs]) for vs in per.values()]) - 1) * 100,
                     mk(lambda v: v[2]), None, trend[fig], VER,
                     "Fig. 22: SPEC records no paper values for it" if fig == "Fig. 22" else
                     ("unlimited Tail = 1,024 entries (SPEC OQ-3)" if fig == "Fig. 20" else "")])
    write_sheet(wb, "Sweeps",
                ["Sensitivity sweeps (Tier 1, 1B): Fig. 20 Tail size, Fig. 22 eviction policy x Tail size, Fig. 23 space-throttle halt. One row per setting and workload, then the 7-app mean.",
                 "Source: sim_run_snake, sim_run_snake_sweep_{tail,evict,throttle}. Paper trends from SPEC §3.4 / §5. The unlimited Tail is 1,024 entries; backprop still evicts."],
                cols, rows)

    # ======================= 6. Ceilings
    cols = [Col("config"), Col("workload"), Col("baseline IPC", "ipc"), Col("perfect-memory IPC", "ipc"),
            Col("perfect-memory gain", "gain"), Col("perfect-L1 IPC", "ipc"), Col("perfect-L1 gain", "gain"),
            Col("Snake gain", "gain"), Col("paper Snake gain", "gain"), Col("Status"), Col("Notes")]
    rows = []
    ceil = {}
    for lab, c, bm in (("PAPER_V100", B, base), ("PAPER_LAT", B + "-PAPER_LAT", base_lat)):
        agg = {}
        for a, g in WL:
            b = bm[(a, g)]
            sn = T1[(c + "-SNAKE-SNAKE_FP1", a, g)][1]
            p1, p2 = pm[(a, g, c)], pl[(a, g, c)]
            v = (gain(p1, b), gain(p2, b), gain(sn, b))
            ceil[(lab, a, g)] = v
            rows.append([lab, wl_label(a, g), b["ipc"], p1["ipc"], v[0], p2["ipc"], v[1], v[2], paper[app_of(a)][2], VER, ""])
            agg.setdefault(app_of(a), []).append([1 + x / 100 for x in v])
        gm3 = [(gmean([gmean([x[i] for x in vs]) for vs in agg.values()]) - 1) * 100 for i in range(3)]
        rows.append([lab, "7-app GMEAN", None, None, gm3[0], None, gm3[1], gm3[2],
                     (gmean([1 + paper[x][2] / 100 for x in APPS]) - 1) * 100, VER, "IPC columns n/a: not meaningful as an average"])
    write_sheet(wb, "Ceilings",
                ["Performance ceilings on exactly Tier 1's config: perfect memory (-gpgpu_perfect_mem: no L2/DRAM/interconnect latency, bandwidth or queueing; L1 misses remain) and perfect L1 (-gpgpu_perfect_l1d: every L1-cached global load hits).",
                 "Source: sim_run_snake_gate/D1_pmem, D1_pl1d (1B), Tier 1 baselines and Snake; see D1_ceilings.md. Both ceilings also remove the loads' memory traffic."],
                cols, rows)

    # ======================= 7. PerKernel
    cols = [Col("workload"), Col("kernel"), Col("launches", "int"), Col("run-time share (baseline)", "pct"),
            Col("Snake speedup", "gain2"), Col("perfect-L1 ceiling", "gain"), Col("coverage_pred", "pct"),
            Col("coverage_paper", "pct"), Col("accuracy_paper", "pct"), Col("L1 hit base", "pct"), Col("L1 hit Snake", "pct"),
            Col("best kernel (by Snake speedup)"), Col("Headroom subset"), Col("Status"), Col("Notes")]
    rows, subset_tot = [], {"b": 0, "s": 0, "p": 0, "n": 0}
    sub_rows = []
    for a, g in WL:
        bp = os.path.join(ROOT, "sim_run_baseline", "runs", a, g, B, "sim.out")
        sp = os.path.join(ROOT, "sim_run_snake", "runs", a, g, S, "sim.out")
        pp = os.path.join(d1, "D1_pl1d", "runs", a, g, B + "-PERFECT_L1D", "sim.out")
        (lb, cb), (ls, cs), (lp, cp) = launches(bp), launches(sp), launches(pp)
        db, ds, dp = deltas(lb), deltas(ls), deltas(lp)
        n = min(len(db), len(ds), len(dp)) - (1 if (cb or cs or cp) else 0)
        tot = sum(db[i][2] for i in range(n))
        per = {}
        for i in range(n):
            assert db[i][1] == ds[i][1] == dp[i][1]
            e = per.setdefault(db[i][1], [{}, {}, {}, 0])
            k_add(e[0], db[i][2], db[i][3], db[i][4])
            k_add(e[1], ds[i][2], ds[i][3], ds[i][4])
            k_add(e[2], dp[i][2], dp[i][3], dp[i][4])
            e[3] += 1
        krows = []
        for name, (eb, es, ep, cnt) in per.items():
            mt = k_metrics(eb, es)
            ceil_k = (eb["cyc"] / ep["cyc"] - 1) * 100 if ep["cyc"] else None
            share = 100.0 * eb["cyc"] / tot
            krows.append([name, cnt, share, mt["gain"], ceil_k, mt, eb, es, ep])
        elig = [k for k in krows if k[2] >= 1.0]
        best = max(elig, key=lambda k: k[3])[0] if elig else None
        if elig:
            bk_ = max(elig, key=lambda k: k[3])
            BEST_K[(a, g)] = (bk_[0], bk_[3], bk_[4])
        if app_of(a) == "lud":
            global LUD_KMAX
            LUD_KMAX = max([LUD_KMAX or -1e9] + [k[4] for k in krows if k[2] >= 1.0])
        sb = {"b": 0, "s": 0, "p": 0, "k": 0}
        for name, cnt, share, gk, ceil_k, mt, eb, es, ep in sorted(krows, key=lambda k: -k[2]):
            head = share >= 1.0 and ceil_k is not None and ceil_k >= 10.0
            if head:
                sb["b"] += eb["cyc"]; sb["s"] += es["cyc"]; sb["p"] += ep["cyc"]; sb["k"] += 1
            isbest = name == best
            rows.append([wl_label(a, g), name, cnt, share, gk, ceil_k, mt["cov_pred"], mt["cov_paper"], mt["acc"],
                         mt["hit_b"], mt["hit_s"], "yes (biased selection)" if isbest else "",
                         "yes" if head else "no", DIAG if isbest else VER,
                         ("best kernel: chosen by Snake's own result, a biased statistic. " if isbest else "") +
                         ("last launch at the 1B cap excluded" if (cb or cs or cp) else "")])
        if sb["k"]:
            sub_rows.append([wl_label(a, g), sb["k"], 100.0 * sb["b"] / tot, (sb["b"] / sb["s"] - 1) * 100,
                             (sb["b"] / sb["p"] - 1) * 100])
            for k in "bsp":
                subset_tot[k] += sb[k]
            subset_tot["n"] += sb["k"]
    perk_rows = rows

    def perk_extra(ws, hr):
        r0 = hr + len(perk_rows) + 2
        ws.cell(row=r0, column=1, value="Snake on the headroom subset (kernels with run-time share >= 1% AND perfect-L1 ceiling >= 10%; rule independent of Snake)").font = Font(bold=True)
        hdr = ["workload", "subset kernels", "subset share of run time", "Snake speedup on subset", "perfect-L1 ceiling on subset"]
        for j, h in enumerate(hdr):
            ws.cell(row=r0 + 1, column=j + 1, value=h).font = Font(bold=True)
        for i, sr in enumerate(sub_rows + [["all subset kernels (cycles summed)", subset_tot["n"], None,
                                             (subset_tot["b"] / subset_tot["s"] - 1) * 100 if subset_tot["s"] else None,
                                             (subset_tot["b"] / subset_tot["p"] - 1) * 100 if subset_tot["p"] else None]]):
            for j, v in enumerate(sr):
                c = ws.cell(row=r0 + 2 + i, column=j + 1, value="n/a" if v is None else v)
                if j == 2 and v is not None:
                    c.number_format = FMT["pct"]
                if j >= 3 and v is not None:
                    c.number_format = FMT["gain2"]
                    if v > 1:
                        c.fill = PatternFill("solid", fgColor="C6EFCE")
                    elif v < -1:
                        c.fill = PatternFill("solid", fgColor="FFC7CE")
        ws.cell(row=r0 + 3 + len(sub_rows) + 1, column=1,
                value="Status of this summary: Verified (1B final run) data, subset defined without using Snake's results.")
    write_sheet(wb, "PerKernel",
                ["Every kernel (function, launches summed) per workload: run-time share, Snake (exact) speedup, perfect-L1 ceiling, coverage, accuracy, L1 hits; from per-launch differences of the per-kernel stats blocks.",
                 "Source: Tier 1 runs and D1_pl1d via util/snake/per_kernel.py functions (K1_best_kernel.md). The 'best kernel' is chosen by Snake's own speedup: Diagnostic only, biased."],
                cols, rows, perk_extra)

    # ======================= 8. Diagnostics_100M
    cols = [Col("stage"), Col("config"), Col("workload"), Col("IPC gain vs 100M baseline", "gain2"),
            Col("coverage_pred", "pct"), Col("coverage_paper", "pct"), Col("accuracy_paper", "pct"),
            Col("pred sectors", "int"), Col("wrong %", "pct"), Col("never issued in time %", "pct"),
            Col("dup %", "pct"), Col("late %", "pct"), Col("timely %", "pct"), Col("live %", "pct"),
            Col("wrong: from inter-warp %", "pct"), Col("wrong: from promotion %", "pct"),
            Col("timely: from inter-warp %", "pct"), Col("timely: from own-warp inter-thread %", "pct"),
            Col("lead pred->demand p50, inter-warp", "int"), Col("lead pred->demand p50, inter-thread", "int"),
            Col("lead issue->demand p50, inter-warp late", "int"), Col("lead issue->demand p50, inter-warp timely", "int"),
            Col("miss-queue wait demand p50", "int"), Col("miss-queue wait prefetch p50", "int"),
            Col("demand MISS_QUEUE_FULL", "int"), Col("miss-queue reserve", "int"), Col("Status"), Col("Notes")]
    rows = []
    ref100 = os.path.join(ROOT, "sim_run_snake_refcache", "bin_ref_v2", "runs")
    mqr_pairs = {}
    for st in ("S3c", "S3d", "S4", "S4b", "S4b_diag"):
        for p in sorted(glob.glob(os.path.join(d1, st, "runs", "*", "*", B100 + "*", "sim.out"))):
            a, g, cfg = p.split(os.sep)[-4:-1]
            r = job(os.path.join(d1, st, "runs"), a, g, cfg)
            if r is None or not r["snake"]:
                continue
            b = job(ref100, a, g, B100)
            s = r["snk"]
            fn, _, _ = funnel(s)
            fs, _, _ = funnel(s, "funnel_src")
            tot = sum(fn.values()) or 1
            src = lambda c, m: s.get("funnel_src_%s_%s" % (m, c), 0)
            wrong_tot = sum(src(c, m) for c in ("wrong_issued", "wrong_not_issued") for m in
                            ("inter_thread", "intra_warp", "promotion", "inter_warp", "inter_warp_chain")) or None
            tim_tot = sum(src("timely", m) for m in ("inter_thread", "intra_warp", "promotion", "inter_warp", "inter_warp_chain")) or None
            dd = r["dists"]
            p50 = lambda k: dd[k][1] if k in dd and dd[k][0] else None
            rt = snake_rates(r)
            mqr = 4 if "MQR4" in cfg else 8
            rows.append([st, cfg[len(B100):] or "baseline", wl_label(a, g), gain(r, b), rt["cov_pred"], rt["cov_paper"], rt["acc"],
                         s.get("pred_sectors"), pct_of(fn["wrong_issued"] + fn["wrong_not_issued"], tot),
                         pct_of(fn["queued"] + fn["drop_qfull"] + fn["drop_mq"] + fn["drop_res"], tot),
                         pct_of(fn["dup_l1"] + fn["dup_mshr"], tot), pct_of(fn["late"], tot), pct_of(fn["timely"], tot),
                         pct_of(fn["live"], tot),
                         pct_of(src("wrong_issued", "inter_warp") + src("wrong_not_issued", "inter_warp"), wrong_tot) if wrong_tot and "funnel_src_inter_warp_late" in s else None,
                         pct_of(src("wrong_issued", "promotion") + src("wrong_not_issued", "promotion"), wrong_tot) if wrong_tot and "funnel_src_inter_warp_late" in s else None,
                         pct_of(src("timely", "inter_warp"), tim_tot) if tim_tot and "funnel_src_inter_warp_late" in s else None,
                         pct_of(src("timely", "inter_thread"), tim_tot) if tim_tot and "funnel_src_inter_warp_late" in s else None,
                         p50("lead_pred_inter_warp"), p50("lead_pred_inter_thread"),
                         p50("lead_issue_inter_warp_late"), p50("lead_issue_inter_warp_timely"),
                         p50("mq_wait_demand"), p50("mq_wait_prefetch"), r["mqf"], mqr, PRE,
                         "per-source columns n/a: funnel by source added in S3d" if "funnel_src_inter_warp_late" not in s else ""])
            if st == "S3c":
                mqr_pairs.setdefault((a, g), {})[mqr] = (rt["cov_paper"], r["mqf"], gain(r, b), b["mqf"])
    diag_rows = rows

    def diag_extra(ws, hr):
        r0 = hr + len(diag_rows) + 2
        ws.cell(row=r0, column=1, value="Miss-queue reserve 4 vs 8 (S3c, Snake-DT, exact footprint, 100M): Preliminary (100M gate run)").font = Font(bold=True)
        hdr = ["workload", "coverage_paper, reserve 8", "coverage_paper, reserve 4", "demand MISS_QUEUE_FULL: baseline",
               "reserve 8", "reserve 4", "IPC gain, reserve 8", "IPC gain, reserve 4"]
        for j, h in enumerate(hdr):
            ws.cell(row=r0 + 1, column=j + 1, value=h).font = Font(bold=True)
        i = 0
        for (a, g), v in sorted(mqr_pairs.items()):
            if 4 in v and 8 in v:
                vals = [wl_label(a, g), v[8][0], v[4][0], v[8][3], v[8][1], v[4][1], v[8][2], v[4][2]]
                for j, x in enumerate(vals):
                    c = ws.cell(row=r0 + 2 + i, column=j + 1, value="n/a" if x is None else x)
                    if j in (1, 2):
                        c.number_format = FMT["pct"]
                    elif j in (3, 4, 5):
                        c.number_format = FMT["int"]
                    elif j >= 6:
                        c.number_format = FMT["gain2"]
                i += 1
    write_sheet(wb, "Diagnostics_100M",
                ["Prediction funnel (% of predicted sectors), by-source shares, lead-time medians (cycles), miss-queue waits (cycles) of the 100M diagnostic stages S3c, S3d, S4, S4b (Snake runs).",
                 "Source: sim_run_snake_gate/{S3c,S3d,S4,S4b,S4b_diag}; IPC vs the cached 100M baselines (sim_run_snake_refcache/bin_ref_v2). Reserve 4 vs 8 table below. All Preliminary (100M)."],
                cols, rows, diag_extra)

    # ======================= 9. Baseline_vs_paper
    mq = mq_collect(os.path.join(ROOT, "sim_run_baseline", "runs"), B)
    cols = [Col("workload"), Col("L1D accesses", "int"), Col("RESERVATION_FAIL", "int"), Col("reservation-fail rate", "dec2"),
            Col("paper Fig. 3", "dec2"), Col("fails: MISS_QUEUE_FULL %", "pct"), Col("fails: MSHR entry %", "pct"),
            Col("fails: MSHR merge %", "pct"), Col("fails: line alloc %", "pct"), Col("L1->L2 request util %", "pct"),
            Col("L2->L1 reply util %", "pct"), Col("paper Fig. 4 %", "pct"), Col("request-slot util %", "pct"),
            Col("Fig. 5 proxy: Scoreboard share", "dec2"), Col("Fig. 5 proxy: Scoreboard + Stall", "dec2"),
            Col("paper Fig. 5", "dec2"), Col("avg memory latency", "int"), Col("Status"), Col("Notes")]
    rows = []
    for r in mq:
        lab = wl_label(*[x for x in (r["app"],)] + [r["args"]]) if False else None
        ap = r["app"]
        lab = ("hotspot " + r["args"].split("_")[0]) if ap == "hotspot" else (
            ("lud 512" if "512" in r["args"] else "lud 256") if ap == "lud" else ap)
        rf = r["rf"] or 0
        c = r["causes"]
        rows.append([lab, r["acc"], r["rf"], r["fig3"], PAPER3.get(ap), pct_of(c["MISS_QUEUE_FULL"], rf) if rf else None,
                     pct_of(c["MSHR_ENRTY_FAIL"], rf) if rf else None, pct_of(c["MSHR_MERGE_ENRTY_FAIL"], rf) if rf else None,
                     pct_of(c["LINE_ALLOC_FAIL"], rf) if rf else None, r["fig4_req"], r["fig4_rep"], PAPER4.get(ap),
                     r["slot"], r["fig5"], r["fig5b"], PAPER5.get(ap), r["mflat"], VER,
                     "fail causes n/a: no reservation fails" if not rf else "paper values read off Figs. 3-5 (missq_check.py)"])
    calib = []
    cdir = os.path.join(ROOT, "sim_run_snake_calib")
    for sub, lab in (("base", "our config (QV100 + PAPER_V100 knobs: L1 20, ROP 160)"),
                     ("paper_lat", "PAPER_LAT as of S3e (L1 28, ROP 199)"), ("paper_lat23", "PAPER_LAT (L1 23, ROP 199)")):
        p = os.path.join(cdir, sub, "sim.out")
        if not os.path.exists(p):
            calib.append([lab, None, None, None, None, TODO, "missing"])
            continue
        t = rd(p)
        cyc = [int(x) for x in re.findall(r"^gpu_sim_cycle = (\d+)", t, re.M)]
        cfgt = rd(os.path.join(cdir, sub, "gpgpusim.config"))
        l1k = last(cfgt, r"^-gpgpu_l1_latency (\d+)")
        rop = last(cfgt, r"^-gpgpu_l2_rop_latency (\d+)")
        calib.append([lab, l1k, rop, cyc[0] / 32768 if len(cyc) > 0 else None, cyc[1] / 32768 if len(cyc) > 1 else None,
                      DIAG, "l2_lat microbenchmark, 32,768 dependent loads per kernel (bin_ref_v2)"])
    readme = rd(os.path.join(cdir, "README.md")) if os.path.exists(os.path.join(cdir, "README.md")) else ""
    mm = re.search(r"L2 hit ([\d.]+) cycles, L1 ([\d.]+) cycles", readme)
    calib.append(["native RTX 2080 Ti (sm_70 build)", None, None, float(mm.group(1)) if mm else None,
                  float(mm.group(2)) if mm else None, DIAG, "hardware reference from sim_run_snake_calib/README.md; knobs n/a"])
    pl_ = re.search(r"Paper \"(\d+)-cycle\" L1", rd(os.path.join(DOCS, "VERSION_DIFF.md")))
    pl2 = re.search(r"Paper \"(\d+)-cycle\" L2", rd(os.path.join(DOCS, "VERSION_DIFF.md")))
    calib.append(["paper Table 1", None, None, float(pl2.group(1)) if pl2 else None, float(pl_.group(1)) if pl_ else None,
                  VER, "Table 1 (VERSION_DIFF.md); knobs n/a"])

    def base_extra(ws, hr):
        r0 = hr + len(rows) + 2
        ws.cell(row=r0, column=1, value="Latency calibration (L1 / L2 hit latency in cycles)").font = Font(bold=True)
        for j, h in enumerate(["setup", "-gpgpu_l1_latency", "-gpgpu_l2_rop_latency", "L2 hit (cycles)", "L1 hit (cycles)", "Status", "Notes"]):
            ws.cell(row=r0 + 1, column=j + 1, value=h).font = Font(bold=True)
        for i, cr in enumerate(calib):
            for j, v in enumerate(cr):
                c = ws.cell(row=r0 + 2 + i, column=j + 1, value="n/a" if v is None else v)
                if j in (3, 4) and v is not None:
                    c.number_format = "0.0"
                if j == 5 and v in FILL:
                    c.fill = PatternFill("solid", fgColor=FILL[v])
    write_sheet(wb, "Baseline_vs_paper",
                ["Our baseline vs the paper's motivation figures (Figs. 3-5) on the Tier 1 baselines (1B, PAPER_V100), and the latency calibration (below).",
                 "Source: sim_run_baseline via util/snake/missq_check.py (definitions in its docstring; paper values read off the figures); calibration from sim_run_snake_calib."],
                cols, rows, base_extra)

    # ======================= 10. Trace_fix
    rc = os.path.join(ROOT, "sim_run_snake_refcmp", "paperera_vs_v2")
    cols = [Col("workload"), Col("cycles: paper-era", "int"), Col("cycles: fixed", "int"), Col("cycles change", "gain2"),
            Col("L1D read hits: paper-era", "int"), Col("L1D read hits: fixed", "int"), Col("L1D read misses: paper-era", "int"),
            Col("L1D read misses: fixed", "int"), Col("DRAM reads: paper-era", "int"), Col("DRAM reads: fixed", "int"),
            Col("lines re-decoded", "int"), Col("Status"), Col("Notes")]
    rows = []
    hit = r"Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[%s\] = (\d+)"
    for p in sorted(glob.glob(os.path.join(rc, "b", "*", "*", B100, "sim.out"))):
        a, g = p.split(os.sep)[-4:-2]
        pa = os.path.join(rc, "a", a, g, B100, "sim.out")
        ta, tb = rd(pa), rd(p)
        ca, cb_ = last(ta, r"^gpu_tot_sim_cycle = (\d+)"), last(tb, r"^gpu_tot_sim_cycle = (\d+)")
        rows.append([wl_label(a, g), ca, cb_, (cb_ / ca - 1) * 100, last(ta, hit % "HIT"), last(tb, hit % "HIT"),
                     last(ta, hit % "MISS"), last(tb, hit % "MISS"), last(ta, r"total dram reads\s*=\s*(\d+)"),
                     last(tb, r"total dram reads\s*=\s*(\d+)"), last(tb, r"re-decoded = (\d+)"), PRE,
                     "bin_ref_paperera (a) vs bin_ref_v2 (b), 100M"])
    TF_ROWS.extend(rows)
    scan = [t for t in md_tables(os.path.join(DOCS, "VERSION_DIFF.md")) if t[1] and t[1][0][0] == "app" and "base+delta fixed" in " ".join(t[1][0])]
    scan_rows = scan[0][1] if scan else []

    def tf_extra(ws, hr):
        r0 = hr + len(rows) + 2
        ws.cell(row=r0, column=1, value="Scan of affected trace lines per app (util/snake/scan_legacy_traces.py; table parsed from VERSION_DIFF.md 'Pre-756ed33 traces'); Status: Verified (1B final run) traces are the same").font = Font(bold=True)
        for i, row in enumerate(scan_rows):
            for j, v in enumerate(row):
                c = ws.cell(row=r0 + 1 + i, column=j + 1, value=v)
                if i == 0:
                    c.font = Font(bold=True)
        r1 = r0 + 2 + len(scan_rows)
        ws.cell(row=r1, column=1, value="Lines re-decoded in each Tier 1 baseline run (accel-sim prints the count):").font = Font(bold=True)
        for i, (a, g) in enumerate(WL):
            ws.cell(row=r1 + 1 + i, column=1, value=wl_label(a, g))
            ws.cell(row=r1 + 1 + i, column=2, value=base[(a, g)]["redecoded"]).number_format = FMT["int"]
    write_sheet(wb, "Trace_fix",
                ["Effect of the pre-756ed33 trace-decoding fix (accel-sim fe8f37e) on the baseline: paper-era decoding (bin_ref_paperera) vs fixed (bin_ref_v2), 100M, PAPER_V100.",
                 "Source: sim_run_snake_refcmp/paperera_vs_v2/{a,b} raw sim.out; per-app scan of affected lines below (VERSION_DIFF.md)."],
                cols, rows, tf_extra)

    # ======================= 11. S6_scope
    cols = [Col("workload"), Col("instructions"), Col("inter-warp scope"), Col("IPC gain vs baseline", "gain2"),
            Col("coverage_pred", "pct"), Col("coverage_paper", "pct"), Col("accuracy_paper", "pct"),
            Col("inter-warp timely (sectors)", "int"), Col("timely lead p50 (cycles)", "int"),
            Col("predictions for other CTAs", "int"), Col("Status"), Col("Notes")]
    rows, s6 = [], {}
    for lab, ins, apps_ in (("100M", B100, ("hotspot-rodinia-3.1", "srad_v1-rodinia-3.1", "parboil-histo")), ("1B", B, ("backprop-rodinia-3.1",))):
        for a, g in WL:
            if a not in apps_ or (a == "hotspot-rodinia-3.1" and not g.startswith("512")):
                continue
            b = job(ref100, a, g, B100) if lab == "100M" else base[(a, g)]
            for sc, (dirr, cfg) in ((0, (os.path.join(d1, "S4b", "runs") if lab == "100M" else os.path.join(ROOT, "sim_run_snake", "runs"),
                                         ins + "-SNAKE-SNAKE_FP1")),
                                    (1, (os.path.join(d1, "S6_scope" if lab == "100M" else "S6_scope_bp", "runs"), ins + "-SNAKE-SNAKE_FP1-SNAKE_IWS1")),
                                    (2, (os.path.join(d1, "S6_scope" if lab == "100M" else "S6_scope_bp", "runs"), ins + "-SNAKE-SNAKE_FP1-SNAKE_IWS2"))):
                r = job(dirr, a, g, cfg)
                rt = snake_rates(r)
                dd = r["dists"].get("lead_issue_inter_warp_timely")
                s6[(app_of(a), sc)] = gain(r, b)
                rows.append([wl_label(a, g), lab, sc, gain(r, b), rt["cov_pred"], rt["cov_paper"], rt["acc"],
                             dd[0] if dd else None, dd[1] if dd and dd[0] else None,
                             r["snk"].get("pred_inter_warp_other_cta", 0), DIAG,
                             {0: "default (OQ-8, same CTA)", 1: "next warps by global id", 2: "same warp index in later CTAs"}[sc]
                             + ("; timely lead n/a: no timely inter-warp sectors" if not (dd and dd[0]) else "")])
    write_sheet(wb, "S6_scope",
                ["S6 experiment -snake_interwarp_scope (0 = paper reading, same CTA; 1 = next warps by global warp id; 2 = same warp index in later resident CTAs). Default stays 0.",
                 "Source: sim_run_snake_gate/S6_scope (100M), S6_scope_bp (backprop, 1B); scope 0 from S4b (100M) and Tier 1 (1B); baselines from refcache (100M) and Tier 1 (1B)."],
                cols, rows)

    # ======================= 12. Interpretations
    oq_paper = {
        "OQ-1": "Head: per-warp register + 2 warps per PC under GTO; 32 rows (Tab. 3) - organisation not fully specified",
        "OQ-2": "32-bit fields in the storage figure", "OQ-3": "evict from an LRU group by fewest warpID bits; group size not given",
        "OQ-4": "prefetch queue size and issue rate not given", "OQ-5": "T1/T2 codes; text and Fig. 15 partly conflict",
        "OQ-6": "inter-warp stride across >= 3 warps", "OQ-7": "chain length variable, bounded by Tail and throttling",
        "OQ-8": "prefetch 'for all future warps'", "OQ-9": "prefetch footprint per warp address not specified",
        "OQ-10": "halt at 70% of peak 'bandwidth', resume at 50%; window not given",
        "OQ-11": "interaction with MSHRs and the miss queue not described",
        "OQ-12": "flagged regions, 25% free, 80% transfer rule, 50-cycle consumption (§3.2, fn. 3)",
        "OQ-13": "inputs not given; 11 apps incl. ISPASS", "OQ-14": "coverage = correct / demand addresses; accuracy = timely correct / demand",
        "OQ-15": "50% L1 cap 'until trained'; training takes 3 (max 10) cycles (§3.2, §3.5)",
        "OQ-16": "6.4 pJ/access, 6 mW static (§5.5)", "OQ-17": "lanes with equal inter-lane strides (§3.4)",
        "OQ-18": "GTO scheduler", "OQ-19": "not discussed", "OQ-20": "not discussed", "OQ-21": "global loads",
        "OQ-22": "Isolated-Snake buffer size not given (§5.7)", "OQ-23": "1530 MHz (Tab. 1)", "OQ-24": "Rodinia version not given",
        "OQ-25": "INTRA, INTER, MTA, CTA-Aware, Tree compared", "OQ-26": "intra-warp stride per PC",
        "OQ-27": "prefetch for all future warps at promotion (§3.2 ¶4)", "OQ-28": ">2 removed warps -> not trained (§3.2)",
        "OQ-29": "Fig. 15: inter-warp targets get PC1 only", "OQ-30": "evict by popcount (§3.1)",
        "OQ-31": "inter-thread before inter-warp (§3.4)", "OQ-32": "promote to prefetch for other warps after 3 warps (§3, §3.2)"}
    oq_default = {
        "OQ-8": "default scope 0 (same CTA); scopes 1/2 experimental (S6)",
        "OQ-9": "default line footprint (0); exact (1) used for all final runs",
        "OQ-10": "request slots (default) vs bytes (-snake_bw_mode 1)",
        "OQ-12": "variant 1 (revised, default) vs 0 (S4 literal)", "OQ-15": "variant 1: <= 10 cycles (default) vs 0: until promotion",
        "OQ-17": "lane mode 1 (run filter, default) vs 0 (paper-literal)", "OQ-29": "depth 0 = paper (default) vs 3 (extension)",
        "OQ-30": "weight 3 (default) vs 0 (paper rule)", "OQ-31": "own-warp priority (extension of §3.4)",
        "OQ-32": "promoted_all_warps 1 (default) vs 0", "OQ-11": "miss-queue reserve 8, MSHR reserve 8 (our choice)",
        "OQ-7": "max chain depth 3; 1 while throttled", "OQ-4": "32 lines, 1 line/cycle", "OQ-3": "group 3",
        "OQ-22": "256 lines", "OQ-23": "1530 MHz main, 1132 MHz sensitivity", "OQ-18": "GTO (PAPER_V100)"}
    oqs = {}
    plan = rd(os.path.join(DOCS, "SNAKE_PLAN.md"))
    for src_txt, srcname in ((spec[spec.index("## 7. Open Questions"):spec.index("## 8.")], "SPEC §7"),
                             (plan[plan.index("## 12. Open Questions"):], "SNAKE_PLAN §12")):
        blocks = re.split(r"(?m)^- \*\*(?=OQ-\d)", src_txt)[1:]
        for bl in blocks:
            m = re.match(r"(OQ-\d+)([^*]*)\*\*(.*)", bl, re.S)
            if not m:
                continue
            oid, title, body = m.group(1), m.group(2).strip(" .:"), m.group(3)
            if oid in oqs:
                continue  # SPEC first; plan entries that only point to SPEC are skipped
            conf = re.search(r"\((low|medium|high)\)", bl)
            decided = bool(re.search(r"\(decided\)|\*\*Decided|Decided:|decided 20", bl))
            text = re.sub(r"\s+", " ", re.sub(r"[*`]", "", body)).strip()
            oqs[oid] = (title or text[:60], text[:600] + ("..." if len(text) > 600 else ""),
                        conf.group(1) if conf else None, decided, srcname)
    rows = []
    for oid in sorted(oqs, key=lambda x: int(x.split("-")[1])):
        title, text, conf, decided, srcname = oqs[oid]
        st = TODO if conf == "low" and not decided else VER
        rows.append([oid, title, oq_paper.get(oid), text, conf, oq_default.get(oid, "single setting"),
                     "decided" if decided else "adopted with the plan (2026-10-08)", st,
                     srcname + ("; confidence n/a: none stated" if conf is None else "")])
    write_sheet(wb, "Interpretations",
                ["Every open question (OQ) on how to read the paper, with our choice. Parsed from SPEC.md §7 (OQ-1..18, 29..32) and SNAKE_PLAN.md §12 (OQ-19..28).",
                 "Status here: Verified (1B final run) = the choice is what the 1B final runs used; To check later = low confidence and not yet decided. 'what the paper says' is a short summary."],
                [Col("OQ"), Col("topic"), Col("what the paper says"), Col("our choice (from the docs)"), Col("confidence"),
                 Col("default vs paper-literal"), Col("decision"), Col("Status"), Col("Notes")], rows)

    # ======================= 13. Discrepancies
    sec8 = spec[spec.index("## 8. Design limitations"):]
    rows = []
    subs = re.split(r"(?m)^### ", sec8)[1:]
    for sub in subs:
        head = sub.split("\n")[0].strip()
        stt = (PRE if "S3e" in head or "S4b" in head or "Observations" in head else
               VER if "S6" in head else VER)
        for t in md_tables_text(sub):
            for cells in t[1:]:
                if len(cells) >= 4:
                    rows.append([cells[0], cells[1], cells[2], cells[3], head, PRE if "S3e" in head else stt, "SPEC §8 table"])
        for item in re.findall(r"(?m)^- (.+(?:\n  .+)*)", sub):
            txt = re.sub(r"\s+", " ", re.sub(r"[*`]", "", item)).strip()
            lead = re.match(r"(.{0,80}?)[.:]\s", txt)
            paperv = None
            stx = DIAG if ("K1" in txt[:20] or "experiment" in txt) else (VER if ("D1" in txt[:10] or "Tier 1" in txt) else stt)
            rows.append([lead.group(1) if lead else txt[:80], paperv, txt[:700], None, head, stx,
                         "paper value and likely cause n/a as separate fields: both are in the finding text"])
    write_sheet(wb, "Discrepancies",
                ["Every SPEC.md §8 entry (discrepancies with the paper and observations per stage), parsed from the markdown.",
                 "Columns for table rows: item | paper | ours | likely cause; for bullet entries the full text is in 'ours / finding'. Evidence = the §8 subsection (it names its results doc)."],
                [Col("item"), Col("paper"), Col("ours / finding"), Col("likely cause"), Col("evidence (SPEC §8 subsection)"),
                 Col("Status"), Col("Notes")], rows)

    # ======================= 14. Open_items
    g = summary_rows[-1]
    hist = [r for r in summary_rows if r[0] == "histo"][0]
    rows = [
        ["Accuracy / timeliness gap", "timely accuracy %.1f%% vs the paper's %.1f%% (7-app means); IPC gains depend on it" % (g[9], g[8]),
         "decide whether a latency-bound baseline or longer lookahead is in scope; see S6_ipc.md", TODO, ""],
        ["D1 Phase 1: known-answer tests", "confirms Snake and the ceilings behave as designed on traces with known answers",
         "not run; awaiting go-ahead on Phase 1", TODO, ""],
        ["D1 Part B: oracle prefetcher with real traffic", "separates latency from bandwidth: a perfect prefetcher that still moves the bytes",
         "not run", TODO, ""],
        ["D1 Part C: 'generous' Snake", "upper bound of Snake's own mechanism with relaxed limits", "not run", TODO, ""],
        ["Tier 2 sensitivity", "117 jobs: interwarp degree, miss-queue reserve, stale cycles, bytes meter, chain depth, variant 0, queue 64, issue 2, 1132 MHz",
         "on hold, awaiting approval (results/S5.md)", TODO, ""],
        ["histo predicted coverage", "%.1f%% vs the paper's %.0f%%; data-dependent accesses (SPEC §8)" % (hist[6], hist[5]),
         "check the paper's histo input and kernels; per-kernel view in PerKernel", TODO, ""],
        ["nw: no promotions", "Snake makes no predictions for nw (paper coverage 12%, IPC +5%)", "inspect nw's load PCs and strides (load_pcs.py)", TODO, ""],
        ["Energy unavailable", "Fig. 19 cannot be compared: AccelWattch costs +55% simulation time (> 30% limit)",
         "decide whether to accept the cost for a subset of runs", TODO, "Snake's own energy is computed (OQ-16)"],
        ["Questions to the authors", "inputs (OQ-13), metric definitions (OQ-14), whole-run vs per-kernel reporting (K1), clock 1530 MHz (VERSION_DIFF), exact figure values",
         "none sent; no draft in docs/snake (candidate questions listed here)", TODO, "n/a: no record of questions sent"],
    ]
    write_sheet(wb, "Open_items",
                ["Unresolved questions to check later: what, why it matters, the planned check.",
                 "Source: SPEC §8, results docs (S5, S6_ipc, D1_ceilings, K1_best_kernel), README status. Numbers in the text are computed from the runs."],
                [Col("what"), Col("why it matters"), Col("planned check"), Col("Status"), Col("Notes")], rows)

    # ======================= 15. Glossary
    gl = [
        ("coverage_pred", "demand sectors named by an earlier prediction within 5,000 cycles / demand sectors (GLOBAL_ACC_R, not failed)", "SPEC OQ-14", VER),
        ("coverage_paper (delivered coverage)", "(timely + late) / demand sectors", "SPEC OQ-14", VER),
        ("accuracy_paper (timely accuracy)", "timely / demand sectors; the paper's 'accuracy' (timely correctly predicted / demand)", "SPEC OQ-14, §5", VER),
        ("accuracy_std", "(timely + late) / issued prefetch sectors", "SPEC OQ-14", VER),
        ("timely", "issued prefetch sector whose first demand access hit the filled sector", "SPEC OQ-14", VER),
        ("late", "issued prefetch sector into whose MSHR entry a demand merged before the fill", "SPEC OQ-14", VER),
        ("wrong", "predicted sector with no demand within the 5,000-cycle window (funnel)", "SPEC OQ-14, S3c_diag.md", DIAG),
        ("evicted_unused", "issued prefetch sector evicted (or found stale) before any demand use", "SPEC OQ-14", VER),
        ("never issued in time", "predicted sectors still queued, dropped (queue full, miss queue, line/MSHR) when the demand came", "S3c_diag.md", DIAG),
        ("IPC gain", "IPC / baseline IPC - 1 (same instructions); averages: GMEAN over apps", "compare_paper.py", VER),
        ("L1 hit rate", "GLOBAL_ACC_R HIT / TOTAL_ACCESS of the L1D (TOTAL_ACCESS excludes reservation fails)", "s4_report.py", VER),
        ("perfect-memory ceiling", "-gpgpu_perfect_mem 1: L1 misses answered at once (no interconnect/L2/DRAM latency, bandwidth, queueing)", "D1_ceilings.md §2", VER),
        ("perfect-L1 ceiling", "-gpgpu_perfect_l1d 1: every L1-cached global load hits at L1 latency", "D1_ceilings.md §3", VER),
        ("exact footprint", "-snake_footprint 1: replay the target load's lane descriptor (sectors it touches)", "SPEC OQ-9", VER),
        ("line footprint", "-snake_footprint 0: one 128 B line per predicted warp address", "SPEC OQ-9", VER),
        ("PAPER_V100", "QV100 config + GTO scheduler + 1530 MHz clocks", "define-standard-cfgs.yml", VER),
        ("PAPER_LAT", "PAPER_V100 + -gpgpu_l1_latency 23 / -gpgpu_l2_rop_latency 199: measured L1 28, L2 212 cycles (Table 1)", "define-standard-cfgs.yml, S4.md §0", VER),
        ("Snake (exact)", "full Snake: decoupling mode 1 (variant 1) + throttling, exact footprint", "SNAKE_PLAN §6", VER),
        ("Snake-DT", "no decoupling, no throttling", "paper §4; SNAKE_DT token", VER),
        ("Snake-T", "decoupling, no throttling", "paper §4; SNAKE_T token", VER),
        ("Snake-ISO", "decoupling mode 2: 256-line isolated prefetch buffer", "paper §5.7; SPEC OQ-22", VER),
        ("s-Snake", "inter-thread chains only (no inter-warp, no intra-warp)", "paper §4; S_SNAKE token", VER),
        ("inter-thread / intra-warp / inter-warp prediction", "next load of the same warp via PC chain / same PC next iteration / same PC for other warps", "paper §3.1-3.2", VER),
        ("Head / Tail tables", "Head: last load per warp and last 2 warps per PC; Tail: (PC1, PC2, strides, warpID vector) entries", "paper §3.1, Tab. 3", VER),
        ("lead time", "cycles from prediction (or issue) to the matching demand", "S3c_diag.md", DIAG),
        ("Headroom subset", "kernels with run-time share >= 1% and perfect-L1 ceiling >= 10%", "this workbook (PerKernel)", VER),
        ("Status labels", "Verified (1B final run) | Preliminary (100M gate run) | Diagnostic only | To check later", "this workbook", VER),
    ]
    write_sheet(wb, "Glossary",
                ["Metrics and terms used in this workbook, with formula and source document.",
                 "Status: Verified (1B final run) = reported for the 1B final runs; Diagnostic only = diagnosis statistics."],
                [Col("term"), Col("definition / formula"), Col("source"), Col("Status"), Col("Notes")],
                [[a, b_, c, d, ""] for a, b_, c, d in gl])

    # ======================= 2. Best_kernel (moved after Summary)
    variants = [("Snake (exact)", "sim_run_snake", S, B), ("Snake (line)", "sim_run_snake", B + "-SNAKE", B),
                ("Snake-DT", "sim_run_snake_dt", B + "-SNAKE_DT-SNAKE_FP1", B),
                ("Snake-T", "sim_run_snake_t", B + "-SNAKE_T-SNAKE_FP1", B),
                ("s-Snake", "sim_run_ssnake", B + "-S_SNAKE-SNAKE_FP1", B),
                ("Snake-ISO", "sim_run_snake", B + "-SNAKE_ISO-SNAKE_FP1", B),
                ("PAPER_LAT Snake (exact)", "sim_run_snake", B + "-PAPER_LAT-SNAKE-SNAKE_FP1", B + "-PAPER_LAT")]
    cols = [Col("workload"), Col("best kernel (Snake exact)"), Col("run-time share", "pct"),
            Col("paper IPC gain (Fig. 18)", "gain"), Col("best-kernel speedup", "gain2"), Col("whole-run speedup (full run, incl. capped last launch)", "gain2"),
            Col("kernel perfect-L1 ceiling", "gain"), Col("paper coverage (Fig. 16)", "pct"),
            Col("best kernel: predicted coverage", "pct"), Col("best kernel: delivered coverage", "pct"),
            Col("paper accuracy (Fig. 17)", "pct"), Col("best kernel: timely accuracy", "pct"),
            Col("best kernel: L1 hit base", "pct"), Col("best kernel: L1 hit Snake", "pct"),
            Col("best over 7 Tier 1 variants: speedup", "gain2"), Col("that variant"), Col("that kernel"),
            Col("Status"), Col("Notes")]
    rows, bk_app, BKV = [], {}, {}
    for a, g in WL:
        ap = app_of(a)
        bpath = lambda c: os.path.join(ROOT, "sim_run_baseline", "runs", a, g, c, "sim.out")
        kr, capped = kernel_rows(bpath(B), os.path.join(ROOT, "sim_run_snake", "runs", a, g, S, "sim.out"),
                                 os.path.join(d1, "D1_pl1d", "runs", a, g, B + "-PERFECT_L1D", "sim.out"))
        bk = best_of(kr)
        bestv = None
        for vlab, vd, vc, vb in variants:
            vk, _ = kernel_rows(bpath(vb), os.path.join(ROOT, vd, "runs", a, g, vc, "sim.out"))
            vb_ = best_of(vk)
            if vb_ and (bestv is None or vb_["m"]["gain"] > bestv[1]["m"]["gain"]):
                bestv = (vlab, vb_)
        BKV[wl_label(a, g)] = bestv
        whole = gain(snake[(a, g)], base[(a, g)])
        m_ = bk["m"]
        rows.append([wl_label(a, g), bk["name"], bk["share"], paper[ap][2], m_["gain"], whole, bk["ceil"],
                     paper[ap][0], m_["cov_pred"], m_["cov_paper"], paper[ap][1], m_["acc"], m_["hit_b"], m_["hit_s"],
                     bestv[1]["m"]["gain"], bestv[0], bestv[1]["name"], DIAG,
                     "best = largest Snake speedup among kernels with >= 1% of the run time (selection uses Snake's own result)"
                     + ("; last launch at the 1B cap excluded" if capped else "")])
        bk_app.setdefault(ap, []).append((m_, bk["ceil"], bestv[1]["m"]["gain"], whole))
    gk = lambda f: (gmean([gmean([1 + f(v) / 100 for v in vs]) for vs in bk_app.values()]) - 1) * 100
    ak = lambda f: mean([mean([f(v) for v in vs]) for vs in bk_app.values()])
    rows.append(["7-app GMEAN / mean", None, None, (gmean([1 + paper[x][2] / 100 for x in APPS]) - 1) * 100,
                 gk(lambda v: v[0]["gain"]), gk(lambda v: v[3]), gk(lambda v: v[1]), mean([paper[x][0] for x in APPS]),
                 ak(lambda v: v[0]["cov_pred"]), ak(lambda v: v[0]["cov_paper"]), mean([paper[x][1] for x in APPS]),
                 ak(lambda v: v[0]["acc"]), ak(lambda v: v[0]["hit_b"]), ak(lambda v: v[0]["hit_s"]),
                 gk(lambda v: v[2]), None, None, DIAG,
                 "speedups: GMEAN over apps (inputs averaged first); rates: mean over apps; kernel and share n/a for an average"])
    bk_rows = rows
    FIND["bk"] = bk_rows[-1]
    FIND["bk_rows"] = {r[0]: r for r in bk_rows}

    def bk_extra(ws, hr):
        n = len(bk_rows)
        cats = Reference(ws, min_col=1, min_row=hr + 1, max_row=hr + n)
        for k, (title, cols_, yt) in enumerate((
                ("Best kernel: IPC gain, paper vs best kernel vs whole run vs kernel ceiling", (4, 5, 6, 7), "IPC gain (%)"),
                ("Best kernel: coverage and accuracy vs the paper", (8, 9, 11, 12), "% of demand sectors"))):
            ch = BarChart()
            ch.type, ch.title, ch.y_axis.title = "col", title, yt
            ch.height, ch.width = 9, 24
            for c in cols_:
                ch.add_data(Reference(ws, min_col=c, min_row=hr, max_row=hr + n), titles_from_data=True)
            ch.set_categories(cats)
            ws.add_chart(ch, "A%d" % (hr + n + 3 + k * 20))
    write_sheet(wb, "Best_kernel",
                ["Best-performing kernel per workload vs the paper's per-benchmark values (the mentor's hypothesis: the paper may report its best kernel). Tier 1 runs, 1B, PAPER_V100; ceilings from D1_pl1d.",
                 "Caveat: the kernel is selected by Snake's own speedup (>= 1% of run time), so it is an optimistic view. Same data as PerKernel and K1_best_kernel.md; last columns: best kernel over all 7 Tier 1 Snake variants."],
                cols, rows, bk_extra)

    # ======================= cross-checks (rule 2)
    do_crosschecks(WL, base, snake, pl, pm, T1, summary_rows, ceil, s6)
    src = "results/K1_best_kernel.md §1"
    n0, n = len(MISMATCH), 0
    for lab_, (gk, ck) in K1_DOC.items():
        k = [w for w in WL if wl_label(*w) == lab_.split(" (")[0]]
        if not k or k[0] not in BEST_K:
            continue
        ours = BEST_K[k[0]]
        dec = lambda t: len(re.sub(r"[^\d.]", "", t).split(".")[1]) if "." in t else 0
        gtxt = gk.replace("−", "-").split("/")[1].strip()
        xcheck("K1", lab_ + " best-kernel speedup", ours[1], num(gtxt), dec(gtxt), src)
        xcheck("K1", lab_ + " best-kernel ceiling", ours[2], num(ck), dec(ck), src)
        n += 2
    CHECKS.append(("PerKernel best kernel vs K1_best_kernel.md §1", n, len(MISMATCH) - n0, src))
    src = "results/K1_best_kernel.md §2"
    n0, n = len(MISMATCH), 0
    for lab_, cell in K1V_DOC.items():
        if lab_ in BKV and BKV[lab_]:
            t = cell.replace("**", "").replace("−", "-")
            xcheck("K1 §2", lab_ + " best over variants", BKV[lab_][1]["m"]["gain"], num(t),
                   len(re.sub(r"[^\d.]", "", t.split("%")[0]).split(".")[1]) if "." in t.split("%")[0] else 0, src)
            n += 1
    CHECKS.append(("Best_kernel (best over 7 variants) vs K1_best_kernel.md §2", n, len(MISMATCH) - n0, src))
    # computed inputs of the key findings
    FIND.update(dict(
        lead=[(T1[(S, a, g)][1]["dists"].get("lead_pred_inter_warp") or (0, None, None))[1] for a, g in WL
              if (T1[(S, a, g)][1]["dists"].get("lead_pred_inter_warp") or (0,))[0]],
        mflat=[base[w]["mflat"] for w in WL],
        abl=[x for x in ABL_GM],
        hot=[ceil[("PAPER_V100",) + w][1] for w in WL if app_of(w[0]) == "hotspot"],
        lud=[ceil[("PAPER_V100",) + w][1] for w in WL if app_of(w[0]) == "lud"],
        lud_k=LUD_KMAX, bp=[w for w in WL if app_of(w[0]) == "backprop"][0],
        bp_dram=(base[[w for w in WL if app_of(w[0]) == "backprop"][0]]["dram_rd"],
                 snake[[w for w in WL if app_of(w[0]) == "backprop"][0]]["dram_rd"]),
        bp_lat=(base[[w for w in WL if app_of(w[0]) == "backprop"][0]]["mflat"],
                snake[[w for w in WL if app_of(w[0]) == "backprop"][0]]["mflat"]),
        tf=TF_ROWS, power=power_slowdown(),
        covr=[r[6] for r in summary_rows[:-1] if r[0] in ("hotspot 512", "hotspot 1024", "srad_v1", "mri-q")]))
    write_sheet(wb, "Summary", *SUMMARY)
    wb.move_sheet("Summary", offset=-(len(wb.sheetnames) - 1))
    wb.move_sheet("Best_kernel", offset=-(wb.sheetnames.index("Best_kernel") - 1))

    # ======================= 16. Provenance
    rows = []
    for t in md_tables(os.path.join(DOCS, "SNAKE_PLAN.md")):
        if t[1] and t[1][0][:2] == ["Stage", "Status"]:
            for cells in t[1][1:]:
                rows.append(["stage", cells[0], cells[2], cells[3], cells[1], (cells[4][:300] if len(cells) > 4 else None),
                             VER if "final" in cells[0].lower() else PRE, "SNAKE_PLAN.md §0"])
    for repo, path in (("accel-sim", ROOT), ("gpgpu-sim", os.path.join(ROOT, "gpu-simulator", "gpgpu-sim"))):
        log = subprocess.run(["git", "-C", path, "log", "--format=%h|%ad|%s", "--date=short", "Snake"],
                             capture_output=True, text=True).stdout.strip().split("\n")
        for l in log:
            h, dte, msg = l.split("|", 2)
            if re.match(r"(snake: |D1|K1)", msg):
                rows.append(["commit", msg[:120], h if repo == "accel-sim" else None, h if repo == "gpgpu-sim" else None,
                             dte, None, VER if re.search(r"Tier 1|S5|S6|D1|K1|S4b", msg) else PRE, repo])
    used = ["sim_run_baseline", "sim_run_snake", "sim_run_snake_dt", "sim_run_snake_t", "sim_run_ssnake",
            "sim_run_snake_sweep_tail", "sim_run_snake_sweep_evict", "sim_run_snake_sweep_throttle",
            "sim_run_snake_gate/D1_pmem", "sim_run_snake_gate/D1_pl1d", "sim_run_snake_gate/S6_ceiling",
            "sim_run_snake_gate/S6_scope", "sim_run_snake_gate/S6_scope_bp", "sim_run_snake_gate/S3c", "sim_run_snake_gate/S3d",
            "sim_run_snake_gate/S4", "sim_run_snake_gate/S4b", "sim_run_snake_gate/S4b_diag", "sim_run_snake_refcache/bin_ref_v2",
            "sim_run_snake_refcmp/paperera_vs_v2/a", "sim_run_snake_refcmp/paperera_vs_v2/b"]
    for d in used:
        sims = glob.glob(os.path.join(ROOT, d, "runs", "*", "*", "*", "sim.out")) or \
               glob.glob(os.path.join(ROOT, d, "*", "*", "*", "sim.out"))
        tags, mt, cfgs = {}, [], set()
        for p in sims:
            t = rd(p)[:20000]
            tg = " ".join(sorted(set(re.findall(r"(?:accelsim-commit|gpgpu-sim_git-commit)-[0-9a-f]+_modified_[0-9.]+", t))))
            tags[tg] = tags.get(tg, 0) + 1
            mt.append(os.path.getmtime(p))
            cfgs.add(p.split(os.sep)[-2])
        dates = ("%s .. %s" % (datetime.date.fromtimestamp(min(mt)), datetime.date.fromtimestamp(max(mt)))) if mt else None
        stt = VER if d.startswith("sim_run_snake_gate/D1") or not d.startswith(("sim_run_snake_gate", "sim_run_snake_ref")) else (
            DIAG if "S6" in d else PRE)
        rows.append(["run dir", d, "; ".join("%s (%d jobs)" % (k, v) for k, v in sorted(tags.items())), "%d configs" % len(cfgs),
                     dates, "%d sim.out" % len(sims), stt, "all build tags _modified_0.0" if all("_modified_0.0" in k for k in tags)
                     else "CHECK: a build tag is not _modified_0.0"])
    toks = set()
    for _, cfg in tier1_list():
        toks.update(cfg.split("-")[2:])
    toks.update(["PERFECT_MEM", "PERFECT_L1D", "SNAKE_IWS1", "SNAKE_IWS2", "100M_INSN", "SNAKE_MQR4", "SNAKE_IWC0", "SNAKE_IWC3", "SNAKE_DV0"])
    yml = rd(os.path.join(ROOT, "util", "job_launching", "configs", "define-standard-cfgs.yml"))
    for tk in sorted(toks):
        mm = re.search(r"(?m)^%s:\n\s+extra_params: \"((?:[^\"])*)\"" % re.escape(tk), yml)
        rows.append(["config token", tk, None, None, None, re.sub(r"\s*\\n", " ", mm.group(1)) if mm else None, VER,
                     "util/job_launching/configs/define-standard-cfgs.yml" + ("" if mm else "; not found")])
    for chk, nn, nm, src in CHECKS:
        rows.append(["cross-check", chk, None, None, None, "%d values compared, %d mismatches" % (nn, nm),
                     VER if nm == 0 else TODO, src])
    for chk, item, ours, doc, src in MISMATCH:
        rows.append(["MISMATCH", "%s: %s" % (chk, item), None, None, None, "ours %s vs doc %s" % (ours, doc), TODO, src])
    write_sheet(wb, "Provenance",
                ["Where every number comes from: stage commits (SNAKE_PLAN §0 and git log), build tags and dates per run dir, config tokens, and the cross-checks against the results docs (rule 2).",
                 "Generated by util/snake/make_report_xlsx.py on %s from the accel-sim tree at %s." % (
                     date, subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())],
                [Col("kind"), Col("item"), Col("accel-sim"), Col("gpgpu-sim"), Col("date / dates"), Col("detail"),
                 Col("Status"), Col("Notes")], rows)

    wb.save(out)
    print(out)
    print("mismatches: %d" % len(MISMATCH))
    for m_ in MISMATCH:
        print("  ", m_)


def md_tables_text(txt):
    out, cur = [], None
    for line in txt.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = []
                out.append(cur)
            if not all(re.fullmatch(r"-+", c) for c in cells if c):
                cur.append(cells)
        else:
            cur = None
    return out


def do_crosschecks(WL, base, snake, pl, pm, T1, summary_rows, ceil, s6):
    lab = {wl_label(a, g): (a, g) for a, g in WL}

    def find(cell):
        c = cell.replace("**", "")
        for (a, g) in WL:
            ap = app_of(a)
            if c.startswith(ap + " / ") and g.startswith(c.split(" / ")[1][:10]):
                return (a, g)
            if c.split(" (")[0].strip() in (wl_label(a, g),) or (c == ap and ap not in ("hotspot", "lud")):
                return (a, g)
        return None

    # final_tier1.md §3
    src = "results/final_tier1.md §3"
    n0 = len(MISMATCH)
    n = 0
    for head, rows in md_tables(os.path.join(DOCS, "results", "final_tier1.md")):
        if not rows or not rows[0][0].startswith("app / input") or not rows[0][1].startswith("coverage: paper / pred"):
            continue
        for cells in rows[1:]:
            k = find(cells[0])
            if not k:
                continue
            sr = [r for r in summary_rows if r[0] == wl_label(*k)][0]
            cov = nums(cells[1])
            acc = nums(cells[2])
            ipcg = nums(cells[3])
            hits = nums(cells[4])
            for item, ours, doc, dec in (("coverage_pred", sr[6], cov[1], 1), ("coverage_paper", sr[7], cov[2], 1),
                                         ("accuracy", sr[9], acc[1], 1), ("IPC gain", sr[2], ipcg[1], 1),
                                         ("L1 hit base", sr[10], hits[0], 1), ("L1 hit Snake", sr[11], hits[1], 1)):
                xcheck("final_tier1", "%s %s" % (cells[0], item), ours, doc, dec, src)
                n += 1
    CHECKS.append(("Summary vs final_tier1.md §3", n, len(MISMATCH) - n0, src))
    # D1_ceilings.md
    src = "results/D1_ceilings.md §5"
    n0, n = len(MISMATCH), 0
    for head, rows in md_tables(os.path.join(DOCS, "results", "D1_ceilings.md")):
        cfg = "PAPER_LAT" if head.startswith("PAPER_LAT") else ("PAPER_V100" if head.startswith("PAPER_V100") else None)
        if not cfg or not rows or rows[0][0] != "workload":
            continue
        for cells in rows[1:]:
            k = find(cells[0])
            if not k:
                continue
            v = ceil[(cfg, k[0], k[1])]
            for item, ours, doc in (("perfect memory", v[0], num(cells[2])), ("perfect L1", v[1], num(cells[3])),
                                    ("Snake", v[2], num(cells[4]))):
                xcheck("D1_ceilings", "%s %s %s" % (cfg, cells[0], item), ours, doc, 1, src)
                n += 1
    CHECKS.append(("Ceilings vs D1_ceilings.md §5", n, len(MISMATCH) - n0, src))
    # S6_ipc.md §3 scope table
    src = "results/S6_ipc.md §3"
    n0, n = len(MISMATCH), 0
    cur = None
    for head, rows in md_tables(os.path.join(DOCS, "results", "S6_ipc.md")):
        if not rows or rows[0][:2] != ["workload", "scope"]:
            continue
        for cells in rows[1:]:
            if cells[0]:
                cur = cells[0].replace("**", "").split(" (")[0]
            sc = int(num(cells[1].replace("**", "")))
            if "identical" in cells[2]:
                continue
            ap = {"hotspot 512": "hotspot"}.get(cur, cur)
            ours = s6.get((ap, sc))
            xcheck("S6_ipc", "%s scope %d IPC gain" % (cur, sc), ours, num(cells[2].replace("**", "")), 2, src)
            n += 1
    CHECKS.append(("S6_scope vs S6_ipc.md §3", n, len(MISMATCH) - n0, src))
    # final_tier1.csv vs raw parse
    import csv
    src = "results/final_tier1.csv"
    n0, n = len(MISMATCH), 0
    for row in csv.DictReader(open(os.path.join(DOCS, "results", "final_tier1.csv"))):
        r = T1.get((row["config"], row["app"], row["args"]))
        if r is None or r[1] is None:
            MISMATCH.append(("final_tier1.csv", row["config"] + " " + row["app"], "missing", "present", src))
            continue
        for k in ("cycles", "insn"):
            n += 1
            if int(row[k]) != r[1][k]:
                MISMATCH.append(("final_tier1.csv", "%s %s %s" % (row["config"], row["app"], k), r[1][k], row[k], src))
    CHECKS.append(("Tier1_all vs final_tier1.csv (cycles, instructions)", n, len(MISMATCH) - n0, src))
    # K1 best kernel table
    src = "results/K1_best_kernel.md §1"
    n0, n = len(MISMATCH), 0
    pk = {}
    for head, rows in md_tables(os.path.join(DOCS, "results", "K1_best_kernel.md")):
        if rows and rows[0][:2] == ["workload", "best gain (variant, kernel)"]:
            for cells in rows[1:]:
                K1V_DOC[cells[0]] = cells[1]
        if not rows or not rows[0][0] == "workload" or "best kernel (share of baseline cycles)" not in rows[0][1]:
            continue
        for cells in rows[1:]:
            pk[cells[0]] = (cells[2], cells[3])
    K1_DOC.update(pk)


K1_DOC = {}
BEST_K = {}     # (app, args) -> (kernel, speedup, ceiling)
FIND = {}
ABL_GM = []
LUD_KMAX = None
TF_ROWS = []


def kernel_rows(bp, sp, pp=None):
    """Per-kernel metrics of a baseline / Snake (/ perfect-L1) run pair, as in
    per_kernel.py; the last launch of a capped run is excluded."""
    (lb, cb), (ls, cs) = launches(bp), launches(sp)
    db, ds = deltas(lb), deltas(ls)
    dp, cp = None, False
    if pp:
        lp, cp = launches(pp)
        dp = deltas(lp)
    n = min(len(db), len(ds), len(dp) if dp else 10 ** 9) - (1 if (cb or cs or cp) else 0)
    tot = sum(db[i][2] for i in range(n))
    per = {}
    for i in range(n):
        assert db[i][1] == ds[i][1] and (dp is None or dp[i][1] == db[i][1])
        e = per.setdefault(db[i][1], [{}, {}, {}, 0])
        k_add(e[0], *db[i][2:5])
        k_add(e[1], *ds[i][2:5])
        if dp:
            k_add(e[2], *dp[i][2:5])
        e[3] += 1
    out = []
    for name, (eb, es, ep, cnt) in per.items():
        out.append(dict(name=name, launches=cnt, share=100.0 * eb["cyc"] / tot, m=k_metrics(eb, es),
                        ceil=(eb["cyc"] / ep["cyc"] - 1) * 100 if ep.get("cyc") else None))
    return out, bool(cb or cs or cp)


def best_of(krows):
    elig = [k for k in krows if k["share"] >= 1.0 and k["m"]["gain"] is not None]
    return max(elig, key=lambda k: k["m"]["gain"]) if elig else None


K1V_DOC = {}


def power_slowdown():
    """AccelWattch wall-time cost from the paired S5 pre-flight runs."""
    d = os.path.join(ROOT, "sim_run_snake_preflight")
    el = []
    for j in ("power_off_paired", "power_on_paired"):
        p = os.path.join(d, j, "time.txt")
        if not os.path.exists(p):
            return None
        m = re.search(r"Elapsed \(wall clock\) time .*: ([\d:.]+)", rd(p))
        parts = [float(x) for x in m.group(1).split(":")]
        el.append(sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts)))
    return (el[1] / el[0] - 1) * 100

if __name__ == "__main__":
    main()
