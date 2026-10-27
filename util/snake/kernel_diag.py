#!/usr/bin/env python3
"""Per-kernel diagnosis of a workload under no prefetching, Tier 1's Snake and
SK_R0_D0 (PAPER_V100, 1B; docs/snake/results/fix_today.md, Phase 0).

    kernel_diag.py [<app dir> <args dir> [<config suffix> ...]]

Default: parboil-histo; configs '' (no prefetching, sim_run_baseline),
-SNAKE-SNAKE_FP1 (Tier 1, sim_run_snake) and the SK_R0_D0 suffix
(sim_run_snake_skm/v100p_1b).  Every stats block a kernel launch prints is
cumulative (except gpu_sim_cycle / gpu_sim_insn), so a launch's counters are
differences of consecutive blocks; a kernel sums its launches; the launch cut
by the 1B cap is excluded (as K1).  Per kernel:
  - IPC gain; prefetches issued per demand sector (snake_pf_issued /
    snake_demand_sector_accesses); outcomes of issued sectors (timely, late,
    evicted unused, other = unused at a kernel end); accuracy_sc and the
    timeliness split (SPEC OQ-36; the Tier 1 build predates snake_sc_*: n/a)
  - demand (GLOBAL_ACC_R) MISS_QUEUE_FULL and RESERVATION_FAIL per access and
    vs no prefetching; L1 hit rate; prefetch-induced demand evictions
    (OQ-36, SK build only)
  - demand miss-queue wait (snake_hist_mq_wait_demand) and memory-fetch
    latency (mf_lat_table, every read the SMs sent, prefetches included):
    mean and p90 from log2 buckets (bucket midpoints / upper bounds, so
    approximate); DRAM reads
  - issued prefetches per prediction source and the source's wrong issued
    sectors (snake_funnel_<source>_wrong_issued)
Per-PC prediction counts are not printed by the simulator; the kernel's load
PCs come from the trace (load_pcs.py's rule).
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P1 = "QV100-SASS-PAPER_V100-1B_INSN"
SK = "-SNAKE-SNAKE_FP1-SK_SC-SK_SC_CTA-SNAKE_HEAD64-SNAKE_MQR0-SNAKE_DM0"
RUNDIR = {"": "sim_run_baseline", "-SNAKE-SNAKE_FP1": "sim_run_snake", SK: "sim_run_snake_skm/v100p_1b"}
LABEL = {"": "no prefetching", "-SNAKE-SNAKE_FP1": "Tier 1 Snake", SK: "SK_R0_D0"}
SRC = ["inter_thread", "intra_warp", "promotion", "inter_warp", "inter_warp_chain"]
NUM = re.compile(r"^\s*([\w\[\]]+) = (\d+)\s*$", re.M)


def blocks(path):
    """[(name, cycles, insn, cumulative stats dict)] per launch; cut by the cap?"""
    t = open(path, errors="replace").read()
    capped = "break due to reaching the maximum" in t
    out = []
    for p in re.split(r"(?m)^kernel_name = ", t)[1:]:
        d = {k: int(v) for k, v in NUM.findall(p)}
        for name, vals in re.findall(r"(?m)^snake_hist_(\w+) = ([\d ]+)$", p):
            d["hist_" + name] = [int(x) for x in vals.split()]
        m = re.search(r"(?m)^mf_lat_table:([\d \t]+)$", p)
        if m:
            d["mf_lat"] = [int(x) for x in m.group(1).split()]
        m = re.search(r"total dram reads = (\d+)", p)
        d["dram_rd"] = int(m.group(1)) if m else 0
        out.append((p.split()[0], d.get("gpu_sim_cycle", 0), d.get("gpu_sim_insn", 0), d))
    return out, capped


def sub(a, b):
    if isinstance(a, list):
        return [x - (b[i] if b and i < len(b) else 0) for i, x in enumerate(a)]
    return a - (b or 0)


def per_kernel(path):
    ls, capped = blocks(path)
    n = len(ls) - (1 if capped else 0)
    agg = {}
    prev = {}
    for i in range(len(ls)):
        name, cyc, ins, cum = ls[i]
        d = {k: sub(v, prev.get(k)) for k, v in cum.items()}
        prev = cum
        if i >= n:
            break
        a = agg.setdefault(name, {"_cyc": 0, "_ins": 0, "_n": 0})
        a["_cyc"] += cyc
        a["_ins"] += ins
        a["_n"] += 1
        for k, v in d.items():
            if k in ("gpu_sim_cycle", "gpu_sim_insn"):
                continue
            if isinstance(v, list):
                a[k] = [x + y for x, y in zip(a[k], v)] if k in a else list(v)
            else:
                a[k] = a.get(k, 0) + v
    return agg


def log2_stats(h, mf=False):
    """mean (bucket midpoints) and p90 (bucket upper bound) of log2 buckets:
    snake_hist [0], [1], [2,3], ...; mf_lat_table [2^i, 2^(i+1))."""
    if not h or sum(h) <= 0:
        return None, None
    lo = (lambda i: 2 ** i) if mf else (lambda i: 0 if i == 0 else 2 ** (i - 1))
    hi = (lambda i: 2 ** (i + 1) - 1) if mf else (lambda i: 0 if i == 0 else 2 ** i - 1)
    n = sum(h)
    mean = sum(c * (lo(i) + hi(i)) / 2 for i, c in enumerate(h)) / n
    acc = 0
    for i, c in enumerate(h):
        acc += c
        if acc >= 0.9 * n:
            return mean, hi(i)
    return mean, hi(len(h) - 1)


def pct(a, b):
    return 100.0 * a / b if b else None


def f(x, fmt="%.1f"):
    return "n/a" if x is None else fmt % x


def kname(m):
    g = re.match(r"_Z(\d+)", m)
    return m[g.end():g.end() + int(g.group(1))] if g else m


def load_pcs(app, args, names):
    """L1-cached global-load PCs per kernel name, from the first trace of each."""
    tdir = os.path.join(ROOT, "sim_run_baseline", "runs", app, args, P1, "traces")
    out = {}
    for f_ in sorted(os.listdir(tdir)):
        if not f_.startswith("kernel-") or not f_.endswith(".traceg"):
            continue
        name, pcs = None, {}
        for line in open(os.path.join(tdir, f_), errors="replace"):
            if line.startswith("-kernel name"):
                name = line.split("=", 1)[1].strip()
                if name in out or name not in names:
                    break
                continue
            tok = line.split()  # PC mask n_dst dst... opcode ...
            if len(tok) > 3 and re.fullmatch(r"[0-9a-f]{4,}", tok[0]) and tok[2].isdigit():
                op = tok[3 + int(tok[2])] if len(tok) > 3 + int(tok[2]) else ""
                ops = op.split(".")
                if ops[0] == "LDG" and not (("STRONG" in ops and "GPU" in ops) or "BYPASS" in ops):
                    pcs[tok[0]] = pcs.get(tok[0], 0) + 1
        if name in names and name not in out:
            out[name] = pcs
    return out


def main():
    app = sys.argv[1] if len(sys.argv) > 1 else "parboil-histo"
    args = sys.argv[2] if len(sys.argv) > 2 else "_i___data_default_input_img_bin__o_ref_bmp____20_4"
    cfgs = sys.argv[3:] or ["", "-SNAKE-SNAKE_FP1", SK]
    runs = {c: per_kernel(os.path.join(ROOT, RUNDIR.get(c, "sim_run_snake_skm/v100p_1b"), "runs", app, args,
                                       P1 + c, "sim.out")) for c in cfgs}
    base = runs[""]
    tot = sum(v["_cyc"] for v in base.values())
    order = sorted(base, key=lambda k: -base[k]["_cyc"])
    A = lambda d, k: d.get("Total_core_cache_stats_breakdown[GLOBAL_ACC_R][%s]" % k, 0)
    MQF = "Total_core_cache_fail_stats_breakdown[GLOBAL_ACC_R][MISS_QUEUE_FULL]"
    print("## %s: per kernel (whole 1B run, PAPER_V100; launch cut by the cap excluded)\n" % app)
    cols = ["kernel (share of base cycles, launches)", "config", "IPC gain", "pf issued / demand sector",
            "issued: timely / late / evicted / other %", "accuracy_sc", "on time / early / late %",
            "demand MQF / access (vs base)", "demand resfail / access", "L1 hit %", "pf-induced evictions %",
            "demand MQ wait mean / p90", "mf latency mean / p90", "DRAM reads (vs base)",
            "issued by source %", "wrong issued by source (% of issued)"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for k in order:
        b = base[k]
        bacc = A(b, "TOTAL_ACCESS")
        for c in cfgs:
            d = runs[c].get(k)
            if d is None:
                continue
            gain = ((d["_ins"] / d["_cyc"]) / (b["_ins"] / b["_cyc"]) - 1) * 100 if c else None
            acc_ = A(d, "TOTAL_ACCESS")
            dem = d.get("snake_demand_sector_accesses", 0)
            iss = d.get("snake_pf_issued", 0)
            tl, lt, ev = (d.get("snake_pf_useful_timely", 0), d.get("snake_pf_late", 0),
                          d.get("snake_pf_evicted_unused", 0))
            oc = "/".join(f(pct(x, iss), "%.0f") for x in (tl, lt, ev, iss - tl - lt - ev)) if iss else "-"
            fills = d.get("snake_sc_pf_fills")
            if fills:
                on, late, early = d["snake_sc_ld_ontime"], d["snake_sc_ld_late"], d["snake_sc_ld_early"]
                used = on + late + early
                accsc = f(pct(on + fills - d.get("snake_pf_filled", 0), fills))
                tim = "/".join(f(pct(x, used), "%.0f") for x in (on, early, late)) if used else "-"
                pfev = f(pct(d["snake_sc_victim_demand_by_pf"], d["snake_sc_victim_sectors"]))
            else:
                accsc = tim = pfev = "n/a" if c else "-"
            mq, bmq = d.get(MQF, 0), b.get(MQF, 0)
            wm, wp = log2_stats(d.get("hist_mq_wait_demand"))
            lm, lp = log2_stats(d.get("mf_lat"), mf=True)
            src = ", ".join("%s %.0f" % (s, pct(d.get("snake_pf_issued_" + s, 0), iss)) for s in SRC
                            if d.get("snake_pf_issued_" + s, 0)) if iss else "-"
            wrong = ", ".join("%s %.0f" % (s, pct(d.get("snake_funnel_%s_wrong_issued" % s, 0), iss)) for s in SRC
                              if d.get("snake_funnel_%s_wrong_issued" % s, 0)) if iss else "-"
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                "`%s` (%.1f%%, %d)" % (kname(k), 100.0 * b["_cyc"] / tot, b["_n"]), LABEL.get(c, c),
                f(gain, "%+.2f%%"), f(iss / dem, "%.2f") if dem else ("-" if not c else "0"), oc, accsc, tim,
                "%s (%s)" % (f(pct(mq, acc_), "%.1f"), f((mq / bmq) if bmq else None, "x%.2f") if c else "base"),
                f(pct(A(d, "RESERVATION_FAIL"), acc_)), f(pct(A(d, "HIT"), acc_)), pfev,
                "%s / %s" % (f(wm), wp) if wm is not None else "n/a",
                "%s / %s" % (f(lm, "%.0f"), lp) if lm is not None else "n/a",
                "%d (%s)" % (d["dram_rd"], f(d["dram_rd"] / b["dram_rd"], "x%.2f") if b["dram_rd"] and c else "base"),
                src, wrong))
    print("\nL1-cached global-load PCs per kernel (trace of its first launch: PC: warp instructions):\n")
    for k, pcs in load_pcs(app, args, set(order)).items():
        print("- `%s`: %s" % (kname(k), ", ".join("%s: %d" % (p, n) for p, n in sorted(pcs.items())) or "none"))


if __name__ == "__main__":
    main()
