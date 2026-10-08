#!/usr/bin/env python3
"""Stall breakdown for the latency-hiding oracle arms, per benchmark and arm.

Reads sim_run_oracle_lh/ and writes, next to this script:

  oracle_lh_stalls.csv   one row per benchmark/input/arm

Two groups of counters are collected:

  issue stage (scheduler_unit::cycle, mutually exclusive, counted once per
  scheduler per cycle) -- w0_idle: no instruction to issue at all (barrier,
  i-cache miss, finished warp); w0_scoreboard: an instruction is there but its
  operands are not ready, i.e. waiting on memory; stall_pipeline: ready to
  issue but the unit is busy.

  memory stage (ldst_unit) -- the gpgpu_stall_shd_mem[...] breakdown, plus
  DRAM-full and interconnect back-pressure.

The issue-stage counters tick once per scheduler, so to compare them with
gpu_tot_sim_cycle they are divided by the number of schedulers on the GPU
(clusters x cores per cluster x schedulers per core, read from each job's
gpgpusim.config).  The *_gpucyc_vs_base columns are that conversion applied to
the arm's difference from its baseline, and net_pred_gpucyc is their sum, which
should land close to cycles_vs_base.

    ./oracle_lh_results/extract_oracle_lh_stalls.py
"""
import csv, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from extract_ipc import find_log  # noqa: E402

RUN_DIR = os.path.join(ROOT, "sim_run_oracle_lh")
ARMS = [("baseline", "A100-SASS"),
        ("uncoal", "A100-SASS-ORACLE_PREF"),
        ("allmiss", "A100-SASS-ORACLE_ALL_MISS")]
DONE_MARKER = "GPGPU-Sim: *** exit detected ***"

# Last value printed wins; every counter is cumulative over the run.
SIMPLE = {
    "cycles": (r"gpu_tot_sim_cycle\s*=\s*(\d+)", int),
    "n_stall_shd_mem": (r"gpgpu_n_stall_shd_mem\s*=\s*(\d+)", int),
    "gl_resource_stall":
        (r"gpgpu_stall_shd_mem\[gl_mem\]\[resource_stall\]\s*=\s*(\d+)", int),
    "gl_coal_stall":
        (r"gpgpu_stall_shd_mem\[gl_mem\]\[coal_stall\]\s*=\s*(\d+)", int),
    "gl_data_port_stall":
        (r"gpgpu_stall_shd_mem\[gl_mem\]\[data_port_stall\]\s*=\s*(\d+)", int),
    "smem_bank_conf_stall":
        (r"gpgpu_stall_shd_mem\[s_mem\]\[bk_conf\]\s*=\s*(\d+)", int),
    "cmem_resource_stall":
        (r"gpgpu_stall_shd_mem\[c_mem\]\[resource_stall\]\s*=\s*(\d+)", int),
    "l1cache_bkconflict": (r"gpgpu_n_l1cache_bkconflict\s*=\s*(\d+)", int),
    "reg_bank_conflict_stalls": (r"gpu_reg_bank_conflict_stalls\s*=\s*(\d+)", int),
    "stall_dramfull": (r"gpu_stall_dramfull\s*=\s*(\d+)", int),
    "stall_icnt2sh": (r"gpu_stall_icnt2sh\s*=\s*(\d+)", int),
    "l1d_res_fails": (r"L1D_total_cache_reservation_fails\s*=\s*(\d+)", int),
    "avg_mem_latency": (r"averagemflatency\s*=\s*(\d+)", int),
}
SIMPLE = {k: (re.compile(r"^\s*" + p), c) for k, (p, c) in SIMPLE.items()}
# "Stall:N\tW0_Idle:N\tW0_Scoreboard:N\t..." - the rest of the line is the
# warp-occupancy histogram, which is not a stall counter.
DISTRO = re.compile(r"^Stall:(\d+)\s+W0_Idle:(\d+)\s+W0_Scoreboard:(\d+)")

ISSUE = ["stall_pipeline", "w0_idle", "w0_scoreboard"]
COLS = (["cycles"] + ISSUE +
        ["n_stall_shd_mem", "gl_resource_stall", "gl_coal_stall",
         "gl_data_port_stall", "smem_bank_conf_stall", "cmem_resource_stall",
         "l1cache_bkconflict", "reg_bank_conflict_stalls", "stall_dramfull",
         "stall_icnt2sh", "l1d_res_fails", "avg_mem_latency"])
DERIVED = ["schedulers", "w0_scoreboard_gpucyc_vs_base",
           "stall_pipeline_gpucyc_vs_base", "w0_idle_gpucyc_vs_base",
           "net_pred_gpucyc", "cycles_vs_base"]


def parse_log(path):
    out = {"complete": False}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if DONE_MARKER in line:
                out["complete"] = True
                continue
            m = DISTRO.match(line)
            if m:
                out["stall_pipeline"] = int(m.group(1))
                out["w0_idle"] = int(m.group(2))
                out["w0_scoreboard"] = int(m.group(3))
                continue
            for key, (pattern, cast) in SIMPLE.items():
                m = pattern.match(line)
                if m:
                    out[key] = cast(m.group(1))
                    break
    return out


def schedulers(config_dir):
    """Scheduler units on the GPU, for scaling the issue-stage counters."""
    path = os.path.join(config_dir, "gpgpusim.config")
    vals = {"-gpgpu_n_clusters": None, "-gpgpu_n_cores_per_cluster": None,
            "-gpgpu_num_sched_per_core": None}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] in vals and vals[parts[0]] is None:
                    vals[parts[0]] = int(parts[1])
    except OSError:
        return None
    if any(v is None for v in vals.values()):
        return None
    n = 1
    for v in vals.values():
        n *= v
    return n


def main():
    rows = []
    for bench in sorted(os.listdir(RUN_DIR)):
        bench_dir = os.path.join(RUN_DIR, bench)
        if not os.path.isdir(bench_dir) or bench == "gpgpu-sim-builds":
            continue
        for args in sorted(os.listdir(bench_dir)):
            arms = {}
            for label, config in ARMS:
                config_dir = os.path.join(bench_dir, args, config)
                log = find_log(config_dir) if os.path.isdir(config_dir) else None
                if log:
                    s = parse_log(log)
                    s["schedulers"] = schedulers(config_dir)
                    arms[label] = s
            base = arms.get("baseline")
            if base is not None and not base["complete"]:
                base = None
            for label, _ in ARMS:
                s = arms.get(label)
                if s is None:
                    continue
                row = {"benchmark": bench, "args": args, "arm": label,
                       "complete": s["complete"], "schedulers": s["schedulers"]}
                row.update({k: s.get(k, "") for k in COLS})
                if (base is not None and label != "baseline" and s["complete"]
                        and s["schedulers"]):
                    n = float(s["schedulers"])
                    net = 0
                    for key in ISSUE:
                        if key in s and key in base:
                            d = round((s[key] - base[key]) / n)
                            row[key + "_gpucyc_vs_base"] = d
                            net += d
                    row["net_pred_gpucyc"] = net
                    row["cycles_vs_base"] = s["cycles"] - base["cycles"]
                rows.append(row)

    out = os.path.join(HERE, "oracle_lh_stalls.csv")
    header = ["benchmark", "args", "arm", "complete"] + COLS + DERIVED
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
