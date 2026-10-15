#!/usr/bin/env python3
"""Wall-time estimate for the S5 final evaluation (docs/snake/SNAKE_PLAN.md §8).

    s5_estimate.py [-p <parallel>] <ref runs> <run dir> [<run dir> ...]

Per workload:
  - total thread instructions to completion = the trace's warp instructions
    (sum of the "insts =" lines of its kernel-*.traceg, measured 2026-10-15,
    WARP_INSTS below) x the thread/warp instruction ratio of its 100M
    baseline run (<ref runs>/<app>/<args>/QV100-SASS-PAPER_V100-100M_INSN)
  - the job length is that total, capped at 1B
  - wall time per instruction = the largest seconds/instruction among the
    100M runs found in <ref runs> and the <run dir>s (any config: baseline,
    Snake variants), which ran 5-15 jobs at a time on this machine
  - wall time per 1B job = rate x job length (linear extrapolation)
Then per tier (configs from s5_runs.sh: Tier 1 21, Tier 2 13; 9 workloads
each): core-hours, and the makespan of a longest-first pool of <parallel>
(default 15) jobs.
"""
import glob
import heapq
import os
import re
import sys

WARP_INSTS = {  # sum of "insts =" in hw_run/.../traces/kernel-*.traceg
    "backprop-rodinia-3.1/65536": 5931031,
    "hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out": 3250800,
    "hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out": 13007672,
    "lud-rodinia-3.1/_i___data_512_dat": 6320384,
    "lud-rodinia-3.1/_s_256__v": 830848,
    "nw-rodinia-3.1/2048_10": 12926976,
    "srad_v1-rodinia-3.1/100_0_5_502_458": 403902810,
    "parboil-histo/_i___data_default_input_img_bin__o_ref_bmp____20_4": 73885150,
    "parboil-mri-q/_i___data_small_input_32_32_32_dataset_bin__o_32_32_32_dataset_out": 45697632,
}
TIERS = {"Tier 1": 21, "Tier 2": 13}
CAP = 1e9


def stat(path, pat, conv=int):
    m = re.findall(pat, open(path, errors="replace").read(), re.M)
    return conv(m[-1]) if m else None


def main():
    argv = sys.argv[1:]
    par = 15
    if argv[:1] == ["-p"]:
        par, argv = int(argv[1]), argv[2:]
    ref, dirs = argv[0], argv[1:]
    rows, per_cfg = [], []
    for w, winsts in WARP_INSTS.items():
        b = os.path.join(ref, w, "QV100-SASS-PAPER_V100-100M_INSN", "sim.out")
        insn = stat(b, r"^gpu_tot_sim_insn = (\d+)")
        wi = stat(b, r"^gpgpu_n_tot_w_icount = (\d+)")
        total = winsts * insn / wi
        rate, n = 0.0, 0
        for d in [ref] + dirs:
            for p in glob.glob(os.path.join(d, w, "QV100-SASS-PAPER_V100-100M_INSN*", "sim.out")) + \
                     glob.glob(os.path.join(d, "runs", w, "QV100-SASS-PAPER_V100-100M_INSN*", "sim.out")):
                s = stat(p, r"gpgpu_simulation_time = .*\((\d+) sec\)")
                i = stat(p, r"^gpu_tot_sim_insn = (\d+)")
                if s and i:
                    rate, n = max(rate, s / i), n + 1
        job = min(total, CAP)
        sec = rate * job
        rows.append((w, total, job, rate * 1e8, n, sec))
        per_cfg.append(sec)
    print("| workload | instructions to completion | 1B job length | s per 100M (max of %s runs) | est. s per 1B job |" % "n")
    print("|---|---|---|---|---|")
    for w, total, job, r100, n, sec in rows:
        print("| %s / %s | %.0fM | %.0fM | %.0f (%d) | %.0f |" % (
            w.split("/")[0].replace("-rodinia-3.1", "").replace("parboil-", ""),
            w.split("/")[1][:12], total / 1e6, job / 1e6, r100, n, sec))
    one = sum(per_cfg)
    print("\nOne config (9 jobs): %.0f s = %.2f core-hours; longest job %.0f s.\n" % (one, one / 3600, max(per_cfg)))
    print("| tier | configs | runs | core-hours | makespan at %d parallel (longest first) |" % par)
    print("|---|---|---|---|---|")
    tot_h = tot_runs = 0
    for t, n in TIERS.items():
        jobs = sorted(per_cfg * n, reverse=True)
        slots = [0.0] * par
        for j in jobs:
            heapq.heappush(slots, heapq.heappop(slots) + j)
        h = one * n / 3600
        tot_h += h
        tot_runs += 9 * n
        print("| %s | %d | %d | %.1f | %.1f h |" % (t, n, 9 * n, h, max(slots) / 3600))
    print("| both | %d | %d | %.1f | %.1f h (sequential tiers) |" % (
        sum(TIERS.values()), tot_runs, tot_h,
        sum(max(_pool(per_cfg * n, par)) for n in TIERS.values()) / 3600))


def _pool(jobs, par):
    slots = [0.0] * par
    for j in sorted(jobs, reverse=True):
        heapq.heappush(slots, heapq.heappop(slots) + j)
    return slots


if __name__ == "__main__":
    main()
