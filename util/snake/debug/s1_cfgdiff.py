#!/usr/bin/env python3
"""Debug plan Stage 1.2: option-by-option diff of our PAPER_V100 baseline
against the v1.2.0 QV100 configuration (docs/snake/results/debug/S1_environment.md).

    python3 util/snake/debug/s1_cfgdiff.py

Ours: the option dump at the top of a PAPER_V100 1B baseline sim.out (every
registered option with its effective value), plus the job's interconnect file.
v1.2.0: the registered defaults (option_parser_register calls) of each source
tree, overlaid with its SM7_QV100 gpgpusim.config and trace.config, all read
from git objects (nothing is checked out):
  release   gpgpu-sim 971dd69e + accel-sim ed6d38b (the pair v1.2.0 ships)
  dev       gpgpu-sim 6b244a5d + accel-sim ed6d38b (dev at the tag date; its L1
            matches Table 1, VERSION_DIFF.md)
Values are compared after normalising numbers (1530 == 1530.0) and spaces.
"""
import glob
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GS = os.path.join(ROOT, "gpu-simulator", "gpgpu-sim")
ACCEL = "ed6d38b"
TREES = {"release": "971dd69e", "dev": "6b244a5d"}
JOB = glob.glob(os.path.join(ROOT, "sim_run_baseline", "runs", "backprop-rodinia-3.1", "*",
                             "QV100-SASS-PAPER_V100-1B_INSN"))[0]
# does the difference plausibly change how memory-bound an app is? (checked by hand)
EFFECT = {
    "-gpgpu_cache:dl1": "**Yes, large, but ours is the paper's.** Release: 32 KB streaming L1D (1 set x 256 ways, "
                        "allocate on fill), MSHRs overridden to 4096 / merge 64 in code, so almost no MSHR "
                        "reservation fails. Ours = dev = Table 1 (128 KB unified, 4 x 256 ways, on miss, 512 / 8)",
    "-gpgpu_unified_l1d_size": "part of the L1D row (unified 128 KB, Table 1); absent in release",
    "-gpgpu_shmem_option": "part of the L1D row (shared-memory carve-out per kernel); absent in release",
    "-gpgpu_l1_cache_write_ratio": "part of the L1D row; absent in release",
    "-gpgpu_clock_domains": "**Small.** Ours is Table 1's 1530 MHz with DRAM at 850 MHz in both: ~6% more core "
                            "cycles per DRAM cycle, so ours is slightly *more* memory-bound",
    "-gpgpu_coalesce_arch": "**No.** 60 and 70 take the same coalescer path (>= 40) in both trees "
                            "(`abstract_hardware_model.cc:491`)",
    "-gpgpu_max_concurrent_kernel": "**No.** All 9 workloads use one stream, and a stream's kernels run one "
                                    "at a time (`gpu-simulator/accel-sim.cc:55-66`)",
    "-gpgpu_max_insn": "**No.** The paper's 1B cap (§4)",
    "-gpgpu_perfect_inst_const_cache": "**Maybe.** v1.2.0 models I-cache and constant-cache misses: they go to L2 "
                                       "(more L1-L2 packets, Fig. 4) and stall fetch or the memory stage (idle "
                                       "and pipeline-stall cycles, not scoreboard ones, Fig. 5). Constant-heavy "
                                       "mri-q is the most exposed. Ours: always hit (`shader.cc:1003`, `:2313`)",
    "-gpgpu_scheduler": "**No** against Table 1 (GTO); dev's lrr is the only difference",
    "-trace_opcode_latency_initiation_int": "**Yes, but the wrong way.** v1.2.0's longer ALU latencies add "
                                            "scoreboard waits on non-memory registers, so v1.2.0 is *less* "
                                            "memory-bound than ours, not more",
    "-trace_opcode_latency_initiation_sp": "as int",
    "-trace_opcode_latency_initiation_spec_op_3": "as int",
    "-trace_opcode_latency_initiation_tensor": "as int (no tensor ops in our apps)",
}
# accel-sim sources that register options (trace front end)
ACCEL_SRC = ["gpu-simulator/trace-driven/trace_driven.cc", "gpu-simulator/main.cc",
             "gpu-simulator/accel-sim.cc"]


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


LIT = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)


def registered(repo, rev, paths):
    """name -> default string of every option_parser_register call whose name and
    default are string literals (a default may be adjacent literals)."""
    out = {}
    for p in paths:
        src = git(repo, "show", "%s:%s" % (rev, p))
        for m in re.finditer(r"option_parser_register\s*\((.*?)\)\s*;", src, re.S):
            lits = []
            args = LIT.sub(lambda l: lits.append(l.group(1)) or "\0%d\0" % (len(lits) - 1), m.group(1)).split(",")
            tok = [re.findall(r"\0(\d+)\0", a) for a in args]
            if len(args) < 3 or not tok[1] or re.sub(r"\0\d+\0|\s", "", args[-1]) or not tok[-1]:
                continue
            out[lits[int(tok[1][0])]] = "".join(lits[int(i)] for i in tok[-1])
    return out


def config_lines(text):
    """name -> value of a gpgpusim.config-style file (later lines win; a quoted
    value may span lines)."""
    text = "\n".join(line.split("#")[0] for line in text.split("\n"))
    return {m.group(1): m.group(2).strip().strip('"') for m in
            re.finditer(r'^\s*(-\S+)[ \t]+("[^"]*"|[^\n]*)', text, re.M)}


def norm(v):
    def one(x):
        try:
            return "%g" % float(x)
        except ValueError:
            return x
    v = re.sub(r"\s+", "", v or "")
    return re.sub(r"[^,:]+", lambda m: one(m.group(0)), v)


def v120(rev):
    srcs = [p for p in git(GS, "ls-tree", "-r", "--name-only", rev).split("\n")
            if p.startswith(("src/", "libcuda/")) and p.endswith(".cc") and p != "src/option_parser.cc"]
    reg = registered(GS, rev, srcs)
    reg.update(registered(ROOT, ACCEL, ACCEL_SRC))
    cfg = config_lines(git(GS, "show", rev + ":configs/tested-cfgs/SM7_QV100/gpgpusim.config"))
    trc = config_lines(git(ROOT, "show", ACCEL + ":gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"))
    eff = dict(reg)
    src = {k: "default" for k in reg}
    for d, lbl in ((cfg, "gpgpusim.config"), (trc, "trace.config")):
        for k, v in d.items():
            eff[k], src[k] = v, lbl
    icnt = git(GS, "show", rev + ":configs/tested-cfgs/SM7_QV100/config_volta_islip.icnt")
    return eff, src, icnt


def ours():
    t = open(os.path.join(JOB, "sim.out"), errors="replace").read()
    eff = {}
    for m in re.finditer(r"^(-\S+)\s+(.*?)\s+#", t, re.M):  # the option dump: "-name value # help"
        eff.setdefault(m.group(1), m.group(2))
    icnt = open(os.path.join(JOB, "config_volta_islip.icnt")).read()
    return eff, icnt


def icnt_kv(text):
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\s*(\w+)\s*=\s*([^;]*);", text, re.M)}


def main():
    me, my_icnt = ours()
    out = ["## Stage 1.2: option-by-option diff vs v1.2.0 (generated by `util/snake/debug/s1_cfgdiff.py`)\n",
           "Ours = the effective options of `%s` (PAPER_V100 baseline, 1B). v1.2.0 = each tree's registered "
           "defaults overlaid with its SM7_QV100 `gpgpusim.config` and accel-sim `ed6d38b`'s `trace.config`, "
           "read from git. 'absent' = the option does not exist in that tree.\n" % os.path.relpath(JOB, ROOT)]
    res = {}
    for tag, rev in TREES.items():
        eff, src, icnt = v120(rev)
        res[tag] = (eff, src, icnt)
    keys = sorted(set(me) | set(res["release"][0]) | set(res["dev"][0]))
    rows, new_only = [], []
    for k in keys:
        a = me.get(k)
        vals = {t: res[t][0].get(k) for t in TREES}
        if a is None and all(v is None for v in vals.values()):
            continue
        if all(v is None for v in vals.values()):
            new_only.append((k, a))
            continue
        if all(v is not None and norm(v) == norm(a) for v in vals.values()):
            continue
        rows.append((k, a, vals, {t: res[t][1].get(k, "") for t in TREES}))
    out.append("### Options whose value differs from v1.2.0 (release or dev)\n")
    out.append("| option | ours (PAPER_V100) | v1.2.0 release `971dd69e` | v1.2.0-era dev `6b244a5d` | "
               "changes how memory-bound an app is? |")
    out.append("|---|---|---|---|---|")
    for k, a, vals, srcs in rows:
        cell = lambda t: "absent" if vals[t] is None else ("`%s` (%s)" % (vals[t], srcs[t]) if norm(vals[t]) != norm(a)
                                                           else "same")
        out.append("| `%s` | %s | %s | %s | %s |" % (k, "absent" if a is None else "`%s`" % a, cell("release"),
                                                     cell("dev"), EFFECT.get(k, "not assessed")))
    out.append("\n### Options that exist only in our tree (%d)\n" % len(new_only))
    out.append(", ".join("`%s %s`" % (k, v) for k, v in new_only))
    out.append("\n### Interconnect file (`config_volta_islip.icnt`)\n")
    mi = icnt_kv(my_icnt)
    for tag in TREES:
        ti = icnt_kv(res[tag][2])
        d = ["`%s` ours %s vs %s" % (k, mi.get(k, "absent"), ti.get(k, "absent")) for k in sorted(set(mi) | set(ti))
             if norm(mi.get(k)) != norm(ti.get(k))]
        out.append("- %s: %s" % (tag, "; ".join(d) if d else "identical"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
