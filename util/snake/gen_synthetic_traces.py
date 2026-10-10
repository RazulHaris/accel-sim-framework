#!/usr/bin/env python3
"""Write synthetic accel-sim traces for the Snake prefetcher tests.

    gen_synthetic_traces.py <trace root>

Creates <root>/<app>/NO_ARGS/traces/{kernelslist.g,kernel-1.traceg} for the
apps of the snake-synth suite (util/job_launching/apps/define-snake-synth.yml)
and <root>/manifest.json with the parameters util/snake/check_synthetic.py
checks against.  Trace format: accel-sim tracer version 3, every load is an
LDG.E with base+stride address compression (mode 1, trace_parser.cc:200).

(a) snake-synth-strided: a loop of three loads P1, P2, P3 per iteration.
    Warp g of the grid (g = cta * WARPS + w) reads
        P1: A + g*WS + i*IT,  P2: P1 + D1,  P3: P2 + D2      (lanes 4 B apart)
    so Snake should learn the inter-thread chain P1->P2 (D1), P2->P3 (D2),
    P3->P1 (IT-D1-D2), the inter-warp stride WS per warp-ID step at every PC,
    and the intra-warp stride IT at every PC.
(b) snake-synth-random: the same loop shape, but every warp-level base is
    random (lanes still 4 B apart), and P3's lanes are scattered (unequal lane
    strides, so it is excluded, OQ-17).  Nothing should be promoted.
(c) snake-synth-fig15: the paper's Fig. 15 example.  One CTA of 5 warps:
    warp 0 only exits, and W1..W4 are warps 1..4, so that warp IDs, warpID
    bits and the inter-warp stride per warp-ID step match the figure and a
    single LRR scheduler issues W1, W2, W3, W4 in order (with warp 0 active it
    issues warp 0 last in each round).  Phase (a): every warp loads PC 520, addresses
    1000/1100/1200/1300; phase (b): PC 540, 1500/1600/1700/1800; phase (c):
    W1 loads PC 520 again at 2500.  The figure's PCs are taken as hex
    (0x520, 0x540), since trace PCs are hex; strides do not involve PCs.
"""
import json
import os
import random
import sys

HEADER = """-kernel name = {name}
-kernel id = 1
-grid dim = ({grid},1,1)
-block dim = ({block},1,1)
-shmem = 0
-nregs = 64
-binary version = 70
-cuda stream id = 0
-shmem base_addr = 0x00007f0675000000
-local mem base_addr = 0x00007f0677000000
-nvbit version = 1.4
-accelsim tracer version = 3

#traces format = threadblock_x threadblock_y threadblock_z warpid_tb PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]

"""
FULL = "ffffffff"


def ldg(pc, dest_reg, base, lane_stride=4):
    # dest register rotates so that loads do not wait on each other (WAW);
    # R2 is never written, so the address operand is always ready.
    return "%04x %s 1 R%d LDG.E 1 R2 4 1 0x%x %d " % (pc, FULL, dest_reg, base,
                                                       lane_stride)


def ldg_list(pc, dest_reg, addrs):
    return "%04x %s 1 R%d LDG.E 1 R2 4 0 %s " % (
        pc, FULL, dest_reg, " ".join("0x%x" % a for a in addrs))


def exit_inst(pc):
    return "%04x %s 0 EXIT 0 0 " % (pc, FULL)


def write_app(root, app, name, grid, block, warps_insts):
    """warps_insts[cta][warp] = list of instruction lines (EXIT appended)."""
    d = os.path.join(root, app, "NO_ARGS", "traces")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "kernelslist.g"), "w") as f:
        f.write("kernel-1.traceg\n")
    with open(os.path.join(d, "kernel-1.traceg"), "w") as f:
        f.write(HEADER.format(name=name, grid=grid, block=block))
        for cta, warps in enumerate(warps_insts):
            f.write("#BEGIN_TB\n\nthread block = %d,0,0\n\n" % cta)
            for w, insts in enumerate(warps):
                f.write("warp = %d\ninsts = %d\n" % (w, len(insts)))
                f.write("\n".join(insts) + "\n\n")
            f.write("#END_TB\n\n")


def strided(root):
    p = dict(ctas=4, warps=8, iters=16, A=0x10000000, WS=512, IT=65536,
             D1=1 << 20, D2=1 << 20, pcs=[0x100, 0x110, 0x120])
    ctas = []
    for c in range(p["ctas"]):
        warps = []
        for w in range(p["warps"]):
            g = c * p["warps"] + w
            insts, k = [], 0
            for i in range(p["iters"]):
                a1 = p["A"] + g * p["WS"] + i * p["IT"]
                for pc, a in zip(p["pcs"], [a1, a1 + p["D1"],
                                            a1 + p["D1"] + p["D2"]]):
                    insts.append(ldg(pc, 4 + k % 48, a))
                    k += 1
            insts.append(exit_inst(0x130))
            warps.append(insts)
        ctas.append(warps)
    write_app(root, "snake-synth-strided", "snake_synth_strided", p["ctas"],
              p["warps"] * 32, ctas)
    return p


def random_gather(root):
    p = dict(ctas=4, warps=8, iters=16, seed=20261010, pcs=[0x100, 0x110, 0x120])
    rng = random.Random(p["seed"])
    ctas = []
    for c in range(p["ctas"]):
        warps = []
        for w in range(p["warps"]):
            insts, k = [], 0
            for i in range(p["iters"]):
                for j, pc in enumerate(p["pcs"]):
                    if j < 2:  # equal lane strides, random warp base
                        insts.append(ldg(pc, 4 + k % 48,
                                         rng.randrange(1 << 20, 1 << 32) & ~127))
                    else:  # scattered lanes: excluded by the lane check
                        insts.append(ldg_list(pc, 4 + k % 48, [
                            rng.randrange(1 << 20, 1 << 32) & ~3
                            for _ in range(32)]))
                    k += 1
            insts.append(exit_inst(0x130))
            warps.append(insts)
        ctas.append(warps)
    write_app(root, "snake-synth-random", "snake_synth_random", p["ctas"],
              p["warps"] * 32, ctas)
    return p


def fig15(root):
    p = dict(pc_a=0x520, pc_b=0x540,
             phase_a=[1000, 1100, 1200, 1300],
             phase_b=[1500, 1600, 1700, 1800],
             phase_c=2500)
    p["warp_of"] = {"W1": 1, "W2": 2, "W3": 3, "W4": 4}
    warps = [[exit_inst(0x560)]]  # warp 0: idle
    for k in range(4):
        insts = [ldg(p["pc_a"], 4, p["phase_a"][k]),
                 ldg(p["pc_b"], 5, p["phase_b"][k])]
        if k == 0:
            insts.append(ldg(p["pc_a"], 6, p["phase_c"]))
        insts.append(exit_inst(0x560))
        warps.append(insts)
    write_app(root, "snake-synth-fig15", "snake_synth_fig15", 1, 160, [warps])
    return p


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    manifest = {"snake-synth-strided": strided(root),
                "snake-synth-random": random_gather(root),
                "snake-synth-fig15": fig15(root)}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("wrote", ", ".join(sorted(manifest)), "under", root)


if __name__ == "__main__":
    main()
