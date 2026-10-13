#!/usr/bin/env python3
"""Count the load PCs Snake trains on, per kernel, from accel-sim traces.

    load_pcs.py <kernel-N.traceg> [...]

For each kernel trace prints the number of distinct L1-cached global-load PCs
(LDG without .STRONG.GPU or .BYPASS, not LDGSTS: the loads Snake observes,
OQ-21) and of distinct consecutive (PC1, PC2) pairs of those loads within a
warp, i.e. the inter-thread edges that compete for Tail entries (Tab. 3: 10).
Strides are not considered, so an edge may need more than one entry.
"""
import sys


def snake_load(op):
    toks = op.split(".")
    if toks[0] != "LDG":
        return False
    return not (("STRONG" in toks and "GPU" in toks) or "BYPASS" in toks)


def scan(path):
    name = "?"
    pcs, pairs = set(), set()
    last = None
    for line in open(path, errors="replace"):
        if line.startswith("-kernel name"):
            name = line.split("=", 1)[1].strip()
            continue
        if line.startswith("warp =") or line.startswith("thread block"):
            last = None  # a new warp's instruction stream
            continue
        f = line.split()
        if len(f) < 4 or line[0] in "#-":
            continue
        try:
            ndst = int(f[2])
        except ValueError:
            continue
        if len(f) <= 3 + ndst or not snake_load(f[3 + ndst]):
            continue
        pc = int(f[0], 16)
        pcs.add(pc)
        if last is not None:
            pairs.add((last, pc))
        last = pc
    return name, pcs, pairs


def main():
    for path in sys.argv[1:]:
        name, pcs, pairs = scan(path)
        print("%s  %s: %d load PCs, %d (PC1, PC2) edges; PCs %s"
              % (path.rsplit("/", 1)[-1], name, len(pcs), len(pairs),
                 " ".join("%x" % p for p in sorted(pcs))))


if __name__ == "__main__":
    main()
