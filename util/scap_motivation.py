#!/usr/bin/env python3
"""
scap_motivation.py - streaming analysis of the Accel-Sim sector-mask dump.

Input : CSV produced by -gpgpu_dump_sector_masks (plain or .gz), columns
        cycle,sm_id,warp_id,uid,pc,num_coalesced_transactions,sector_mask,sector,addr,status
Output: a text report on stdout + one CSV with per-PC statistics.

The file is read one line at a time. Memory holds only:
  - the warp-loads that are still "open" (rows seen < num_coalesced_transactions),
  - small fixed-size summaries per PC and per (SM, PC).
It never loads the dump into RAM.

Usage:
  python3 scap_motivation.py sector_masks.csv
  python3 scap_motivation.py sector_masks.csv.gz --out per_pc.csv --max-rows 50000000
"""
import argparse, collections, gzip, sys, time

SECTOR = 32
LINE = 128
TEMPLATE_CAP = 256          # max distinct footprint shapes remembered per PC
DELTA_CAP = 256             # max distinct inter-warp deltas remembered per PC
STALE_CYCLES = 200_000      # an open warp-load older than this is closed as incomplete


# ---------------------------------------------------------------- helpers
def bounded_inc(counter, key, cap):
    """Counter with a size cap: when full, drop the least frequent half (approximate top-k)."""
    counter[key] += 1
    if len(counter) > cap:
        for k, _ in counter.most_common()[cap // 2:]:
            del counter[k]


def mask_sectors(mask_str):
    """'0010' (MSB first) -> [1]"""
    n = len(mask_str)
    return [n - 1 - i for i, ch in enumerate(mask_str) if ch == '1']


def d_class(d):
    return 'dense(1-4)' if d <= 4 else ('moderate(5-12)' if d <= 12 else 'scattered(13+)')


class PCStats:
    __slots__ = ('loads', 'sectors', 'lines', 'd_hist', 'templates', 'line_masks',
                 'deltas', 'delta_pairs', 'delta_repeat', 'status', 'pred')

    def __init__(self):
        self.loads = 0
        self.sectors = 0
        self.lines = 0
        self.d_hist = collections.Counter()
        self.templates = collections.Counter()   # footprint shape -> count (bounded)
        self.line_masks = collections.Counter()  # 4-bit mask per touched line (16 keys max)
        self.deltas = collections.Counter()      # inter-warp base delta -> count (bounded)
        self.delta_pairs = 0                     # consecutive-load pairs with a previous delta
        self.delta_repeat = 0                    # ... where delta == previous delta
        self.status = collections.Counter()
        # predictor counters: [predicted_sectors_fetched, useful_fetched, actual_sectors, covered]
        self.pred = {'scap': [0, 0, 0, 0], 'line': [0, 0, 0, 0], 'firstline': [0, 0, 0, 0]}


class Stream:
    """Last-value predictor state for one (SM, PC) stream of warp-loads."""
    __slots__ = ('base', 'delta', 'template')

    def __init__(self):
        self.base = None
        self.delta = None
        self.template = None


# ---------------------------------------------------------------- per warp-load
def close_load(key, rec, pcs, streams, totals):
    sm, _uid = key
    pc, d_reported, secs, statuses = rec[0], rec[1], rec[3], rec[4]
    if not secs:
        return
    st = pcs.setdefault(pc, PCStats())
    sectors = sorted(secs)
    base = sectors[0]
    template = tuple((s - base) // SECTOR for s in sectors)       # shape relative to first sector
    lines = collections.defaultdict(int)
    for s in sectors:
        lines[s // LINE] |= 1 << ((s % LINE) // SECTOR)

    st.loads += 1
    st.sectors += len(sectors)
    st.lines += len(lines)
    st.d_hist[len(sectors)] += 1
    bounded_inc(st.templates, template, TEMPLATE_CAP)
    for m in lines.values():
        st.line_masks[m] += 1
    st.status.update(statuses)
    if d_reported != len(sectors):
        totals['d_mismatch'] += 1

    # ---- inter-warp delta + three simple last-value predictors (same SM, same PC)
    sp = streams.setdefault((sm, pc), Stream())
    actual = set(sectors)
    if sp.base is not None:
        delta = base - sp.base
        bounded_inc(st.deltas, delta, DELTA_CAP)
        if sp.delta is not None:
            st.delta_pairs += 1
            st.delta_repeat += (delta == sp.delta)
            nb = sp.base + sp.delta                                  # predicted first sector
            # (a) SCAP: learned shape, fetch only those sectors
            pred = {nb + o * SECTOR for o in sp.template}
            hit = len(pred & actual)
            p = st.pred['scap']; p[0] += len(pred); p[1] += hit; p[2] += len(actual); p[3] += hit
            # (b) same shape, but fetched as whole 128 B lines
            plines = {s // LINE for s in pred}
            hit_l = sum(1 for s in actual if s // LINE in plines)
            p = st.pred['line']; p[0] += 4 * len(plines); p[1] += hit_l; p[2] += len(actual); p[3] += hit_l
            # (c) one line at the predicted first address (first-address prefetcher)
            fl = nb // LINE
            hit_f = sum(1 for s in actual if s // LINE == fl)
            p = st.pred['firstline']; p[0] += 4; p[1] += hit_f; p[2] += len(actual); p[3] += hit_f
        sp.delta = delta
    sp.base, sp.template = base, template


# ---------------------------------------------------------------- main loop
def run(path, max_rows, progress_every):
    opener = gzip.open if path.endswith('.gz') else open
    open_loads = {}            # (sm, uid) -> [pc, D, first_cycle, set(sector_addr), [status]]
    pcs, streams = {}, {}
    totals = collections.Counter()
    t0 = time.time()
    last_flush_cycle = 0

    with opener(path, 'rt') as f:
        header = f.readline().strip().split(',')
        col = {name: i for i, name in enumerate(header)}
        need = ['cycle', 'sm_id', 'uid', 'pc', 'num_coalesced_transactions', 'sector_mask', 'addr', 'status']
        missing = [c for c in need if c not in col]
        if missing:
            sys.exit(f'missing columns: {missing}')
        iC, iS, iU, iP, iD, iM, iA, iT = (col[c] for c in need)

        for n, line in enumerate(f, 1):
            parts = line.rstrip('\n').split(',')
            if len(parts) < len(header):
                totals['bad_rows'] += 1
                continue
            cycle = int(parts[iC]); sm = int(parts[iS]); uid = int(parts[iU])
            pc = parts[iP]; d = int(parts[iD]); addr = int(parts[iA], 16)
            key = (sm, uid)
            rec = open_loads.get(key)
            if rec is None:
                rec = open_loads[key] = [pc, d, cycle, set(), []]
            line_base = addr - addr % LINE
            secs = mask_sectors(parts[iM])
            if len(secs) == 1 and (addr % LINE) // SECTOR != secs[0]:
                totals['addr_vs_mask_mismatch'] += 1      # sanity check; the mask wins
            for s in secs:
                rec[3].add(line_base + s * SECTOR)
            rec[4].append(parts[iT])
            totals['rows'] += 1

            if len(rec[4]) >= rec[1]:                     # all D transactions seen
                close_load(key, open_loads.pop(key), pcs, streams, totals)
                totals['loads'] += 1

            if cycle - last_flush_cycle > STALE_CYCLES:   # close warp-loads that never completed
                for k in [k for k, r in open_loads.items() if cycle - r[2] > STALE_CYCLES]:
                    close_load(k, open_loads.pop(k), pcs, streams, totals)
                    totals['loads'] += 1; totals['incomplete'] += 1
                last_flush_cycle = cycle

            if progress_every and n % progress_every == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f'  {n:,} rows  {rate/1e6:.2f} M rows/s  open={len(open_loads)}', file=sys.stderr)
            if max_rows and n >= max_rows:
                break

    for k in list(open_loads):
        close_load(k, open_loads.pop(k), pcs, streams, totals)
        totals['loads'] += 1; totals['incomplete'] += 1
    totals['sms'] = len({sm for sm, _ in streams})
    return pcs, totals


# ---------------------------------------------------------------- report
def pct(a, b):
    return 100.0 * a / b if b else 0.0


def report(pcs, totals, out_csv, min_loads):
    T = sum(s.sectors for s in pcs.values())
    L = sum(s.loads for s in pcs.values())
    print('\n=== 0. Dataset ===')
    print(f"rows {totals['rows']:,} | warp-loads {L:,} | static load PCs {len(pcs):,} | SMs {totals['sms']}")
    print(f"incomplete warp-loads {totals['incomplete']:,} | D column != rows seen {totals['d_mismatch']:,} | "
          f"addr/mask sector mismatch {totals['addr_vs_mask_mismatch']:,} | bad rows {totals['bad_rows']:,}")

    print('\n=== 1. Coalescing degree D (sectors per warp-load) ===')
    cls_l, cls_s = collections.Counter(), collections.Counter()
    for s in pcs.values():
        for d, c in s.d_hist.items():
            cls_l[d_class(d)] += c; cls_s[d_class(d)] += c * d
    for c in ('dense(1-4)', 'moderate(5-12)', 'scattered(13+)'):
        print(f'{c:16s} {pct(cls_l[c], L):6.1f}% of warp-loads   {pct(cls_s[c], T):6.1f}% of sector traffic')

    print('\n=== 2. L1D outcome per sector ===')
    stat = collections.Counter()
    for s in pcs.values():
        stat.update(s.status)
    for k, v in stat.most_common():
        print(f'{k:16s} {pct(v, sum(stat.values())):6.1f}%')

    print('\n=== 3. Does the footprint shape repeat across warps? ===')
    same_shape = sum(s.templates.most_common(1)[0][1] for s in pcs.values() if s.templates)
    print(f'warp-loads whose shape equals their PC\'s most common shape: {pct(same_shape, L):.1f}%')
    for thr in (0.9, 0.5):
        tr = sum(s.sectors for s in pcs.values() if s.loads and s.templates.most_common(1)[0][1] / s.loads >= thr)
        print(f'sector traffic from PCs whose top shape covers >= {int(thr*100)}% of their warp-loads: {pct(tr, T):.1f}%')

    print('\n=== 4. Sector masks inside 128 B lines ===')
    lm = collections.Counter()
    for s in pcs.values():
        lm.update(s.line_masks)
    nlines = sum(lm.values())
    for m, c in lm.most_common(6):
        print(f'mask {m:04b} (s3..s0)  {pct(c, nlines):6.1f}% of touched lines')
    print(f'partial lines (mask != 1111): {pct(nlines - lm[0b1111], nlines):.1f}%')
    print(f'bytes a whole-line fetcher moves vs sectors actually requested: {4 * nlines / max(T, 1):.2f}x')

    print('\n=== 5. Inter-warp delta (consecutive warp-loads, same SM and PC) ===')
    dp = sum(s.delta_pairs for s in pcs.values()); dr = sum(s.delta_repeat for s in pcs.values())
    print(f'delta equals the previous delta: {pct(dr, dp):.1f}% of {dp:,} pairs')

    print('\n=== 6. Last-value predictor: predict warp n+1 from warp n ===')
    print(f"{'predictor':34s}{'coverage':>10s}{'accuracy':>10s}{'sectors fetched / used':>24s}")
    names = {'scap': 'SCAP (shape + delta, sectors)', 'line': 'same shape, whole lines',
             'firstline': 'first address only, 1 line'}
    for k, label in names.items():
        f_, u, a, c = (sum(s.pred[k][i] for s in pcs.values()) for i in range(4))
        print(f'{label:34s}{pct(c, a):9.1f}%{pct(u, f_):9.1f}%{f_ / max(u, 1):>20.2f}x')

    rows = []
    for pc, s in pcs.items():
        if s.loads < min_loads:
            continue
        top_t = s.templates.most_common(1)[0][1] / s.loads if s.templates else 0
        top_m, top_mc = s.line_masks.most_common(1)[0]
        top_d, top_dc = s.deltas.most_common(1)[0] if s.deltas else (0, 0)
        p = s.pred['scap']
        rows.append([pc, s.loads, s.sectors, round(s.sectors / s.loads, 2), d_class(round(s.sectors / s.loads)),
                     round(top_t, 3), f'{top_m:04b}', round(top_mc / max(s.lines, 1), 3),
                     round(4 * s.lines / s.sectors, 2), top_d, round(top_dc / max(sum(s.deltas.values()), 1), 3),
                     round(s.delta_repeat / max(s.delta_pairs, 1), 3),
                     round(p[3] / max(p[2], 1), 3), round(p[1] / max(p[0], 1), 3),
                     round(s.status['SECTOR_MISS'] / max(sum(s.status.values()), 1), 3)])
    rows.sort(key=lambda r: -r[2])
    hdr = ['pc', 'warp_loads', 'sectors', 'mean_D', 'D_class', 'top_shape_share', 'top_line_mask',
           'top_line_mask_share', 'line_overfetch_x', 'top_delta', 'top_delta_share', 'delta_repeat_rate',
           'scap_coverage', 'scap_accuracy', 'sector_miss_share']
    with open(out_csv, 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')

    print(f'\n=== 7. Top PCs by sector traffic (all PCs with >= {min_loads} warp-loads in {out_csv}) ===')
    print(f"{'pc':>10s}{'traffic%':>9s}{'meanD':>7s}{'shape%':>8s}{'mask':>6s}{'mask%':>7s}{'lineX':>7s}{'dRep%':>7s}{'cov%':>7s}{'acc%':>7s}")
    for r in rows[:15]:
        print(f'{r[0]:>10s}{pct(r[2], T):9.1f}{r[3]:7.1f}{100*r[5]:8.1f}{r[6]:>6s}{100*r[7]:7.1f}'
              f'{r[8]:7.2f}{100*r[11]:7.1f}{100*r[12]:7.1f}{100*r[13]:7.1f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', help='sector-mask CSV (.csv or .csv.gz)')
    ap.add_argument('--out', default='scap_per_pc.csv', help='per-PC CSV output')
    ap.add_argument('--max-rows', type=int, default=0, help='stop after N rows (0 = whole file)')
    ap.add_argument('--min-loads', type=int, default=32, help='skip PCs with fewer warp-loads in the CSV')
    ap.add_argument('--progress', type=int, default=10_000_000, help='progress line every N rows (0 = off)')
    a = ap.parse_args()
    pcs, totals = run(a.dump, a.max_rows, a.progress)
    report(pcs, totals, a.out, a.min_loads)
