#!/usr/bin/env python3
"""Replay ExLlamaV3's dynamic CPU-split placement (block_sparse_mlp_cpu.py, _split_swap_tick /
_split_sweep_layer / run_pending_swap_sweeps) on an r859 route trace, for a grid of its knobs (R860b, 2026-10-07).

Policy as served: a histogram of selections per layer; every INTERVAL decode steps a sweep becomes pending and runs
at the next generation boundary (or inline once 4*INTERVAL steps are overdue). A sweep walks layers in order with a
global BUDGET of swaps, swapping the hottest CPU expert with the coldest GPU expert while
hot >= max(HYST * max(cold, 1), FLOOR * hist.sum() / E); then the histogram halves.
Reported: mean CPU share of decode selections (lower is better) and swaps per 1k decode steps (each swap is two
checkpoint reads between generations).

usage: sim_placement.py <route-traces/run> [--grid]
"""
import argparse, collections, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--grid', action='store_true')
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.run, 'route-*.npz')), key=lambda f: int(re.search(r'-(\d+)\.npz$', f).group(1)))
steps = []          # per decode step of the first layer: dict layer -> ids ; plus new-generation flag
init = {}
cur = None; first_layer = None; last_pos = None
for f in files:
    z = np.load(f, allow_pickle=True)
    m = json.loads(str(z['meta']))
    if m.get('component') != 'text' or m.get('phase') != 'decode' or m.get('kind') != 'actual':
        continue
    L = int(re.search(r'layers\.(\d+)\.', m['layer']).group(1))
    if L not in init:
        init[L] = z['cpu_mask'].copy()
    if first_layer is None:
        first_layer = L
    if L == first_layer:
        pos = int(z['positions'][0])
        newgen = last_pos is None or pos <= last_pos
        last_pos = pos
        cur = {'new': newgen, 'ids': {}}
        steps.append(cur)
    cur['ids'][L] = z['ids'][0].astype(np.int64)
layers = sorted(init)
E = 288
print(f'decode steps {len(steps)}, generations {sum(s["new"] for s in steps)}, layers {len(layers)}')

def simulate(interval, floor, hyst, budget, decay=0.5):
    cpu = {L: init[L].copy() for L in layers}
    hist = {L: np.zeros(E) for L in layers}
    tick = 0; pending = False; swaps = 0; hits = 0; total = 0
    def sweep():
        nonlocal swaps
        b = budget
        for L in layers:
            if b <= 0:
                break
            h = hist[L]; fl = floor * h.sum() / E
            head = sorted((h[e], e) for e in range(E) if not cpu[L][e])
            tail = sorted(((h[e], e) for e in range(E) if cpu[L][e]), reverse=True)
            for (cc, ec), (ch, eh) in zip(head, tail):
                if b <= 0 or ch < max(hyst * max(cc, 1.0), fl):
                    break
                cpu[L][ec] = True; cpu[L][eh] = False; b -= 1; swaps += 1
            h *= decay
    for s in steps:
        if s['new'] and pending:
            sweep(); pending = False; tick = 0
        for L, ids in s['ids'].items():
            hits += int(cpu[L][ids].sum()); total += len(ids)
            np.add.at(hist[L], ids, 1)
        tick += 1
        if tick >= interval:
            pending = True
            if tick >= 4 * interval:
                sweep(); pending = False; tick = 0
    return hits / total, 1000 * swaps / len(steps)

print('interval floor hyst budget   cpu_share  swaps/1k-steps')
base = simulate(128, 8.0, 2.0, 64)
print(f'{128:8d} {8.0:5.1f} {2.0:4.1f} {64:6d}   {base[0]:9.3f}  {base[1]:8.1f}   <- served defaults')
if a.grid:
    for interval in (16, 32, 64, 128):
        for floor in (0.0, 1.0, 2.0, 4.0):
            for hyst in (1.2, 1.5, 2.0):
                for budget in (64, 512, 4096):
                    r = simulate(interval, floor, hyst, budget)
                    print(f'{interval:8d} {floor:5.1f} {hyst:4.1f} {budget:6d}   {r[0]:9.3f}  {r[1]:8.1f}', flush=True)
