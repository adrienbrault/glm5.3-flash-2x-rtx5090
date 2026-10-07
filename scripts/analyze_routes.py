#!/usr/bin/env python3
"""GLM-5.3 routing analysis from an r859 route-trace run directory (R860b, 2026-10-07).

Per text MoE layer, over decode records (kind=actual), it answers:
  - observed CPU share: fraction of selected experts that were in the live cpu_mask (dynamic placement);
  - uniform share: what a placement that ignores routing would give (split_n / 288);
  - ideal static share: CPU share if the GPU held the (288 - split_n) most-selected experts of this trace
    (an upper bound for any hot-set file, since it is fitted on the same trace);
  - reuse: fraction of a step's experts that were also selected at the previous step of the same layer;
  - LRU(k): hit rate of a k-slot LRU per layer over the experts that the static ideal leaves on the CPU,
    i.e. what a VRAM cache of k extra slots per layer would catch.

usage: analyze_routes.py <route-traces/run> [--max-files N] [--lru 4,8,16,32]
"""
import argparse, collections, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--max-files', type=int, default=0)
ap.add_argument('--lru', default='4,8,16,32')
ap.add_argument('--holdout', action='store_true', help='fit the static hot set on even steps, score on odd steps (cross-validation)')
ap.add_argument('--split-half', action='store_true', help='fit on the first half of the trace, score on the second (different prompts)')
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.run, 'route-*.npz')), key=lambda f: int(re.search(r'-(\d+)\.npz$', f).group(1)))
if a.max_files:
    files = files[:a.max_files]
seqs = collections.defaultdict(list)    # layer -> list of selected-id arrays, in step order
cpu_hits = collections.Counter(); sel = collections.Counter(); split_n = {}
for f in files:
    z = np.load(f, allow_pickle=True)
    m = json.loads(str(z['meta']))
    if m.get('component') != 'text' or m.get('phase') != 'decode' or m.get('kind') != 'actual':
        continue
    L = int(re.search(r'layers\.(\d+)\.', m['layer']).group(1))
    ids = z['ids'].astype(np.int64)
    mask = z['cpu_mask']
    for row in ids:
        seqs[L].append(row)
        cpu_hits[L] += int(mask[row].sum()); sel[L] += len(row)
    split_n[L] = int(m['split_n'])
ks = [int(k) for k in a.lru.split(',') if k]
tot = collections.Counter()
print(f'files {len(files)}, layers {len(seqs)}')
print('layer steps  observed  uniform  ideal-static  reuse(t-1)  ' + '  '.join(f'LRU{k}' for k in ks))
for L in sorted(seqs):
    rows = seqs[L]; n = split_n[L]; E = 288
    fit = rows[0::2] if a.holdout else rows[:len(rows) // 2] if a.split_half else rows
    test = rows[1::2] if a.holdout else rows[len(rows) // 2:] if a.split_half else rows
    freq = np.bincount(np.concatenate(fit), minlength=E)
    gpu = set(np.argsort(-freq)[:E - n].tolist())
    ideal = sum(1 for r in test for e in r if e not in gpu) / sum(len(r) for r in test)
    reuse = np.mean([len(set(r) & set(p)) / len(r) for p, r in zip(rows, rows[1:])]) if len(rows) > 1 else 0
    lru_rates = []
    for k in ks:
        cache = collections.OrderedDict(); hit = miss = 0
        for r in rows:
            for e in r:
                if e in gpu:
                    continue
                if e in cache:
                    hit += 1; cache.move_to_end(e)
                else:
                    miss += 1; cache[e] = 1
                    if len(cache) > k:
                        cache.popitem(last=False)
        lru_rates.append(hit / max(1, hit + miss))
    obs = cpu_hits[L] / sel[L]
    for key, v in (('obs', obs), ('uni', n / E), ('ideal', ideal), ('reuse', reuse)):
        tot[key] += v
    for k, v in zip(ks, lru_rates):
        tot[f'lru{k}'] += v
    print(f'{L:5d} {len(rows):5d}  {obs:8.3f}  {n/E:7.3f}  {ideal:12.3f}  {reuse:10.3f}  ' + '  '.join(f'{v:5.3f}' for v in lru_rates))
nl = len(seqs)
print(f'MEAN        {tot["obs"]/nl:8.3f}  {tot["uni"]/nl:7.3f}  {tot["ideal"]/nl:12.3f}  {tot["reuse"]/nl:10.3f}  '
      + '  '.join(f'{tot[f"lru{k}"]/nl:5.3f}' for k in ks))
