#!/usr/bin/env python3
"""Per-layer expert selection counts from an r859 route trace, in the format EXL3_MOE_CPU_SPLIT_STATS reads
(block_sparse_mlp_cpu.py: json {module key: [count per checkpoint expert id]}). Decode records only (kind=actual),
optionally restricted to the first fraction of decode steps per layer so a later slice can serve as held-out data.
usage: build_split_stats.py <route-traces/run> <out.json> [--fraction 1.0]"""
import argparse, collections, glob, json, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('out'); ap.add_argument('--fraction', type=float, default=1.0)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.run, 'route-*.npz')), key=lambda f: int(re.search(r'-(\d+)\.npz$', f).group(1)))
rows = collections.defaultdict(list)
for f in files:
    z = np.load(f, allow_pickle=True); m = json.loads(str(z['meta']))
    if m.get('component') == 'text' and m.get('phase') == 'decode' and m.get('kind') == 'actual' and m.get('expert_space') == 'checkpoint':
        rows[m['layer']].extend(z['ids'].astype(np.int64))
out = {}
for key, rs in rows.items():
    rs = rs[:max(1, int(len(rs) * a.fraction))]
    out[key] = np.bincount(np.concatenate(rs), minlength=288).astype(int).tolist()
json.dump(out, open(a.out, 'w'))
print(f'{len(out)} layers, {sum(sum(v) for v in out.values())} selections -> {a.out}')
