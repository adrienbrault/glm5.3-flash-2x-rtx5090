#!/usr/bin/env python3
"""How often does a decode token need NO CPU expert? (2026-10-08, user question on gating MTP to all-GPU tokens.)

Reads an r859 route-trace run (R860b) the same way analyze_routes.py does: per text MoE layer, decode records, selected
expert ids and the LIVE cpu_mask at that step. Aligns token t across layers by per-layer order and reports:
  - mean CPU picks per token per layer, and the per-layer distribution of CPU picks (0..8);
  - P(layer has zero CPU picks) and what independence would predict from the mean share;
  - per token: number of layers with >= 1 CPU pick, and P(all layers CPU-free);
  - the same for a hypothetical placement where the GPU holds each layer's (288 - N) most-selected experts of this trace
    (upper bound for any placement, fitted on the same trace).
usage: route_allgpu.py <route-traces/run>
"""
import collections, glob, json, os, re, sys
import numpy as np

run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, 'route-*.npz')), key=lambda f: int(re.search(r'-(\d+)\.npz$', f).group(1)))
ids_by_layer = collections.defaultdict(list)
cpu_by_layer = collections.defaultdict(list)
split_n = {}
for f in files:
    z = np.load(f, allow_pickle=True)
    m = json.loads(str(z['meta']))
    if m.get('component') != 'text' or m.get('phase') != 'decode' or m.get('kind') != 'actual':
        continue
    L = int(re.search(r'layers\.(\d+)\.', m['layer']).group(1))
    ids = z['ids'].astype(np.int64)
    mask = z['cpu_mask'].astype(bool)
    split_n[L] = int(mask.sum())
    for row in ids:
        ids_by_layer[L].append(row)
        cpu_by_layer[L].append(int(mask[row].sum()))
layers = sorted(ids_by_layer)
T = min(len(cpu_by_layer[L]) for L in layers)
print(f'layers {len(layers)}, aligned decode tokens {T} (per-layer counts {min(len(v) for v in cpu_by_layer.values())}-{max(len(v) for v in cpu_by_layer.values())}), CPU experts per layer {sorted(set(split_n.values()))}')

def report(name, C):  # C: [T, layers] CPU picks
    k = C.shape[1]
    hist = np.bincount(C.reshape(-1), minlength=9)[:9] / C.size
    share = C.mean() / 8
    zero_layer = (C == 0).mean()
    per_tok_layers_with_cpu = (C > 0).sum(1)
    print(f'\n[{name}] mean CPU picks/token/layer {C.mean():.2f} (share {share:.3f})')
    print('  per-layer CPU-pick histogram 0..8: ' + ' '.join(f'{i}:{h:.3f}' for i, h in enumerate(hist)))
    print(f'  P(layer CPU-free) observed {zero_layer:.3f} vs independent picks {(1 - share) ** 8:.3f}')
    print(f'  P(token CPU-free in all {k} layers) observed {(per_tok_layers_with_cpu == 0).mean():.5f} '
          f'(independent layers would give {np.prod([(C[:, j] == 0).mean() for j in range(k)]):.2e})')
    q = np.percentile(per_tok_layers_with_cpu, [5, 25, 50, 75, 95])
    print(f'  layers with >= 1 CPU pick per token: mean {per_tok_layers_with_cpu.mean():.1f} of {k}; p5/p25/p50/p75/p95 {q}')
    print(f'  tokens with <= 5 such layers: {(per_tok_layers_with_cpu <= 5).mean():.4f}; <= 10: {(per_tok_layers_with_cpu <= 10).mean():.4f}')

C = np.array([[cpu_by_layer[L][t] for L in layers] for t in range(T)], dtype=np.int64)
report('live placement (as served in the trace)', C)
# Upper bound: GPU holds each layer's most-selected (288 - N) experts of this very trace.
Cb = np.zeros_like(C)
for j, L in enumerate(layers):
    rows = np.array(ids_by_layer[L][:T])
    counts = np.bincount(rows.reshape(-1), minlength=288)
    cold = np.argsort(counts)[:split_n[L]]
    is_cpu = np.zeros(288, bool); is_cpu[cold] = True
    Cb[:, j] = is_cpu[rows].sum(1)
report('ideal static hot set fitted on this trace (upper bound)', Cb)
