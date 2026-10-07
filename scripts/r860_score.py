#!/usr/bin/env python3
"""R860 scoring. `r860_score.py <c1.jsonl> <tag>` prints a one-line arm summary.
`r860_score.py --best <results dir> <tag>...` prints the tag with the highest score (empty if none measured).
Score = mean over content kinds of the per-kind median c1 decode rate; arms that measured fewer kinds than the
others are not comparable and are skipped by --best."""
import json, os, statistics as st, sys

def load(path):
    if not os.path.exists(path):
        return {}
    by = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('phase') == 'c1-decode':
            by.setdefault(r['kind'], []).append(r)
    return by

def score(by):
    return st.mean(st.median(x['tps'] for x in v) for v in by.values()) if by else None

if sys.argv[1] == '--best':
    root, tags = sys.argv[2], sys.argv[3:]
    rows = [(t, load(f'{root}/{t}/c1.jsonl')) for t in tags]
    kinds = max((len(b) for _, b in rows), default=0)
    scored = [(t, score(b)) for t, b in rows if b and len(b) == kinds]
    print(max(scored, key=lambda x: x[1])[0] if scored else '')
else:
    by = load(sys.argv[1])
    parts = []
    for k, v in by.items():
        acc = [x['accept_rate'] for x in v if x.get('accept_rate') is not None]
        parts.append(f"{k} {st.median(x['tps'] for x in v):.1f}" + (f" (acc {st.median(acc):.2f})" if acc else ''))
    s = score(by)
    deps = []
    for line in (open(sys.argv[1]) if os.path.exists(sys.argv[1]) else []):
        r = json.loads(line)
        if r.get('phase') == 'depth':
            deps.append(f"{r['target']//1024}k pf {r['engine_prefill_tps']:.0f} t/s dec {r['decode_tps']:.1f}")
    print(f"{sys.argv[2]} c1 score {s:.1f} t/s: " + '; '.join(parts) + (' | ' + '; '.join(deps) if deps else '')
          if s is not None else f"{sys.argv[2]}: no c1 decode samples")
