"""R892: per-arm MTP step cost from one arm's engine log (fresh container per arm, so the log holds only that arm).

Server-side R828_COUNTER rows (gen_time, decode_steps, gen_tokens), in request order: warmup rows are short (< 512
tokens) and skipped; then c1 (--runs 2 x five kinds), fp (x five kinds), then c4 rounds of four concurrent requests.
Usage: python3 mtp_steps.py <engine.log> [c1_requests=15]
"""
import json
import sys

rows = []
for line in open(sys.argv[1], errors='replace'):
    if 'R828_COUNTER' in line:
        try:
            rows.append(json.loads(line.split('R828_COUNTER', 1)[1]))
        except ValueError:
            pass
n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 15
big = [r for r in rows if r['gen_tokens'] >= 512]
c1, c4 = big[:n1], big[n1:]
if not c1:
    sys.exit('no R828_COUNTER rows')
s = sum(r['decode_steps'] for r in c1); t = sum(r['gen_time'] for r in c1); n = sum(r['gen_tokens'] for r in c1)
kinds = {}
for r in c1:
    k = r['label'] if 'label' in r else '?'
    kinds.setdefault(k.split()[0], []).append(r['gen_time'] * 1000 / r['decode_steps'])
out = f'c1 n={len(c1)} ms/step={t * 1000 / s:.2f} tok/step={n / s:.3f} ms/tok={t * 1000 / n:.2f}'
if c4:
    rounds = [c4[i:i + 4] for i in range(0, len(c4) - len(c4) % 4, 4)]
    ms = [max(r['gen_time'] for r in g) * 1000 / max(r['decode_steps'] for r in g) for g in rounds]
    agg = [sum(r['gen_tokens'] for r in g) / max(r['gen_time'] for r in g) for g in rounds]
    out += ' | c4 ' + ' '.join(f'{m:.1f}ms/round {a:.1f}tok/s' for m, a in zip(ms, agg))
print(out)
