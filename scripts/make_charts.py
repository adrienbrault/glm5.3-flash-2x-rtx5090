#!/usr/bin/env python3
"""Draw the README and HISTORY charts as standalone SVG files in docs/img/.

The README charts are read from the raw records of the round named in README.md (c1.jsonl and measure.jsonl).
The history chart takes one value per served configuration from the write-ups listed in HISTORY below.
Colors: categorical slots 1 and 2 of the dataviz reference palette, light and dark steps (validated for
adjacent-pair CVD and contrast); text uses neutral ink, never the series color.

    python3 scripts/make_charts.py            # writes docs/img/*.svg
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVED = ROOT / 'bench/results/2026-10-07-r882b-glm53-swap-agent/XA'
OUT = ROOT / 'docs/img'

# (label, date, c1 score tok/s or None, sum over 4 streams tok/s or None, write-up)
HISTORY = [
    ('R858', '10-06', None, 68.6, 'r858-glm53-audition.md'),       # MTP depth 1, 104 on the CPU, dynamic, 64k cache
    ('R860', '10-07', 46.2, None, 'r860-glm53-chain.md'),          # MTP off, 104 on the CPU, dynamic
    ('R864', '10-07', 48.4, None, 'r864-glm53-levers.md'),         # 96 on the CPU, dynamic
    ('R869', '10-07', 57.5, None, 'r869-glm53-hotset.md'),         # static placement, broad counts
    ('R873', '10-07', None, 86.3, 'r873-glm53-c4.md'),             # static broad + agent overlay
    ('R882b', '10-07', 59.9, 109.9, 'r882b-glm53-swap-agent.md'),  # exchange swaps + agent overlay (served)
]

STYLE = '''<style>
  .bg { fill: #fcfcfb; }
  .t1 { fill: #0b0b0b; font: 600 15px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .t2 { fill: #52514e; font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .t3 { fill: #0b0b0b; font: 600 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .grid { stroke: #e6e5e0; stroke-width: 1; }
  .axis { stroke: #bdbcb5; stroke-width: 1; }
  .s1 { fill: #2a78d6; } .s1l { stroke: #2a78d6; }
  .s2 { fill: #eb6834; } .s2l { stroke: #eb6834; }
  .ring { stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .t1, .t3 { fill: #ffffff; } .t2 { fill: #c3c2b7; }
    .grid { stroke: #2e2e2c; } .axis { stroke: #55544f; }
    .s1 { fill: #3987e5; } .s1l { stroke: #3987e5; }
    .s2 { fill: #d95926; } .s2l { stroke: #d95926; }
    .ring { stroke: #1a1a19; }
  }
</style>'''


def svg(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="t d">\n<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n{STYLE}\n'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n{body}\n</svg>\n')


def ticks(vmax, step):
    return [step * i for i in range(int(vmax // step) + 1)]


def col(x, y0, y1, w, cls, tip):
    """Column from baseline y0 up to y1, square at the baseline, 4px rounded top."""
    r = min(4, (y0 - y1) / 2, w / 2)
    d = (f'M{x:.1f},{y0:.1f} V{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} H{x + w - r:.1f} '
         f'Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} V{y0:.1f} Z')
    return f'<path class="{cls}" d="{d}"><title>{tip}</title></path>'


def served_numbers():
    by_kind = {}
    for line in open(SERVED / 'c1.jsonl'):
        r = json.loads(line)
        by_kind.setdefault(r['kind'], []).append(r['tps'])
    kinds = {k: statistics.median(v) for k, v in by_kind.items()}
    conc = {}
    for line in open(SERVED / 'measure.jsonl'):
        r = json.loads(line)
        if r.get('phase') == 'decode-summary':
            for s in r['summaries']:
                conc[s['c']] = (s['ss_per_stream_tps_median'], s['ss_agg_tps_median'])
    return kinds, conc


def chart_concurrency(conc):
    w, h, l, r, t, b = 640, 350, 56, 70, 96, 46
    xs = sorted(conc)
    vmax = 120
    px = lambda i: l + (w - l - r) * i / (len(xs) - 1)
    py = lambda v: h - b - (h - t - b) * v / vmax
    out = ['<text class="t1" x="20" y="30">Decode rate by number of concurrent streams</text>',
           '<text class="t2" x="20" y="50">tok/s, chat requests forced to 1,024 tokens, temperature 0, median of 3 rounds (R882b)</text>']
    for v in ticks(vmax, 20):
        out.append(f'<line class="grid" x1="{l}" x2="{w - r}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        out.append(f'<text class="t2" x="{l - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for i, c in enumerate(xs):
        out.append(f'<text class="t2" x="{px(i):.1f}" y="{h - b + 20}" text-anchor="middle">{c} stream{"s" if c > 1 else ""}</text>')
    for k, cls, name in ((1, 's2', 'sum of the streams'), (0, 's1', 'per stream')):
        pts = ' '.join(f'{px(i):.1f},{py(conc[c][k]):.1f}' for i, c in enumerate(xs))
        out.append(f'<polyline class="{cls}l" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="{pts}"/>')
        for i, c in enumerate(xs):
            out.append(f'<circle class="{cls} ring" cx="{px(i):.1f}" cy="{py(conc[c][k]):.1f}" r="5">'
                       f'<title>{c} stream{"s" if c > 1 else ""}, {name}: {conc[c][k]:.1f} tok/s</title></circle>')
        end = conc[xs[-1]][k]
        out.append(f'<text class="t3" x="{px(len(xs) - 1) + 12:.1f}" y="{py(end) + 4:.1f}">{end:.1f}</text>')
    out.append(f'<text class="t3" x="{px(0):.1f}" y="{py(conc[xs[0]][0]) - 12:.1f}" text-anchor="middle">{conc[xs[0]][0]:.1f}</text>')
    out.append(f'<line class="axis" x1="{l}" x2="{w - r}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    # legend
    out.append('<circle class="s1" cx="26" cy="72" r="5"/><text class="t2" x="36" y="76">per stream</text>')
    out.append('<circle class="s2" cx="126" cy="72" r="5"/><text class="t2" x="136" y="76">sum of the streams</text>')
    return svg(w, h, 'Decode rate by number of concurrent streams',
               'Per-stream decode 59.9, 41.1, 27.5 tok/s and sum of the streams 59.9, 82.2, 109.9 tok/s at 1, 2, 4 streams.',
               '\n'.join(out))


def chart_kinds(kinds):
    order = ['code', 'prose', 'chat', 'html', 'edit']
    w, l, r, t, bar, gap = 640, 70, 70, 70, 22, 14
    h = t + len(order) * (bar + gap) + 30
    vmax = 70
    px = lambda v: l + (w - l - r) * v / vmax
    out = ['<text class="t1" x="20" y="30">Single-stream decode by content kind</text>',
           '<text class="t2" x="20" y="50">tok/s, median of 2 runs, 1,024 forced tokens (html 2,048), reasoning effort low (R882b)</text>']
    for v in ticks(vmax, 10):
        out.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{t - 6}" y2="{h - 26}"/>')
        out.append(f'<text class="t2" x="{px(v):.1f}" y="{h - 10}" text-anchor="middle">{v}</text>')
    for i, k in enumerate(order):
        y = t + i * (bar + gap)
        x1 = px(kinds[k])
        rr = 4
        d = f'M{l},{y} H{x1 - rr:.1f} Q{x1:.1f},{y} {x1:.1f},{y + rr} V{y + bar - rr} Q{x1:.1f},{y + bar} {x1 - rr:.1f},{y + bar} H{l} Z'
        out.append(f'<path class="s1" d="{d}"><title>{k}: {kinds[k]:.1f} tok/s</title></path>')
        out.append(f'<text class="t2" x="{l - 10}" y="{y + bar / 2 + 4}" text-anchor="end">{k}</text>')
        out.append(f'<text class="t3" x="{x1 + 8:.1f}" y="{y + bar / 2 + 4}">{kinds[k]:.1f}</text>')
    out.append(f'<line class="axis" x1="{l}" x2="{l}" y1="{t - 6}" y2="{h - 26}"/>')
    return svg(w, h, 'Single-stream decode by content kind',
               ', '.join(f'{k} {kinds[k]:.1f} tok/s' for k in order) + '.', '\n'.join(out))


def chart_history():
    w, h = 760, 340
    out = ['<text class="t1" x="20" y="30">Served configurations, 2026-10-06 to 2026-10-07</text>',
           '<text class="t2" x="20" y="50">tok/s; each column is the round that measured the configuration it served (docs/HISTORY.md)</text>']
    panels = (('Single stream, c1 score (mean over the content kinds)', 2, 70, 's1'),
              ('Four streams, sum of the stream rates', 3, 120, 's2'))
    pw = (w - 60) / 2
    for p, (name, idx, vmax, cls) in enumerate(panels):
        x0, t, b = 30 + p * (pw + 20), 96, 56
        rows = [hst for hst in HISTORY if hst[idx] is not None]
        l = x0 + 34
        r = x0 + pw - 10
        py = lambda v: h - b - (h - t - b) * v / vmax
        out.append(f'<text class="t3" x="{x0}" y="78">{name}</text>')
        for v in ticks(vmax, 20 if vmax > 80 else 10):
            out.append(f'<line class="grid" x1="{l}" x2="{r:.1f}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
            out.append(f'<text class="t2" x="{l - 6}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
        band = (r - l) / len(rows)
        cw = min(24, band * 0.5)
        for i, row in enumerate(rows):
            cx = l + band * (i + 0.5)
            out.append(col(cx - cw / 2, py(0), py(row[idx]), cw, cls, f'{row[0]} ({row[1]}): {row[idx]} tok/s'))
            out.append(f'<text class="t3" x="{cx:.1f}" y="{py(row[idx]) - 6:.1f}" text-anchor="middle">{row[idx]}</text>')
            out.append(f'<text class="t2" x="{cx:.1f}" y="{h - b + 18}" text-anchor="middle">{row[0]}</text>')
        out.append(f'<line class="axis" x1="{l}" x2="{r:.1f}" y1="{py(0):.1f}" y2="{py(0):.1f}"/>')
    desc = '; '.join(f'{x[0]}: c1 {x[2]}, 4-stream sum {x[3]}' for x in HISTORY)
    return svg(w, h, 'Served configurations over time', desc, '\n'.join(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds, conc = served_numbers()
    (OUT / 'decode-concurrency.svg').write_text(chart_concurrency(conc))
    (OUT / 'c1-by-kind.svg').write_text(chart_kinds(kinds))
    (OUT / 'history.svg').write_text(chart_history())
    print('kinds', {k: round(v, 1) for k, v in kinds.items()}, 'conc', {c: tuple(round(x, 1) for x in v) for c, v in conc.items()})


if __name__ == '__main__':
    main()
