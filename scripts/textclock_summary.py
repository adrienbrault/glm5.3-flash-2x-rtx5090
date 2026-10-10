#!/usr/bin/env python3
"""Fail-closed R968 report. Schema is the installed glm53_probe/r860_score schema."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import sys

KINDS = ('code', 'prose', 'chat', 'html', 'edit')
# REVIEW-r2 B1: glm53_probe.py KIND_TOKENS forces html to 2048 tokens, every other kind 1024.
KIND_TOKENS = {'html': 2048}
TAGS = ('A1', 'B1', 'B2', 'A2')


def require(condition, message):
    if not condition: raise ValueError(message)


def rows(path):
    require(path.is_file(), f'missing {path}')
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def positive(x):
    require(isinstance(x, (float, int)) and math.isfinite(x) and x > 0, f'invalid throughput {x}')
    return x


def c4(path):
    data = rows(path)
    samples = [r for r in data if r.get('phase') == 'decode' and r.get('c') == 4]
    require(len(samples) == 2 and {r['run'] for r in samples} == {0, 1}, f'{path}: expected two c4 runs')
    require(all(r['tokens'] == 1024 and r['ss_window_s'] > 0 for r in samples), f'{path}: incomplete c4')
    values = [positive(r['ss_agg_tps']) for r in samples]
    summaries = [r for r in data if r.get('phase') == 'decode-summary']
    require(len(summaries) == 1, f'{path}: expected one decode summary')
    s = [s for s in summaries[0]['summaries'] if s['c'] == 4]
    require(len(s) == 1 and s[0]['distinct'] is True, f'{path}: missing distinct c4 summary')
    rate = positive(s[0]['ss_agg_tps_median'])
    require(math.isclose(rate, st.median(values), rel_tol=1e-9), f'{path}: summary disagrees with samples')
    return rate


def c1(path):
    samples = [r for r in rows(path) if r.get('phase') == 'c1-decode']
    require(len(samples) == 10 and {(r['kind'], r['run']) for r in samples} ==
            {(k, n) for k in KINDS for n in (0, 1)}, f'{path}: expected ten c1 samples')
    require(all(r['tokens'] == KIND_TOKENS.get(r['kind'], 1024) for r in samples), f'{path}: incomplete c1')
    return st.median(positive(r['tps']) for r in samples)


def fingerprints(directory):
    samples = [r for r in rows(directory / 'fp.jsonl') if r.get('phase') == 'c1-decode']
    require(len(samples) == 5 and {(r['kind'], r['run']) for r in samples} ==
            {(k, 0) for k in KINDS}, f'{directory}: expected five fingerprint samples')
    hashes = {}
    for r in samples:
        k = r['kind']; digest = r['content_sha256']
        require(r['tokens'] == KIND_TOKENS.get(k, 1024) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                f'{directory}: invalid {k} fingerprint')
        reasoning, content = [], []
        for e in rows(directory / f'c1-{k}-r0.events.jsonl'):
            event = e['event']
            require('error' not in event, f'{directory}: fingerprint stream error')
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning.append(delta.get('reasoning_content') or '')
                content.append(delta.get('content') or '')
        require(any(reasoning) or any(content), f'{directory}: empty {k} fingerprint')
        actual = hashlib.sha256((''.join(reasoning) + '\0' + ''.join(content)).encode()).hexdigest()
        require(actual == digest, f'{directory}: {k} hash disagrees with archived stream')
        hashes[k] = digest
    return hashes


def report(root):
    arms = {}
    for tag in TAGS:
        directory = root / tag
        require((directory / 'engine.log').is_file() and (directory / 'engine-boot.log').is_file(),
                f'{tag}: engine logs missing')
        logs = (directory / 'engine.log').read_text()
        require('[SWAP-CLOCK]' not in logs and ' -- exchange sweep:' not in logs,
                f'{tag}: DEBUG on in measured arm')
        arms[tag] = dict(c4_cold=c4(directory / 'c4-cold.jsonl'), c1_median=c1(directory / 'c1.jsonl'),
                         c4_after=c4(directory / 'c4-after.jsonl'), fingerprints=fingerprints(directory))
        a = arms[tag]
        print(f'{tag}: c4 cold={a["c4_cold"]:.2f} c1 median={a["c1_median"]:.2f} c4 after={a["c4_after"]:.2f} tok/s')
        print(f'{tag} fingerprint hashes: ' + ' '.join(f'{k}={a["fingerprints"][k]}' for k in KINDS))
    aggregated = {label: {metric: st.median(arms[t][metric] for t in tags)
                          for metric in ('c4_cold', 'c1_median', 'c4_after')}
                  for label, tags in (('A', ('A1', 'A2')), ('B', ('B1', 'B2')))}
    cold_ratio = aggregated['B']['c4_cold'] / aggregated['A']['c4_cold']
    c1_ratio = aggregated['B']['c1_median'] / aggregated['A']['c1_median']
    identical = all(a['fingerprints'] == arms['A1']['fingerprints'] for a in arms.values())
    fatal = []
    for p in root.rglob('*.log'):
        for n, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
            if 'FATAL' in line: fatal.append(f'{p.relative_to(root)}:{n}')
    if (root / 'summary.txt').is_file():
        text = (root / 'summary.txt').read_text()
        require(not any(f'{t} fp rc=' in text for t in TAGS), 'fingerprint command failed')
        if 'FATAL' in text: fatal.append('summary.txt')
    gates = {'cold_c4_gain_at_least_10_percent': cold_ratio >= 1.10,
             'c1_within_2_percent': 0.98 <= c1_ratio <= 1.02,
             'no_FATAL_lines': not fatal}
    # REVIEW-r2 B2: greedy output is not boot-to-boot stable under exchange on this image (LEARNINGS), so
    # fingerprint identity is reported as information only, not gated.
    print(f'INFO fingerprints_identical={identical}')
    print('Aggregation: median of the two arm medians per A/B (order A1 B1 B2 A2)')
    print(f'A medians={aggregated["A"]}; B medians={aggregated["B"]}')
    print(f'B/A cold c4={cold_ratio:.4f} ({(cold_ratio-1)*100:+.2f}%); c1={c1_ratio:.4f} ({(c1_ratio-1)*100:+.2f}%)')
    for k, ok in gates.items(): print(f'GATE {k}: {"PASS" if ok else "FAIL"}')
    if fatal: print('FATAL locations: ' + ', '.join(fatal))
    passed = all(gates.values())
    print('OVERALL: ' + ('PASS' if passed else 'FAIL'))
    (root / 'gate.json').write_text(json.dumps(dict(arms=arms, aggregate=aggregated, gates=gates,
                                                  fatal_locations=fatal, passed=passed), indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        sys.exit(report(Path(sys.argv[1])))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as e:
        print(f'OVERALL: FAIL (missing/invalid evidence: {e})')
        sys.exit(2)
