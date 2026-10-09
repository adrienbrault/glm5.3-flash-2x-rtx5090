#!/usr/bin/env python3
"""Offline-testable environment checks and same-slot ABA reporting for R929."""
import argparse
import json
from pathlib import Path
import re
import shlex

DAILY_IMAGE = 'tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2'
PAIRS = ('96,100', '97,101', '98,102', '99,103')
# 2026-10-09 run 3: shadow walk-up pairs (108,112 warmed with GPU1 at 59-61 MiB free).
SHADOW_PAIRS = ('108,116', '112,120')
ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=')


def daily_values(text):
    values = {}
    for word in shlex.split(text):
        if not ASSIGNMENT.match(word):
            raise ValueError('daily contains a non-assignment token')
        key, value = word.split('=', 1)
        values[key] = value
    if values.get('GLM_IMG') != DAILY_IMAGE:
        raise ValueError('daily image differs from the pinned splitdev2 image')
    flags = dict(item.split('=', 1) for item in values.get('EXL3_EXTRA', '').split(';') if item)
    if flags.get('EXL3_MOE_CPU_SPLIT_BY_DEVICE') != '100,104':
        raise ValueError('daily per-device split must be 100,104')
    return values


def extra(text, pair, ring, shadow):
    values = daily_values(text)
    if pair not in ('100,104', '108,112') + PAIRS + SHADOW_PAIRS or (shadow and not ring):
        raise ValueError('invalid R929 arm')
    overridden = ('EXL3_DSA_INDEX_RING', 'EXL3_DSA_INDEX_RING_SHADOW', 'EXL3_MOE_CPU_SPLIT_BY_DEVICE')
    flags = [item for item in values.get('EXL3_EXTRA', '').split(';')
             if item and item.split('=', 1)[0] not in overridden]
    flags += [f'EXL3_MOE_CPU_SPLIT_BY_DEVICE={pair}', f'EXL3_DSA_INDEX_RING={ring}',
              f'EXL3_DSA_INDEX_RING_SHADOW={shadow}']
    return ';'.join(flags)


def effective_env(inspect):
    """Docker image/container Config.Env, then launcher env Args, in execution order."""
    if isinstance(inspect, list):
        if len(inspect) != 1:
            raise ValueError('expected one inspected container')
        inspect = inspect[0]
    values = {}
    for item in inspect.get('Config', {}).get('Env') or []:
        key, value = item.split('=', 1)
        values[key] = value
    args = iter(inspect.get('Args') or [])
    for item in args:
        if item in ('-i', '--ignore-environment'):
            values.clear()
        elif item in ('-u', '--unset'):
            values.pop(next(args), None)
        elif item.startswith('--unset='):
            values.pop(item.split('=', 1)[1], None)
        elif ASSIGNMENT.match(item):
            key, value = item.split('=', 1)
            values[key] = value
    return values


def check_env(inspect, pair, ring, shadow, image=None):
    values = effective_env(inspect)
    expected = {'EXL3_MOE_CPU_SPLIT_BY_DEVICE': pair, 'EXL3_DSA_INDEX_RING': str(ring),
                'EXL3_DSA_INDEX_RING_SHADOW': str(shadow)}
    errors = [f'{key}: effective={values.get(key)!r}, expected={value!r}'
              for key, value in expected.items() if values.get(key) != value]
    container = inspect[0] if isinstance(inspect, list) else inspect
    if image is not None and container.get('Config', {}).get('Image') != image:
        errors.append(f'container image differs from expected {image}')
    return {'passed': not errors, 'effective': values, 'expected': expected, 'errors': errors}


def decode(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get('error') for row in rows):
        raise ValueError(f'{path}: probe error')
    summaries = [row for row in rows if row.get('phase') == 'decode-summary']
    if not summaries:
        raise ValueError(f'{path}: no decode-summary')
    rates = {str(row['c']): float(row['ss_agg_tps_median']) for row in summaries[-1]['summaries']}
    if set(rates) != {'1', '2', '4'} or any(value <= 0 for value in rates.values()):
        raise ValueError(f'{path}: incomplete c1/c2/c4 matrix')
    return rates


def probe_check(paths):
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or any(row.get('error') or row.get('ok') is False for row in rows):
            raise ValueError(f'{path}: empty or failed probe')


def report(root, btag):
    rates = {tag: decode(root / tag / 'dec.jsonl') for tag in ('A1', btag, 'A2')}
    result = {'arms': rates, 'B_vs_A_mean': {}, 'A2_vs_A1': {},
              'note': 'Same-slot throughput only; exactness comes from the served shadow gate.'}
    for c in ('1', '2', '4'):
        control = (rates['A1'][c] + rates['A2'][c]) / 2
        result['B_vs_A_mean'][c] = rates[btag][c] / control
        result['A2_vs_A1'][c] = rates['A2'][c] / rates['A1'][c]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('extra', 'env'):
        p = sub.add_parser(name)
        p.add_argument('--pair', required=True, choices=('100,104', '108,112') + PAIRS + SHADOW_PAIRS)
        p.add_argument('--ring', required=True, type=int, choices=(0, 1))
        p.add_argument('--shadow', required=True, type=int, choices=(0, 1))
        if name == 'extra':
            p.add_argument('--daily', required=True)
        else:
            p.add_argument('inspect', type=Path)
            p.add_argument('--out', type=Path, required=True)
            p.add_argument('--image', required=True)
    p = sub.add_parser('report')
    p.add_argument('root', type=Path)
    p.add_argument('btag')
    p = sub.add_parser('probes')
    p.add_argument('paths', nargs='+', type=Path)
    a = parser.parse_args()
    if a.command == 'extra':
        print(extra(a.daily, a.pair, a.ring, a.shadow))
    elif a.command == 'env':
        result = check_env(json.loads(a.inspect.read_text()), a.pair, a.ring, a.shadow, a.image)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': result['passed'], 'expected': result['expected'], 'errors': result['errors']}))
        raise SystemExit(0 if result['passed'] else 1)
    elif a.command == 'probes':
        probe_check(a.paths)
        print('PASS nonempty probes, no request errors')
    else:
        print(json.dumps(report(a.root, a.btag), indent=2))
