#!/usr/bin/env python3
"""R969: fail-closed evidence checks and the pre-registered c1-first decision.

No automatic promotion/reversion. A valid REVERT candidate needs operator review.
Exit 0 = KEEP, 1 = REVERT candidate/INCONCLUSIVE, 2 = invalid evidence.
"""
import json
import math
from pathlib import Path
import re
import shlex
import statistics as st
import sys

IMAGE = 'tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1'
CLOCK = 'EXL3_MOE_CPU_SWAP_TEXT_CLOCK'
DEBUG = 'EXL3_MOE_CPU_SWAP_DEBUG'
KINDS = ('code', 'prose', 'chat', 'html', 'edit')
TAGS = ('A1', 'B1', 'B2', 'A2', 'B3', 'A3')
PHASES = ('c1', 'c2', 'c3', 'c4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    require(path.is_file(), f'missing {path}')
    data = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    require(all(isinstance(r, dict) for r in data), f'{path}: non-object row')
    return data


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f'invalid positive number {value!r}')
    return value


def daily(text):
    words = shlex.split(text)
    require(len(text.strip().splitlines()) == 1 and words == text.split(),
            'daily must be the launcher-compatible single whitespace-split line')
    require(words and all('=' in w for w in words), 'invalid daily env')
    env = dict(w.split('=', 1) for w in words)
    require(len(env) == len(words), 'duplicate daily key')
    require(env.get('GLM_IMG') == IMAGE, 'daily is not the served _textclock1 image')
    extra_words = env.get('EXL3_EXTRA', '').split(';')
    require(all(re.fullmatch(r'EXL3_[A-Z0-9_]+=[A-Za-z0-9_./:,=-]*', w)
                for w in extra_words if w), 'invalid daily EXL3_EXTRA')
    extra = dict(w.split('=', 1) for w in extra_words if w)
    require(len(extra) == len([w for w in extra_words if w]), 'duplicate daily EXL3_EXTRA key')
    require(extra.get(CLOCK) == '1', 'daily must serve TEXT_CLOCK=1 via EXL3_EXTRA')
    require(not extra.get(DEBUG, ''), 'served daily DEBUG must be off (empty/absent, not 0)')
    require(CLOCK not in env and DEBUG not in env, 'clock/debug must pass through EXL3_EXTRA')
    return env, extra


def effective_config(container):
    """The launcher uses /usr/bin/env -u KEY KEY=value... in Config.Cmd.

    Apply the R968 fix: inherited Config.Env alone is insufficient. GNU env
    unsets inherited variables before applying assignments; assignment wins.
    """
    c = container['Config']
    require(c['Image'] == IMAGE, 'wrong container image')
    require(c.get('Entrypoint') == ['/usr/bin/env'], 'unexpected container entrypoint')
    env = dict(w.split('=', 1) for w in c.get('Env') or [])
    cmd = list(c.get('Cmd') or [])
    i = 0
    while i < len(cmd) and cmd[i] == '-u':
        require(i + 1 < len(cmd), 'truncated env -u')
        env.pop(cmd[i + 1], None)
        i += 2
    while i < len(cmd) and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', cmd[i]):
        k, v = cmd[i].split('=', 1)
        env[k] = v
        i += 1
    require(cmd[i:i+2] == ['python3', 'main.py'], 'unexpected container command')
    return env, cmd[i:]


def check_container(path, clock, debug):
    data = json.loads(path.read_text())
    require(isinstance(data, list) and len(data) == 1, f'{path}: expected one container')
    c = data[0]
    env, command = effective_config(c)
    require(env.get(CLOCK) == clock, f'{path}: wrong clock')
    require(env.get(DEBUG, '') == debug, f'{path}: wrong DEBUG (0 is enabled)')
    for key, default, value in (('MODE', 'checkpoint', 'exchange'), ('POLICY', 'histogram', 'histogram'),
                                ('CADENCE', 'exact', 'exact')):
        require(env.get('EXL3_MOE_CPU_SWAP_' + key, default) == value, f'{path}: wrong {key}')
    require(env.get('EXL3_MOE_CPU_SWAP') == '1', f'{path}: exchange disabled')
    require(env.get('EXL3_MTP_MAX_BATCH') == '1', f'{path}: MTP batch differs from R968')
    require(env.get('EXL3_MOE_CPU_SWAP_INTERVAL') == '64', f'{path}: wrong interval')
    require(isinstance(c.get('Id'), str) and c['Id'], f'{path}: missing container ID')
    require(isinstance(c.get('Image'), str) and c['Image'], f'{path}: missing image ID')
    comparable = dict(env)
    comparable.pop(CLOCK, None)
    comparable.pop(DEBUG, None)
    return c, (comparable, command)


def c1(path):
    data = rows(path)
    samples = [r for r in data if r.get('phase') == 'c1-decode']
    require(len(samples) == 10 and {(r['kind'], r['run']) for r in samples} ==
            {(k, n) for k in KINDS for n in (0, 1)}, f'{path}: expected ten c1 samples')
    require(all(type(r['run']) is int and r['tokens'] == (2048 if r['kind'] == 'html' else 1024)
                for r in samples), f'{path}: incomplete c1')
    require(not any(r.get('error') for r in data), f'{path}: probe error')
    values = [positive(r['tps']) for r in samples]
    return dict(median=positive(st.median(values)), samples=[dict(kind=r['kind'], run=r['run'], tps=v)
                                               for r, v in zip(samples, values)])


def decode(path, concurrency):
    data = rows(path)
    samples = [r for r in data if r.get('phase') == 'decode']
    require(len(samples) == 2 and {r['run'] for r in samples} == {0, 1}, f'{path}: expected two decode runs')
    require(all(type(r['run']) is int and r['c'] == concurrency and r['tokens'] == 1024
                for r in samples), f'{path}: wrong concurrency/incomplete decode')
    require(not any(r.get('error') for r in data), f'{path}: probe error')
    values = [positive(r['ss_agg_tps']) for r in samples]
    for r in samples:
        positive(r['ss_window_s'])
    summaries = [r for r in data if r.get('phase') == 'decode-summary']
    require(len(summaries) == 1, f'{path}: expected one decode summary')
    summary = summaries[0]['summaries']
    require(len(summary) == 1 and summary[0]['c'] == concurrency and summary[0]['distinct'] is True,
            f'{path}: expected distinct c{concurrency} summary')
    rate = positive(summary[0]['ss_agg_tps_median'])
    require(math.isclose(rate, st.median(values), rel_tol=1e-9), f'{path}: summary/sample disagreement')
    return dict(median=rate, samples=values)


def text_log(path):
    require(path.is_file(), f'missing {path}')
    return path.read_text(errors='replace')


def debug_phase(directory, phase):
    before = text_log(directory / f'engine-before-{phase}.log')
    after = text_log(directory / f'engine-after-{phase}.log')
    require(after.startswith(before), f'{phase}: engine log prefix changed')
    window = after[len(before):]
    require(text_log(directory / f'engine-{phase}-window.log') == window, f'{phase}: wrong debug window')
    sweeps = [s for s in window.splitlines() if ' -- exchange sweep:' in s]
    counters = [s for s in window.splitlines() if '[SWAP-CLOCK]' in s and 'event=sweep' in s]
    require(len(sweeps) > 0 and len(sweeps) == len(counters), f'{phase}: missing/unmatched debug sweeps')
    require(all('component=text' in s for s in counters), f'{phase}: non-text sweep owner')
    totals = []
    for line in sweeps:
        m = re.search(r'exchange sweep: (\d+) swaps wall_ms=([\d.]+) fence_ms=([\d.]+)', line)
        require(m is not None, f'{phase}: malformed sweep line')
        values = tuple(map(float, m.groups()))
        require(all(math.isfinite(v) and v >= 0 for v in values), f'{phase}: invalid sweep metrics')
        totals.append(values)
    return dict(sweeps=len(sweeps), clock_sweeps=len(counters), swaps=int(sum(v[0] for v in totals)),
                wall_ms=sum(v[1] for v in totals), fence_ms=sum(v[2] for v in totals))


def decision(a, b):
    ratio = positive(st.median(b) / st.median(a))
    separated = max(b) < min(a)
    verdict = 'KEEP' if ratio >= 0.985 else ('REVERT candidate' if separated else 'INCONCLUSIVE')
    return ratio, separated, verdict


def report(root):
    # Remove stale successful evidence before validating a rerun.
    gate_path = root / 'gate.json'
    gate_path.unlink(missing_ok=True)
    _, extra = daily((root / 'daily.env').read_text())
    boots = rows(root / 'boots.jsonl')
    require([r.get('tag') for r in boots] == ['Bdebug', *TAGS], 'wrong fresh-boot order')
    order = rows(root / 'phases.jsonl')
    expected = [('Bdebug', p) for p in ('c1', 'c2')] + [(t, p) for t in TAGS for p in PHASES]
    require([(r.get('tag'), r.get('phase')) for r in order] == expected, 'wrong workload order or incomplete phase')
    image_records = json.loads((root / 'image.json').read_text())
    require(len(image_records) == 1 and image_records[0].get('Id'), 'missing pinned image ID')
    image_id = image_records[0]['Id']
    reference = None
    ids = set()
    arms = {}
    for boot in boots:
        tag = boot['tag']
        directory = root / tag
        clock = '0' if tag.startswith('A') else '1'
        debug = '1' if tag == 'Bdebug' else ''
        container, comparable = check_container(directory / 'container.json', clock, debug)
        require(boot.get('container_id') == container['Id'], f'{tag}: boot ID differs')
        require(container['Id'] not in ids, f'{tag}: reused container')
        ids.add(container['Id'])
        require(container['Image'] == image_id, f'{tag}: image ID changed between arms')
        if reference is None:
            reference = comparable
        require(comparable == reference, f'{tag}: container config differs beyond clock/debug')
        env, _ = effective_config(container)
        require(all(env.get(k, '') == v for k, v in extra.items() if k not in (CLOCK, DEBUG)),
                f'{tag}: daily EXL3_EXTRA was not preserved')
        before = text_log(directory / 'engine-boot.log')
        after = text_log(directory / 'engine.log')
        require(after.startswith(before), f'{tag}: boot log prefix changed')
        if clock == '1':
            owners = [s for s in before.splitlines() if '[SWAP-OWNER]' in s]
            require(owners and 'component=text' in owners[-1] and 'text_clock=1' in owners[-1],
                    f'{tag}: missing text startup owner')
        if tag != 'Bdebug':
            require('[SWAP-CLOCK]' not in after and ' -- exchange sweep:' not in after,
                    f'{tag}: DEBUG in measured arm')
        metrics = {'c1': c1(directory / 'c1/samples.jsonl')}
        for p in (('c2',) if tag == 'Bdebug' else PHASES[1:]):
            metrics[p] = decode(directory / p / 'samples.jsonl', int(p[1]))
        if tag != 'Bdebug':
            arms[tag] = metrics
    d = root / 'Bdebug'
    require(text_log(d / 'engine-after-c1.log') == text_log(d / 'engine-before-c2.log'),
            'debug c1/c2 log boundary differs')
    require(text_log(d / 'engine.log') == text_log(d / 'engine-after-c2.log'), 'debug final log differs')
    debug = {p: debug_phase(d, p) for p in ('c1', 'c2')}
    # Scan experiment logs only; copied historical tool logs are not evidence.
    paths = list(root.glob('*.log'))
    for tag in ('Bdebug', *TAGS):
        paths.extend((root / tag).rglob('*.log'))
    if (root / 'summary.txt').is_file():
        paths.append(root / 'summary.txt')
    fatal = [f'{p.relative_to(root)}:{n}' for p in paths
             for n, line in enumerate(text_log(p).splitlines(), 1) if 'FATAL' in line]
    require(not fatal, 'FATAL in experiment logs: ' + ', '.join(fatal))
    aggregate = {side: {p: st.median(arms[t][p]['median'] for t in TAGS if t.startswith(side))
                        for p in PHASES} for side in ('A', 'B')}
    ratios = {p: positive(aggregate['B'][p] / aggregate['A'][p]) for p in PHASES}
    ratio, separated, verdict = decision([arms[t]['c1']['median'] for t in TAGS if t.startswith('A')],
                                         [arms[t]['c1']['median'] for t in TAGS if t.startswith('B')])
    for tag, a in arms.items():
        print(tag + ': ' + ' '.join(f'{p}={a[p]["median"]:.2f}' for p in PHASES) + ' tok/s')
    for p, counts in debug.items():
        print(f'INFO Bdebug {p}: {counts} (excluded from gates)')
    print('Aggregation: median of three boot medians per A/B; order A1 B1 B2 A2 B3 A3; c1 first')
    print(f'A medians={aggregate["A"]}; B medians={aggregate["B"]}')
    print(f'GATE c1 B/A={ratio:.6f} ({(ratio-1)*100:+.2f}%); every B below every A={separated}')
    for p in PHASES[1:]:
        print(f'INFO {p} B/A={ratios[p]:.6f} ({(ratios[p]-1)*100:+.2f}%)')
    print(f'DECISION: {verdict}' + (' (operator decides; no automatic rollback)' if verdict == 'REVERT candidate' else ''))
    gate_path.write_text(json.dumps(dict(valid_evidence=True, decision=verdict, arms=arms, aggregate=aggregate,
                                         ratios=ratios, every_B_below_every_A=separated, debug=debug,
                                         order=list(TAGS), threshold=0.985), indent=2) + '\n')
    return 0 if verdict == 'KEEP' else 1


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if args[0] == '--daily':
            daily(Path(args[1]).read_text())
            return 0
        if args[0] == '--container':
            check_container(Path(args[1]), args[2], args[3])
            return 0
        if args[0] == '--debug-window':
            d, phase = Path(args[1]), args[2]
            before = text_log(d / f'engine-before-{phase}.log')
            after = text_log(d / f'engine-after-{phase}.log')
            require(after.startswith(before), 'container log prefix changed')
            (d / f'engine-{phase}-window.log').write_text(after[len(before):])
            print(f'INFO Bdebug {phase}: {debug_phase(d, phase)}')
            return 0
        return report(Path(args[0]))
    except (ValueError, KeyError, TypeError, OSError, IndexError) as e:
        print(f'INVALID EVIDENCE: {e}; no KEEP/REVERT decision')
        return 2


if __name__ == '__main__':
    sys.exit(main())
