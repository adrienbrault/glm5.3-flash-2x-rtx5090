#!/usr/bin/env python3
"""Exercise the actual R929 phase logic with shell mocks; no service/GPU/network calls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

PACKET = Path(__file__).resolve().parents[1]

MOCK_SHELL = r'''
set -uo pipefail
R=$FIXTURE/results
HERE=$FIXTURE/tools
PACKET=$REAL_PACKET
HELPER=$PACKET/r929_helpers.py
IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3
DAILY_IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2
MODEL=fixture
SLOT_START=$SECONDS
CURRENT_TAG=
note(){ printf '%s\n' "$*"; }
sleep(){ :; }
timeout(){ [[ $1 == -k ]] && shift 2; shift; "$@"; }
sudo(){
    [[ $1 == -n ]] && shift
    [[ $1 == docker ]] || return 90
    shift
    case $1 in
        rm|stop) return 0;;
        inspect) cat "$FIXTURE/inspect.json";;
        logs)
            if [[ $CURRENT_TAG == SH ]]; then cat "$FIXTURE/engine.log"; else echo 'healthy fixture'; fi;;
        run)
            [[ $TEST_CASE != kernel_fail ]] || return 1
            echo 'PASS synthetic kernel result';;
        *) return 91;;
    esac
}
nvidia-smi(){
    case "$*" in
        *index,memory.used,memory.free*) echo 'index, memory.used, memory.free'; echo '0, 30000, 500'; echo '1, 30000, 350';;
        *memory.used*) printf '0\n0\n';;
        *memory.free*)
            case $CURRENT_TAG in
                B-97-101) printf '449\n350\n';;
                B-98-102|B-99-103)
                    if [[ $TEST_CASE == all_rejected ]]; then printf '200\n350\n'; else printf '450\n300\n'; fi;;
                *) printf '850\n365\n';;
            esac;;
        *) return 92;;
    esac
}
boot(){
    CURRENT_TAG=$(basename "$(dirname "$1")")
    command python3 "$FIXTURE/boot_mock.py" "$@" || return 1
    [[ $CURRENT_TAG != B-96-100 ]]
}
python3(){
    local -a words=("$@")
    local index=0 out=
    [[ ${words[0]} != -u ]] || index=1
    if [[ ${words[index]} == */serve_workload.py ]]; then
        command python3 "$FIXTURE/traffic_mock.py" "${words[@]:index+1}"
    else
        command python3 "$@"
    fi
}
'''

BOOT_MOCK = r'''
import json, os, sys
from pathlib import Path
root = Path(os.environ['FIXTURE'])
tag = Path(sys.argv[1]).parent.name
values = dict(word.split('=', 1) for word in sys.argv[3:] if '=' in word)
with (root / 'events.jsonl').open('a') as output:
    output.write(json.dumps({'tag': tag, 'values': values}) + '\n')
args = values['EXL3_EXTRA'].split(';')
if os.environ['TEST_CASE'] == 'wrong_env' and tag.startswith('B-'):
    args.append('EXL3_DSA_INDEX_RING=0')
obj = [{'Config': {'Image': values['GLM_IMG'], 'Env': ['EXL3_DSA_INDEX_RING=0',
        'EXL3_DSA_INDEX_RING_SHADOW=0', 'EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104']},
        'Args': ['-u', 'EXL3_DSA_INDEX_RING'] + args + ['python3', 'main.py']}]
(root / 'inspect.json').write_text(json.dumps(obj))
'''

PROBE_MOCK = r'''
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--phase', required=True)
a, other = p.parse_known_args()
if os.environ['TEST_CASE'] == 'request_error' and a.out.parent.name.startswith('B-') and a.phase == 'c1':
    rows = [{'error': 'HTTP fixture failure'}]  # Intentionally exit zero: the unit must reject the row.
elif a.phase == 'decode':
    rows = [{'phase': 'decode-summary', 'summaries':
             [{'c': c, 'ss_agg_tps_median': 80*c} for c in (1,2,4)]}]
else:
    rows = [{'phase': a.phase, 'ok': True}]
a.out.write_text(''.join(json.dumps(row)+'\n' for row in rows))
with (Path(os.environ['FIXTURE']) / 'probes.jsonl').open('a') as f:
    f.write(json.dumps({'tag': a.out.parent.name, 'phase': a.phase, 'args': other})+'\n')
'''

TRAFFIC_MOCK = r'''
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--minutes', type=int, required=True)
p.add_argument('--cache-tokens', type=int, required=True)
a, _ = p.parse_known_args()
assert a.minutes == 15 and a.cache_tokens == 262144
tags = ('long8192','long16384','long32768','resume','c4','mixed','churn','vision','agentic_tool','agentic_resume')
a.out.write_text(''.join(json.dumps({'tag':tag})+'\n' for tag in tags))
'''


class UnitTests(unittest.TestCase):
    def run_unit(self, case):
        import check_shadow
        with tempfile.TemporaryDirectory(prefix='r929-offline-') as temp:
            root = Path(temp)
            (root / 'results').mkdir()
            (root / 'tools').mkdir()
            for name, text in (('boot_mock.py', BOOT_MOCK), ('traffic_mock.py', TRAFFIC_MOCK),
                               ('tools/glm53_probe.py', PROBE_MOCK),
                               ('tools/r860_score.py', "print('fixture c1 score')\n"),
                               ('tools/mtp_steps.py', "print('fixture server step')\n")):
                (root / name).write_text(text)
            encode = lambda obj: json.dumps(obj, separators=(',', ':'))
            layers = {f'model.layers.{i}.self_attn': 1 for i in range(11)}
            layers['model.layers.45.self_attn'] = 1
            mismatch = int(case == 'shadow_fail')
            summary = (f'[RING-SHADOW] calls=123 compared_rows=456 mismatches={mismatch} layers=12 '
                       f'coverage={encode(dict.fromkeys(check_shadow.REQUIRED, 1))} '
                       f'kinds={encode(dict.fromkeys(check_shadow.KINDS, 1))} '
                       f'layer_calls={encode(layers)} reason=shutdown\n')
            (root / 'engine.log').write_text(summary)
            source = (PACKET / 'r929-glm53-ring-splitdev.sh').read_text()
            # Startup/build need the operator host. Execute the actual functions and phase body below them.
            body = source[source.index('\ncap(){'):]
            script = root / 'unit.sh'
            script.write_text(MOCK_SHELL + body)
            result = subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=30,
                                    env={**os.environ, 'FIXTURE': str(root), 'REAL_PACKET': str(PACKET),
                                         'TEST_CASE': case, 'PYTHONDONTWRITEBYTECODE': '1',
                                         'DENV': (PACKET / 'tests/daily.env').read_text().strip()})
            events_path = root / 'events.jsonl'
            events = [json.loads(row) for row in events_path.read_text().splitlines()] if events_path.exists() else []
            probes_path = root / 'probes.jsonl'
            probes = [json.loads(row) for row in probes_path.read_text().splitlines()] if probes_path.exists() else []
            aba = root / 'results/aba.json'
            report = json.loads(aba.read_text()) if aba.exists() else None
            return result, events, probes, report

    def test_selects_first_headroom_pair_then_returns_to_daily(self):
        result, events, probes, report = self.run_unit('success')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([e['tag'] for e in events], ['SH','A1','B-96-100','B-97-101','B-98-102','A2'])
        for event in events:
            self.assertEqual(event['values']['CACHE_TOKENS'], '262144')
            self.assertEqual(event['values']['MAX_SEQ'], '262144')
            self.assertIn('EXL3_MTP_CACHED_REWIND=1', event['values']['EXL3_EXTRA'])
        a1 = next(e['values'] for e in events if e['tag'] == 'A1')
        a2 = next(e['values'] for e in events if e['tag'] == 'A2')
        self.assertEqual(a1, a2)
        self.assertEqual(a1['GLM_IMG'], 'tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2')
        for tag in ('A1','B-98-102','A2'):
            c1 = next(p for p in probes if p['tag'] == tag and p['phase'] == 'c1')
            self.assertIn('code,prose,chat,html,edit', c1['args'])
            self.assertIn('2', c1['args'])
            dec = next(p for p in probes if p['tag'] == tag and p['phase'] == 'decode')
            self.assertIn('1,2,4', dec['args'])
            self.assertIn('--distinct', dec['args'])
        self.assertEqual(report['B_vs_A_mean'], {'1':1.,'2':1.,'4':1.})

    def test_shadow_failure_blocks_aba(self):
        result, events, _, report = self.run_unit('shadow_fail')
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual([e['tag'] for e in events], ['SH'])
        self.assertIsNone(report)

    def test_kernel_failure_blocks_serving(self):
        result, events, _, report = self.run_unit('kernel_fail')
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(events, [])
        self.assertIsNone(report)

    def test_silent_request_error_fails_b_and_still_runs_a2(self):
        result, events, _, report = self.run_unit('request_error')
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertEqual(events[-1]['tag'], 'A2')
        self.assertIsNone(report)

    def test_wrong_effective_env_stops_search_and_still_runs_a2(self):
        result, events, _, report = self.run_unit('wrong_env')
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertEqual([e['tag'] for e in events], ['SH','A1','B-96-100','B-97-101','A2'])
        self.assertIsNone(report)

    def test_exhausts_exact_pair_list_if_headroom_never_passes(self):
        result, events, _, report = self.run_unit('all_rejected')
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertEqual([e['tag'] for e in events],
                         ['SH','A1','B-96-100','B-97-101','B-98-102','B-99-103','A2'])
        self.assertIsNone(report)


if __name__ == '__main__':
    unittest.main()
