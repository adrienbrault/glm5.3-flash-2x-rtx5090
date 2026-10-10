#!/usr/bin/env python3
"""Run the real R968 shell unit and summary on synthetic CPU-only server tools.
The temporary copy rewrites /srv/qwen5090; fake docker never contacts a daemon.
"""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

OUT = Path(__file__).resolve().parent
BASE = 'tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1'

MOCK = r'''
import argparse, hashlib, json, os, pathlib, sys
ROOT = pathlib.Path(__ROOT__)
STATE = ROOT/'state.json'
TRACE = ROOT/'trace.jsonl'
CONTROL = json.loads((ROOT/'control.json').read_text())
def trace(action, **kw):
    with TRACE.open('a') as f: f.write(json.dumps(dict(action=action, **kw))+'\n')
def save(s): STATE.write_text(json.dumps(s))
def state(): return json.loads(STATE.read_text())
name = pathlib.Path(sys.argv[0]).name
if name == 'sudo':
    a=sys.argv[1:]
    if a[0]=='-n': a=a[1:]
    assert a.pop(0)=='docker', a
    cmd=a.pop(0); trace('docker', command=cmd, args=a)
    if cmd=='build':
        assert '--network=none' in a and '--pull=false' in a, a
    elif cmd=='image': print('[]')
    elif cmd=='rm': pass
    elif cmd=='inspect': print(json.dumps([{'Config':state()['config']}]))
    elif cmd=='logs': print(state()['log'], end='')
    else: raise AssertionError(cmd)
elif name == 'timeout':
    a=sys.argv[1:]
    if a[0]=='-k': a=a[2:]
    a=a[1:]
    os.execvp(a[0], a)
elif name in ('sleep','flock'): pass
elif name == 'launch.py':
    extra=dict(kv.split('=',1) for kv in os.environ.get('EXL3_EXTRA','').split(';') if kv)
    clock=extra.get('EXL3_MOE_CPU_SWAP_TEXT_CLOCK','0'); debug=extra.get('EXL3_MOE_CPU_SWAP_DEBUG','')
    img=os.environ['GLM_IMG']; tag=pathlib.Path(os.environ['RUN_DIR']).parent.name
    trace('boot', tag=tag, image=img, clock=clock, debug=debug)
    e=dict(EXL3_MOE_CPU_SWAP_MODE='exchange', EXL3_MOE_CPU_SWAP_POLICY='histogram', EXL3_MOE_CPU_SWAP_CADENCE='exact', **extra)
    log='[SWAP-OWNER] owner=text component=text modules=46 text_clock=1\n' if clock=='1' else 'daily startup\n'
    if CONTROL.get('fatal')==tag: log+='FATAL synthetic engine failure\n'
    save(dict(config=dict(Image=img, Env=[k+'='+v for k,v in e.items()]), log=log, tag=tag))
elif name == 'glm53_probe.py':
    p=argparse.ArgumentParser()
    for a in ('url','model','out','phase','concurrency','runs','kinds','salt'): p.add_argument('--'+a)
    p.add_argument('--distinct', action='store_true'); a=p.parse_args()
    dest=pathlib.Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    s=state(); tag=s['tag']; clock='textclock1' in s['config']['Image']
    trace('probe', tag=tag, phase=a.phase, out=dest.name, runs=a.runs, distinct=a.distinct,
          flan_power=os.environ.get('FLAN_POWER'), salt=a.salt)
    if CONTROL.get('probe_failure')==tag: sys.exit(7)
    data=[]
    if a.phase=='decode':
        rate=(CONTROL.get('b_c4',92) if clock else 80)
        data=[dict(phase='decode',c=4,run=n,tokens=1024,ss_agg_tps=rate,ss_window_s=10) for n in range(int(a.runs))]
        data.append(dict(phase='decode-summary', summaries=[dict(c=4,distinct=a.distinct,ss_agg_tps_median=rate)]))
        if tag=='Bdebug':
            s['log']+=' -- exchange sweep: 1 swaps wall_ms=1 fence_ms=1\n[SWAP-CLOCK] owner=text component=text owner_tick=0 ticks=64 pending=0 sweeps=1 event=sweep\n'
            save(s)
    elif a.phase=='c1':
        for kind in a.kinds.split(','):
            for n in range(int(a.runs)):
                text='greedy '+kind
                if CONTROL.get('fingerprint_diff')==tag and a.salt: text+=' different'
                digest=hashlib.sha256(('\0'+text).encode()).hexdigest()
                data.append(dict(phase='c1-decode',kind=kind,run=n,tokens={'html':2048}.get(kind,1024),
                                 tps=(CONTROL.get('b_c1',40.2) if clock else 40),content_sha256=digest))
                (dest.parent/f'c1-{kind}-r{n}.events.jsonl').write_text(json.dumps(dict(event=dict(choices=[dict(delta=dict(content=text))])))+'\n')
    else: raise AssertionError(a.phase)
    dest.write_text(''.join(json.dumps(r)+'\n' for r in data))
else: raise AssertionError(name)
'''


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


@contextlib.contextmanager
def server(control=None):
    with tempfile.TemporaryDirectory(prefix='r968-cpu-') as td:
        root=Path(td); srv=root/'srv'; packet=srv/'r968'; lib=srv/'lib'; tools=srv/'glm-daily-tools'; shim=root/'bin'
        for d in (packet, lib, tools, shim): d.mkdir(parents=True)
        for name in ('r968-glm53-textclock.sh', 'glm_arms.sh'):
            (packet/name).write_text((OUT/name).read_text().replace('/srv/qwen5090',str(srv)))
        for name in ('Dockerfile.textclock', 'textclock_summary.py'):
            (packet/name).write_text((OUT/name).read_text())
        (root/'control.json').write_text(json.dumps(control or {}))
        (srv/'glm-daily.env').write_text((OUT.parent/'ref/glm-daily.env').read_text())
        (lib/'gpu-queue.sh').write_text(f'GPU_QUEUE_MARK={root}/queue-mark\n: > "$GPU_QUEUE_MARK"\ngpu_lock(){{ exec 9>{root}/gpu.lock; }}\ngpu_queue_others(){{ :; }}\n')
        (lib/'gateway-drain.sh').write_text('gateway_drain(){ :; }\ngateway_wait_idle(){ :; }\n')
        mock='#!/usr/bin/env python3\n'+MOCK.replace('__ROOT__',repr(str(root)))
        for n in ('sudo','timeout','sleep','flock'): write_executable(shim/n, mock)
        write_executable(tools/'launch.py', mock)
        write_executable(tools/'launch-glm53.sh', f'#!/bin/bash\nexec /usr/bin/env python3 "{tools}/launch.py"\n')
        write_executable(tools/'glm53_probe.py', mock)
        for n in ('glm53_plan.py','mtp_steps.py'): (tools/n).write_text('# synthetic tool\n')
        env=dict(os.environ, PATH=str(shim)+os.pathsep+os.environ['PATH'])
        r=subprocess.run(['bash',str(packet/'r968-glm53-textclock.sh')], env=env, text=True,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
        results=list((srv/'results').iterdir())[0]
        yield r, trace, results


class UnitTests(unittest.TestCase):
    def test_abba_cold_first_debug_isolated_restore_and_gates(self):
        with server() as (r, trace, results):
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            boots=[e for e in trace if e['action']=='boot']
            self.assertEqual([e['tag'] for e in boots[:-1]],['Bdebug','A1','B1','B2','A2'])
            self.assertEqual(boots[-1]['image'],BASE)
            self.assertEqual(boots[-1]['clock'],'0')
            for tag in ('A1','B1','B2','A2'):
                probes=[e for e in trace if e['action']=='probe' and e['tag']==tag]
                self.assertEqual([e['out'] for e in probes],['c4-cold.jsonl','c1.jsonl','c4-after.jsonl','fp.jsonl'])
                self.assertTrue(all(e['flan_power']=='0' for e in probes))
                self.assertTrue(all(e['runs']=='2' for e in probes[:3]))
                self.assertEqual(probes[-1]['salt'],'r968fp')
                self.assertEqual(next(e for e in boots if e['tag']==tag)['debug'],'')
            self.assertTrue(json.loads((results/'gate.json').read_text())['passed'])
            self.assertIn('restore_rc=0',(results/'last.txt').read_text())

    def test_failed_probe_restores_daily(self):
        with server(dict(probe_failure='B1')) as (r, trace, results):
            self.assertNotEqual(r.returncode,0)
            self.assertEqual([e for e in trace if e['action']=='boot'][-1]['image'],BASE)
            self.assertFalse((results/'gate.json').exists())

    def test_each_gate_fails_independently(self):
        for control, gate in ((dict(b_c4=87.9),'cold_c4_gain_at_least_10_percent'),
                              (dict(b_c1=41),'c1_within_2_percent'),
                              (dict(fatal='B1'),'no_FATAL_lines')):
            with self.subTest(gate=gate), server(control) as (r, trace, results):
                self.assertNotEqual(r.returncode,0,r.stdout+r.stderr)
                g=json.loads((results/'gate.json').read_text())
                self.assertFalse(g['gates'][gate])
                self.assertEqual(sum(not x for x in g['gates'].values()),1)
                self.assertEqual([e for e in trace if e['action']=='boot'][-1]['image'],BASE)

    def test_fingerprint_difference_is_information_only(self):
        # REVIEW-r2 B2: greedy output differs across boots under exchange; never a gate.
        with server(dict(fingerprint_diff='B1')) as (r, trace, results):
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            g=json.loads((results/'gate.json').read_text())
            self.assertNotIn('fingerprints_identical',g['gates'])
            self.assertIn('INFO fingerprints_identical=False',r.stdout+r.stderr+(results/'summary.txt').read_text())

    def test_summary_rejects_missing_or_bad_fingerprint(self):
        with server() as (r, trace, results):
            self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            path=results/'B1/c1-code-r0.events.jsonl'
            original=path.read_text()
            for body in ('',json.dumps(dict(event=dict(choices=[dict(delta=dict(content='tampered'))])))+'\n'):
                path.write_text(body)
                check=subprocess.run([sys.executable,str(OUT/'textclock_summary.py'),str(results)],capture_output=True,text=True)
                self.assertEqual(check.returncode,2,check.stdout+check.stderr)
            path.write_text(original)
            (results/'B1/c4-cold.jsonl').unlink()
            check=subprocess.run([sys.executable,str(OUT/'textclock_summary.py'),str(results)],capture_output=True,text=True)
            self.assertEqual(check.returncode,2)


if __name__=='__main__':
    print('CPU-only orchestration fixtures: fake Docker/launcher/probe; no daemon, network or GPU.',flush=True)
    unittest.main(verbosity=2)
