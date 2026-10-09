#!/usr/bin/env python3
"""A cumulative mismatch or absent path fails; output similarity is never a gate."""
import argparse
import json
from pathlib import Path
import re

REQUIRED=('prefill','decode','c1','c4','mtp_draft','verify2','rewind','accept',
          'checkpoint_stash','checkpoint_restore','prefix_hit','page_reuse','ragged',
          'mixed_prefill_decode','vision','cuda_graph','dispatch','scoring')
KINDS=('keys','gates','pools','scored_pools','scores','topk')
SUMMARY=re.compile(r'\[RING-SHADOW\] calls=(\d+) compared_rows=(\d+) mismatches=(\d+) layers=(\d+) '
                   r'coverage=(\S+) kinds=(\S+) layer_calls=(\S+) reason=(\S+)')

def check(logs,traffic=None):
    coverage={};kinds={};layers=set();calls=rows=0;finals=[];errors=[]
    for path,text in logs:
        matches=list(SUMMARY.finditer(text))
        if not matches:errors.append(f'{path}: no shadow summaries');continue
        if '[RING-SHADOW-MISMATCH]' in text:errors.append(f'{path}: mismatch logged')
        if re.search(r'Traceback \(most recent call last\)|CUDA error:|illegal memory access|out of memory',text):
            errors.append(f'{path}: runtime error')
        for m in matches:
            c,r,mm,n,cov,ks,ls,reason=m.groups()
            if int(mm):errors.append(f'{path}: {mm} mismatched bytes')
        m=matches[-1];c,r,mm,n,cov,ks,ls,reason=m.groups()
        if reason not in ('idle','shutdown'):errors.append(f'{path}: missing final idle/shutdown summary')
        if not int(c) or not int(r):errors.append(f'{path}: zero comparisons')
        calls+=int(c);rows+=int(r);finals.append(reason)
        for key,value in json.loads(cov).items():coverage[key]=coverage.get(key,0)+value
        for key,value in json.loads(ks).items():kinds[key]=kinds.get(key,0)+value
        layers.update(json.loads(ls))
    missing=[k for k in REQUIRED if coverage.get(k,0)<=0]
    missing_kinds=[k for k in KINDS if kinds.get(k,0)<=0]
    if missing:errors.append('missing served paths: '+','.join(missing))
    if missing_kinds:errors.append('missing comparison kinds: '+','.join(missing_kinds))
    if len(layers)<12:errors.append(f'only {len(layers)} distinct layers; need 11 trunk full indexers + MTP 45')
    if not any('.layers.45.' in layer for layer in layers):errors.append('MTP layer 45 absent')
    if traffic is not None:
        if any(row.get('error') for row in traffic):errors.append('workload request failed')
        tags={row.get('tag') for row in traffic}
        for tag in ('long8192','long16384','long32768','resume','c4','mixed','churn','vision','agentic_tool','agentic_resume'):
            if tag not in tags:errors.append('missing workload '+tag)
    return dict(passed=not errors,calls=calls,compared_rows=rows,coverage=coverage,kinds=kinds,
                layers=sorted(layers),finals=finals,errors=sorted(set(errors)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('logs',nargs='+',type=Path)
    p.add_argument('--traffic',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();traffic=[json.loads(s) for s in a.traffic.read_text().splitlines() if s.strip()]
    result=check([(str(path),path.read_text(errors='replace')) for path in a.logs],traffic)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 1)
