#!/usr/bin/env python3
"""Compare actual sampled token IDs, including requeue segments, from gpu_probe.py logs."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');args=p.parse_args()
def read(path):
    requests={}
    for line in Path(path).read_text(errors='replace').splitlines():
        if '[MTP-EXACT] ' not in line: continue
        row=json.loads(line.split('[MTP-EXACT] ',1)[1]);key=(row.get('prompt_key', row['identifier']),row['serial'])
        requests.setdefault(key,[]).extend(row['ids'])
    if not requests:raise SystemExit('FAIL: no actual token-ID records (text re-tokenization is not an exactness check)')
    # Compare per-prompt complete trajectories; UUIDs and c4 enqueue order can vary.
    by_prompt={}
    for (prompt, serial),ids in requests.items():by_prompt.setdefault(prompt,[]).append(ids)
    return {prompt:sorted(runs) for prompt,runs in by_prompt.items()}
ref,candidate=read(args.reference),read(args.candidate)
if ref.keys()!=candidate.keys():raise SystemExit('FAIL: request identifiers/serials differ; use the same fixed workload/order')
total=0
for key in ref:
    if len(ref[key]) != len(candidate[key]):raise SystemExit(f'FAIL: prompt {key}: run counts differ')
    for a,b in zip(ref[key],candidate[key]):
        total+=len(a)
        if a!=b:
            index=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
            raise SystemExit(f'FAIL: prompt {key} first divergence {index}; lengths {len(a)}/{len(b)}')
print(f'PASS: {sum(map(len,ref.values()))} requests, {total} actual sampled token IDs identical')
