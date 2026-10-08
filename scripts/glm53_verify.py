#!/usr/bin/env python3
"""Guards evaluated against actual engine output and Docker inspection."""
import json
import os
import re
import sys
from pathlib import Path

PIN = 'sha256:dfaed2cba56f353a99589549b6ccb971fe137f5e7d11cbeed522f207fb8b8724'

def offload(settings, log):
    s = json.loads(Path(settings).read_text())
    text = Path(log).read_text(errors='replace')
    if re.search(r'CPU (?:offload|split) skipped', text):
        raise ValueError('requested offload was skipped')
    if s['mode'] == 'layers':
        ls = {int(x) for x in re.findall(r'CPU-offloaded experts \(worker\): model\.language_model\.layers\.(\d+)\.mlp', text)}
        expected = set(range(3, 3 + s['n']))
        if ls != expected:
            raise ValueError(f'actual whole-layer registrations {sorted(ls)} != {sorted(expected)}')
    else:
        mode = s.get('placement', 'dynamic')
        rows = re.findall(r'CPU split experts \(worker, ' + mode + r'\): model\.language_model\.layers\.(\d+)\.mlp\s+\[(\d+)\.\.(\d+)\) of (\d+)', text)
        layers = {int(l) for l, *_ in rows}
        expected = set(range(3, 45 + s['draft']))
        # split-by-device-r1 (R915): EXL3_MOE_CPU_SPLIT_BY_DEVICE="N0,N1" in EXL3_EXTRA gives each GPU's layers their own N
        extra = os.environ.get('EXL3_EXTRA', '')
        # dflash-adaptive-r1 (R921): DRAFT_MODEL selects draft_mode model; the main model's MTP layer 45 is loaded only
        # with EXL3_DRAFTER_CHOICE=1, so a DFlash-only boot may register layers 3..44 alone.
        ok_sets = [expected]
        if s.get('draft_model') and 'EXL3_DRAFTER_CHOICE=1' not in extra:
            ok_sets.append(set(range(3, 45)))
        # the engine applies the LAST occurrence (a unit appending its own value to the daily's EXL3_EXTRA, R925)
        ms = re.findall(r'EXL3_MOE_CPU_SPLIT_BY_DEVICE=([0-9,]+)', extra)
        ns = {int(v) for v in ms[-1].split(',') if v} if ms else {s['n']}
        if layers not in ok_sets or any(not (int(b) == int(e) == 288 and 288 - int(a) in ns) for _,a,b,e in rows):
            raise ValueError(f'actual {mode} split registrations {sorted(layers)} != {sorted(expected)} or bad expert interval')
    print('OFFLOAD VERIFIED', s['mode'], s['n'], 'MTP', s['draft'], s.get('placement', 'dynamic'), flush=True)

def daily(path):
    d = json.loads(Path(path).read_text())[0]
    if d['Image'] != PIN:
        raise ValueError(f'restored image {d["Image"]} != {PIN}')
    if any(kv.partition('=')[0] == 'TABBY_OUTPUT_CHUNK_TOKENS' for kv in d['Config'].get('Env') or []):
        raise ValueError('restored daily has TABBY_OUTPUT_CHUNK_TOKENS env')
    if not d['State']['Running'] or d['State'].get('OOMKilled'):
        raise ValueError('restored daily is not running cleanly')
    print('DAILY IMAGE/ENV VERIFIED', PIN, flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'offload': offload(*sys.argv[2:])
    elif sys.argv[1] == 'daily': daily(sys.argv[2])
    else: raise SystemExit('unknown guard')
