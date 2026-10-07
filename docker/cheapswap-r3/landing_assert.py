#!/usr/bin/env python3
import argparse, ast, hashlib, importlib.util, json, os
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root');a=ap.parse_args()
packet=Path(__file__).resolve().parent
root=Path(a.root) if a.root else Path(next(iter(importlib.util.find_spec('exllamav3').submodule_search_locations)))
for rel,expected in json.loads((packet/'patched_hashes.json').read_text()).items():
    p=root/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,f'landing content differs: {p}'
    ast.parse(p.read_text(),filename=str(p))
# Import just the standalone helper, so this assertion does not require a GPU.
spec=importlib.util.spec_from_file_location('landing_exchange',root/'model/moe_exchange.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
old=os.environ.pop('EXL3_MOE_CPU_SWAP_MODE',None)
assert not m.enabled(), 'exchange must default off'
os.environ['EXL3_MOE_CPU_SWAP_MODE']='exchange';assert m.enabled()
if old is None: os.environ.pop('EXL3_MOE_CPU_SWAP_MODE',None)
else: os.environ['EXL3_MOE_CPU_SWAP_MODE']=old
spec=importlib.util.spec_from_file_location('landing_score',root/'model/moe_score.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
keys=('EXL3_MOE_CPU_SWAP_POLICY','EXL3_MOE_CPU_SWAP_MAX','EXL3_MOE_CPU_SWAP_INTERVAL',
      'EXL3_MOE_CPU_SWAP_CADENCE','EXL3_MOE_CPU_SWAP_BUDGET_SCOPE','EXL3_MOE_CPU_SCORE_HALF_LIFE',
      'EXL3_MOE_CPU_SCORE_PREFILL','EXL3_MOE_CPU_SCORE_PRIOR','EXL3_MOE_CPU_SCORE_HYST')
saved={k:os.environ.get(k) for k in keys}
try:
    for k in keys:os.environ.pop(k,None)
    assert score.settings() is None, 'score must default off'
    os.environ['EXL3_MOE_CPU_SWAP_POLICY']='score'
    cfg=score.settings()
    assert (cfg['k'],cfg['H'],cfg['wp'],cfg['wb'],cfg['M'],cfg['rho'])==(32,512,32,256,32,2)
    assert score.initial_map([2,0,1])==[1,2,0]
    assert score.select_pairs([0,1,2,3],2,[0,0,2,3],1,2)==[(0,3)]
finally:
    for k,v in saved.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v
print('Landing PASS: exact files, syntax, default opt-in gating, score defaults and inverse profile map; GPU tests remain separate.')
