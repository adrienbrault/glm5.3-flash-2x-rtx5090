#!/usr/bin/env python3
"""Operator: compare the actual R869 serving profile with the reconstructed trace profile."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('profile',type=Path);a=p.parse_args()
base=Path(__file__).resolve().parent/'r869-split-stats.json'
x=json.loads(a.profile.read_text());y=json.loads(base.read_text());out={}
for key,c in y.items():
    if key not in x:out[key]={'missing':True};continue
    hot=lambda v:set(sorted(range(len(v)),key=lambda e:-v[e])[:192])
    out[key]={'expert_count':len(x[key]),'mass':sum(x[key]),'same_gpu_hot_set':hot(x[key])==hot(c),'gpu_hot_set_overlap':len(hot(x[key])&hot(c))}
print(json.dumps({'path':str(a.profile),'sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'layers':out},indent=2))
