#!/usr/bin/env python3
"""Weighted phase aggregates. Parent/child and different devices must not be added."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--skip',type=int,default=1,help='skip first aggregate line per mode/batch/depth per log');args=p.parse_args()
for path in args.logs:
    totals={};seen=defaultdict(int)
    for line in Path(path).read_text(errors='replace').splitlines():
        if '[MTP-PHASE] ' not in line:continue
        r=json.loads(line.split('[MTP-PHASE] ',1)[1]);key=(r['mode'],r['batch'],r['depth']);seen[key]+=1
        if seen[key]<=args.skip:continue
        acc=totals.setdefault(key,{'steps':0,'host':defaultdict(float),'cuda':defaultdict(float)})
        acc['steps']+=r['steps']
        for domain in ('host','cuda'):
            for name,v in r[domain].items():acc[domain][name]+=v['ms_per_step']*r['steps']
    for key,acc in totals.items():
        for domain in ('host','cuda'):
            acc[domain]={name:round(ms/acc['steps'],5) for name,ms in sorted(acc[domain].items())}
        print(json.dumps({'file':path,'mode':key[0],'batch':key[1],'depth':key[2],**acc},sort_keys=True))
