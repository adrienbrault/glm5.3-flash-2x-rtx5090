#!/usr/bin/env python3
"""Measure rather than assume D2H, serialized bidirectional exchange and two-link totals."""
import argparse,json,time
import torch
ap=argparse.ArgumentParser();ap.add_argument('--mb',type=float,default=6.33)
ap.add_argument('--iters',type=int,default=256);a=ap.parse_args()
n=int(a.mb*1e6)//2;count=min(2,torch.cuda.device_count());assert count
host=[torch.randn(n,dtype=torch.half,pin_memory=True) for _ in range(count)]
gpu=[torch.empty(n,dtype=torch.half,device=f'cuda:{i}') for i in range(count)]
streams=[torch.cuda.Stream(device=i) for i in range(count)]
results={}
for cards in (1,count):
    for direction in ('h2d','d2h','exchange'):
        for i in range(cards): gpu[i].copy_(host[i])
        for s in streams[:cards]: s.synchronize()
        start=time.perf_counter()
        for _ in range(a.iters):
            for i in range(cards):
                with torch.cuda.stream(streams[i]):
                    if direction in ('h2d','exchange'): gpu[i].copy_(host[i],non_blocking=True)
                    if direction in ('d2h','exchange'): host[i].copy_(gpu[i],non_blocking=True)
        for s in streams[:cards]: s.synchronize()
        seconds=time.perf_counter()-start
        bytes_=a.iters*cards*n*2*(2 if direction=='exchange' else 1)
        results[f'{cards}card_{direction}']=dict(gbs=bytes_/seconds/1e9,ms_per_iteration=seconds*1000/a.iters)
print(json.dumps(results,indent=2))
