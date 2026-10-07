#!/usr/bin/env python3
"""Operator only: extra score-count decay kernel cost, no model or worker.
Reports device event timings AND total host-wall cost; not a worker-contention probe.
"""
import argparse,json,time
import torch
p=argparse.ArgumentParser();p.add_argument('--devices',default='cuda:0,cuda:1');p.add_argument('--iters',type=int,default=1000);p.add_argument('--layers',type=int,default=42);p.add_argument('--half-life',type=float,default=512);a=p.parse_args()
devices=a.devices.split(',');vectors=[torch.ones(288,device=devices[l%len(devices)],dtype=torch.float32) for l in range(a.layers)]
def run(decay):
    for d in devices:torch.cuda.synchronize(d)
    events={d:(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for d in devices}
    for d,(b,e) in events.items():
        with torch.cuda.device(d):b.record()
    started=time.perf_counter()
    for _ in range(a.iters):
        for v in vectors:
            with torch.cuda.device(v.device):
                if decay:v.mul_(2**(-1/a.half_life))
    for d,(b,e) in events.items():
        with torch.cuda.device(d):e.record()
    for _,e in events.values():e.synchronize()
    return {'wall_ms_per_call':1000*(time.perf_counter()-started)/a.iters,
            'device_ms_per_call':{d:b.elapsed_time(e)/a.iters for d,(b,e) in events.items()}}
for _ in range(10):
    for v in vectors:v.mul_(.99)
control=run(False);decay=run(True)
print(json.dumps({'control':control,'decay':decay,'extra_wall_ms_per_call':decay['wall_ms_per_call']-control['wall_ms_per_call'],'warning':'isolated launch cost; full-model no-swap score/static A/B must measure exposure and worker contention'},indent=2))
