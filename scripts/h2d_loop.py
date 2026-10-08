#!/usr/bin/env python3
"""R897: sustained host->device copy from pinned memory, for a fixed time, on the given GPUs (one stream each).
Prints per-second and mean GB/s per device as JSON lines. usage: h2d_loop.py --devices 0,1 --seconds 300 --mib 512"""
import argparse, json, time, torch
ap = argparse.ArgumentParser(); ap.add_argument('--devices', default='0'); ap.add_argument('--seconds', type=float, default=300)
ap.add_argument('--mib', type=int, default=512); a = ap.parse_args()
devs = [int(d) for d in a.devices.split(',')]
n = a.mib << 20
host = {d: torch.empty(n, dtype=torch.uint8).pin_memory() for d in devs}
dst = {d: torch.empty(n, dtype=torch.uint8, device=f'cuda:{d}') for d in devs}
streams = {d: torch.cuda.Stream(device=d) for d in devs}
copied = {d: 0 for d in devs}; t0 = time.time(); last = t0; last_copied = dict(copied)
while time.time() - t0 < a.seconds:
    for d in devs:
        with torch.cuda.stream(streams[d]):
            dst[d].copy_(host[d], non_blocking=True)
    for d in devs:
        streams[d].synchronize(); copied[d] += n
    now = time.time()
    if now - last >= 1.0:
        print(json.dumps({'t': round(now - t0, 1), **{f'gpu{d}_GBps': round((copied[d] - last_copied[d]) / (now - last) / 1e9, 2) for d in devs}}), flush=True)
        last, last_copied = now, dict(copied)
el = time.time() - t0
print(json.dumps({'summary': True, 'seconds': round(el, 1), **{f'gpu{d}_mean_GBps': round(copied[d] / el / 1e9, 2) for d in devs}}), flush=True)
