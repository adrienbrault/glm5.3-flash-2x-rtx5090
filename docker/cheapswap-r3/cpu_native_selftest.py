#!/usr/bin/env python3
"""Operator CPU-only native ABI/layout test: registered pointers survive arena rewrites.
Runs inside the image without allocating any CUDA tensors. Local NumPy tests are separate.
"""
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_exchange import band_view

def register(experts,swz):
    args=[]
    for j in range(3):
        for field in ('trellis','suh','svh'): args.append([p[j][field] for p in experts])
    args.extend([[p[j]['bias'] for p in experts] for j in range(3)])
    return ext.exl3_moe_cpu_make_layer(*args,0,0.,int(swz))

def run():
    torch.manual_seed(4107)
    for bits in (2,8):
        for swz in ((False,True) if ext.exl3_moe_cpu_has_avx512_bw() else (False,)):
            # Include more than one band and k tile, or permutation mistakes become identity.
            D=256
            experts=[[dict(trellis=torch.randint(-32768,32767,(16,16,16*bits),dtype=torch.int16),
                suh=torch.ones(D,dtype=torch.half)*.035,svh=torch.ones(D,dtype=torch.half),
                bias=torch.randn(D,dtype=torch.half)*.002) for _ in range(3)] for _ in range(2)]
            band=[[{k:band_view(t,swz).contiguous().clone() if k=='trellis' else t.clone()
                    for k,t in p.items()} for p in pp] for pp in experts]
            arena=[[{k:t.clone() for k,t in p.items()} for p in band[0]]]
            original_ptrs=[t.data_ptr() for p in arena[0] for t in p.values()]
            reference=register(band,swz);live=register(arena,swz)
            try:
                for p,q in zip(arena[0],band[1]):
                    for k,t in p.items(): t.copy_(q[k])
                assert original_ptrs==[t.data_ptr() for p in arena[0] for t in p.values()]
                for rows in (1,4,8):
                    x=torch.randn(rows,D,dtype=torch.half)*.2;w=torch.ones(rows,1,dtype=torch.half)
                    y=torch.empty(rows,D,dtype=torch.float32);expected=torch.empty_like(y)
                    ext.exl3_moe_cpu_forward(reference,x,torch.ones(rows,1,dtype=torch.long),w,expected,2)
                    ext.exl3_moe_cpu_forward(live,x,torch.zeros(rows,1,dtype=torch.long),w,y,2)
                    torch.testing.assert_close(y,expected,rtol=0,atol=0)
            finally:
                ext.exl3_moe_cpu_free_layer(reference);ext.exl3_moe_cpu_free_layer(live)
    print('Native CPU PASS: K2/K8, native/band layout, 1/4/8 rows, fixed arena pointers.')

if __name__=='__main__': run()
