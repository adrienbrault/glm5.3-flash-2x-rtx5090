#!/usr/bin/env python3
"""Operator-only integration probe: real child, registered shared arena, shared job ring.
Two layers/devices, rotating producer streams, decode/c4/verify-sized rows and pinned
streamed prefill; pending outputs must survive exchanges. Run with free GPUs, no model.
No full-model MTP acceptance/rejection simulation is claimed.
"""
import os
for k,v in {'EXL3_MOE_CPU_SWAP_MODE':'exchange','EXL3_MOE_PINNED_ARENA':'1',
            'EXL3_MOE_CPU_SWAP_INTERVAL':'100000','EXL3_MOE_CPU_SWAP_FLOOR':'0',
            'EXL3_MOE_CPU_SWAP_HYST':'1.2','EXL3_MOE_CPU_SWAP_MAX':'1',
            'EXL3_MOE_CPU_SWAP_BUDGET_SCOPE':'layer','EXL3_MOE_CPU_WSLOT_MB':'1',
            'EXL3_MOE_CPU_THREADS':'2','EXL3_MOE_ARENA_HUGEPAGE':'0',
            'EXL3_MOE_RECON_TILES':'0','EXL3_MOE_STREAM_T':'8'}.items():
    os.environ[k]=v
import argparse
import json
import tempfile
import types
import torch
from safetensors.torch import save_file
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_host import MoeCpuHost
from exllamav3.model.moe_exchange import arena_tensor, native_view
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU, run_pending_swap_sweeps


class Fixture(BlockSparseMLP_CPU): pass


def register_parent_cpu(experts,swz):
    from exllamav3.model.moe_exchange import band_view
    refs=[[{k:(band_view(v,swz).contiguous().clone() if k=='trellis' else v.clone())
            for k,v in p.items()} for p in pp] for pp in experts]
    args=[]
    for j in range(3):
        for f in ('trellis','suh','svh'): args.append([p[j][f] for p in refs])
    args.extend([[],[],[]])
    return ext.exl3_moe_cpu_make_layer(*args,0,0.,int(swz)),refs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--devices',default='cuda:0,cuda:1')
    ap.add_argument('--steps',type=int,default=12);a=ap.parse_args()
    devices=[torch.device(d) for d in a.devices.split(',')]
    assert devices and all(d.type=='cuda' for d in devices)
    torch.manual_seed(8761);E,N,D,K=4,2,256,2
    originals=[];checkpoint={}
    for l in range(2):
        experts=[]
        for e in range(E):
            pp=[]
            for name in ('g','u','d'):
                p=dict(trellis=torch.randint(-32768,32767,(16,16,32),dtype=torch.int16),
                       suh=(torch.randint(0,2,(D,))*2-1).half()*.035,
                       svh=(torch.randint(0,2,(D,))*2-1).half())
                pp.append(p)
                for f,t in p.items(): checkpoint[f'probe.{l}.{e}.{name}.{f}']=t
            experts.append(pp)
        originals.append(experts)
    handles=[];refs=[];host=None;mods=[];streams=[]
    with tempfile.TemporaryDirectory(prefix='exchange-ring-') as directory:
        save_file(checkpoint,directory+'/fixture.safetensors')
        ip=types.SimpleNamespace(moe_cpu_threads=2,moe_cpu_component='text',moe_cpu_swap_modules=[],moe_cpu_swap_pending=False)
        cfg=types.SimpleNamespace(directory=directory,infer_params=ip)
        host=MoeCpuHost(cfg)
        try:
            for l in range(2):
                dev=devices[l%len(devices)]
                with torch.cuda.device(dev):
                    m=Fixture();m.key=f'probe.{l}';m.device=dev;m.gated=True
                    m.num_experts=E;m.cpu_split_first=N;m.config=cfg;m.cpu_host=host
                    gpu=[[{f:t.to(dev) for f,t in p.items()} for p in pp] for pp in originals[l][:N]]
                    m.gates=[types.SimpleNamespace(inner=types.SimpleNamespace(**pp[0])) for pp in gpu]
                    m.ups=[types.SimpleNamespace(inner=types.SimpleNamespace(**pp[1])) for pp in gpu]
                    m.downs=[types.SimpleNamespace(inner=types.SimpleNamespace(**pp[2])) for pp in gpu]
                    aux={f+'_'+name:[originals[l][e][j][f].to(dev) for e in range(N,E)]
                         for j,name in enumerate(('g','u','d')) for f in ('suh','svh')}
                    aux.update({f'bias_{p}':None for p in ('g','u','d')})
                    keys=lambda name:[f'probe.{l}.{e}.{name}' for e in range(N,E)]
                    m.cpu_layer_idx=host.register_layer(m.key,keys('g'),keys('u'),keys('d'),0,0.,D,D,2,
                        proj_dims={p:(D,D,K) for p in ('g','u','d')},aux=aux)
                    m._split_map=torch.arange(E,device=dev);m._split_hist=torch.zeros(E,device=dev)
                    m._split_selcpu_t=None;m._swap_tick_count=0;m.batch_recon=None
                    mods.append(m);ip.moe_cpu_swap_modules.append(m)
                    streams.append([torch.cuda.Stream(device=dev),torch.cuda.Stream(device=dev)])
            host.ensure_started()
            # Real child's swizzle selection, not the parent's requested flag, is authoritative.
            for l in range(2):
                h,r=register_parent_cpu(originals[l],host.exchange_swizzled);handles.append(h);refs.append(r)
            # Folded=False creates copied scale tables during streamed prefill. The resident
            # copied-row case is separately tested in CPU regressions and the standalone fixture.
            def linear(x,p):
                xh=torch.empty_like(x);ext.had_r_128(x,xh,p.suh,None,1.)
                w=torch.empty((D,D),device=x.device,dtype=torch.half);ext.reconstruct(w,p.trellis,K,False,True)
                y=torch.empty_like(x);ext.hgemm(xh,w,y);ext.had_r_128(y,y,None,p.svh,1.)
                return y
            def expert(x,slot,m):
                u=linear(x,m.ups[slot].inner);g=linear(x,m.gates[slot].inner)
                return linear(torch.nn.functional.silu(g.float()).half()*u,m.downs[slot].inner)
            # Graph fixed-address map+trellis readers, replayed on the tracked producer stream.
            graphs=[]
            for m in mods:
                with torch.cuda.device(m.device):
                    capture=torch.cuda.Stream(device=m.device)
                    capture.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(capture):
                        for _ in range(3):
                            fingerprint=torch.stack([ls[0].inner.trellis.long().sum() for ls in (m.gates,m.ups,m.downs)])
                            map_snapshot=m._split_map.clone()
                    capture.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=capture):
                        fingerprint=torch.stack([ls[0].inner.trellis.long().sum() for ls in (m.gates,m.ups,m.downs)])
                        map_snapshot=m._split_map.clone()
                    graphs.append((graph,fingerprint,map_snapshot))
            swaps=0
            with torch.inference_mode():
                for rows in (1,4,8,128):
                    for step in range(a.steps):
                        host.begin_pass();pending=[];snapshots=[]
                        for l,m in enumerate(mods):
                            mp=m._split_map.cpu()
                            cold=int(torch.where(mp<N)[0][0]);hot=int(torch.where(mp>=N)[0][-1])
                            old_tenant=int(torch.where(mp==0)[0][0])
                            expected_oldfp=torch.tensor([originals[l][old_tenant][j]['trellis'].long().sum() for j in range(3)])
                            with torch.cuda.device(m.device),torch.cuda.stream(streams[l][step%2]):
                                # Weight allocations/map commits from the setup stream are ready.
                                torch.cuda.current_stream().wait_stream(torch.cuda.default_stream(m.device))
                                xh=torch.randn(rows,D,dtype=torch.half,pin_memory=True);xh.mul_(.2)
                                x=xh.to(m.device,non_blocking=True)
                                ids_h=torch.tensor([[0,2],[1,3]]).repeat((rows+1)//2,1)[:rows].contiguous().pin_memory()
                                ids=ids_h.to(m.device,non_blocking=True)
                                w=torch.full((rows,2),.5,device=m.device,dtype=torch.half)
                                part,job=m.cpu_split_submit(x,rows,ids,w)
                                out=torch.zeros(rows,D,device=m.device,dtype=torch.float32)
                                for slot in range(N):
                                    mask=(ids==slot).float().sum(dim=1)*.5
                                    out+=expert(x,slot,m).float()*mask[:,None]
                                out=m.cpu_split_combine(out,part,job,(rows,D))
                                graph,fp,ms=graphs[l];graph.replay()
                                pending.append((l,xh,ids_h,out))
                                # These output clones are still enqueued before the sweep fence.
                                snapshots.append((m,fp.clone(),ms.clone(),expected_oldfp,mp.clone()))
                                m._split_hist.zero_();m._split_hist[hot]=100
                        # Both devices/layers share one native worker ring. Leave their collect
                        # work queued and let production fences protect the exchange.
                        ip.moe_cpu_swap_pending=True;run_pending_swap_sweeps(ip);swaps+=len(mods)
                        for l,x,ids,out in pending:
                            expected=torch.empty(rows,D,dtype=torch.float32)
                            ext.exl3_moe_cpu_forward(handles[l],x,ids,torch.full((rows,2),.5,dtype=torch.half),expected,2)
                            torch.testing.assert_close(out.cpu(),expected,rtol=.03,atol=.003)
                        for m,oldfp,oldmap,expected_oldfp,expected_oldmap in snapshots:
                            torch.testing.assert_close(oldfp.cpu(),expected_oldfp,rtol=0,atol=0)
                            torch.testing.assert_close(oldmap.cpu(),expected_oldmap,rtol=0,atol=0)
                            # Replays after publication must read both new map and new slot bytes.
                            with torch.cuda.device(m.device),torch.cuda.stream(streams[mods.index(m)][step%2]):
                                graph,fp,ms=graphs[mods.index(m)];graph.replay()
                                expected=torch.stack([ls[0].inner.trellis.long().sum() for ls in (m.gates,m.ups,m.downs)])
                                torch.testing.assert_close(fp,expected,rtol=0,atol=0)
                                torch.testing.assert_close(ms,m._split_map,rtol=0,atol=0)
                            mp=m._split_map.cpu()
                            for e in range(E):
                                slot=int(mp[e])
                                for j,ls in enumerate((m.gates,m.ups,m.downs)):
                                    for f in ('trellis','suh','svh'):
                                        if slot<N: t=getattr(ls[slot].inner,f).cpu()
                                        else:
                                            d=host.exchange_views[m.cpu_layer_idx][slot-N][j][('trellis','suh','svh','bias').index(f)]
                                            t=arena_tensor(host,d)
                                            if f=='trellis': t=native_view(t,host.exchange_swizzled)
                                        assert torch.equal(t,originals[mods.index(m)][e][j][f]),(m.key,e,f)
            print(json.dumps(dict(pass_test=True,swaps=swaps,rows=[1,4,8,128],devices=[str(d) for d in devices],
                folded=os.environ.get('EXL3_MOE_RECON_FOLDED','1'),fused=os.environ.get('EXL3_MOE_SPLIT_FUSED','1'))))
        finally:
            for dev in devices:
                with torch.cuda.device(dev): torch.cuda.synchronize()
            if host is not None: host.shutdown()
            for h in handles: ext.exl3_moe_cpu_free_layer(h)


if __name__=='__main__': main()
