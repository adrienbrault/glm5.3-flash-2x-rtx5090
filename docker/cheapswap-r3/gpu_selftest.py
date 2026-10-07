#!/usr/bin/env python3
"""Random K2 MoE fixture: real CPU kernel + real GPU reconstruct/Hadamard/GEMM.
Forced swaps through production mixin every step, c1/c4/MTP-sized rows, byte equality,
aux mirror integrity, stable addresses, device numeric calibration. No model files needed.
"""
import argparse, os, types, json
os.environ['EXL3_MOE_CPU_SWAP_MODE']='exchange'
os.environ['EXL3_MOE_PINNED_ARENA']='1'
os.environ['EXL3_MOE_CPU_SWAP_INTERVAL']='1'
os.environ['EXL3_MOE_CPU_SWAP_FLOOR']='0'
os.environ['EXL3_MOE_CPU_SWAP_HYST']='1.2'
os.environ['EXL3_MOE_CPU_SWAP_MAX']='2'
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_exchange import band_view, native_view
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU, run_pending_swap_sweeps

class Fixture(BlockSparseMLP_CPU): pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--bits',type=int,choices=range(1,9),default=2);ap.add_argument('--device',default='cuda:0');ap.add_argument('--swizzle',type=int,choices=[0,1],default=1)
    ap.add_argument('--rtol',type=float,default=.03);ap.add_argument('--atol',type=float,default=.003)
    a=ap.parse_args();torch.manual_seed(1979);dev=torch.device(a.device)
    E,N,D=6,3,256; K=a.bits; shape=(D//16,D//16,16*K)
    swz=bool(a.swizzle) and ext.exl3_moe_cpu_has_avx512_bw()
    originals=[]
    for e in range(E):
        originals.append([dict(trellis=torch.randint(-32768,32767,shape,dtype=torch.int16),
            suh=(torch.randint(0,2,(D,))*2-1).half()*.035,
            svh=(torch.randint(0,2,(D,))*2-1).half(),bias=torch.randn(D,dtype=torch.half)*.002)
            for _ in range(3)])
    # Build actual independently aligned arena descriptors, rather than assuming aux offsets.
    cursor=0; plans=[]
    for e in range(N,E):
        projs=[]
        for p in originals[e]:
            ds=[]
            for field in ('trellis','suh','svh','bias'):
                t=p[field];cursor=(cursor+63)&~63
                ds.append((0,cursor,tuple(t.shape),str(t.dtype).split('.')[-1]));cursor+=t.numel()*t.element_size()
            projs.append(ds)
        plans.append(projs)
    arena=torch.empty(cursor,dtype=torch.uint8,pin_memory=True);cpu=[]
    for e,desc in zip(range(N,E),plans):
        pp=[]
        for p,dd in zip(originals[e],desc):
            q={}
            for field,d in zip(('trellis','suh','svh','bias'),dd):
                _,off,sh,dt=d;t=p[field];q[field]=arena[off:off+t.numel()*t.element_size()].view(getattr(torch,dt)).view(sh)
                q[field].copy_(band_view(t,swz) if field=='trellis' else t)
            pp.append(q)
        cpu.append(pp)
    # Native registration preserves these arena pointers through every exchange.
    args=[]
    for j in range(3):
        for field in ('trellis','suh','svh'): args.append([p[j][field] for p in cpu])
    args.extend([[p[j]['bias'] for p in cpu] for j in range(3)])
    handle=ext.exl3_moe_cpu_make_layer(*args,0,0.,int(swz))
    m=Fixture(); m.key='random.moe';m.gated=True;m.device=dev;m.num_experts=E;m.cpu_split_first=N;m.cpu_layer_idx=0
    gpu=[[{k:v.to(dev) for k,v in p.items()} for p in originals[e]] for e in range(N)]
    m.gates=[types.SimpleNamespace(inner=types.SimpleNamespace(**p[0])) for p in gpu]
    m.ups=[types.SimpleNamespace(inner=types.SimpleNamespace(**p[1])) for p in gpu]
    m.downs=[types.SimpleNamespace(inner=types.SimpleNamespace(**p[2])) for p in gpu]
    aux={field+'_'+name:[p[j][field].to(dev) for p in cpu]
         for j,name in enumerate(('g','u','d')) for field in ('suh','svh','bias')}
    m.cpu_host=types.SimpleNamespace(pinned=True,exchange_views=[plans],exchange_swizzled=swz,
                                   arena_views=[arena.view(torch.int16)],aux={0:aux},_dev_bufs={})
    ip=types.SimpleNamespace(moe_cpu_swap_modules=[m],moe_cpu_swap_pending=False)
    # Any checkpoint access in this fixture is a hard failure.
    class NoCheckpoint:
        def __getattr__(self,name): raise AssertionError('checkpoint access: '+name)
    m.config=types.SimpleNamespace(infer_params=ip,stc=NoCheckpoint())
    m._split_map=torch.arange(E,device=dev);m._split_hist=torch.zeros(E,device=dev);m._swap_tick_count=0
    m.batch_recon=types.SimpleNamespace(folded=False,scales={name:(torch.stack([p[j]['suh'] for p in gpu]),
        torch.stack([p[j]['svh'] for p in gpu])) for j,name in enumerate(('g','u','d'))})
    m._exchange_read_streams={};addresses=[t.data_ptr() for pp in gpu for p in pp for t in p.values()]
    def linear(x,p):
        xh=torch.empty_like(x);ext.had_r_128(x,xh,p['suh'],None,1.)
        w=torch.empty((D,D),device=dev,dtype=torch.half)
        ext.reconstruct(w,p['trellis'],K,False,True)
        y=torch.empty_like(x);ext.hgemm(xh,w,y);ext.had_r_128(y,y,None,p['svh'],1.)
        return y+p['bias']
    def expert(x,pp): return linear(torch.nn.functional.silu(linear(x,pp[0]))*linear(x,pp[1]),pp[2])
    original_gpu=[[{k:v.to(dev) for k,v in p.items()} for p in pp] for pp in originals]
    # No-swap CPU baseline has all original experts; also calibrates GPU/CPU numerical difference.
    baseline_cpu=[[{k:(band_view(v,swz).contiguous() if k=='trellis' else v.clone()) for k,v in p.items()} for p in pp] for pp in originals]
    ba=[]
    for j in range(3):
        for f in ('trellis','suh','svh'): ba.append([p[j][f] for p in baseline_cpu])
    ba.extend([[p[j]['bias'] for p in baseline_cpu] for j in range(3)])
    basehandle=ext.exl3_moe_cpu_make_layer(*ba,0,0.,int(swz))
    max_error=0.; total_swaps=0
    try:
        with torch.inference_mode():
            for rows in (1,4,8):  # c1, c4, c4 x draft/verify rows (no autoregressive feedback)
                for step in range(a.steps):
                    mp=m._split_map.cpu();cold=int(torch.where(mp<N)[0][0]);hot=int(torch.where(mp>=N)[0][-1])
                    m._split_hist.zero_();m._split_hist[hot]=100
                    ip.moe_cpu_swap_pending=True
                    old_gpu=[{k:v.clone() for k,v in p.items()} for p in gpu[int(mp[cold])]]
                    source=[{k:(native_view(v,swz).clone() if k=='trellis' else v.clone()) for k,v in p.items()} for p in cpu[int(mp[hot])-N]]
                    run_pending_swap_sweeps(ip); total_swaps+=1
                    mp=m._split_map.cpu();slot=int(mp[hot]);local=int(mp[cold])-N
                    for j in range(3):
                        for f in ('trellis','suh','svh','bias'):
                            assert torch.equal(gpu[slot][j][f].cpu(),source[j][f]),('promotion',rows,step,f)
                            h=native_view(cpu[local][j][f],swz) if f=='trellis' else cpu[local][j][f]
                            assert torch.equal(h,old_gpu[j][f].cpu()),('demotion',rows,step,f)
                            if f!='trellis': assert torch.equal(aux[f+'_'+('g','u','d')[j]][local],old_gpu[j][f])
                            if f in ('suh','svh'):
                                assert torch.equal(m.batch_recon.scales[('g','u','d')[j]][('suh','svh').index(f)][slot],gpu[slot][j][f])
                    assert addresses==[t.data_ptr() for pp in gpu for p in pp for t in p.values()]
                    x=torch.randn(rows,D,dtype=torch.half)*.2
                    ids=torch.stack([torch.randperm(E)[:2] for _ in range(rows)]);wts=torch.full((rows,2),.5,dtype=torch.half)
                    allcpu=torch.empty(rows,D,dtype=torch.float32);ext.exl3_moe_cpu_forward(basehandle,x,ids,wts,allcpu,2)
                    # No-swap mixed-device reference, same routes/input as swapped run.
                    base=torch.zeros(rows,D,device=dev,dtype=torch.float32)
                    allgpu=torch.zeros_like(base)
                    sel=mp[ids];cpuids=(sel-N).clamp_min(-1);out=torch.empty(rows,D)
                    ext.exl3_moe_cpu_forward(handle,x,cpuids,wts,out,2);actual=out.to(dev)
                    # Compute initial GPU-resident contribution + original CPU contribution.
                    baseids=ids.clone();baseids[baseids<N]=-1
                    baseout=torch.empty(rows,D,dtype=torch.float32);ext.exl3_moe_cpu_forward(basehandle,x,baseids,wts,baseout,2);base+=baseout.to(dev)
                    for r in range(rows):
                        for e in ids[r].tolist():
                            v=expert(x[r:r+1].to(dev),original_gpu[e]).float()*.5
                            allgpu[r:r+1]+=v
                            if e<N: base[r:r+1]+=v
                            if int(mp[e])<N: actual[r:r+1]+=expert(x[r:r+1].to(dev),gpu[int(mp[e])]).float()*.5
                    # First establish a numeric envelope against all-original CPU/GPU paths.
                    torch.testing.assert_close(allgpu,allcpu.to(dev),rtol=a.rtol,atol=a.atol)
                    torch.testing.assert_close(actual,base,rtol=a.rtol,atol=a.atol)
                    max_error=max(max_error,float((actual-base).abs().max()))
        print(json.dumps(dict(pass_test=True,swaps=total_swaps,max_abs_error=max_error,swizzle=swz,rows=[1,4,8])))
    finally:
        ext.exl3_moe_cpu_free_layer(handle);ext.exl3_moe_cpu_free_layer(basehandle)

if __name__=='__main__': main()
