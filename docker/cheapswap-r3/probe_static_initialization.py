#!/usr/bin/env python3
"""Operator-only GLM N96 probe, run in the r3 image with GPUs free.
Loads the real text model, checks every initialized expert's bytes and native map,
then runs pinned streamed CPU-tail prefill on one layer per GPU. No serving traffic.
Requires the actual model and stats mounts and explicit matching loader kwargs.
Not executed on GPU by the reviewer. Numeric checks cover the tail path, not full logits.
"""
import argparse
import json
import os

# These must precede every exllamav3 import (also in the spawned child).
os.environ.update(EXL3_MOE_CPU_SWAP_MODE='exchange', EXL3_MOE_CPU_SWAP='1',
                  EXL3_MOE_CPU_SWAP_POLICY='histogram', EXL3_MOE_PINNED_ARENA='1',
                  EXL3_MOE_CPU_SWAP_CADENCE='exact', EXL3_MOE_CPU_SWAP_INTERVAL='1000000000',
                  EXL3_MOE_STREAM_T='8', EXL3_MOE_STREAM_MIN_ROWS='32')
# Force heavy reconstruction, so folded/unfolded reconstruction is actually exercised.
os.environ.setdefault('EXL3_MOE_STREAM_FUSED_T', '0')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--stats',default='/app/split-stats.json')
    p.add_argument('--load-kwargs',required=True,help='JSON of the operator\'s matching Model.load kwargs, e.g. reserve_per_device/max_chunk_size/max_batch_size')
    p.add_argument('--rows',type=int,default=128)
    p.add_argument('--rtol',type=float,default=.03)
    p.add_argument('--atol',type=float,default=.003)
    a=p.parse_args()
    assert a.rows>=32
    os.environ['EXL3_MOE_CPU_SPLIT_STATS']=a.stats
    os.environ['EXL3_MOE_CPU_SPLIT']='96'
    from exllamav3 import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.moe_exchange import arena_tensor, native_view
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    import torch
    profile=json.load(open(a.stats))
    cfg=Config.from_directory(a.model)
    cfg.infer_params.moe_cpu_split=96
    cfg.infer_params.moe_cpu_threads=8
    model=Model.from_config(cfg)
    candidates={m.key:m for m in model if isinstance(m,BlockSparseMLP)}
    expected={f'model.language_model.layers.{i}.mlp' for i in range(3,45)}
    assert set(candidates)==expected,('target layer keys',set(candidates)^expected)
    original={key:{name:[l.key for l in getattr(m,ls)] for name,ls in
                     (('g','gates'),('u','ups'),('d','downs'))} for key,m in candidates.items()}
    for key in expected:
        counts=profile[key]
        assert len(counts)==288 and all(isinstance(x,(int,float)) and 0<=x<float('inf') for x in counts),(key,'invalid profile')
    kw=json.loads(a.load_kwargs)
    assert not kw.get('tensor_p'), 'probe targets served layer-split placement'
    assert kw.get('max_chunk_size',2048)>=a.rows, 'loader must budget the probe row count'
    stream_layers={}
    stc=cfg.stc
    def source(key,field):
        return stc.get_tensor(key+'.'+field,'cpu',optional=field=='bias',float2half=field!='trellis',no_defer=True)
    def equal(actual,want,label):
        if actual is None or want is None: assert actual is want,label
        else: assert torch.equal(actual.cpu(),want),label
    try:
        with torch.inference_mode():
            model.load(**kw)
            reg=cfg.infer_params.moe_cpu_swap_modules
            assert len(reg)==42 and {m.key for m in reg}==expected,'not every layer registered'
            for m in reg:
                assert m.num_experts==288 and m.cpu_split_first==192,m.key
                counts=profile[m.key];order=sorted(range(288),key=lambda e:-counts[e])
                inverse=torch.empty(288,dtype=torch.long)
                inverse[torch.tensor(order)]=torch.arange(288)
                equal(m._split_map,inverse,(m.key,'inverse map'))
                h=m.cpu_host;li=m.cpu_layer_idx
                assert h.pinned and h.layer_blocks[li] is not None,'not pinned DMA'
                # Router must retain original columns; target router is ordinary fp16.
                rg=m.routing_gate
                assert rg.quant_type=='fp16' and not rg.is_sliced
                original_router=stc.get_tensor(rg.key+'.weight','cpu',transpose=rg.transposed_load,float2half=True,no_defer=True)
                equal(rg.inner.weight,original_router,(m.key,'original router columns'))
                equal(rg.inner.bias,stc.get_tensor(rg.key+'.bias','cpu',optional=True,float2half=True,no_defer=True),(m.key,'router bias'))
                for j,(name,ls) in enumerate((('g',m.gates),('u',m.ups),('d',m.downs))):
                    assert h.specs[li][{'g':'gate_keys','u':'up_keys','d':'down_keys'}[name]]==[original[m.key][name][e] for e in order[192:]]
                    for slot,e in enumerate(order):
                        key=original[m.key][name][e]
                        for fi,field in enumerate(('trellis','suh','svh','bias')):
                            want=source(key,field)
                            if slot<192:
                                equal(getattr(ls[slot].inner,field),want,(m.key,e,name,field,'GPU'))
                            else:
                                local=slot-192;d=h.exchange_views[li][local][j][fi]
                                actual=arena_tensor(h,d) if d is not None else None
                                if field=='trellis':actual=native_view(actual,h.exchange_swizzled)
                                equal(actual,want,(m.key,e,name,field,'worker'))
                                if field!='trellis':
                                    aux=h.aux[li].get(field+'_'+name)
                                    equal(aux[local] if aux is not None else None,want,(m.key,e,name,field,'GPU aux'))
                # Real CUDA map kernel, preserving histogram indexing in router space.
                with torch.cuda.device(m.device):
                    ids=torch.arange(288,device=m.device);hist=torch.zeros(288,device=m.device)
                    tail=torch.empty_like(ids)
                    ext.moe_split_map(ids,m._split_map,hist,tail,192)
                    equal(ids,inverse,(m.key,'native physical IDs'))
                    equal(tail,torch.where(inverse>=192,inverse-192,-1),(m.key,'native tail IDs'))
                    equal(hist,torch.ones(288),(m.key,'router histogram'))
                stream_layers.setdefault(str(m.device),m)
                print(json.dumps({'layer':m.key,'bytes_and_map':'PASS','device':str(m.device),'swizzled':h.exchange_swizzled}),flush=True)
            for device,m in stream_layers.items():
                h=m.cpu_host;li=m.cpu_layer_idx
                order=m._split_initial_order;expert=order[192]
                refs=[{f:source(original[m.key][name][expert],f) for f in ('trellis','suh','svh','bias')} for name in ('g','u','d')]
                args=[]
                for ref in refs:
                    args.extend([[ref[f]] for f in ('trellis','suh','svh')])
                args.extend([[r['bias']] if r['bias'] is not None else [] for r in refs])
                spec=h.specs[li]
                handle=ext.exl3_moe_cpu_make_layer(*args,spec['activation'],spec['act_limit'],0)
                try:
                    with torch.cuda.device(m.device):
                        torch.manual_seed(9107)
                        xh=torch.randn(a.rows,spec['hi'],dtype=torch.half).mul_(.2)
                        x=xh.to(m.device)
                        topk=m.num_experts_per_tok
                        raw=torch.tensor([expert]+order[:topk-1],dtype=torch.long).repeat(a.rows,1)
                        selected=raw.to(m.device)
                        wh=torch.zeros(a.rows,topk,dtype=torch.half);wh[:,0]=1
                        w=wh.to(m.device)
                        h.begin_pass()
                        part,job=m.cpu_split_submit(x,a.rows,selected,w)
                        out=m.cpu_split_combine(torch.zeros(a.rows,spec['ho'],device=m.device),part,job,(a.rows,spec['ho']))
                        local=torch.full((a.rows,topk),-1,dtype=torch.long);local[:,0]=0
                        refout=torch.empty(a.rows,spec['ho'],dtype=torch.float32)
                        ext.exl3_moe_cpu_forward(handle,xh,local,wh,refout,8)
                        torch.testing.assert_close(out.cpu(),refout,rtol=a.rtol,atol=a.atol)
                        equal(selected,m._split_map.cpu()[raw],(m.key,'submission map'))
                        st=h.sstate[m.device.index or 0]
                        assert st['wslot_used'] and any(st['wslot_used']),'streamed path not exercised'
                        # A single selected CPU expert fits one block, so last ring slot is its DMA.
                        ws=(h.next_wslot-1)%h.num_wslots
                        rawslot=st['native_slots'][ws] if st['swz'] else st['vram_slots'][ws]
                        off=0
                        for ref in refs:
                            t=ref['trellis'];n=t.numel()
                            equal(rawslot[off:off+n].view(t.shape),t,(m.key,'GPU streamed native trellis'))
                            off+=n
                        print(json.dumps({'layer':m.key,'streamed_tail_numeric':'PASS','GPU_unswizzle_bytes':'PASS','rows':a.rows,'device':device}),flush=True)
                finally:
                    ext.exl3_moe_cpu_free_layer(handle)
            print('PASS: 42 initialized layers, all GPU/worker/aux expert bytes, native map; streamed tail per GPU. Full-model logits/lifecycle remain separate.',flush=True)
    finally:
        for d in getattr(model,'active_devices',[]):torch.cuda.synchronize(d)
        model.unload()

if __name__=='__main__':main()
