#!/usr/bin/env python3
"""CUDA graph + oracle, exact native shapes/strides. Run separately on BOTH devices."""
import argparse
import os
os.environ['EXL3_DSA_INDEX_RING']='1'
os.environ['EXL3_DSA_INDEX_RING_SHADOW']='1'
from types import SimpleNamespace
import torch
from exllamav3.cache.mla_index_ring import MLAIndexRingState
from exllamav3.cache.mla_index_shadow import RingShadow, STATS
from exllamav3.modules.attention_fn.mla_triton import _mla_plane_update_kernel
from exllamav3.modules.attention_fn.dsa_triton import (
    _dsa_pool_update_kernel, _dsa_indexer_fewq_kernel, _dsa_pool_expand_kernel)
from exllamav3.ext import exllamav3_ext as ext


def prepare_history(start, prefix, ring, old, pool, shadow, module, table, seq, slot):
    """Build both histories independently; then restore only the unfinished tail.

    Warmup/capture and previous cases must not become historical scorer inputs.
    Never copy served pools into the oracle: each layout runs its own pool builder.
    """
    P, D = module.index_kpool, module.index_head_dim
    for tensor in (ring.ring, old, pool, shadow.pool):
        tensor.zero_()
    for pos in range(0, start, ring.max_chunk_size):
        length = min(ring.max_chunk_size, start - pos)
        seq.fill_(pos)
        data = prefix[:, pos:pos+length].contiguous()
        for plane, destination, nr in ((old, shadow.pool, 0),
                                       (ring.ring[slot:slot+1], pool, ring.rows)):
            _mla_plane_update_kernel[(length,)](data, plane, table, seq, table.shape[1], length,
                page_size=256, D=2*D, RING_ROWS=nr, num_warps=2, num_stages=2)
            _dsa_pool_update_kernel[(1, length//P+1)](plane, destination, module.idx_kpool_ape,
                table, seq, table.shape[1], length, page_size=256, P=P, D=D,
                MAXPOOLS=length//P+1, RING_ROWS=nr, num_warps=2, num_stages=1)
    # Model a checkpoint/resume: completed pools persist, only raw residue is restored.
    ring.ring[slot].zero_()
    ids = torch.arange(start - start % P, start, device=old.device)
    tail = prefix[0, ids]
    ring.ring[slot, ids % ring.rows] = tail
    seq.fill_(start)


def run(device):
    torch.manual_seed(919)
    P,D,H,PAGE=4,128,16,256
    module=SimpleNamespace(index_kpool=P,idx_plane_dim=2*D,index_head_dim=D,index_n_heads=H,
                           index_topk=128,index_kpool_tail=True,key='model.layers.45.self_attn',layer_idx=0)
    ring=MLAIndexRingState(module,4,1,0,2048);ring.alloc(device)
    old=torch.zeros((16,PAGE,2*D),dtype=torch.half,device=device)
    pool=torch.zeros((16,PAGE//P,D),dtype=torch.half,device=device)
    layer=SimpleNamespace(k_idx=old,k_pool=pool,index_ring=ring)
    shadow=RingShadow(layer);STATS.active=True
    module.idx_kpool_ape=torch.randn((P,D),device=device)
    slot=3;length=2;table=torch.randperm(16,device=device).int().view(1,16)
    seq=torch.tensor([0],dtype=torch.int32,device=device)
    keys=torch.randn((length,D),dtype=torch.half,device=device)
    gates=torch.randn((8,D),dtype=torch.half,device=device)
    query=torch.randn((length,H*D),dtype=torch.half,device=device)
    weights=torch.randn((8,H),dtype=torch.half,device=device)
    scores=torch.full((length,1024),-float('inf'),dtype=torch.half,device=device)
    chosen=torch.empty((length,32),dtype=torch.int32,device=device)
    indices=torch.empty((length,160),dtype=torch.int32,device=device)
    starts=(250,251,252,253,254,255,256,ring.rows-1,ring.rows+1,3003,3004,3005)
    prefix=torch.randn((1,max(starts)+length,2*D),dtype=torch.half,device=device)

    def serve():
        for data,off in ((keys,0),(gates,D)):
            _mla_plane_update_kernel[(length,)](data,ring.ring[slot:slot+1],table,seq,16,length,
                page_size=PAGE,D=D,DST_D=2*D,DST_OFF=off,RING_ROWS=ring.rows,num_warps=2,num_stages=2)
        _dsa_pool_update_kernel[(1,1)](ring.ring[slot:slot+1],pool,module.idx_kpool_ape,table,seq,16,length,
            page_size=PAGE,P=P,D=D,MAXPOOLS=1,RING_ROWS=ring.rows,num_warps=2,num_stages=1)

    def select(start,backend):
        T=(start+length)//P;scores.fill_(-float('inf'))
        _dsa_indexer_fewq_kernel[(length,8)](query,weights,pool.view(-1,D),scores,T,length,start,T,table,0,
            H_i=H,H_pad=H,D_i=D,S_stride=1024,compress_rate=P,scale=D**-0.5*H**-0.5,
            BLOCK_N=128,SEQ=length,MULTIROW=0,EPP=PAGE//P,DEBUG_BOUNDS=0,DEBUG_PAGES=0,
            SHARED_BOUNDS=0,num_warps=8,num_stages=2)
        ext.dsa_topk(scores if backend=='native_eager' else scores[:,:T],chosen,32,None,0)
        _dsa_pool_expand_kernel[(length,1)](chosen,indices,start,P=P,SEL=32,K_pad=160,KP_pool=32,
                                         TAIL=True,SEQ=2,MULTIROW=0,BLOCK=256,num_warps=4,num_stages=1)

    # Warm kernels on a stream created on the requested device, including device 1.
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):serve()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):serve()
    for start in starts:
        for backend in ('native_eager','cuda_graph'):
            prepare_history(start,prefix,ring,old,pool,shadow,module,table,seq,slot)
            if backend=='native_eager':serve()
            else:graph.replay()
            select(start,backend)
            before=[x.clone() for x in (ring.ring,pool,keys,gates,query,weights,scores,chosen,indices)]
            shadow.native(module,(keys,gates,query,weights,scores,indices,chosen),table,seq,[start],length,
                          (slot,),backend,{'recurrent_history':True})
            for value,saved in zip((ring.ring,pool,keys,gates,query,weights,scores,chosen,indices),before):
                assert torch.equal(value.view(torch.uint8),saved.view(torch.uint8)), 'oracle mutated served storage'
    torch.cuda.synchronize(device)
    assert STATS.mismatches==0, f'{STATS.mismatches} shadow mismatches'
    assert all(STATS.kinds[k]>0 for k in ('keys','gates','pools','scored_pools','scores','pool_topk','topk'))
    assert STATS.coverage['native_eager']==STATS.coverage['cuda_graph']==len(starts)
    STATS.summary('graph_probe_clean')
    # Exercise the long-scan split/merge branch with private native workspace on this device.
    long_scores=torch.randn((2,65536),dtype=torch.half,device=device)
    reference=torch.empty((2,512),dtype=torch.int32,device=device)
    private=torch.empty_like(reference);snapshot=long_scores.clone()
    ext.dsa_topk(long_scores,reference,512,None,0)
    ext.dsa_topk_shadow(long_scores,private,512,65536)
    assert torch.equal(reference,private),'private top-k split/merge differs'
    assert torch.equal(long_scores.view(torch.uint8),snapshot.view(torch.uint8))
    # Negative control proves the checker is live, not just counting calls.
    # Corrupt every gate dimension so even fp16 rounding cannot erase the control.
    ring.ring[slot,3005%ring.rows,D:]+=1
    shadow.members(module,table,[3005],length,(slot,))
    assert STATS.mismatches>0,'corrupted gate was not detected'
    print(f'PASS graph/oracle/read-only/negative-control device={device}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA required; no skipped pass')
    with torch.cuda.device(a.device):run(torch.device(a.device))
