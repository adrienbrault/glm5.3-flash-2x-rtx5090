#!/usr/bin/env python3
"""Actual Python pool/scorer, ragged slots, tiled merge, and private oracle scratch."""
import argparse
import os
os.environ['EXL3_DSA_INDEX_RING']='1'
os.environ['EXL3_DSA_INDEX_RING_SHADOW']='1'
from types import SimpleNamespace, MethodType
import torch
import exllamav3.modules.mla_attn as mla
from exllamav3.cache.mla_index_ring import MLAIndexRingState
from exllamav3.cache.mla_index_shadow import RingShadow, STATS
from exllamav3.modules.attention_fn.mla_triton import mla_plane_append


def run(device):
    P,D,H,B=4,128,16,4
    mla._score_tile=256  # Force the production multi-tile merge algorithm in a small probe.
    mod=SimpleNamespace(index_kpool=P,idx_plane_dim=2*D,index_head_dim=D,index_n_heads=H,
                        index_topk=128,index_kpool_tail=True,key='model.layers.7.self_attn',layer_idx=7)
    ring=MLAIndexRingState(mod,8,1,0,512);ring.alloc(device)
    mod.idx_kpool_ape=torch.randn((P,D),device=device)
    old=torch.zeros((B*32,256,2*D),dtype=torch.half,device=device)
    pool=torch.zeros((B*32,64,D),dtype=torch.half,device=device)
    table=torch.randperm(B*32,device=device).int().view(B,32)
    slots=(6,2,7,0)
    class Layer:
        index_ring=ring;k_idx=old;k_pool=pool
        def get_idx(self):return ring.ring
        def update_pool_direct(self,seq,bt,keys):mla_plane_append(keys,pool,bt,seq)
    layer=Layer();shadow=RingShadow(layer);STATS.active=True
    starts=[0]*B
    for length in (511,512,512,511,7):
        packed=torch.randn((B,length,2*D),dtype=torch.half,device=device)
        seq=torch.tensor(starts,dtype=torch.int32,device=device)
        mla_plane_append(packed,old,table,seq)
        for b,slot in enumerate(slots):
            mla_plane_append(packed[b:b+1],ring.ring[slot:slot+1],table[b:b+1],seq[b:b+1],ring_rows=ring.rows)
        mla.MLAttention._update_pool_plane(mod,layer,table,starts,length,slots)
        mla.MLAttention._update_pool_plane(mod,shadow,table,starts,length)
        shadow.begin(mod,table,starts,length,slots,'dispatch',{})
        shadow.pools(mod,table,starts,length)
        starts=[s+length for s in starts]
    # Replay two tokens at different positions (ragged decode / MTP verify boundary).
    starts=[s-b for b,s in enumerate(starts)]
    length=2;packed=torch.randn((B,length,2*D),dtype=torch.half,device=device)
    seq=torch.tensor(starts,dtype=torch.int32,device=device)
    mla_plane_append(packed,old,table,seq)
    for b,slot in enumerate(slots):
        mla_plane_append(packed[b:b+1],ring.ring[slot:slot+1],table[b:b+1],seq[b:b+1],ring_rows=ring.rows)
    mla.MLAttention._update_pool_plane(mod,layer,table,starts,length,slots)
    mla.MLAttention._update_pool_plane(mod,shadow,table,starts,length)
    shadow.begin(mod,table,starts,length,slots,'dispatch',{'recurrent_history':True})
    shadow.pools(mod,table,starts,length)
    mod._indexer_topk_kpool=MethodType(mla.MLAttention._indexer_topk_kpool,mod)
    for qlen in (1,2,256,300):
        q=torch.randn((B,qlen,H*D),dtype=torch.half,device=device)
        w=torch.randn((B,qlen,H),dtype=torch.half,device=device)
        counts=[0,0]
        def q_forward(*args):counts[0]+=1;return q
        def w_forward(*args):counts[1]+=1;return w
        mod.idx_wq_b=SimpleNamespace(forward=q_forward)
        mod.idx_weights=SimpleNamespace(forward=w_forward)
        image=pool.clone();queries=q.clone();weights=w.clone()
        mod._indexer_topk_kpool(q,{},q,B,qlen,starts,pool_plane=pool,block_table=table,_shadow=shadow)
        assert counts==[1,1],'oracle reprojected inputs'
        assert torch.equal(pool.view(torch.uint8),image.view(torch.uint8)),'oracle wrote served pool'
        assert torch.equal(q.view(torch.uint8),queries.view(torch.uint8))
        assert torch.equal(w.view(torch.uint8),weights.view(torch.uint8))
    torch.cuda.synchronize(device)
    assert STATS.mismatches==0,f'{STATS.mismatches} mismatched bytes'
    assert all(STATS.kinds[k]>0 for k in ('keys','gates','pools','scored_pools','scores','topk'))
    print(f'PASS dispatch/tiled-merge/ragged/read-only device={device}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',required=True);a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA required; no skipped pass')
    with torch.cuda.device(a.device):run(torch.device(a.device))
