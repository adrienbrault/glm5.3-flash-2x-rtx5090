#!/usr/bin/env python3
"""PyTorch CPU tensors: execute ring/checkpoint and Python-pool code without importing the GPU extension."""
import argparse
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace
import unittest
import os
import json
import struct
try:
    import torch
except ModuleNotFoundError:
    if os.environ.get('RING2_NUMPY_TESTS')!='1':raise
    from numpy_torchshim import torch
    print('CPU NumPy adapter: source lifecycle checks; real PyTorch rerun required in Docker',flush=True)

ROOT=Path(__file__).resolve().parents[2]/'src/exllamav3'

def load_ring():
    tree=ast.parse((ROOT/'cache/mla_index_ring.py').read_text())
    tree.body=[n for n in tree.body if not isinstance(n,(ast.ImportFrom,ast.Import))]
    ns={'PAGE_SIZE':256,'torch':torch,'os':os};exec(compile(tree,'mla_index_ring.py','exec'),ns)
    return ns['MLAIndexRingState']


def method(file,cls,name,ns):
    tree=ast.parse((ROOT/file).read_text())
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    f.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(file),'exec',
                 flags=__future__.annotations.compiler_flag),ns)
    return ns[name]

class StateTests(unittest.TestCase):
    def test_mtp_peer_stash_roundtrips_existing_disk_serializer(self):
        names={'_align','_json_bytes','_enc_key','_dec_key','serialize_stash','deserialize_stash'}
        tree=ast.parse((ROOT/'generator/disk_cache.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        ns=dict(torch=torch,json=json,struct=struct,ALIGNMENT=4096,_CP_ALIGN=64,
                _CP_MAGIC=b'RCP1',_CP_PREFIX=struct.Struct('<4sI'))
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'disk serializer','exec',
                     flags=__future__.annotations.compiler_flag),ns)
        Ring=load_ring();mod=SimpleNamespace(index_kpool=4,idx_plane_dim=256)
        target=Ring(mod,4,1,7);draft=Ring(mod,4,1,8)
        target.alloc('cpu');draft.alloc('cpu');target.peers=(draft,)
        for remainder in range(4):
            pos=512+remainder;ids=torch.arange(pos-remainder,pos)%target.rows
            target.ring[2,ids]=1;draft.ring[2,ids]=2
            stash={'position':pos,'checkpoint_size':target.get_checkpoint_size(),(7,0):target.stash(2,pos)}
            self.assertTrue(all(isinstance(t,torch.Tensor) for t in stash[(7,0)]))
            blob=bytearray(b''.join(ns['serialize_stash'](stash)))
            restored=ns['deserialize_stash'](blob)
            target.clear(2);target.unstash(1,restored[(7,0)],pos)
            self.assertTrue(torch.all(target.ring[1,ids]==1))
            self.assertTrue(torch.all(draft.ring[1,ids]==2))

    def test_mtp_peer_checkpoint_migrates_and_owns_tail(self):
        Ring=load_ring();mod=SimpleNamespace(index_kpool=4,idx_plane_dim=256)
        target=Ring(mod,4,1,7,1);draft=Ring(mod,4,1,8,1)
        target.alloc('cpu');draft.alloc('cpu');target.peers=(draft,)
        self.assertEqual(target.max_chunk_size,2)
        self.assertEqual(target.get_checkpoint_size(),3072)
        for remainder in range(4):
            pos=512+remainder;ids=torch.arange(pos-remainder,pos)%target.rows
            target.ring[2,ids]=1;draft.ring[2,ids]=2
            saved=target.stash(2,pos);target.clear(2)
            target.unstash(1,saved,pos)
            self.assertTrue(torch.equal(target.ring[1,ids],saved[0]))
            self.assertTrue(torch.equal(draft.ring[1,ids],saved[1]))
            target.clear(1)
            self.assertTrue(torch.all(saved[0]==1))
            self.assertTrue(torch.all(saved[1]==2))

    def test_checkpoint_ownership_migration_rewind_and_sizes(self):
        Ring=load_ring();mod=SimpleNamespace(index_kpool=4,idx_plane_dim=256)
        ring=Ring(mod,8,16,7);ring.configure(4096);ring.alloc('cpu')
        self.assertGreaterEqual(ring.rows,4096+16+3)
        self.assertEqual(ring.rows%256,0)
        self.assertEqual(ring.storage_size(),8*ring.rows*256*2)
        self.assertEqual(ring.get_checkpoint_size(),3*256*2)
        for r in (0,1,2,3):
            pos=2*ring.rows+r
            ids=torch.arange(pos-r,pos)%ring.rows
            values=torch.randn((r,256),dtype=torch.half)
            ring.ring[6,ids]=values
            tail=ring.stash(6,pos)
            ring.clear(6)
            ring.unstash(1,tail,pos)
            self.assertTrue(torch.equal(ring.ring[1,ids],values))
            self.assertTrue(torch.equal(tail[0],values))
            ring.clear(1);self.assertTrue(torch.equal(tail[0],values))
        ring.rewind(1,16,16)
        with self.assertRaises(ValueError):ring.rewind(1,16,17)
        with self.assertRaises(RuntimeError):ring.configure(8192)
        ring.free();ring.configure(1024)
        self.assertEqual(ring.ring.device.type,'meta')

    def test_gdn_checkpoint_passes_absolute_position_local_and_tp(self):
        Ring=load_ring();ring=Ring(SimpleNamespace(index_kpool=4,idx_plane_dim=256),2,0,7)
        ring.alloc('cpu');ring.ring[0,:3]=1
        stash=method('modules/gated_delta_net.py','GDNState','stash',{
            '_pm':SimpleNamespace(async_stash=lambda _:None),'new_checkpoint_handle':lambda:42,
            'mp_cache_recurrent_stash':'stash-dispatch'})
        cache=SimpleNamespace(model=SimpleNamespace(loaded_tp=False),get_all_recurrent_layers=lambda:{(1,0):ring})
        state=SimpleNamespace(cache=cache,position=3,slot=0,checkpoint_size=1536)
        got=stash(state);self.assertEqual(got[(1,0)][0].shape,(3,256))
        calls=[];cache.model.loaded_tp=True
        cache.model.tp_dispatch_all=lambda f,args:calls.append((f,args))
        stash(state);self.assertEqual(calls[-1][1][-1],3)
        restore=method('modules/gated_delta_net.py','GDNState','unstash',{
            '_pm':SimpleNamespace(wait_stash=lambda _:None),'mp_cache_recurrent_unstash':'unstash-dispatch'})
        restore(state,dict(position=3,tp_handle=42));self.assertEqual(calls[-1][1][-1],3)

    def test_python_pool_builder_with_nontrivial_slots_and_pages(self):
        fn=method('modules/mla_attn.py','MLAttention','_update_pool_plane',{'torch':torch})
        D,P,PAGE=128,4,256;B,npr=4,16;nr=512
        torch.manual_seed(22)
        bt=torch.randperm(B*npr).view(B,npr).int()
        full=torch.zeros((B*npr,PAGE,2*D),dtype=torch.half)
        ring=torch.zeros((8,nr,2*D),dtype=torch.half)
        slots=(6,2,7,0);starts=(1025,1538,2047,255);length=257
        for b,pos in enumerate(starts):
            ts=torch.arange(pos-pos%P,pos+length)
            physical=bt[b,ts//PAGE].long()*PAGE+ts%PAGE
            vals=torch.randn((len(ts),2*D),dtype=torch.half)
            full.view(-1,2*D)[physical]=vals;ring[slots[b],ts%nr]=vals
        class Layer:
            def __init__(self,plane,pooled,index_ring):self.plane,self.pool,self.index_ring=plane,pooled,index_ring
            def get_idx(self):return self.plane
            def update_pool_direct(self,seq,table,keys):
                ts=torch.arange(keys.shape[1])+seq[0];epp=PAGE//P
                rows=table[0,ts//epp].long()*epp+ts%epp
                self.pool.view(-1,D)[rows]=keys[0]
        pooled=torch.zeros((B*npr,PAGE//P,D),dtype=torch.half)
        a=Layer(full,pooled,None);b=Layer(ring,pooled.clone(),object())
        mod=SimpleNamespace(index_kpool=P,index_head_dim=D,idx_kpool_ape=torch.randn((P,D)))
        fn(mod,a,bt,starts,length);fn(mod,b,bt,starts,length,slots)
        self.assertTrue(torch.equal(a.pool.view(torch.int16),b.pool.view(torch.int16)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT);a,left=p.parse_known_args();ROOT=a.source
    unittest.main(argv=['test_state.py']+left)
