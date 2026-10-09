#!/usr/bin/env python3
"""No dependencies. Execute the supplied kernel bodies with a tiny vector/pointer emulator.
Checks address semantics, not Triton compilation or GPU floating-point behavior.
"""
import argparse
import __future__
import contextlib
import io
from types import SimpleNamespace
import ast
import math
import random
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'src/exllamav3'

def rounded(x, code='f'):
    return struct.unpack(code,struct.pack(code,x))[0]

class V:
    def __init__(self, vals): self.v=list(vals)
    def op(self,other,fn):
        rhs=other.v if isinstance(other,V) else [other]*len(self.v)
        return V(fn(x,y) for x,y in zip(self.v,rhs))
    def __add__(self,o): return self.op(o,lambda a,b:a+b)
    __radd__=__add__
    def __sub__(self,o): return self.op(o,lambda a,b:a-b)
    def __mul__(self,o): return self.op(o,lambda a,b:a*b)
    __rmul__=__mul__
    def __truediv__(self,o): return self.op(o,lambda a,b:a/b)
    def to(self,dtype): return V(rounded(x,'e' if dtype=='half' else 'f') for x in self.v)

class Ptr:
    def __init__(self, data,offset=0): self.data,self.offset=data,offset
    def __add__(self,n): return Ptr(self.data,self.offset+n)

class TL:
    constexpr=int;float32='float';float16='half'
    def __init__(self): self.pid=(0,0)
    def program_id(self,i): return self.pid[i]
    def arange(self,start,end): return V(range(start,end))
    def load(self,p):
        return V(p.data[int(i)] for i in p.offset.v) if isinstance(p.offset,V) else p.data[int(p.offset)]
    def store(self,p,values):
        for i,v in zip(p.offset.v,values.v): p.data[int(i)]=v
    def full(self,shape,value,dtype): return V([value]*shape[0])
    def zeros(self,shape,dtype): return self.full(shape,0,dtype)
    def maximum(self,a,b): return a.op(b,max)
    def exp(self,a): return V(math.exp(x) for x in a.v)


def kernel(file,name,tl):
    node=next(n for n in ast.parse((ROOT/file).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    ns={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(file),'exec'),ns)
    return ns[name]

class KernelTests(unittest.TestCase):
    def test_default_off_kernel_bytes_match_current_base(self):
        # The paged branch must preserve the supplied daily kernel output, independently of ring.
        base=ROOT.parents[1]/'base/exllamav3'
        if not base.exists():
            # Docker receives the immutable base kernel bodies as fixtures below.
            base=Path(__file__).resolve().parent/'baseline'
        rng=random.Random(824);tl=TL();D,P,PAGE=8,4,256
        funcs=[]
        for source in (base,ROOT):
            pair=[]
            for file,name in (('modules/attention_fn/mla_triton.py','_mla_plane_update_kernel'),
                              ('modules/attention_fn/dsa_triton.py','_dsa_pool_update_kernel')):
                tree=ast.parse((source/file).read_text())
                node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                node.decorator_list=[];ns={'tl':tl}
                exec(compile(ast.Module(body=[node],type_ignores=[]),'kernel','exec'),ns)
                pair.append(ns[name])
            funcs.append(pair)
        bt=[2,0,3,1];positions=[255];length=17
        planes=[[0.]*(4*PAGE*2*D) for _ in range(2)]
        pools=[[0.]*(4*PAGE//P*D) for _ in range(2)]
        seed=[rounded(rng.uniform(-1,1),'e') for _ in planes[0]]
        for plane in planes:plane[:]=seed
        ape=[rng.uniform(-1,1) for _ in range(P*D)]
        for off in (0,D):
            data=[rounded(rng.uniform(-1,1),'e') for _ in range(length*D)]
            for t in range(length):
                tl.pid=(t,0)
                for j,(app,pool) in enumerate(funcs):
                    app(Ptr(data),Ptr(planes[j]),Ptr(bt),Ptr(positions),4,length,PAGE,D,2*D,off)
        for pi in range(length//P+1):
            tl.pid=(0,pi)
            for j,(app,pool) in enumerate(funcs):
                pool(Ptr(planes[j]),Ptr(pools[j]),Ptr(ape),Ptr(bt),Ptr(positions),4,length,PAGE,P,D,1)
        self.assertEqual(planes[0],planes[1]);self.assertEqual(pools[0],pools[1])

    def test_real_kernel_addressing_and_pool_reads(self):
        rng=random.Random(827);tl=TL()
        app=kernel('modules/attention_fn/mla_triton.py','_mla_plane_update_kernel',tl)
        pool=kernel('modules/attention_fn/dsa_triton.py','_dsa_pool_update_kernel',tl)
        D,P,PAGE=8,4,256
        for slots,chunk in ((1,2048),(4,255),(8,513)):
            nr=-(-(chunk+16+P-1)//PAGE)*PAGE
            pages=128; npr=pages//slots
            bt=list(range(pages));rng.shuffle(bt)
            old=[0.0]*(pages*PAGE*2*D);ring=[0.0]*(slots*nr*2*D)
            old_pool=[0.0]*(pages*(PAGE//P)*D);new_pool=old_pool.copy()
            ape=[rng.uniform(-1,1) for _ in range(P*D)]
            positions=[i%P for i in range(slots)]
            # Seed incomplete residues; completed historical pools need no raw rows.
            for b in range(slots):
                for t in range(positions[b]):
                    phys=bt[b*npr+t//PAGE];row=phys*PAGE+t%PAGE
                    vals=[rounded(rng.uniform(-1,1),'e') for _ in range(2*D)]
                    old[row*2*D:(row+1)*2*D]=vals
                    k=b*nr+t%nr;ring[k*2*D:(k+1)*2*D]=vals
            for step in range(14):
                length=chunk if step%3==0 else (1,2,3,7)[step%4]
                for off in (0,D):
                    data=[rounded(rng.uniform(-1,1),'e') for _ in range(slots*length*D)]
                    for r in range(slots*length):
                        tl.pid=(r,0)
                        for dst,rows in ((old,0),(ring,nr)):
                            app(Ptr(data),Ptr(dst),Ptr(bt),Ptr(positions),npr,length,PAGE,D,2*D,off,rows)
                for b in range(slots):
                    for pi in range(length//P+1):
                        tl.pid=(b,pi)
                        for src,dst,rows in ((old,old_pool,0),(ring,new_pool,nr)):
                            pool(Ptr(src),Ptr(dst),Ptr(ape),Ptr(bt),Ptr(positions),npr,length,PAGE,P,D,1,rows)
                self.assertEqual(old_pool,new_pool,(slots,chunk,step))
                positions=[p+length for p in positions]

    def test_actual_cache_allocations_exclude_paged_index_plane(self):
        ledger=[]
        half=SimpleNamespace(itemsize=2)
        integer=SimpleNamespace(itemsize=4)
        class Tensor:
            def __init__(self,shape,dtype):self.shape,self.dtype,self.device=shape,dtype,'cpu'
            def numel(self):return math.prod(self.shape)
            def element_size(self):return self.dtype.itemsize
        def zeros(shape,dtype,device):
            ledger.append(tuple(shape));return Tensor(tuple(shape),dtype)
        fake_torch=SimpleNamespace(half=half,int=integer,zeros=zeros)
        class CacheLayer:
            def __init__(self,config,attention,cache_id,max_num_tokens):
                self.cache_id,self.max_num_tokens=cache_id,max_num_tokens
        tree=ast.parse((ROOT/'cache/mla.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        for cls in nodes:
            for fn in cls.body:
                if isinstance(fn,ast.FunctionDef) and fn.name=='alloc':
                    # Use a pool allocation spy instead of importing the GPU package.
                    for branch in ast.walk(fn):
                        if isinstance(branch,ast.If):
                            branch.body=[n for n in branch.body if not isinstance(n,ast.ImportFrom)]
        class RingShadow:
            def __init__(self,layer):self.pool=zeros(layer.shape_p,half,'cpu')
        ns={'torch':fake_torch,'CacheLayer':CacheLayer,'PAGE_SIZE':256,
            'np':SimpleNamespace(prod=math.prod),'override':lambda f:f,'RingShadow':RingShadow}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'cache/mla.py','exec',
                     flags=__future__.annotations.compiler_flag),ns)
        for cls_name in ('CacheLayer_MLA_fp16','CacheLayer_MLA_quant'):
            for enabled,shadow in ((False,False),(True,False),(True,True)):
                attn=SimpleNamespace(kv_lora_rank=512,qk_rope_head_dim=64,idx_plane_dim=256,
                                     index_kpool=4,index_head_dim=128,index_ring_enabled=enabled,
                                     index_ring_shadow=shadow,layer_idx=7)
                kwargs={'k_bits':8} if cls_name.endswith('quant') else {}
                cache=ns[cls_name](None,attn,17,524288,**kwargs)
                ring=SimpleNamespace(storage_size=lambda:4*2304*256*2,max_batch_size=4,rows=2304,max_chunk_size=2048)
                if enabled:cache.index_ring=ring
                ledger.clear()
                with contextlib.redirect_stdout(io.StringIO()):cache.alloc('cpu')
                paged=(2048,256,256)
                self.assertEqual(paged in ledger,not enabled or shadow)
                self.assertEqual(cache.k_idx is None,enabled and not shadow)
                self.assertNotIn(ring,cache.get_tensors())
                if shadow:
                    self.assertEqual(ledger.count(cache.shape_p),2)
                    self.assertIn(cache.shadow.pool,cache.get_tensors())
                if enabled:self.assertEqual(cache.index_plane_bytes_saved,268435456)

    def test_retention_bound_and_checkpoint_tail(self):
        rng=random.Random(15)
        for chunk in (1,16,255,256,1024,2048,4096):
            for history in (0,1,16):
                nr=-(-(chunk+history+3)//256)*256
                for _ in range(100):
                    pos=rng.randrange(0,100000)
                    beg=pos-pos%4;end=pos+chunk
                    self.assertEqual(len(set(t%nr for t in range(beg,end))),end-beg)
                    # Any supported rewind's incomplete pool also survives the full window.
                    rw=min(history,chunk-1)
                    target=end-rw
                    needed=range(target-target%4,target)
                    self.assertTrue(all(t>=end-nr for t in needed))
                    tail=list(range(pos-pos%4,pos))
                    restored={t%nr:t for t in tail}
                    self.assertEqual([restored[t%nr] for t in tail],tail)

    def test_depth_one_verify_rejection_and_slot_reassignment(self):
        # Execute both real kernel bodies through the emulator after rejection at all pool tails.
        rng=random.Random(72);tl=TL()
        app=kernel('modules/attention_fn/mla_triton.py','_mla_plane_update_kernel',tl)
        pool=kernel('modules/attention_fn/dsa_triton.py','_dsa_pool_update_kernel',tl)
        D,P,PAGE,nr=8,4,256,256
        bt=[2,0,3,1];old=[0.]*(4*PAGE*2*D);ring=[0.]*(nr*2*D)
        op=[0.]*(4*PAGE//P*D);rp=op.copy();ape=[0.]*(P*D)
        def advance(start,length):
            seq=[start];data=[rounded(rng.uniform(-1,1),'e') for _ in range(length*2*D)]
            for t in range(length):
                tl.pid=(t,0)
                for dst,rows in ((old,0),(ring,nr)):
                    app(Ptr(data),Ptr(dst),Ptr(bt),Ptr(seq),4,length,PAGE,2*D,0,0,rows)
            for pi in range(length//P+1):
                tl.pid=(0,pi)
                for src,dst,rows in ((old,op,0),(ring,rp,nr)):
                    pool(Ptr(src),Ptr(dst),Ptr(ape),Ptr(bt),Ptr(seq),4,length,PAGE,P,D,1,rows)
            self.assertEqual(op,rp,(start,length))
        advance(0,253)
        pos=253
        for _ in range(70):
            advance(pos,2)
            pos+=1  # reject draft: preserve only the first verified row, then overwrite draft
            advance(pos,1);pos+=1
        # Clear/reassign a slot at a page checkpoint; historical raw rows are not required.
        ring[:]=[0.]*len(ring)
        advance(512,7)

    def test_actual_ring_geometry_rewind_and_slots(self):
        tree=ast.parse((ROOT/'cache/mla_index_ring.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
        class T:
            def __init__(self,shape):self.shape=shape
            def numel(self):return math.prod(self.shape)
            def element_size(self):return 2
        ns={'PAGE_SIZE':256,'INDEX_RING_SHADOW':False,
            'torch':SimpleNamespace(half='half',zeros=lambda shape,**kw:T(shape))}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'ring','exec'),ns)
        Ring=ns['MLAIndexRingState']
        for chunk in (1,2,255,256,2048):
            ring=Ring(SimpleNamespace(index_kpool=4,idx_plane_dim=256),4,1,0,chunk)
            self.assertGreaterEqual(ring.max_chunk_size,2)
            self.assertGreaterEqual(ring.rows,max(chunk,2)+1+3)
            ring.rewind(0,1,1);ring.rewind(0,1,0)
            for history,count in ((0,1),(1,2),(1,-1)):
                with self.assertRaises(ValueError):ring.rewind(0,history,count)
        slots=ns['ring_slots']
        self.assertEqual(slots({'recurrent_states':[SimpleNamespace(slot=3),SimpleNamespace(slot=1)]},2),(3,1))
        self.assertEqual(slots({'_index_ring_slots':(2,0)},2),(2,0))
        for params in ({},{'_index_ring_slots':(1,1)},{'_index_ring_slots':(1,)}):
            with self.assertRaises(ValueError):slots(params,2)

    def test_shadow_requires_ring_and_off_does_not_allocate(self):
        tree=ast.parse((ROOT/'cache/mla_index_ring.py').read_text())
        nodes=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef))]
        for ring,shadow in (('0','0'),('1','0'),('1','1'),('0','1')):
            env={'EXL3_DSA_INDEX_RING':ring,'EXL3_DSA_INDEX_RING_SHADOW':shadow}
            ns={'os':SimpleNamespace(environ=env)}
            if ring=='0' and shadow=='1':
                with self.assertRaises(ValueError):exec(compile(ast.Module(body=nodes,type_ignores=[]),'flags','exec'),ns)
            else:
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'flags','exec'),ns)
                self.assertEqual(ns['INDEX_RING'],ring=='1')
                self.assertEqual(ns['INDEX_RING_SHADOW'],shadow=='1')

    def test_cached_gdn_rewind_keeps_descriptor_cache_with_ring(self):
        tree=ast.parse((ROOT/'modules/gated_delta_net.py').read_text())
        f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cached_rewind_jobs')
        class GDN: pass
        class Tensor:
            device='cuda:1';shape=(4,2);stride=lambda self:(2,1);dtype='fp32'
            def data_ptr(self):return id(self)
        g=GDN();g.conv_state=Tensor();g.recurrent_state=Tensor();g.device='cuda:1'
        g.module=SimpleNamespace(conv_kernel_size=4)
        collected=[];rewound=[]
        def collect(*args):collected.append(args);return {1:([],[])}
        ns={'GDNLayerState':GDN,'_collect_rewind_jobs':collect}
        exec(compile(ast.Module(body=[f],type_ignores=[]),'rewind','exec'),ns)
        ring=SimpleNamespace(is_index_ring=True,rewind=lambda *a:rewound.append(a))
        state=SimpleNamespace(slot=2,last_history=1)
        fn=ns['cached_rewind_jobs']
        fn(state,(g,ring),1);fn(state,(g,ring),1)
        self.assertEqual(len(collected),1)
        self.assertEqual(rewound,[(2,1,1)]*2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT);a,remaining=p.parse_known_args();ROOT=a.source
    unittest.main(argv=['test_cpu.py']+remaining)
