#!/usr/bin/env python3
"""CPU-only tests, no Torch/extension import required. Run from out/ with unittest."""
import importlib.util, sys, unittest, os, types, contextlib, ast
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m
ex=module('exchange',ROOT/'implementation/model/moe_exchange.py')
sim=module('sim_cost',ROOT/'sim_cost.py')

class ArrayTensor:
    def __init__(self,a): self.a=np.asarray(a)
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    def view(self,*shape): return ArrayTensor(self.a.reshape(*shape))
    def permute(self,*axes): return ArrayTensor(self.a.transpose(axes))
    def reshape(self,shape): return ArrayTensor(self.a.reshape(shape))
    def is_contiguous(self): return self.a.flags.c_contiguous
    def copy_(self,other,non_blocking=False): self.a[...] = other.a; return self
    def cpu(self): return self

class LayoutTests(unittest.TestCase):
    def test_native_band_roundtrip_all_quant_widths(self):
        for bits in range(1,9):
            a=np.arange(3*16*16*bits,dtype=np.int16).reshape(3,16,16*bits)
            t=ArrayTensor(a)
            band=ex.band_view(t,True)
            if bits==8: np.testing.assert_array_equal(band.a,a)
            else:
                expected=a.reshape(3,2,8,16*bits).transpose(1,0,2,3).reshape(a.shape)
                np.testing.assert_array_equal(band.a,expected)
            np.testing.assert_array_equal(ex.native_view(band,True).a,a)
            np.testing.assert_array_equal(ex.native_view(t,False).a,a)
    def test_rejects_shape_and_dtype_mismatch(self):
        self.assertFalse(ex.compatible(ArrayTensor(np.zeros(3)),ArrayTensor(np.zeros(4))))
        self.assertFalse(ex.compatible(ArrayTensor(np.zeros(3,dtype='int16')),ArrayTensor(np.zeros(3,dtype='float16'))))
        self.assertFalse(ex.compatible(None,ArrayTensor(np.zeros(3))))
    def test_opt_in_only(self):
        old=os.environ.pop('EXL3_MOE_CPU_SWAP_MODE',None)
        try:
            self.assertFalse(ex.enabled())
            os.environ['EXL3_MOE_CPU_SWAP_MODE']='exchange'; self.assertTrue(ex.enabled())
        finally:
            if old is None: os.environ.pop('EXL3_MOE_CPU_SWAP_MODE',None)
            else: os.environ['EXL3_MOE_CPU_SWAP_MODE']=old

class ExchangeTests(unittest.TestCase):
    def test_real_exchange_routine_with_cpu_tensor_double(self):
        # Tests production exchange ordering/copies without loading CUDA. GPU test uses Torch.
        calls=[]
        class Event:
            def record(self,stream): calls.append('record')
            def synchronize(self): calls.append('sync')
        fake=types.SimpleNamespace(empty_like=lambda t:ArrayTensor(np.empty_like(t.a)),
            equal=lambda a,b:np.array_equal(a.a,b.a),
            cuda=types.SimpleNamespace(stream=lambda _:contextlib.nullcontext(),Event=Event))
        old=sys.modules.get('torch');sys.modules['torch']=fake
        try:
            for bits in (2,8):
                native=np.arange(3*16*16*bits,dtype=np.int16).reshape(3,16,16*bits)
                g=ArrayTensor(native.copy()); hot=ArrayTensor((native+200).copy())
                h=ex.band_view(hot,True); h=ArrayTensor(h.a.copy())
                ga=ArrayTensor(np.arange(16,dtype=np.float16))
                ha=ArrayTensor(np.arange(16,dtype=np.float16)+20)
                mirror=ArrayTensor(ha.a.copy())
                self.assertTrue(ex.exchange_tensors([(g,h,True,None),(ga,ha,False,mirror)],None,True,True))
                np.testing.assert_array_equal(g.a,hot.a)
                np.testing.assert_array_equal(ex.native_view(h,True).a,native)
                np.testing.assert_array_equal(mirror.a,np.arange(16,dtype=np.float16))
                np.testing.assert_array_equal(ha.a,mirror.a)
                self.assertEqual(calls[-1],'sync')
        finally:
            if old is None: sys.modules.pop('torch',None)
            else: sys.modules['torch']=old
    def test_mismatch_is_atomic(self):
        g=ArrayTensor(np.array([1,2]));h=ArrayTensor(np.array([3,4,5]))
        old=sys.modules.get('torch');sys.modules['torch']=types.SimpleNamespace()
        try:
            self.assertFalse(ex.exchange_tensors([(g,h,False,None)],None))
            np.testing.assert_array_equal(g.a,[1,2]);np.testing.assert_array_equal(h.a,[3,4,5])
        finally:
            if old is None: sys.modules.pop('torch',None)
            else: sys.modules['torch']=old
    def test_fences_old_producers_and_real_prefill_state(self):
        def stream(n,d): return types.SimpleNamespace(cuda_stream=n,device=d)
        oldstream,current,prefill,other=stream(1,0),stream(2,0),stream(3,0),stream(4,1)
        recorded=[]; waited=[]
        class Event:
            def record(self,s): self.s=s;recorded.append(s.cuda_stream)
            def synchronize(self): waited.append(self.s.cuda_stream)
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda d:current,
            device=lambda d:contextlib.nullcontext(),Event=Event))
        old=sys.modules.get('torch');sys.modules['torch']=fake
        try:
            m=types.SimpleNamespace(device=0,_exchange_read_streams={1:oldstream},
                cpu_host=types.SimpleNamespace(sstate={0:{'copy_stream':prefill},1:{'copy_stream':other}}))
            ex.fence_layer(m)
            self.assertEqual(recorded,[1,2,3]);self.assertEqual(waited,recorded)
        finally:
            if old is None: sys.modules.pop('torch',None)
            else: sys.modules['torch']=old
    def test_allocation_failure_precedes_writes(self):
        attempts=[]
        def allocate(t):
            attempts.append(1)
            if len(attempts)==2: raise MemoryError('simulated GPU OOM')
            return ArrayTensor(np.empty_like(t.a))
        old=sys.modules.get('torch');sys.modules['torch']=types.SimpleNamespace(empty_like=allocate)
        g=ArrayTensor(np.array([1,2]));h=ArrayTensor(np.array([3,4]))
        try:
            with self.assertRaises(MemoryError): ex.exchange_tensors([(g,h,False,None)],None)
            np.testing.assert_array_equal(g.a,[1,2]);np.testing.assert_array_equal(h.a,[3,4])
        finally:
            if old is None: sys.modules.pop('torch',None)
            else: sys.modules['torch']=old
    def test_legacy_checkpoint_body_preserved(self):
        base=ROOT.parents[1]/'exllamav3-full/modules/block_sparse_mlp_cpu.py'
        if not base.exists(): self.skipTest('exact original tree not bundled inside image')
        def body(path):
            tree=ast.parse(path.read_text())
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_split_swap_experts')
            start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='stc' for t in n.targets))
            return [ast.dump(n,include_attributes=False) for n in fn.body[start:]]
        self.assertEqual(body(base),body(ROOT/'implementation/modules/block_sparse_mlp_cpu.py'))

class ReplayTests(unittest.TestCase):
    def fixture(self):
        return np.tile(np.array([[[3,4],[3,4]]]),(12,1,1)),np.array([True]+[False]*11),np.array([[False]*3+[True]*3]*2)
    def test_no_swap_counts(self):
        ids,new,cpu=self.fixture();r=sim.replay(ids,new,cpu,sim.Policy(budget=0))
        self.assertEqual(r['train']['cpu_share'],1); self.assertEqual(r['test']['swaps'],0)
    def test_exact_cadence_and_layer_budget(self):
        ids,new,cpu=self.fixture();r=sim.replay(ids,new,cpu,sim.Policy(2,0,1.2,1,'layer','exact'))
        self.assertLess(r['test']['cpu_share'],1)
        self.assertLessEqual((r['train']['swaps']+r['test']['swaps'])*6,10)
    def test_heldout_does_not_change_training(self):
        ids,new,cpu=self.fixture(); p=sim.Policy(2,0,1.2,2,'layer','exact')
        r=sim.replay(ids,new,cpu,p); ids[6:]=0
        self.assertEqual(r['train'],sim.replay(ids,new,cpu,p)['train'])
    def test_dma_both_directions_and_contention(self):
        r=dict(cpu_picks=0,swaps=1,sweeps=0)
        expected=6.33/28.8+6.33/28.8+4*6.33/600+.08
        self.assertAlmostEqual(sim.price(r,'exchange',0,50),expected)
        r=dict(cpu_picks=100,swaps=100,sweeps=0)
        self.assertGreaterEqual(sim.price(r,'exchange',0,50,True), (100*6.33+200*6.33)/60)

if __name__=='__main__': unittest.main()
