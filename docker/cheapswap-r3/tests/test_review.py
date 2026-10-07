"""Regression tests at production sweep, descriptor and cache boundaries; no CUDA needed."""
import contextlib
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from test_cpu import ex, ArrayTensor, ROOT


class IndexedTensor(ArrayTensor):
    def __getitem__(self, i):
        return IndexedTensor(self.a[i])


class SweepTests(unittest.TestCase):
    def fixture(self, failure=None, aborted=False):
        calls=[]
        host=types.SimpleNamespace(v_abort=[int(aborted)])
        class Event:
            def record(self,s): calls.append('commit')
            def synchronize(self):
                if failure=='commit': raise RuntimeError('commit failure')
        fake=types.SimpleNamespace(inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,
                device=lambda _:contextlib.nullcontext(),current_stream=lambda _:None,Event=Event))
        def layer(budget):
            calls.append('changed bytes')
            if failure=='publish': raise RuntimeError('map publish failure')
            return 1
        def reset(): calls.append('reset')
        m=types.SimpleNamespace(cpu_host=host,device=0,_split_sweep_layer=layer,_split_sweep_layer_reset=reset)
        ip=types.SimpleNamespace(moe_cpu_swap_pending=False)
        return fake,m,ip,calls

    def test_publish_and_commit_failures_poison_entire_sweep(self):
        for stage in ('publish','commit'):
            fake,m,ip,calls=self.fixture(stage)
            with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',lambda _:None):
                with self.assertRaisesRegex(RuntimeError,'failure'): ex.run_sweep(ip,[m])
            self.assertTrue(getattr(m.cpu_host,'exchange_failed',False),stage)
            self.assertEqual(m.cpu_host.v_abort[0],1)

    def test_aborted_worker_is_rejected_before_exchange(self):
        fake,m,ip,calls=self.fixture(aborted=True)
        with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',lambda _:None):
            with self.assertRaises(RuntimeError): ex.run_sweep(ip,[m])
        self.assertNotIn('changed bytes',calls)

    def test_abort_during_fence_is_rejected_before_exchange(self):
        fake,m,ip,calls=self.fixture()
        def fence(_): m.cpu_host.v_abort[0]=1
        with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',fence):
            with self.assertRaises(RuntimeError): ex.run_sweep(ip,[m])
        self.assertNotIn('changed bytes',calls)

    def test_poisoned_host_cannot_retry_a_sweep(self):
        fake,m,ip,calls=self.fixture()
        m.cpu_host.exchange_failed=True
        with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',lambda _:None):
            with self.assertRaises(RuntimeError): ex.run_sweep(ip,[m])
        self.assertNotIn('changed bytes',calls)

    def test_capturing_defers_without_reset_or_fence(self):
        fake,m,ip,calls=self.fixture()
        fake.cuda.is_current_stream_capturing=lambda:True
        with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',lambda _:self.fail('capture fence')):
            ex.run_sweep(ip,[m])
        self.assertTrue(ip.moe_cpu_swap_pending)
        self.assertEqual(calls,[])

    def test_success_commits_before_return_without_poison(self):
        fake,m,ip,calls=self.fixture()
        with patch.dict(sys.modules,torch=fake), patch.object(ex,'fence_layer',lambda _:calls.append('fence')):
            ex.run_sweep(ip,[m])
        self.assertEqual(calls,['reset','fence','changed bytes','commit'])
        self.assertFalse(getattr(m.cpu_host,'exchange_failed',False))


class CacheTests(unittest.TestCase):
    def test_unfolded_resident_and_streamed_scale_rows_refresh_in_place(self):
        def table(): return IndexedTensor(np.zeros((3,8),dtype=np.float16))
        resident=types.SimpleNamespace(folded=False,scales={p:(table(),table()) for p in ('u','d')})
        streamed=types.SimpleNamespace(folded=False,scales={p:(table(),table()) for p in ('u','d')})
        folded=types.SimpleNamespace(folded=True,scales={})
        lists=[[types.SimpleNamespace(inner=types.SimpleNamespace(suh=IndexedTensor(np.full(8,10+j)),
                svh=IndexedTensor(np.full(8,20+j)))) for i in range(3)] for j in range(2)]
        aux={f+'_'+p:[IndexedTensor(np.full(8,30+j+(f=='svh'))) for i in range(3)]
                for j,p in enumerate(('u','d')) for f in ('suh','svh')}
        host=types.SimpleNamespace(aux={2:aux},sstate={0:{'recon':{2:streamed}}},
            _dev_bufs={0:{'recon':{2:streamed}},1:{'recon':{2:folded}}})
        m=types.SimpleNamespace(gated=False,ups=lists[0],downs=lists[1],batch_recon=resident,
            cpu_host=host,cpu_layer_idx=2)
        original=[id(t.a) for r in (resident,streamed) for pp in r.scales.values() for t in pp]
        ex.refresh_recon_scales(m,1,2)
        for j,p in enumerate(('u','d')):
            np.testing.assert_array_equal(resident.scales[p][0].a[1],np.full(8,10+j))
            np.testing.assert_array_equal(streamed.scales[p][1].a[2],np.full(8,31+j))
            np.testing.assert_array_equal(resident.scales[p][0].a[0],np.zeros(8))
        self.assertEqual(original,[id(t.a) for r in (resident,streamed) for pp in r.scales.values() for t in pp])


class TransferFailureTests(unittest.TestCase):
    def test_enqueue_failure_drains_side_stream_before_releasing_scratch(self):
        calls=[]
        class BadTensor(ArrayTensor):
            def copy_(self,*a,**k):
                calls.append('write failure');raise RuntimeError('enqueue failure')
        stream=types.SimpleNamespace(synchronize=lambda:calls.append('drained'))
        fake=types.SimpleNamespace(empty_like=lambda t:ArrayTensor(np.empty_like(t.a)),
            cuda=types.SimpleNamespace(stream=lambda _:contextlib.nullcontext()))
        g=BadTensor(np.array([1,2]));h=ArrayTensor(np.array([3,4]))
        with patch.dict(sys.modules,torch=fake):
            with self.assertRaisesRegex(RuntimeError,'enqueue failure'):
                ex.exchange_tensors([(g,h,False,None)],stream)
        self.assertEqual(calls,['write failure','drained'])




class ByteTensor:
    def __init__(self,a): self.a=a
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    def data_ptr(self): return self.a.ctypes.data
    def numel(self): return self.a.size
    def element_size(self): return self.a.itemsize
    def is_contiguous(self): return self.a.flags.c_contiguous
    def view(self,*args):
        if len(args)==1 and isinstance(args[0],np.dtype): return ByteTensor(self.a.view(args[0]))
        if len(args)==1 and isinstance(args[0],tuple): return ByteTensor(self.a.reshape(args[0]))
        return ByteTensor(self.a.reshape(*args))
    def __getitem__(self,i): return ByteTensor(self.a[i])


class DescriptorTests(unittest.TestCase):
    def test_chunk_edges_padding_aux_spill_and_bias_absence(self):
        chunks=[bytearray(256),bytearray(512),bytearray(256)]
        torch=types.SimpleNamespace(uint8=np.dtype('uint8'),int16=np.dtype('int16'),float16=np.dtype('float16'))
        host=types.SimpleNamespace(arena_views=[ByteTensor(np.frombuffer(c,dtype=np.int16)) for c in chunks])
        for ci,off,shape,dtype in ((0,64,(2,16),np.int16),(1,0,(32,),np.float16),
                                  (1,384,(64,),np.float16),(2,0,(16,),np.float16)):
            tensor=ByteTensor(np.frombuffer(chunks[ci],offset=off,count=int(np.prod(shape)),dtype=dtype).reshape(shape))
            tensor.a[:]=np.arange(tensor.numel()).reshape(shape)
            descriptor=ex.arena_descriptor(tensor,chunks)
            self.assertEqual(descriptor,(ci,off,shape,np.dtype(dtype).name))
            with patch.dict(sys.modules,torch=torch): restored=ex.arena_tensor(host,descriptor)
            self.assertEqual(restored.data_ptr(),tensor.data_ptr())
            np.testing.assert_array_equal(restored.a,tensor.a)
        self.assertIsNone(ex.arena_descriptor(None,chunks))
        with self.assertRaises(RuntimeError): ex.arena_descriptor(ByteTensor(np.zeros(16,dtype=np.int16)),chunks)

    def test_bad_descriptors_reject_before_views_are_written(self):
        fake=types.SimpleNamespace(uint8=np.dtype('uint8'),int16=np.dtype('int16'),float16=np.dtype('float16'))
        host=types.SimpleNamespace(arena_views=[ByteTensor(np.zeros(16,dtype=np.int16))])
        for desc in ((-1,0,(2,),'int16'),(1,0,(2,),'int16'),(0,-2,(2,),'int16'),
                     (0,1,(2,),'int16'),(0,30,(2,),'int16'),(0,0,(-1,),'int16'),(0,0,(2,),'float32')):
            with patch.dict(sys.modules,torch=fake):
                with self.assertRaises(RuntimeError): ex.arena_tensor(host,desc)
            self.assertEqual(host.arena_views[0].a.sum(),0)


class OffPathTests(unittest.TestCase):
    def test_entire_legacy_modules_match_after_disabled_flag_specialization(self):
        import ast
        base=ROOT.parents[1]/'exllamav3-full'
        if not base.exists(): self.skipTest('pristine source required for differential test')
        class Disable(ast.NodeTransformer):
            def visit_If(self,node):
                if isinstance(node.test,ast.Name) and node.test.id in ('_exchange_mode','_score_mode'):
                    return [self.visit(n) for n in node.orelse]
                return self.generic_visit(node)
            def visit_IfExp(self,node):
                if isinstance(node.test,ast.Name) and node.test.id in ('_exchange_mode','_score_mode'):
                    return self.visit(node.orelse)
                return self.generic_visit(node)
            def visit_ImportFrom(self,node):
                node.names=[n for n in node.names if n.name!='_score_mode']
                return node
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id in ('_exchange_mode','_score_mode') for t in node.targets): return None
                return self.generic_visit(node)
        for rel in ('model/moe_cpu_host.py','modules/block_sparse_mlp_cpu.py','modules/block_sparse_mlp.py'):
            original=ast.parse((base/rel).read_text())
            specialized=Disable().visit(ast.parse((ROOT/'implementation'/rel).read_text()))
            self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(specialized,include_attributes=False),rel)




class AdditionalFailureTests(unittest.TestCase):
    def test_undrainable_transfer_retains_scratch_and_operands(self):
        class BadTensor(ArrayTensor):
            def copy_(self,*a,**k): raise RuntimeError('original enqueue failure')
        def fail_drain(): raise RuntimeError('CUDA context lost')
        stream=types.SimpleNamespace(synchronize=fail_drain)
        fake=types.SimpleNamespace(empty_like=lambda t:ArrayTensor(np.empty_like(t.a)),
            cuda=types.SimpleNamespace(stream=lambda _:contextlib.nullcontext()))
        g=BadTensor(np.array([1,2]));h=ArrayTensor(np.array([3,4]));pairs=[(g,h,False,None)]
        before=len(ex._failed_transfers)
        try:
            with patch.dict(sys.modules,torch=fake):
                with self.assertRaisesRegex(RuntimeError,'original enqueue failure'):
                    ex.exchange_tensors(pairs,stream)
            saved=ex._failed_transfers[-1]
            self.assertIs(saved[0],stream);self.assertIs(saved[1],pairs)
            self.assertEqual(len(saved[2]),1)
        finally:
            del ex._failed_transfers[before:]

    def test_worker_death_is_not_a_successful_reader_fence(self):
        h=types.SimpleNamespace(proc=types.SimpleNamespace(is_alive=lambda:False))
        with self.assertRaisesRegex(RuntimeError,'died'): ex.require_healthy(h)

    def test_later_layer_failure_poisons_previously_committed_host(self):
        fake,m1,ip,calls=SweepTests().fixture()
        _,m2,_,_=SweepTests().fixture('publish')
        with patch.dict(sys.modules,torch=fake),patch.object(ex,'fence_layer',lambda _:None):
            with self.assertRaises(RuntimeError): ex.run_sweep(ip,[m1,m2])
        self.assertTrue(m1.cpu_host.exchange_failed);self.assertTrue(m2.cpu_host.exchange_failed)


class StoredModelTests(unittest.TestCase):
    def test_every_supplied_grid_price_and_training_choice(self):
        import json
        from test_cpu import sim
        r=json.loads((ROOT/'r2-results.json').read_text());c=r['calibration']
        self.assertEqual(len(r['grid']),501)
        for row in r['grid']:
            for mode in ('today','exchange','overlap'):
                for part in ('train','test'):
                    ms=sim.price(row[part],'exchange' if mode=='overlap' else mode,c['fixed_ms'],
                        c['legacy_effective_ms_per_swap'],mode=='overlap',c['expert_mb'],c['d2h_gbs_assumed'],
                        c['sweep_overhead_ms_assumed'],c['launch_ms_assumed'],c['vram_gbs_assumed'],
                        c['ddr_gbs'],c['exposure'],c['sync_ms_assumed'])
                    self.assertAlmostEqual(1000/ms,row[mode+'_'+part+'_tps'])
        for mode in ('today','exchange','overlap'):
            candidates=[v for v in r['grid'] if mode!='today' or
                (v['policy']['cadence']=='served' and v['policy']['scope']=='global')]
            winner=max(candidates,key=lambda v:v[mode+'_train_tps'])
            self.assertEqual(winner['policy'],r['best_fit_first_half'][mode]['policy'])
        self.assertAlmostEqual(1000/sim.price(r['baseline']['train'],'today',c['fixed_ms'],
            c['legacy_effective_ms_per_swap'],exposure=c['exposure']),48.75)
        static=r['static_train_fit']['train']
        self.assertAlmostEqual(1000/sim.price(static,'exchange',c['fixed_ms'],0,exposure=c['exposure']),63.2)




if __name__=='__main__': unittest.main()
