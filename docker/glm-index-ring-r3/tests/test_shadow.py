#!/usr/bin/env python3
"""PyTorch CPU: execute oracle comparisons without importing CUDA extension."""
import argparse
import ast
import atexit
import contextlib
from collections import Counter
import json
import io
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
import os
try:
    import torch
except ModuleNotFoundError:
    if os.environ.get('RING2_NUMPY_TESTS')!='1':raise
    from numpy_torchshim import torch
    print('CPU NumPy adapter: source byte-observer checks; real PyTorch rerun required in Docker',flush=True)

ROOT=Path(__file__).resolve().parents[2]/'src/exllamav3'

class ShadowTests(unittest.TestCase):
    def load(self):
        tree=ast.parse((ROOT/'cache/mla_index_shadow.py').read_text())
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom)) and
                   not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                        isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='register')]
        ns=dict(torch=torch,Counter=Counter,json=json,time=time,INDEX_RING_SHADOW=True,PAGE_SIZE=256)
        exec(compile(tree,'shadow','exec'),ns)
        ns['STATS'].active=True
        return ns

    def test_native_observer_selects_layer_device(self):
        ns=self.load();events=[]
        class Guard:
            def __enter__(self):events.append('enter')
            def __exit__(self,*args):events.append('exit')
        def select(device):events.append(device);return Guard()
        ns['torch']=SimpleNamespace(cuda=SimpleNamespace(device=select))
        obj=SimpleNamespace(pool=SimpleNamespace(device='cuda:1'),
                            _native=lambda *args:events.append('native'))
        ns['RingShadow'].native(obj,None,None,None,None,None,None,None,None,None)
        self.assertEqual(events,['cuda:1','enter','native','exit'])

    def test_bitwise_mismatch_signed_zero_and_no_writes(self):
        ns=self.load();layer=SimpleNamespace(k_pool=torch.zeros((1,64,8),dtype=torch.half))
        shadow=ns['RingShadow'](layer)
        a=torch.tensor([[0.0,1.0]],dtype=torch.half);b=a.clone();b[0,0]=-0.0
        before=(a.clone(),b.clone())
        shadow.compare('scores',a,a.clone());self.assertEqual(ns['STATS'].mismatches,0)
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.compare('scores',a,b)
        self.assertEqual(ns['STATS'].mismatches,1)
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual(first['index'],[0,0])
        self.assertEqual(first['byte_in_element'],1)
        self.assertEqual((first['served_byte'],first['oracle_byte']),(0,128))
        self.assertEqual(first['served'],0.0);self.assertEqual(first['oracle'],-0.0)
        self.assertTrue(torch.equal(a.view(torch.int16),before[0].view(torch.int16)))
        self.assertTrue(torch.equal(b.view(torch.int16),before[1].view(torch.int16)))
        with self.assertRaises(RuntimeError):shadow.compare('bad',a,b.flatten())

    def test_corrupted_slot_gate_is_detected_and_oracle_owns_pool(self):
        ns=self.load();D=8
        ring=torch.zeros((4,256,2*D),dtype=torch.half)
        old=torch.zeros((4,256,2*D),dtype=torch.half)
        table=torch.tensor([[2,0],[1,3]],dtype=torch.int32);starts=[255,257];slots=(3,1)
        for b,start in enumerate(starts):
            tok=torch.arange(start//4*4,start+2)
            ids=table[b,tok//256].long()*256+tok%256
            vals=torch.randn((len(tok),2*D),dtype=torch.half)
            old.view(-1,2*D)[ids]=vals;ring[slots[b],tok%256]=vals
        layer=SimpleNamespace(k_idx=old,k_pool=torch.zeros((4,64,D),dtype=torch.half),
                              index_ring=SimpleNamespace(ring=ring,rows=256))
        shadow=ns['RingShadow'](layer);mod=SimpleNamespace(index_kpool=4,index_head_dim=D,
                                                         key='model.layers.7.self_attn',layer_idx=7)
        shadow.begin(mod,table,starts,2,slots,'dispatch',{'recurrent_history':True})
        self.assertEqual(ns['STATS'].mismatches,0)
        self.assertGreater(ns['STATS'].coverage['ragged'],0)
        ring[3,255,D]+=1
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.members(mod,table,starts,2,slots)
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual(first['layer'],'model.layers.7.self_attn')
        self.assertEqual((first['slot'],first['position'],first['row_index']),(3,255,3))
        self.assertEqual((first['ring_row_index'],first['paged_row_index']),(255,767))
        self.assertNotEqual(first['served'],first['oracle'])
        self.assertGreater(ns['STATS'].mismatches,0)
        self.assertNotEqual(shadow.pool.data_ptr(),layer.k_pool.data_ptr())

    def test_scored_history_reports_logical_and_physical_pool_row(self):
        ns=self.load();D=8
        pool=torch.zeros((4,64,D),dtype=torch.half)
        shadow=ns['RingShadow'](SimpleNamespace(k_pool=pool))
        mod=SimpleNamespace(index_kpool=4,index_head_dim=D)
        shadow.context=dict(layer='model.layers.45.self_attn',layer_idx=0,
                            slots=[3],positions=[250],length=2,backend='cuda_graph',pool_size=4)
        table=torch.tensor([[2,0,3,1]],dtype=torch.int32)
        pool[2,0,7]=1
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.scoring_pools(mod,table,[250],2)
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual((first['slot'],first['position'],first['pool_index']),(3,0,0))
        self.assertEqual(first['index'],[0,7])
        self.assertEqual((first['served_row_index'],first['oracle_row_index']),(128,128))
        self.assertEqual((first['served'],first['oracle']),(1.0,0.0))
        self.assertEqual(ns['STATS'].compared_rows,63)
        self.assertEqual(ns['STATS'].mismatches,1)
        with contextlib.redirect_stdout(log):ns['STATS'].summary('idle')
        self.assertIn('mismatch_unit=bytes compared_row_unit=tensor_rows',log.getvalue())

    def test_equal_infinity_padding_does_not_inflate_finite_delta(self):
        ns=self.load();shadow=ns['RingShadow'](SimpleNamespace(k_pool=torch.zeros((1,64,8))))
        served=torch.tensor([[-float('inf'),1.0]],dtype=torch.half)
        oracle=torch.tensor([[-float('inf'),3.0]],dtype=torch.half)
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.compare('scores',served,oracle)
        self.assertIn('max_abs_delta=2.0',log.getvalue())
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual(first['index'],[0,1])
        self.assertEqual((first['served'],first['oracle']),(1.0,3.0))

    def test_ragged_topk_and_tiled_score_positions(self):
        ns=self.load();shadow=ns['RingShadow'](SimpleNamespace(k_pool=torch.zeros((1,64,8))))
        shadow.context=dict(layer='model.layers.7.self_attn',layer_idx=7,
                            slots=[3,1],positions=[255,257],length=2,backend='dispatch',pool_size=4)
        served=torch.zeros((4,2),dtype=torch.int32);oracle=served.clone();served[3,1]=99
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.compare('topk',served,oracle)
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual((first['batch'],first['slot'],first['position'],first['query_row']),(1,1,258,1))
        self.assertEqual(first['index'],[3,1])
        served=torch.zeros((2,4),dtype=torch.half);oracle=served.clone();served[1,2]=1
        log=io.StringIO()
        with contextlib.redirect_stdout(log):shadow.compare('scores',served,oracle,dict(batch=1,row=256,tile=128))
        first=json.loads(log.getvalue().split(' first=')[1])
        self.assertEqual((first['slot'],first['position'],first['pool_index'],first['pool_position']),(1,514,130,520))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT)
    a,left=p.parse_known_args();ROOT=a.source
    unittest.main(argv=['test_shadow.py']+left)
