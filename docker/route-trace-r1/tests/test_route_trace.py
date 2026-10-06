import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'src/exllamav3/route_trace.py'

def module(enabled=True):
    with patch.dict(os.environ,{'EXL3_ROUTE_TRACE':'test' if enabled else ''}):
        spec=importlib.util.spec_from_file_location('trace_under_test',MODULE)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

class Tests(unittest.TestCase):
    def test_off_has_no_numpy_torch_or_worker(self):
        code=f"import runpy,sys; m=runpy.run_path({str(MODULE)!r}); assert not m['ENABLED']; assert m['_writer'] is None; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules; m['record'](None,None,None,None); m['begin'](None,None)"
        env=dict(os.environ,EXL3_ROUTE_TRACE='')
        subprocess.run([sys.executable,'-c',code],env=env,check=True)

    def test_off_ast_preserves_every_original_statement(self):
        class Strip(ast.NodeTransformer):
            def visit_ImportFrom(self,node):
                return None if node.module=='exllamav3.route_trace' else node
            def visit_If(self,node):
                if isinstance(node.test,ast.Name) and node.test.id=='_route_trace_enabled': return None
                return self.generic_visit(node)
        for base in (ROOT/'base').rglob('*.py'):
            src=ROOT/'src'/base.relative_to(ROOT/'base')
            a=ast.dump(ast.parse(base.read_text()),include_attributes=False)
            b=ast.dump(Strip().visit(ast.parse(src.read_text())),include_attributes=False)
            self.assertEqual(a,b,src)

    def test_row_positions_requests_and_unknown(self):
        m=module()
        ctx=dict(batch=2,q=3,keys=('a','b'),starts=np.array([4,10]))
        pos,keys,rows=m.metadata(ctx,6)
        self.assertEqual(pos.tolist(),[4,5,6,10,11,12])
        self.assertEqual(keys.tolist(),['a','a','a','b','b','b'])
        self.assertEqual(rows.tolist(),[0,0,0,1,1,1])
        ctx['starts']=None
        self.assertEqual(m.metadata(ctx,6)[0].tolist(),[-1]*6)
        with self.assertRaises(ValueError): m.metadata(ctx,5)

    def test_begin_snapshots_cpu_starts_and_labels_calls(self):
        from types import SimpleNamespace
        class Starts:
            device=SimpleNamespace(type='cpu')
            def __init__(self,data): self.data=np.array(data)
            def clone(self): return Starts(self.data.copy())
        m=module()
        starts=Starts([8,11]); params={'cache_seqlens':starts,'_route_trace_keys':['a','b'],
                                      '_route_trace_phase':'verify'}
        m.begin(params,SimpleNamespace(shape=(2,3)))
        ctx=params['_route_trace_context']; first=ctx['call']; starts.data[:]=99
        self.assertEqual(ctx['starts'].data.tolist(),[8,11])
        self.assertEqual(ctx['phase'],'verify')
        m.begin(params,SimpleNamespace(shape=(2,1)))
        self.assertNotEqual(first,params['_route_trace_context']['call'])
        p2={'past_len':21}
        m.begin(p2,SimpleNamespace(shape=(1,2)))
        self.assertEqual(m.metadata(p2['_route_trace_context'],2)[0].tolist(),[21,22])

    def test_writer_waits_before_serializing_and_restores_ids(self):
        m=module()
        class Event:
            def synchronize(self):
                ids[:]=[[0,1],[2,0]]
                weights[:]=[[.7,.3],[.5,.5]]
        with tempfile.TemporaryDirectory() as t:
            w=m.Writer(t)
            ids=np.full((2,2),-1,dtype=np.int64)
            weights=np.zeros((2,2),dtype=np.float16)
            ctx=dict(call='1:2',batch=2,q=1,keys=('a','b'),starts=np.array([7,12]),phase='decode',component='text')
            w.submit((ctx,'model.layers.3.mlp','cuda:0',0,[2,0,1],ids,weights,Event()))
            w.close()
            files=list(Path(t).glob('route-*.npz')); self.assertEqual(len(files),1)
            with np.load(files[0],allow_pickle=False) as z:
                self.assertEqual(z['ids'].tolist(),[[2,0],[1,2]])
                self.assertEqual(z['positions'].tolist(),[7,12])
                self.assertEqual(z['weights'].dtype,np.float16)
                self.assertEqual(json.loads(str(z['meta']))['expert_space'],'checkpoint')
            self.assertEqual(len(list(Path(t).glob('complete-*.json'))),1)
            self.assertEqual(list(Path(t).glob('*.tmp')),[])

    def test_error_is_reported_no_clean_marker(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            w=m.Writer(t)
            w.submit(({},'layer','cpu',0,None,np.array([[1]]),np.array([[1.]]),None))
            with self.assertRaisesRegex(RuntimeError,'writer failed'): w.close()
            self.assertEqual(list(Path(t).glob('complete-*.json')),[])

    def test_queue_full_fails_instead_of_silent_loss(self):
        import queue
        m=module(); w=object.__new__(m.Writer)
        w.error=None; w.closed=False; w.queue=queue.Queue(1)
        w.submit('one')
        with self.assertRaisesRegex(RuntimeError,'queue full'): w.submit('two')

    def test_hook_precedes_inplace_mapping(self):
        s=(ROOT/'src/exllamav3/modules/block_sparse_mlp.py').read_text()
        self.assertLess(s.index('record(self, selected_experts'),s.index('self.cpu_split_submit(y, bsz'))
        self.assertGreater(s.index('record(self, selected_experts'),s.index('broadcast(routing_weights'))

if __name__=='__main__': unittest.main()
