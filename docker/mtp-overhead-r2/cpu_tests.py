"""Tests actual source definitions with CPU fakes at CUDA/loader boundaries; stdlib only."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT/'src'

def definition(path, name, cls=None, namespace=None):
    tree = ast.parse((SRC/path).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ns = namespace if namespace is not None else {}
    exec(compile(ast.fix_missing_locations(mod),str(path),'exec'),ns)
    return ns[name]

class CpuTests(unittest.TestCase):
    def test_device_verify_enqueued_before_host_commit(self):
        tree=ast.parse((SRC/'exllamav3/generator/generator.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='iterate_gen')
        a=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='forward_ids')
        b=next(i for i,n in enumerate(fn.body[a:],a) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='p_export_states')
        calls=[]
        class Tensor:
            def __init__(self,values,device='cpu'): self.values=values;self.device=device;self.shape=(len(values),len(values[0]))
            def __getitem__(self,key):
                rows,cols=key;return Tensor([r[cols] for r in self.values[rows]],self.device)
            def copy_(self,x): self.values=[r[:] for r in x.values];self.shape=x.shape;return self
            def to(self,device,non_blocking=False): calls.append(('upload',non_blocking));return Tensor(self.values,device)
        def cat(items,dim=0,out=None):
            values=([sum((t.values[row] for t in items),[]) for row in range(items[0].shape[0])] if dim==-1 else sum((t.values for t in items),[]))
            result=Tensor(values,items[0].device)
            if out is not None:out.copy_(result);return out
            return result
        def forward(input_ids,params):
            calls.append(('forward',input_ids.values));self.assertEqual(input_ids.values,[[5,7]]);return Tensor([[0]],'cuda:1')
        def commit(tokens,idx,add_to_cache):
            calls.append(('commit',add_to_cache));self.assertEqual(tokens.values,[[7]]);return [Tensor([[5,7]])]
        obj=NS(model=NS(forward=forward),_staging=lambda *args:Tensor([[0]]),draft_ids_pinned=Tensor([[7]]),_mtp_draft_copy_done=NS(synchronize=lambda:calls.append(('sync',))),_mtp_gpu_round=True)
        ns=dict(self=obj,gpu_tokens=Tensor([[7]],'cuda:1'),batch_ids=Tensor([[5,0]]),params={},mtp_phase=NS(ENABLED=False),_NGRAM_PREFETCH2=False,batch_size=1,batch_jobs=[NS(get_input_ids_list=commit)],torch=NS(long='long',cat=cat))
        exec(compile(ast.Module(body=fn.body[a:b],type_ignores=[]),'<actual device verify handoff>','exec'),ns)
        self.assertEqual([x[0] for x in calls],['upload','forward','sync','commit'])
        self.assertEqual(ns['batch_ids'].values,[[5,7]]);self.assertFalse(obj._mtp_gpu_round)

    def test_embedding_mirror_declines_low_vram_without_allocation(self):
        fn=definition('exllamav3/modules/embedding.py','prepare_mtp_mirror','Embedding',namespace={'os':os,'torch':NS(device=lambda d:d,cuda=NS(mem_get_info=lambda d:(0,0)))})
        obj=NS(can_embed_device_ids=lambda d:True,_gpu_mirror=None,embedding=NS(weight=NS(numel=lambda:100,element_size=lambda:2)))
        with contextlib.redirect_stdout(io.StringIO()):self.assertFalse(fn(obj,'cuda:0'))
        self.assertIn('cuda:0',obj._mtp_mirror_declined)

    def test_embedding_preserves_cuda_ids_at_verify_entry(self):
        # The real override must bypass Module's CPU transfer for GPU draft verification.
        fn=definition('exllamav3/modules/embedding.py','prepare_for_device','Embedding')
        x=NS(is_cuda=True)
        self.assertIs(fn(object(),x,{'mtp_gpu_ids':True}),x)

    def test_syntax_all(self):
        paths=list(SRC.rglob('*.py'))
        for path in paths: ast.parse(path.read_bytes(),str(path))
        self.assertGreater(len(paths),300)

    def test_dirty_hashes_survive_cpu_tier_eviction(self):
        tree=ast.parse((SRC/'exllamav3/generator/generator.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='on_queue_drained')
        guard=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self._mtp_invalid_hashes')
        obj=NS(_mtp_invalid_hashes={b'gpu',b'cpu',b'gone'},pagetable=NS(all_pages=[NS(phash=b'gpu')]),cpu_page_cache=NS(entries={b'cpu':{}}))
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'<actual validity pruning>','exec'),{'self':obj})
        self.assertEqual(obj._mtp_invalid_hashes,{b'gpu',b'cpu'})

    def test_suspended_prefill_marks_full_prompt_pages(self):
        tree=ast.parse((SRC/'exllamav3/generator/job.py').read_text())
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'page.kv_position == PAGE_SIZE' in ast.unparse(n.test) and '_mtp_suspended' in ast.unparse(n.test))
        obj=NS(_mtp_suspended=True,generator=NS(_mtp_invalid_hashes=set()))
        for count in (255,256):
            exec(compile(ast.Module(body=[guard],type_ignores=[]),'<actual prefill validity>','exec'),{'self':obj,'page':NS(kv_position=count,phash=b'full'),'PAGE_SIZE':256})
            self.assertEqual(bool(obj.generator._mtp_invalid_hashes),count==256)

    def test_actual_gpu_probe_taps_seqtensor_prompt_and_cpu_sample(self):
        import hashlib
        tree=ast.parse((ROOT/'out/gpu_probe.py').read_text())
        tap=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tapped_sample')
        flat=NS(reshape=lambda *a:NS(tolist=lambda:[3,5]))
        seq=NS(input_ids=NS(torch=lambda:flat))
        token=NS(reshape=lambda *a:NS(tolist=lambda:[7]))
        obj=NS(sequences=[seq],serial_number=2,identifier=None)
        ns={'_original_sample':lambda *a,**kw:(True,token,False),'json':json,'hashlib':hashlib}
        exec(compile(ast.Module(body=[tap],type_ignores=[]),'<actual token tap>','exec'),ns)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):result=ns['tapped_sample'](obj)
        record=json.loads(output.getvalue().split('[MTP-EXACT] ',1)[1]);self.assertEqual(record['ids'],[7])
        self.assertEqual(record['prompt_key'],hashlib.sha256(json.dumps([3,5]).encode()).hexdigest())
        self.assertIs(result[1],token)

    def test_depth_policy(self):
        fn=definition('exllamav3/generator/generator.py','_get_draft_depth','Generator')
        obj=NS(num_draft_tokens_by_batch=((1,1),(4,0)),num_draft_tokens=1)
        self.assertEqual([fn(obj,i) for i in (1,2,3,4,5)], [1,0,0,0,1])

    def gate(self, cap, jobs, policy=None):
        # Compile the real iterate decision and dispatch block, with CUDA-free collaborators.
        tree=ast.parse((SRC/'exllamav3/generator/generator.py').read_text())
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
        fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='iterate')
        a=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='batch_policy_work' for t in n.targets))
        b=next(i for i,n in enumerate(fn.body[a:],a) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.draft_model')
        calls=[]
        obj=NS(mtp_draft=True,num_draft_tokens_by_batch=policy,_mtp_invalid_hashes=set(),active_jobs=jobs,
               recurrent_cache=object(),draft_model=object(),dflash_draft=False)
        obj.recurrent_checkpoint=lambda:None
        obj._get_draft_depth=lambda n: next((d for lim,d in policy if n<=lim),1) if policy else 1
        obj.iterate_draftmodel_mtp_gen=lambda results: calls.append('draft') or 123
        obj.iterate_gen=lambda results, tokens=None: calls.append(('verify',tokens))
        ns={'self':obj,'_MTP_MAX_BATCH':cap,'mtp_phase':NS(ENABLED=False),'results':[]}
        exec(compile(ast.Module(body=fn.body[a:b+1],type_ignores=[]),'<actual iterate gate>','exec'),ns)
        return obj,calls

    def job(self, ready=True, suspended=False):
        obj=NS(mtp_last_hidden=123,sequences=[object()],is_prefill_done=lambda:ready)
        if suspended: obj._mtp_suspended=True
        return obj

    def test_cap_c1_drafts(self):
        obj,calls=self.gate(1,[self.job()]); self.assertFalse(obj._mtp_no_draft); self.assertIn('draft',calls)

    def test_cap_c2_c4_plain(self):
        for n in (2,3,4):
            jobs=[self.job() for _ in range(n)]; obj,calls=self.gate(1,jobs)
            self.assertEqual(calls,[('verify',None)]); self.assertTrue(all(j._mtp_suspended for j in jobs))

    def test_cap_counts_prefilling_active(self):
        obj,calls=self.gate(1,[self.job(),self.job(ready=False)])
        self.assertEqual(calls,[('verify',None)])

    def test_cap_sticky_and_fresh_job(self):
        job=self.job(); self.gate(1,[job,self.job()]); _,calls=self.gate(1,[job])
        self.assertEqual(calls,[('verify',None)])
        _,calls=self.gate(1,[self.job()]); self.assertIn('draft',calls)

    def test_zero_policy(self):
        _,calls=self.gate(0,[self.job(),self.job()],((1,1),(4,0)))
        self.assertEqual(calls,[('verify',None)])

    def test_default_unmodified_dispatch(self):
        _,calls=self.gate(0,[self.job(),self.job()]); self.assertIn('draft',calls)

    def test_dirty_prefix_plain_before_prefill(self):
        tree=ast.parse((SRC/'exllamav3/generator/job.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Job')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prefill')
        guard=next(n for n in fn.body if isinstance(n,ast.If) and '_mtp_invalid_hashes' in ast.unparse(n.test))
        for h,expected in ((b'dirty',True),(b'clean',False)):
            obj=NS(generator=NS(mtp_draft=True,_mtp_invalid_hashes={b'dirty'}),sequences=[NS(allocated_pages=[NS(phash=h)])])
            exec(compile(ast.Module(body=[guard],type_ignores=[]),'<dirty prefix guard>','exec'),{'self':obj})
            self.assertEqual(getattr(obj,'_mtp_suspended',False),expected)

    def test_accept_eligibility_and_rng(self):
        spec=importlib.util.spec_from_file_location('draft_overlap',SRC/'exllamav3/generator/draft_overlap.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        job=NS(sampler=NS(batch_verify_mode='greedy',reqs_past_ids=False),new_tokens=0,forced_ids=None,filters=[],return_probs=False,return_top_tokens=0,sequences=[1],rng=random.Random(42))
        self.assertEqual(mod.verification_batch_mode(job),'greedy')
        for attr,value in (('forced_ids',[1]),('filters',[1]),('return_probs',True),('return_top_tokens',1),('new_tokens',-1),('sequences',[1,2])):
            old=getattr(job,attr);setattr(job,attr,value);self.assertIsNone(mod.verification_batch_mode(job));setattr(job,attr,old)
        reference=random.Random(42)
        self.assertEqual([mod.draw_sampling_seed(job) for _ in range(30)],[reference.randint(0,(1<<32)-1) for _ in range(30)])

    def rewind_setup(self):
        class Tensor:
            def __init__(self,pointer): self.pointer=pointer;self.device='cuda:0';self.shape=(2,2,8);self.dtype='float32'
            def data_ptr(self): return self.pointer
            def stride(self): return (16,8,1)
        class Layer:
            def __init__(self): self.conv_state=Tensor(100);self.recurrent_state=Tensor(200);self.module=NS(conv_kernel_size=4);self.device='cuda:0'
        calls=[]
        def collect(layers,slot,history,n):
            calls.append((slot,history,n)); return {0:([('conv',layers[0].conv_state.pointer,slot,history,n)],[('state',layers[0].recurrent_state.pointer,slot,history,n)])}
        fn=definition('exllamav3/modules/gated_delta_net.py','cached_rewind_jobs',namespace={'GDNLayerState':Layer,'_collect_rewind_jobs':collect})
        return fn,Layer(),calls

    def test_rewind_cache_matches_real_dispatch_parameters(self):
        fn,l,calls=self.rewind_setup(); state=NS(slot=1,last_history=1)
        for n in (0,1,0,1):
            jobs=fn(state,[l],n); self.assertEqual(jobs[0][0][0],('conv',100,1,1,n))
        self.assertEqual(len(calls),2)

    def test_rewind_storage_change_invalidates_and_releases(self):
        fn,l,calls=self.rewind_setup();state=NS(slot=0,last_history=1)
        fn(state,[l],1);l.conv_state.pointer=900
        jobs=fn(state,[l],1);self.assertEqual(jobs[0][0][0][1],900);self.assertEqual(len(state._mtp_rewind_jobs),1)

    def test_rewind_non_gdn_fallback(self):
        calls=[]
        fn=definition('exllamav3/modules/gated_delta_net.py','cached_rewind_jobs',namespace={'GDNLayerState':type('GDN',(),{}),'_collect_rewind_jobs':lambda *args:calls.append(args) or 'fallback'})
        self.assertEqual(fn(NS(slot=0,last_history=1),[object()],1),'fallback');self.assertEqual(len(calls),1)

    def load_profiler(self, enabled):
        class Event:
            created=0;syncs=0
            def __init__(self,**kw): Event.created+=1;self.ready=True
            def record(self,stream): self.stream=stream
            def query(self): return self.ready
            def synchronize(self): Event.syncs+=1
            def elapsed_time(self,other): self.assert_same(other);return 1.25
            def assert_same(self,other): assert self.stream is other.stream
        class Device:
            def __init__(self,s): self.s=str(s);self.type='cuda' if self.s.startswith('cuda') else 'cpu'
            def __str__(self): return self.s
        streams={s:NS(device=s,cuda_stream=i) for i,s in enumerate(('cuda:0','cuda:1'))}
        fake=NS(device=Device,cuda=NS(Event=Event,current_stream=lambda d:streams[str(d)]))
        old=sys.modules.get('torch');sys.modules['torch']=fake
        oldenv=os.environ.get('EXL3_MTP_PHASE_PROF');os.environ['EXL3_MTP_PHASE_PROF']='1' if enabled else '0'
        try:
            spec=importlib.util.spec_from_file_location('phase_under_test',SRC/'exllamav3/util/mtp_phase.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        finally:
            if old is None: del sys.modules['torch']
            else: sys.modules['torch']=old
            if oldenv is None: os.environ.pop('EXL3_MTP_PHASE_PROF',None)
            else:os.environ['EXL3_MTP_PHASE_PROF']=oldenv
        return mod,Event,streams

    def test_profiler_off_returns_original_and_no_events(self):
        p,Event,_=self.load_profiler(False);fn=lambda:42
        self.assertIs(p.instrument('test')(fn),fn);self.assertIs(p.step_guard(fn),fn);self.assertEqual(Event.created,0)

    def test_profiler_nonblocking_and_aggregation(self):
        mod,Event,streams=self.load_profiler(True);p=mod.PhaseProfiler(['cuda:0','cuda:1'])
        mod.INTERVAL=2;output=io.StringIO()
        with contextlib.redirect_stdout(output):
            for _ in range(2):
                p.begin('mtp',1,1);t=p.start('verify.forward_device',('cuda:1',));p.stop(t);p.end()
        record=json.loads(output.getvalue().split('[MTP-PHASE] ',1)[1]);self.assertEqual(record['steps'],2)
        self.assertEqual(record['cuda']['verify.forward_device@cuda:1']['ms_per_step'],1.25)
        self.assertEqual(Event.syncs,0)

    def test_profiler_uses_explicit_copy_stream(self):
        mod,_,_=self.load_profiler(True);p=mod.PhaseProfiler(['cuda:0']);p.begin('mtp',1,1)
        copy=NS(device='cuda:0',cuda_stream=999);t=p.start('copy',(copy,));p.stop(t)
        self.assertIs(t[2][0][1],copy);p.end()

    def test_profiler_defers_unready_events(self):
        mod,Event,_=self.load_profiler(True);p=mod.PhaseProfiler(['cuda:0']);p.begin('mtp',1,1)
        p.total[2][0][3].ready=False;p.end();self.assertEqual(len(p.pending),1);self.assertEqual(Event.syncs,0)
        with contextlib.redirect_stdout(io.StringIO()):p.drain(force=True)
        self.assertEqual(len(p.pending),0);self.assertGreater(Event.syncs,0)

if __name__=='__main__':unittest.main(verbosity=2)
