"""Real selector, initialization and real trace replay, with CPU CUDA/tensor doubles."""
import ast, contextlib, json, os, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from test_cpu import ROOT, module, ArrayTensor, ex
score=module('r3_score',ROOT/'implementation/model/moe_score.py')
replay=module('r3_replay',ROOT/'sim_policy.py')

class Tensor(ArrayTensor):
    device='cpu'
    def __len__(self):return len(self.a)
    def cpu(self):return Tensor(self.a.copy())
    def to(self,*a,**k):return self
    def tolist(self):return self.a.tolist()
    def __getitem__(self,i):return self.a[i]
    def __setitem__(self,i,v):self.a[i]=v
    def mul_(self,v):self.a*=np.float32(v);return self
    def zero_(self):self.a.fill(0);return self

class Event:
    def record(self,*a):pass
    def synchronize(self):pass

def torch_double():
    return types.SimpleNamespace(inference_mode=contextlib.nullcontext,device=lambda x:x,
        long=np.int64,float=np.float32,arange=lambda n,**k:Tensor(np.arange(n)),
        tensor=lambda x,**k:Tensor(np.array(x)),zeros=lambda n,**k:Tensor(np.zeros(n,dtype=np.float32)),
        zeros_like=lambda t:Tensor(np.zeros_like(t.a)),
        cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,
          device=lambda _:contextlib.nullcontext(),current_stream=lambda _:None,Event=Event))

def production_method(name):
    tree=ast.parse((ROOT/'implementation/modules/block_sparse_mlp_cpu.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    ns={'_exchange_mode':True,'_score_mode':True,'_split_prof':False,'_split_fused':True,'torch':torch_double(),'os':os,'__package__':'fixture.modules'}
    node.decorator_list=[]
    ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),'production','exec'),ns)
    return ns[name]

# Relative imports in the actual production methods use these explicitly isolated stubs.
sys.modules['fixture.model.moe_score']=score
sys.modules['fixture.model.moe_exchange']=ex

class PolicyTests(unittest.TestCase):
    def test_profile_map_and_prior_keep_original_router_ids(self):
        tree=ast.parse((ROOT/'implementation/modules/block_sparse_mlp_cpu.py').read_text())
        f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='load_cpu_split')
        def assign_attr(n,attr):
            return isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr==attr for t in n.targets)
        start=next(i for i,n in enumerate(f.body) if assign_attr(n,'_split_perm'))
        end=next(i for i,n in enumerate(f.body[start:],start) if assign_attr(n,'device'))
        block=ast.FunctionDef(name='init',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=f.body[start:end],decorator_list=[])
        ast.fix_missing_locations(block)
        ns={'_exchange_mode':True,'os':os,'__package__':'fixture.modules'}
        exec(compile(ast.Module(body=[block],type_ignores=[]),'production init','exec'),ns)
        m=types.SimpleNamespace(key='layer',num_experts=6,num_experts_per_tok=2,tid2eid_key=None,gated=True,
            gates=list(range(6)),ups=list(range(6)),downs=list(range(6)),cpu_split_first=3,device=0,
            config=types.SimpleNamespace(infer_params=types.SimpleNamespace()))
        counts=[1,10,3,8,7,2];order=[1,3,4,2,5,0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json';p.write_text(json.dumps({'layer':counts}))
            with patch.dict(os.environ,EXL3_MOE_CPU_SWAP='1',EXL3_MOE_CPU_SPLIT_STATS=str(p),EXL3_MOE_CPU_SWAP_POLICY='score',EXL3_MOE_CPU_SWAP_MAX='32',EXL3_MOE_CPU_SWAP_CADENCE='exact',EXL3_MOE_CPU_SWAP_BUDGET_SCOPE='global'):
                ns['init'](m);production_method('cpu_post_load')(m)
        self.assertEqual(m.gates,order);self.assertIsNone(m._split_perm)
        np.testing.assert_array_equal(m._split_map.a,score.initial_map(order))
        self.assertTrue(m._score_policy);self.assertEqual(m._split_hist.a.sum(),0)
        np.testing.assert_allclose(m._split_stats_prior,score.rate(counts,2))
    def test_admission_threshold_ties_and_zero_churn(self):
        mp=list(range(6));value=[1,1,1,2,2,2]
        self.assertEqual(score.select_pairs(mp,3,value,2,2),[(0,5),(1,4)])
        self.assertEqual(score.select_pairs(mp,3,value,2,2.01),[])
        self.assertEqual(score.select_pairs(mp,3,[0]*6,3,2),[])
        self.assertEqual(score.select_pairs(mp,3,value,0,2),[])
    def test_invalid_inputs_are_rejected(self):
        for counts in ([1,-1],[1,float('nan')],[1,float('inf')]):
            with self.assertRaises(ValueError):score.validated_counts(counts,2)
        with self.assertRaises(ValueError):score.initial_map([0,0])
        for env in ({'EXL3_MOE_CPU_SCORE_HALF_LIFE':'0'},{'EXL3_MOE_CPU_SCORE_PREFILL':'-1'},
                    {'EXL3_MOE_CPU_SWAP_BUDGET_SCOPE':'layer'},{'EXL3_MOE_CPU_SWAP_CADENCE':'served'}):
            with patch.dict(os.environ,dict(EXL3_MOE_CPU_SWAP_POLICY='score',**env),clear=True):
                with self.assertRaises(ValueError):score.settings()
    def test_prefill_chunk_accumulation_and_replacement(self):
        m=types.SimpleNamespace(_score_prompt_hist=Tensor(np.zeros(6)),_split_hist=Tensor(np.zeros(6)),_score_previous_prefill=False)
        score.prepare_signal(m,True,3);m._score_count_hist.a[4]=6
        score.prepare_signal(m,True,2);m._score_count_hist.a[4]+=2
        self.assertEqual(m._score_prompt_rows,5);self.assertEqual(m._score_prompt_hist.a[4],8)
        score.prepare_signal(m,False,1);self.assertIs(m._score_count_hist,m._split_hist)
        score.prepare_signal(m,True,4);self.assertEqual(m._score_prompt_hist.a.sum(),0)
        self.assertEqual(m._score_prompt_rows,4)
    def test_expired_prompt_is_ignored(self):
        cfg={'wp':32,'wb':256}
        self.assertEqual(score.scores([1],[100],0,[.5],cfg),[129])
    def test_real_sweep_publication_failure_poisoned(self):
        class BadMap(Tensor):
            def copy_(self,*a,**k):raise RuntimeError('publication failure')
        mp=BadMap(np.arange(6))
        m=types.SimpleNamespace(_split_map=mp,_split_hist=Tensor(np.array([0,0,0,5,4,3],dtype=np.float32)),
            _score_prompt_hist=Tensor(np.zeros(6)),_score_prompt_rows=0,_split_stats_prior=[0]*6,
            _score_settings={'wp':0,'wb':0,'rho':2},cpu_split_first=3,cpu_host=types.SimpleNamespace(),device=0)
        def exchange(a,b,mp):mp[a],mp[b]=int(mp[b]),int(mp[a]);return True
        m._split_swap_experts=exchange;m._split_sweep_layer=lambda b:score.sweep_layer(m,b)
        m._split_sweep_layer_reset=lambda:None
        with patch.dict(sys.modules,torch=torch_double()),patch.object(ex,'fence_layer',lambda _:None),patch.dict(os.environ,EXL3_MOE_CPU_SWAP_MAX='2',EXL3_MOE_CPU_SWAP_BUDGET_SCOPE='global'):
            with self.assertRaisesRegex(RuntimeError,'publication'):ex.run_sweep(types.SimpleNamespace(),[m])
        self.assertTrue(m.cpu_host.exchange_failed)
    def test_real_submission_decays_decode_and_redirects_prefill_counts(self):
        m=types.SimpleNamespace(device=0,_split_map=Tensor(np.arange(6)),_split_hist=Tensor(np.array([4,0,0,0,0,0],dtype=np.float32)),
            _score_policy=True,_score_prompt_hist=Tensor(np.zeros(6,dtype=np.float32)),_score_previous_prefill=False,
            _score_settings={'decay':.5},cpu_split_first=3,cpu_layer_idx=0)
        tick=[];m._split_swap_tick=lambda:tick.append('tick')
        def fused(layer,y,ids,weights,mp,hist,first):
            np.add.at(hist.a,ids.a.ravel(),1);return 1
        m.cpu_host=types.SimpleNamespace(stream_min_rows=100,submit_issue_fused=fused)
        prepare=production_method('cpu_score_prepare');submit=production_method('cpu_split_submit')
        prepare(m,{'prefill':False},1)  # False is still a prefill phase marker.
        submit.__globals__['torch'].cuda.current_stream=lambda _:types.SimpleNamespace(cuda_stream=1)
        submit(m,Tensor(np.zeros((1,4))),1,Tensor(np.array([[4,5]])),None)
        self.assertEqual(tick,[]);self.assertEqual(m._split_hist.a[0],4)
        np.testing.assert_array_equal(m._score_prompt_hist.a,[0,0,0,0,1,1])
        prepare(m,{},1);submit(m,Tensor(np.zeros((1,4))),1,Tensor(np.array([[3,4]])),None)
        self.assertEqual(tick,['tick']);np.testing.assert_array_equal(m._split_hist.a,[2,0,0,1,1,0])
    def test_inline_sweep_retains_prompt_and_queue_drain_expires_it(self):
        m=types.SimpleNamespace(_score_policy=True,_score_settings={'k':32},_swap_tick_count=32,
            _score_prompt_rows=10,_score_previous_prefill=False)
        ip=types.SimpleNamespace(moe_cpu_swap_modules=[m],moe_cpu_swap_pending=False);m.config=types.SimpleNamespace(infer_params=ip)
        tick=production_method('_split_swap_tick');pending=production_method('run_pending_swap_sweeps')
        seen=[]
        def inline(ip):
            seen.append(ip._score_inline_sweep);ip.moe_cpu_swap_pending=False;m._swap_tick_count=0
        tick.__globals__['run_pending_swap_sweeps']=inline
        with patch.dict(os.environ,EXL3_MOE_CPU_SWAP_CADENCE='exact'):
            tick(m)
        self.assertEqual(seen,[True]);self.assertEqual(m._score_prompt_rows,10)
        self.assertEqual(m._swap_tick_count,1);self.assertFalse(ip._score_inline_sweep)
        pending(ip);self.assertEqual(m._score_prompt_rows,0)

    def test_unload_releases_score_vectors_and_registry(self):
        m=types.SimpleNamespace(cpu_split_first=192,cpu_host=types.SimpleNamespace(unregister=lambda:None),
            _split_saved=([],[],[],[],288,None,None),_split_map=Tensor(np.arange(288)),
            _split_hist=Tensor(np.zeros(288)),_score_prompt_hist=Tensor(np.zeros(288)),
            _score_count_hist=Tensor(np.zeros(288)),_score_policy=True,_score_settings={},cpu_offload=False)
        ip=types.SimpleNamespace(moe_cpu_swap_modules=[m]);m.config=types.SimpleNamespace(infer_params=ip)
        production_method('cpu_unload')(m)
        self.assertEqual(ip.moe_cpu_swap_modules,[]);self.assertIsNone(m._score_prompt_hist)
        self.assertIsNone(m._score_count_hist);self.assertFalse(m._score_policy)
        self.assertIsNone(m._split_stats_prior)

    def test_exchange_transaction_and_host_identical_to_r2(self):
        base=ROOT.parents[1]/'r2/implementation'
        if not base.exists():self.skipTest('workspace-only source differential')
        for rel in ('model/moe_exchange.py','model/moe_cpu_host.py'):
            self.assertEqual((base/rel).read_bytes(),(ROOT/'implementation'/rel).read_bytes())

class RealTraceReplayTests(unittest.TestCase):
    def test_real_selection_plus_r2_global_sweep_matches_independent_simulator(self):
        z=np.load(ROOT/'replay-slice.npz');data={k:z[k] for k in ('ids','new','prompt')};prior=z['prior']
        p=replay.Policy(32,512,32,256,32,2,'static')
        expect=next(v for v in replay.replay_group(data,prior,p.k,p.H,p.wp,p.wb,p.init) if v[0]==p)
        real_sweep=production_method('_split_sweep_layer')
        reg=[];fake=torch_double();orders=np.argsort(-prior,axis=1,kind='stable')
        for l in range(42):
            mp=Tensor(np.array(score.initial_map(orders[l].tolist())))
            m=types.SimpleNamespace(_split_map=mp,_split_hist=Tensor(np.zeros(288,dtype=np.float32)),
                _score_policy=True,_score_prompt_hist=Tensor(data['prompt'][0,l].copy()),_score_prompt_rows=1,
                _split_stats_prior=prior[l].tolist(),_score_settings={'wp':p.wp,'wb':p.wb,'rho':p.rho},
                cpu_split_first=192,num_experts=288,cpu_host=types.SimpleNamespace(),device=0,
                _split_sweep_layer_reset=lambda:None)
            def exchange(a,b,mp):mp[a],mp[b]=int(mp[b]),int(mp[a]);return True
            m._split_swap_experts=exchange;m._split_sweep_layer=lambda budget,m=m:real_sweep(m,budget)
            reg.append(m)
        picks=[];swaps=[]
        with patch.dict(sys.modules,torch=fake),patch.object(ex,'fence_layer',lambda _:None),patch.dict(os.environ,EXL3_MOE_CPU_SWAP_MAX=str(p.M),EXL3_MOE_CPU_SWAP_BUDGET_SCOPE='global',EXL3_MOE_CPU_SWAP_VERIFY=''):
            for t,x in enumerate(data['ids']):
                before=np.array([m._split_map.a.copy() for m in reg])
                if t and t%p.k==0:ex.run_sweep(types.SimpleNamespace(),reg)
                after=np.array([m._split_map.a for m in reg]);swaps.append(np.count_nonzero(before!=after)//2)
                picks.append(sum((m._split_map.a[r]>=192).sum() for m,r in zip(reg,x)))
                for m,r in zip(reg,x):
                    m._split_hist.mul_(2**(-1/p.H));np.add.at(m._split_hist.a,r,1)
                    self.assertEqual(np.unique(m._split_map.a).size,288)
        np.testing.assert_array_equal(picks,expect[1]);np.testing.assert_array_equal(swaps,expect[2])
        self.assertLessEqual(max(swaps),p.M)

if __name__=='__main__':unittest.main()
