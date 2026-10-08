#!/usr/bin/env python3
"""CPU regression harness: execute production AST methods without importing CUDA packages.
The bounded vectors implement only the Torch operations at the tested call sites.
Use --tree base to demonstrate the pre-fix failures. --torch also verifies real CPU scatter.
"""
import argparse
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import random
import sys
sys.dont_write_bytecode = True
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'src'
BASE_TREE = ROOT / 'base'
REAL_TORCH = False


def extract(file, name, cls=None, namespace=None):
    tree = ast.parse((TREE / 'exllamav3' / file).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ns = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(mod), str(TREE / 'exllamav3' / file), 'exec'), ns)
    return ns[name]


class Vec:
    def __init__(self, values, shape=None):
        self.v = list(values)
        self.shape = shape or (len(self.v),)
        self.device = 'cpu'
    def numel(self): return len(self.v)
    def reshape(self, *shape): return Vec(self.v)
    def view(self, *shape): return Vec(self.v, shape)
    def __add__(self, value): return Vec([x + value for x in self.v], self.shape)
    def __sub__(self, value): return Vec([x - value for x in self.v], self.shape)
    def clamp_min(self, value): return Vec([max(value, x) for x in self.v], self.shape)
    def __getitem__(self, i): return Vec(self.v[i]) if isinstance(i, slice) else self.v[i]
    def __setitem__(self, i, value): self.v[i] = value
    def cpu(self): return self
    def detach(self): return self
    def to(self, *a, **kw): return self
    def tolist(self): return self.v[:]
    def sum(self): return sum(self.v)
    def mul_(self, x): self.v = [v*x for v in self.v]; return self
    def copy_(self, other): self.v[:] = other.v; return self
    def contiguous(self): return self
    def is_contiguous(self): return True
    def scatter_add_(self, dim, ids, src):
        for i, x in zip(ids.v, src.v):
            if i < 0 or i >= len(self.v):
                raise IndexError(f'scatter_add_: index {i} out of bounds for size {len(self.v)}')
            self.v[i] += x
        return self


class Cuda:
    @staticmethod
    def device(*a): return nullcontext()
    @staticmethod
    def stream(*a): return nullcontext()
    @staticmethod
    def is_current_stream_capturing(): return False
    @staticmethod
    def Stream(*a, **kw): return SimpleNamespace(synchronize=lambda: None)
    @staticmethod
    def Event(): return SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
    @staticmethod
    def current_stream(*a): return None


TORCH = SimpleNamespace(cuda=Cuda(), long='long', half='half', inference_mode=lambda: nullcontext(),
                        zeros=lambda n, **kw: Vec([0]*n), ones_like=lambda v: Vec([1]*v.numel()))
TORCH.device = lambda d: SimpleNamespace(type=str(d).split(':')[0], index=int(str(d).split(':')[1]) if ':' in str(d) else None)


def checks_module():
    path = TREE / 'exllamav3/model/moe_split_check.py'
    if not path.exists():
        return SimpleNamespace(ENABLED=False)
    spec = importlib.util.spec_from_file_location('splitdev_checks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKS = None


def host():
    h = SimpleNamespace(specs=[], by_key={}, live_layers=0, started=False, aux={},
                        _dev_bufs={}, sstate={}, messages=[], stream_min_rows=16,
                        wslot_size=100000, pinned=True, layer_blocks=[], exchange_views=[],
                        acked=0, exchange_swizzled=False)
    h.conn = SimpleNamespace(send=h.messages.append)
    h._spawn = lambda: None
    ns = {'os': os, 'torch': TORCH, 'split_check': CHECKS}
    h.register_layer = MethodType(extract('model/moe_cpu_host.py', 'register_layer', 'MoeCpuHost', ns), h)
    h._ensure_stream_state = lambda device: {'stream_t': 10**9}
    h.submit = lambda idx, y, sel, weights: ('cpu', idx)
    h.submit_prefill = MethodType(extract('model/moe_cpu_host.py', 'submit_prefill', 'MoeCpuHost',
        {**ns, 'TUNING': SimpleNamespace(stream_debug=False)}), h)
    h.shutdown = lambda: None
    h.unregister = MethodType(extract('model/moe_cpu_host.py', 'unregister', 'MoeCpuHost', ns), h)
    return h


def register(h, key, n, order=None, dims=None):
    order = order or list(range(288))
    keys = order[288-n:]
    return h.register_layer(key, [f'g.{i}' for i in keys], [f'u.{i}' for i in keys],
                            [f'd.{i}' for i in keys], 0, 0., 128, 128, 8,
                            proj_dims=dims or {'g': (128,128,2), 'u': (128,128,2), 'd': (128,128,2)},
                            aux={'suh_u': list(keys)})


def prefill(h, idx, first):
    physical = list(range(288)) * 8
    local = Vec([max(-1, x-first) for x in physical], (288,8))
    return h.submit_prefill(idx, SimpleNamespace(shape=(288,128), device='cpu'), local, None)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'EXL3_MOE_CPU_SPLIT_BY_DEVICE': '98,104',
                                          'EXL3_MOE_CPU_SPLIT_CHECK': '0'})
        self.env.start()
    def tearDown(self): self.env.stop()
    def test_rollback_98_to_104_real_registration_and_scatter(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98)
        h.unregister()
        new=register(h,'L23',104)
        # Exercise the actual histogram before asserting cache identity: base raises
        # scatter_add_: index 99 out of bounds for size 99 at this production seam.
        self.assertEqual(prefill(h,new,184),('cpu',new))
        self.assertNotEqual(new,old)
        self.assertEqual(h.specs[new]['num_experts'],104)
        self.assertEqual(h.specs[new]['up_keys'][0],'u.184')
    def test_reverse_rollback_104_to_98_has_correct_experts(self):
        h=host(); register(h,'L45',104); old=register(h,'L23',104); h.unregister()
        new=register(h,'L23',98)
        self.assertNotEqual(new,old)
        self.assertEqual(h.specs[new]['up_keys'][0],'u.190')
        self.assertEqual(prefill(h,new,190),('cpu',new))
    def test_same_count_changed_profile_order_is_new_registration(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98); h.unregister()
        new=register(h,'L23',98,list(reversed(range(288))))
        self.assertNotEqual(old,new)
        self.assertEqual(h.specs[new]['up_keys'][0],'u.97')
    def test_identical_retry_reuses_worker_and_refreshes_aux(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98); h.unregister()
        count=len(h.messages); h.aux[old]={'old_device': True}
        new=register(h,'L23',98)
        self.assertEqual(old,new); self.assertEqual(count,len(h.messages))
        self.assertEqual(h.live_layers,2); self.assertNotIn('old_device',h.aux[new])
    def test_identical_retry_drops_cached_aux_pointer_tables_on_all_devices(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98); h.unregister()
        h._dev_bufs={0:{'recon':{old:object()}},1:{'recon':{old:object()}}}
        register(h,'L23',98)
        self.assertTrue(all(old not in s['recon'] for s in h._dev_bufs.values()))
    def test_superseded_aux_and_reconstruction_are_released(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98); h.unregister()
        cache={old: object()}; h._dev_bufs={0: {'recon': cache}}; h.sstate={0: {'recon': cache}}
        new=register(h,'L23',104)
        self.assertNotIn(old,h.aux); self.assertNotIn(old,cache); self.assertIn(new,h.aux)
    def test_no_by_device_keeps_original_retry_behavior(self):
        with patch.dict(os.environ, {'EXL3_MOE_CPU_SPLIT_BY_DEVICE': ''}):
            h=host(); register(h,'L45',104); old=register(h,'L23',104); h.unregister()
            self.assertEqual(register(h,'L23',104),old)
            self.assertEqual(len(h.specs),2); self.assertEqual(len(h.messages),2)
    def test_changed_layout_cannot_register_after_start(self):
        h=host(); register(h,'L23',98); h.started=True
        with self.assertRaises((AssertionError,RuntimeError)): register(h,'L23',104)
        self.assertEqual(len(h.specs),1)
    def test_mtp_and_main_mixed_counts_both_mtp_placements(self):
        for mtp in (98,104):
            h=host()
            for key,n in (('L45',mtp),('L3',98),('L24',104)):
                idx=register(h,key,n)
                self.assertEqual(prefill(h,idx,288-n),('cpu',idx))
    def test_superseded_ack_indices_stay_append_only(self):
        h=host(); register(h,'L45',98); old=register(h,'L23',98); h.unregister()
        new=register(h,'L23',104); following=register(h,'L24',104)
        self.assertEqual([old,new,following],[1,2,3])
        self.assertEqual([msg[1]['num_experts'] for msg in h.messages],[98,98,104,104])
        self.assertEqual(h.by_key['L23'],new); self.assertEqual(h.live_layers,3)


class IndexTests(unittest.TestCase):
    def test_no_override_differential_with_base_registration_and_translation(self):
        global TREE
        original=TREE
        traces=[]
        try:
            with patch.dict(os.environ,{'EXL3_MOE_CPU_SPLIT_BY_DEVICE':'','EXL3_MOE_CPU_SPLIT_CHECK':'0'}):
                for location in (BASE_TREE,original):
                    TREE=location; h=host()
                    register(h,'L45',104); register(h,'L3',104); idx=register(h,'L23',104)
                    h.unregister(); register(h,'L23',104)
                    translate=extract('modules/block_sparse_mlp_cpu.py','_split_translate','BlockSparseMLP_CPU',
                                      {'torch':TORCH,'split_check':CHECKS})
                    local=translate(SimpleNamespace(_split_map=None,cpu_split_first=184),Vec(range(288))).tolist()
                    traces.append((h.specs,h.by_key,h.messages,h.aux,h.live_layers,local,prefill(h,idx,184)))
        finally: TREE=original
        self.assertEqual(traces[0],traces[1])

    def test_per_device_resolution_on_each_attempt_and_base_fallback(self):
        if TREE.name=='base': self.skipTest('BY_DEVICE introduced by r1')
        method=extract('modules/block_sparse_mlp_cpu.py','cpu_maybe_split_load','BlockSparseMLP_CPU',
                       {'os':os,'torch':TORCH})
        for env,device,expected in (('98,104','cuda:0',98),('98,104','cuda:1',104),
                                    ('100,104','cuda:0',100),('96,104','cuda:0',96),
                                    ('98','cuda:1',104),('0,104','cuda:0',104),('','cuda:0',104),
                                    ('999,104','cuda:0',287)):
            ip=SimpleNamespace(moe_cpu_split=104,moe_cpu_offload=0,draft_moe_cpu_offload=0,moe_cpu_split_assigned=0)
            m=SimpleNamespace(config=SimpleNamespace(infer_params=ip),num_experts=288,num_local_experts=288,
                              cpu_split_first=None,routing_first=None,gated=True,activation_fn='silu')
            calls=[]; m.load_cpu_split=lambda d,n,**kw:calls.append(n) or True
            with patch.dict(os.environ,{'EXL3_MOE_CPU_SPLIT_BY_DEVICE':env,'EXL3_MOE_CPU_SPLIT_LAYERS':'0'}):
                method(m,device)
            self.assertEqual(calls,[expected]); self.assertEqual(ip.moe_cpu_split_assigned,1)

    def test_rollback_releases_split_layer_budget_in_by_device_mode(self):
        if TREE.name=='base': self.skipTest('rollback budget correction introduced by splitdev2')
        ip=SimpleNamespace(moe_cpu_split_assigned=2)
        m=SimpleNamespace(config=SimpleNamespace(infer_params=ip),cpu_split_first=190,
                          cpu_host=SimpleNamespace(unregister=lambda:None),cpu_layer_idx=1,
                          _split_saved=([],[],[],[],288,None,None),_split_map=None,cpu_offload=False,
                          _exchange_copy_stream=object(),_exchange_read_streams={1:object()},_split_selcpu_t=object())
        unload=extract('modules/block_sparse_mlp_cpu.py','cpu_unload','BlockSparseMLP_CPU',
                       {'os':os,'_exchange_mode':True})
        with patch.dict(os.environ,{'EXL3_MOE_CPU_SPLIT_BY_DEVICE':'98,104'}): unload(m)
        self.assertEqual(ip.moe_cpu_split_assigned,1)
        self.assertIsNone(m.cpu_layer_idx); self.assertIsNone(m.cpu_host)
        self.assertIsNone(m._exchange_copy_stream); self.assertIsNone(m._split_selcpu_t)
        self.assertEqual(m._exchange_read_streams,{})
        self.assertEqual(m.num_local_experts,288)

    def test_static_translation_all_experts_c1_to_c4(self):
        translate=extract('modules/block_sparse_mlp_cpu.py','_split_translate','BlockSparseMLP_CPU',
                          {'torch': TORCH, 'ext': None, '_score_mode': False, 'split_check': CHECKS})
        for n in (96,98,100,104):
            for c in (1,2,3,4,16,2048):
                module=SimpleNamespace(_split_map=None,cpu_split_first=288-n)
                ids=Vec(list(range(288))*c)
                local=translate(module,ids).tolist()
                self.assertEqual(set(local),{-1,*range(n)})
                self.assertTrue(all(0<=x+1<n+1 for x in local))
    def test_profile_permutation_boundaries_and_swaps(self):
        initial=extract('model/moe_score.py','initial_map')
        select=extract('model/moe_score.py','select_pairs',namespace={'math': __import__('math')})
        rng=random.Random(915)
        for n in (98,104):
            order=list(range(288)); rng.shuffle(order); mp=initial(order); first=288-n
            for c in (1,2,3,4):
                score=[rng.randrange(100) for _ in range(288)]
                pairs=select(mp,first,score,64,2.)
                for cold,hot in pairs:
                    self.assertTrue(0<=mp[cold]<first)
                    self.assertTrue(0<=mp[hot]-first<n)
                    mp[cold],mp[hot]=mp[hot],mp[cold]
                self.assertEqual(sorted(mp),list(range(288)))
                local=[p-first if p>=first else -1 for p in mp]
                self.assertTrue(all(0<=x+1<=n for x in local))
    def test_base_defect_exact_index_set_and_silent_wrong_expert(self):
        # Same values produced by the original layer23 rollback: 184 is translated as 0
        # but old worker local0 contains expert190; highest six slots exceed the histogram.
        first=184; stale_n=98
        bad=[p-first+1 for p in range(288) if p-first+1>=stale_n+1]
        self.assertEqual(bad,[99,100,101,102,103,104])
        self.assertNotEqual(190+(184-first),184)
    def test_real_torch_cpu_scatter_if_requested(self):
        if not REAL_TORCH: self.skipTest('use --torch on a Torch-equipped CPU host')
        import torch
        shifted=torch.arange(-1,104,dtype=torch.long)+1
        with self.assertRaises(RuntimeError):
            torch.zeros(99,dtype=torch.long).scatter_add_(0,shifted,torch.ones_like(shifted))
        result=torch.zeros(105,dtype=torch.long).scatter_add_(0,shifted,torch.ones_like(shifted))
        self.assertEqual(result.tolist(),[1]*105)


class ExchangeTests(unittest.TestCase):
    def test_production_global_histogram_sweep_mixed_layers_and_mtp(self):
        initial=extract('model/moe_score.py','initial_map')
        healthy=extract('model/moe_exchange.py','require_healthy')
        fences=[]
        run=extract('model/moe_exchange.py','run_sweep',namespace={
            'os':os, 'require_healthy':healthy, 'fence_layer':lambda m:fences.append(m.key),
            'poison':lambda *a:None})
        sweep=extract('modules/block_sparse_mlp_cpu.py','_split_sweep_layer','BlockSparseMLP_CPU',
                      {'os':os,'_exchange_mode':True})
        reset=extract('modules/block_sparse_mlp_cpu.py','_split_sweep_layer_reset','BlockSparseMLP_CPU')
        for mtp in (98,104):
            order=list(range(288)); random.Random(mtp).shuffle(order)
            h=host(); modules=[]
            for key,n in (('L3',98),('L24',104),('L45',mtp)):
                idx=register(h,key,n,order)
                m=SimpleNamespace(key=key,device='cpu',cpu_host=h,cpu_layer_idx=idx,
                                  cpu_split_first=288-n,num_experts=288,_score_policy=False,
                                  _split_map=Vec(initial(order)),_split_hist=Vec([0.]*288),
                                  residents=order[:],swapped=[])
                for r in order[-3:]: m._split_hist[r]=1000.
                def swap(this,cold,hot,mp):
                    slot=mp[cold]; local=mp[hot]-this.cpu_split_first
                    self.assertTrue(0<=slot<this.cpu_split_first)
                    self.assertTrue(0<=local<this.cpu_host.specs[this.cpu_layer_idx]['num_experts'])
                    self.assertEqual(this.residents[slot],cold)
                    self.assertEqual(this.residents[mp[hot]],hot)
                    this.residents[slot],this.residents[mp[hot]]=hot,cold
                    this.swapped.append((cold,hot))
                    mp[cold],mp[hot]=mp[hot],slot
                    return True
                m._split_swap_experts=MethodType(swap,m)
                m._split_sweep_layer=MethodType(sweep,m)
                m._split_sweep_layer_reset=MethodType(reset,m)
                modules.append(m)
            with patch.dict(sys.modules,{'torch':TORCH}), patch.dict(os.environ,{
                    'EXL3_MOE_CPU_SWAP_MAX':'9','EXL3_MOE_CPU_SWAP_BUDGET_SCOPE':'global',
                    'EXL3_MOE_CPU_SWAP_VERIFY':'','EXL3_MOE_CPU_SWAP_DEBUG':''}):
                run(SimpleNamespace(),modules)
            self.assertEqual([len(m.swapped) for m in modules],[3,3,3])
            for m in modules:
                self.assertEqual(sorted(m._split_map.v),list(range(288)))
                for r,p in enumerate(m._split_map.v): self.assertEqual(m.residents[p],r)
                for c in (1,2,3,4):
                    local=[p-m.cpu_split_first if p>=m.cpu_split_first else -1 for p in m._split_map.v]*c
                    self.assertTrue(all(-1<=e<288-m.cpu_split_first for e in local))
        self.assertEqual(fences,['L3','L24','L45']*2)

    def test_real_exchange_call_selects_layer_local_descriptor_after_rollback(self):
        with patch.dict(os.environ, {'EXL3_MOE_CPU_SPLIT_BY_DEVICE':'98,104'}):
            h=host(); register(h,'L45',98); register(h,'L23',98); h.unregister(); idx=register(h,'L23',104)
        h.exchange_views=[[[[(i,e,'u'), None,None,None], [(i,e,'d'),None,None,None]]
                           for e in range(s['num_experts'])] for i,s in enumerate(h.specs)]
        h.arena={}
        for i,s in enumerate(h.specs):
            for e in range(s['num_experts']):
                for p in ('u','d'): h.arena[(i,e,p)]=Vec([int(s['up_keys'][e].split('.')[1])])
        module=SimpleNamespace(cpu_host=h,cpu_layer_idx=idx,cpu_split_first=184,
                               key='L23',gated=False,device='cpu',num_experts=288,num_local_experts=184,
                               ups=[SimpleNamespace(inner=SimpleNamespace(trellis=Vec([p]))) for p in range(184)],
                               downs=[SimpleNamespace(inner=SimpleNamespace(trellis=Vec([p]))) for p in range(184)])
        healthy=extract('model/moe_exchange.py','require_healthy')
        def swap(pairs,*a):
            for gpu,arena,*unused in pairs:
                gpu.v,arena.v=arena.v[:],gpu.v[:]
            return True
        fn=extract('model/moe_exchange.py','exchange_experts',namespace={
            'require_healthy':healthy,'arena_tensor':lambda host,d:host.arena[d],
            'exchange_tensors':swap,'refresh_recon_scales':lambda *a:False,'os':os,
            'poison':lambda *a:None,'split_check':CHECKS})
        mp=Vec(range(288))
        with patch.dict(sys.modules, {'torch': TORCH}):
            for cold,hot in ((0,184),(183,287)):
                self.assertTrue(fn(module,cold,hot,mp))
                self.assertEqual(module.ups[cold].inner.trellis.v,[hot])
                self.assertEqual(h.arena[(idx,hot-184,'u')].v,[cold])
        self.assertEqual(sorted(mp.v),list(range(288)))


class CheckTests(unittest.TestCase):
    def setUp(self):
        if not hasattr(CHECKS,'index_range'): self.skipTest('CHECK is introduced by splitdev2')
    def test_checks_name_layer_tensor_and_bad_range(self):
        for ids in ([-2,0],[-1,104]):
            with self.assertRaisesRegex(RuntimeError,r'layer=L23 tensor=shifted'):
                CHECKS.index_range('L23','shifted',Vec(ids),-1,104)
    def test_module_contract_rejects_stale_worker_before_scatter(self):
        h=host(); idx=register(h,'L23',98)
        m=SimpleNamespace(key='L23',cpu_split_first=184,num_local_experts=184,
                          num_experts=288,cpu_host=h,cpu_layer_idx=idx,gated=True,_split_map=None)
        with self.assertRaisesRegex(RuntimeError,r'layer=L23 tensor=spec.num_experts.*worker=98 module=104'):
            CHECKS.module_layout(m)
    def test_host_check_runs_at_production_prefill_call_site(self):
        h=host(); idx=register(h,'L23',98)
        with patch.object(CHECKS,'ENABLED',True):
            with self.assertRaisesRegex(RuntimeError,r'layer=L23 tensor=selected_experts.cpu_local'):
                prefill(h,idx,184)
    def test_invalid_map_and_exchange_slot_are_rejected(self):
        h=host(); idx=register(h,'L23',104)
        m=SimpleNamespace(key='L23',cpu_split_first=184,num_local_experts=184,
                          num_experts=288,cpu_host=h,cpu_layer_idx=idx,gated=True,_split_map=Vec(range(288)))
        CHECKS.module_layout(m)
        with self.assertRaisesRegex(RuntimeError,'exchange.cpu_local'):
            CHECKS.exchange_slots(m,Vec(range(288)),0,183)
        m._split_map[287]=286
        with self.assertRaisesRegex(RuntimeError,'_split_map'): CHECKS.module_layout(m)
    def test_bad_descriptor_and_aux_counts_are_rejected(self):
        h=host(); idx=register(h,'L23',104)
        m=SimpleNamespace(key='L23',cpu_split_first=184,num_local_experts=184,
                          num_experts=288,cpu_host=h,cpu_layer_idx=idx,gated=True,_split_map=None)
        h.exchange_views=[[None]*98]
        with self.assertRaisesRegex(RuntimeError,'exchange_views'): CHECKS.module_layout(m)
        h.exchange_views=[[None]*104]; h.aux[idx]['suh_u']=[None]*98
        with self.assertRaisesRegex(RuntimeError,'aux.suh_u'): CHECKS.module_layout(m)

    def test_actual_folded_reconstruct_gather_names_layer_and_table(self):
        fn=extract('modules/moe_batch_recon.py','_recon','BatchReconLayer',{'split_check':CHECKS})
        layer=SimpleNamespace(folded=True,check_layer='L23',scale_ptrs={'u':(Vec(range(98)),Vec(range(98)))})
        with patch.object(CHECKS,'ENABLED',True):
            with self.assertRaisesRegex(RuntimeError,r'layer=L23 tensor=ids_d -> u.suh_p.index_select'):
                fn(layer,None,Vec([1]),'u',Vec([103]),2,None)
        linear=extract('modules/moe_batch_recon.py','_linear','BatchReconLayer',
                       {'split_check':CHECKS,'torch':TORCH})
        layer=SimpleNamespace(folded=False,check_layer='L23',scales={'u':(Vec(range(98)),Vec(range(98)))})
        with patch.object(CHECKS,'ENABLED',True):
            with self.assertRaisesRegex(RuntimeError,r'layer=L23 tensor=ids -> u.suh rows'):
                linear(layer,SimpleNamespace(shape=(1,1,128)),None,'u',Vec([103]),128)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tree',default='src'); ap.add_argument('--torch',action='store_true')
    ap.add_argument('--base-tree',default=str(BASE_TREE))
    args,remaining=ap.parse_known_args(); TREE=ROOT/args.tree; BASE_TREE=Path(args.base_tree); REAL_TORCH=args.torch; CHECKS=checks_module()
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)
