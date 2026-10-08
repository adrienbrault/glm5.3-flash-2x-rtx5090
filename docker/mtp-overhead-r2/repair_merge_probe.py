"""GPU differential probe of repair(q1) + draft(q1) versus a merged causal q2.
Diagnostic ONLY: extra real forwards, snapshot/restore of affected draft-cache pages.
A mismatch falsifies bit-exact folding, even when greedy tokens happen to match.
"""
import json
import os
import torch
from exllamav3.generator import Generator
from exllamav3.constants import PAGE_SIZE


def fresh(params):
    # Per-forward derived data must be recomputed for the changed q_len.
    return {k: params[k] for k in ('cache','attn_mode','block_table','cache_seqlens','target_hidden')}


def install():
    if os.environ.get('EXL3_MOE_CPU_SWAP', '1') != '0':
        raise RuntimeError('repair merge probe requires EXL3_MOE_CPU_SWAP=0 and fixed static placement')
    orig_gen = Generator.iterate_gen
    orig_draft = Generator.iterate_draftmodel_mtp_gen

    def gen(self, *args, **kwargs):
        if (self.mtp_draft and self.draft_window is None and not self.model.loaded_tp and
            len(self.active_jobs) == 1 and not getattr(self, '_repair_probe_installed', False)):
            self._repair_probe_installed = True
            model = self.draft_model
            orig_prefill = model.prefill
            def prefill(input_ids, params=None):
                if (getattr(self, '_repair_probe_in_verify', False) and input_ids.shape == (1,1) and
                    getattr(self, '_repair_probe_count', 0) < 8 and not self._mtp_no_draft):
                    tensors = self.draft_cache.get_all_tensors()
                    pages = self.draft_cache.max_num_tokens // PAGE_SIZE
                    assert all(t.shape[0] == pages for t in tensors), 'unsupported nonpaged draft state'
                    p = fresh(params)
                    for key in ('block_table','cache_seqlens','target_hidden'): p[key] = p[key].clone()
                    start = int(p['cache_seqlens'][0])
                    logical = {start//PAGE_SIZE, (start+1)//PAGE_SIZE}
                    physical = sorted({int(p['block_table'][0,k]) for k in logical})
                    before = [(t, index, t[index].clone()) for t in tensors for index in physical]
                    self._repair_probe_pending = (input_ids.clone(), p, before)
                return orig_prefill(input_ids, params)
            model.prefill = prefill
        self._repair_probe_in_verify = True
        try: return orig_gen(self, *args, **kwargs)
        finally: self._repair_probe_in_verify = False

    def draft(self, *args, **kwargs):
        pending = getattr(self, '_repair_probe_pending', None)
        self._repair_probe_pending = None
        if pending is not None and len(self.active_jobs) == 1 and not self._mtp_no_draft:
            ids0, p0, before = pending
            job = self.active_jobs[0]
            pos = job.sequences[0].kv_position
            if job.mtp_last_hidden is not None and pos == int(p0['cache_seqlens'][0])+1:
                self._repair_probe_count = getattr(self, '_repair_probe_count',0)+1
                postrepair = [(t,k,t[k].clone()) for t,k,_ in before]
                ids1 = job.get_input_ids_list()[0]
                p1 = fresh(p0); p1['cache_seqlens'] = p0['cache_seqlens']+1
                p1['target_hidden'] = job.mtp_last_hidden
                try:
                    with torch.inference_mode():
                        ref = self.draft_model.forward(ids1,p1).clone()
                        refcache = [(t,k,t[k].clone()) for t,k,_ in before]
                        for t,k,v in before: t[k].copy_(v)
                        merged = fresh(p0)
                        merged['target_hidden'] = torch.cat((p0['target_hidden'],job.mtp_last_hidden),dim=1)
                        actual = self.draft_model.forward(torch.cat((ids0,ids1),dim=1),merged)[:, -1:]
                        same_hidden = torch.equal(ref, actual)
                        same_cache = all(torch.equal(t[k],v) for t,k,v in refcache)
                        print('[MTP-REPAIR-PROBE] '+json.dumps({'position':pos,'mod4':pos%4,
                             'hidden_equal':same_hidden,'cache_pages_equal':same_cache,
                             'hidden_max_abs':float((ref.float()-actual.float()).abs().max())}),flush=True)
                finally:
                    # Restore the exact real-path state: the generator will run its normal draft next.
                    with torch.inference_mode():
                        for t,k,v in postrepair: t[k].copy_(v)
        return orig_draft(self,*args,**kwargs)

    Generator.iterate_gen = gen
    Generator.iterate_draftmodel_mtp_gen = draft
