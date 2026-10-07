"""Opt-in score selection only. No Torch dependency until an actual layer sweep.
Transaction, reader fencing, payload conversion and poisoning are owned by r2 unchanged.
All IDs remain in original router/checkpoint order; maps encode physical placement.
"""
import math
import os


def settings():
    policy=os.environ.get('EXL3_MOE_CPU_SWAP_POLICY','histogram')
    if policy not in ('histogram','score'):
        raise ValueError('swap policy must be histogram or score')
    if policy!='score':return None
    if os.environ.get('EXL3_MOE_CPU_SWAP_CADENCE','exact')!='exact':
        raise ValueError('score requires exact cadence')
    if os.environ.get('EXL3_MOE_CPU_SWAP_BUDGET_SCOPE','global')!='global':
        raise ValueError('score requires a global swap budget')
    os.environ.setdefault('EXL3_MOE_CPU_SWAP_MAX','32')
    cfg=dict(k=int(os.environ.get('EXL3_MOE_CPU_SWAP_INTERVAL',32)),
             H=float(os.environ.get('EXL3_MOE_CPU_SCORE_HALF_LIFE',512)),
             wp=float(os.environ.get('EXL3_MOE_CPU_SCORE_PREFILL',32)),
             wb=float(os.environ.get('EXL3_MOE_CPU_SCORE_PRIOR',256)),
             rho=float(os.environ.get('EXL3_MOE_CPU_SCORE_HYST',2)),
             M=int(os.environ.get('EXL3_MOE_CPU_SWAP_MAX',32)))
    if cfg['k']<=0 or cfg['M']<0 or not all(math.isfinite(cfg[x]) for x in ('H','wp','wb','rho')) \
            or cfg['H']<=0 or cfg['wp']<0 or cfg['wb']<0 or cfg['rho']<=1:
        raise ValueError('invalid score settings: positive interval/half-life, nonnegative weights/budget, hysteresis >1 required')
    cfg['decay']=2**(-1/cfg['H'])
    return cfg


def validated_counts(counts,experts):
    if len(counts)!=experts:raise ValueError('profile expert count differs')
    out=[float(x) for x in counts]
    if any(not math.isfinite(x) or x<0 for x in out):raise ValueError('profile counts must be finite and nonnegative')
    return out


def rate(counts,topk):
    counts=validated_counts(counts,len(counts));mass=sum(counts)
    return [x*topk/mass for x in counts] if mass else [0.]*len(counts)


def initial_map(order):
    if sorted(order)!=list(range(len(order))):raise ValueError('initial order must be a permutation')
    mp=[0]*len(order)
    for slot,expert in enumerate(order):mp[expert]=slot
    return mp


def scores(decode,prompt_counts,prompt_rows,prior,cfg):
    if not (len(decode)==len(prompt_counts)==len(prior)):raise ValueError('score vector sizes differ')
    denom=max(prompt_rows,1)
    if prompt_rows<=0:prompt_counts=[0.]*len(decode)
    return [float(d)+cfg['wp']*float(p)/denom+cfg['wb']*float(b)
            for d,p,b in zip(decode,prompt_counts,prior)]


def select_pairs(mp,first,score,budget,rho):
    """Stable served tie order, disjoint pairs, strict improvement prevents zero churn.
    Global allocation remains registry ordered in r2 run_sweep.
    """
    if budget<0 or rho<=1:raise ValueError('invalid score selection budget/hysteresis')
    if len(score)!=len(mp) or any(not math.isfinite(float(x)) or float(x)<0 for x in score):
        raise ValueError('invalid score vector')
    head=sorted((float(score[r]),r) for r in range(len(mp)) if int(mp[r])<first)
    tail=sorted(((float(score[r]),r) for r in range(len(mp)) if int(mp[r])>=first),reverse=True)
    out=[]
    for (cold,a),(hot,b) in zip(head,tail):
        if len(out)>=budget or hot<=cold or hot<rho*cold:break
        out.append((a,b))
    return out


def prepare_signal(module,is_prefill,rows):
    """The presence of params['prefill'] marks LS prefill, including short chunks.
    Router/map kernels already count uses. Redirect those counts to a separate fixed GPU
    vector for prefill. No router ID D2H or new histogram/scatter kernel is added.
    """
    module._score_is_prefill=bool(is_prefill)
    if is_prefill:
        if not module._score_previous_prefill:
            module._score_prompt_hist.zero_();module._score_prompt_rows=0
        module._score_prompt_rows+=int(rows)
        module._score_count_hist=module._score_prompt_hist
    else:
        module._score_count_hist=module._split_hist
    module._score_previous_prefill=bool(is_prefill)


def sweep_layer(module,budget):
    """Called only inside r2's fenced, healthy-worker, fail-closed sweep boundary."""
    import torch
    mp=module._split_map.cpu()
    dec=module._split_hist.cpu().tolist()
    prompt=module._score_prompt_hist.cpu().tolist()
    cfg=module._score_settings
    value=scores(dec,prompt,module._score_prompt_rows,module._split_stats_prior,cfg)
    pairs=select_pairs(mp,module.cpu_split_first,value,budget,cfg['rho'])
    n=0
    for cold,hot in pairs:
        if module._split_swap_experts(cold,hot,mp):n+=1
    if n:
        if os.environ.get('EXL3_MOE_CPU_SWAP_VERIFY'):
            assert mp.sort().values.equal(torch.arange(module.num_experts)), \
                f'{module.key}: placement map is not a permutation after sweep'
        module._split_map.copy_(mp.to(module._split_map.device))
    # No sweep-based histogram decay: each decode submit multiplies by the H-token decay.
    return n
