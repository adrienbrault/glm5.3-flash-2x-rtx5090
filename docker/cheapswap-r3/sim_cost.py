#!/usr/bin/env python3
"""Chronological placement replay with calibrated, explicitly conditional time model.
Run: python sim_cost.py ../trace-sample/run --output results.json
Needs numpy; cache conversion takes time on the 456k-file trace. No GPU needed.
"""
import argparse, glob, json, os, re, itertools, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Policy:
    interval: int = 128
    floor: float = 8.0
    hyst: float = 2.0
    budget: int = 64
    scope: str = 'global'
    cadence: str = 'served'


def load_trace(run, cache):
    if Path(cache).exists():
        z = np.load(cache)
        return z['ids'], z['new'], z['initial'], z['layers'], int(z['records'])
    files = sorted(glob.glob(str(Path(run) / 'route-*.npz')),
                   key=lambda f: int(re.search(r'-(\d+)\.npz$', f).group(1)))
    steps, initial = [], {}; first = None; last_pos = None; discarded = 0
    for i, f in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            m = json.loads(str(z['meta']))
            if m.get('component') != 'text' or m.get('phase') != 'decode' or m.get('kind') != 'actual':
                continue
            L = int(re.search(r'layers\.(\d+)\.', m['layer']).group(1))
            if L not in initial: initial[L] = z['cpu_mask'].astype(bool)
            if first is None: first = L
            if L == first:
                pos = int(z['positions'][0]); new = last_pos is None or pos <= last_pos
                steps.append({'new': new, 'ids': {}}); last_pos = pos
            # Match sim_placement's first-row semantics, but reject incomplete steps below.
            if steps: steps[-1]['ids'][L] = z['ids'][0].astype(np.int16)
        if i % 25000 == 0: print(f'loaded {i}/{len(files)} files', flush=True)
    layers = sorted(initial)
    complete = [s for s in steps if set(s['ids']) == set(layers)]
    discarded = len(steps) - len(complete)
    print(f'complete steps={len(complete)} discarded={discarded}', flush=True)
    ids = np.array([[s['ids'][L] for L in layers] for s in complete])
    new = np.array([s['new'] for s in complete]); init = np.array([initial[L] for L in layers])
    np.savez_compressed(cache, ids=ids, new=new, initial=init, layers=layers, records=len(files))
    return ids, new, init, np.array(layers), len(files)


def replay(ids, new, initial, p):
    T, L, K = ids.shape; E = initial.shape[1]; cut = T // 2
    cpu = initial.copy(); hist = np.zeros_like(cpu, dtype=float)
    picks = np.zeros(T); swaps = np.zeros(T); sweeps = np.zeros(T)
    tick = 0; pending = False
    def sweep(t):
        nonlocal tick, pending
        b = p.budget; sweeps[t] += 1
        for l in range(L):
            if p.scope == 'layer': b = p.budget
            if b <= 0: break  # Faithful served decay: unvisited layers are not decayed.
            h = hist[l]; fl = p.floor * h.sum() / E
            # Same tie ordering as served: cold IDs ascending, hot IDs descending.
            ce=np.flatnonzero(~cpu[l]); he=np.flatnonzero(cpu[l])[::-1]
            ce=ce[np.argsort(h[ce],kind='stable')]
            he=he[np.argsort(-h[he],kind='stable')]
            q=min(len(ce),len(he),b)
            stop=np.flatnonzero(h[he[:q]] < np.maximum(p.hyst*np.maximum(h[ce[:q]],1.),fl))
            n=int(stop[0]) if len(stop) else q
            cpu[l,ce[:n]]=True; cpu[l,he[:n]]=False
            swaps[t]+=n; b-=n
            h *= .5
        tick = 0; pending = False
    rows = np.arange(L)[:, None]
    for t in range(T):
        if p.cadence == 'served' and new[t] and pending: sweep(t)
        # Exchange: sweep before the first layer of the next call, after exactly interval calls.
        if p.cadence == 'exact' and tick >= p.interval: sweep(t)
        picks[t] = cpu[rows, ids[t]].sum()
        np.add.at(hist, (rows, ids[t]), 1)
        tick += 1
        if p.cadence == 'served' and tick >= p.interval:
            pending = True
            if tick >= 4 * p.interval: sweep(t)
    def part(s):
        return dict(cpu_picks=float(picks[s].mean()), cpu_share=float(picks[s].mean()/(L*K)),
                    swaps=float(swaps[s].mean()), sweeps=float(sweeps[s].mean()))
    return dict(policy=asdict(p), train=part(slice(0,cut)), test=part(slice(cut,None)))


def price(r, mode, fixed_ms, legacy_ms, overlap=False, expert_mb=6.33, d2h_gbs=28.8,
          fence_ms=10., launch_ms=.08, vram_gbs=600., ddr_gbs=60., exposure=1., sync_ms=1.):
    # ms: MB / GB/s. Every exchange has H2D AND D2H, unlike today's two checkpoint reads.
    cpu = r['cpu_picks'] * (.36/2.75)
    serial_dma = expert_mb/28.8 + expert_mb/d2h_gbs
    convert = 4*expert_mb/vram_gbs  # two lossless permutations, one read + write each
    copies = r['swaps'] * (serial_dma + convert + launch_ms)
    if mode == 'today':
        # Today's promotion is H2D; demotion comes from checkpoint, so there is no D2H.
        checkpoint_and_cpu_repack=max(0.,legacy_ms-expert_mb/28.8)
        return fixed_ms + exposure*cpu + r['swaps']*(expert_mb/28.8+checkpoint_and_cpu_repack) + r['sweeps']*sync_ms
    if not overlap:
        return fixed_ms + exposure*cpu + copies + r['sweeps']*fence_ms
    # Ideal overlap bound, constrained by shared DDR; no extra CPU memcpy in this design.
    ddr = (r['cpu_picks']*expert_mb + r['swaps']*2*expert_mb)/ddr_gbs
    return fixed_ms + exposure*cpu + max(0.,copies-(1-exposure)*cpu,ddr-cpu) + r['sweeps']*fence_ms


def frontier(rows, mode):
    key = mode+'_train_tps'; out=[]; best=-1
    for r in sorted(rows, key=lambda r:r['train']['swaps']):
        if r[key] > best:
            out.append(dict(policy=r['policy'],train=r['train'],test=r['test'],
                            train_tps=r[key],test_tps=r[mode+'_test_tps'])); best=r[key]
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('run')
    ap.add_argument('--cache',default='routes-cache.npz'); ap.add_argument('--output',default='results.json')
    ap.add_argument('--split',type=int,default=96); ap.add_argument('--d2h-gbs',type=float,default=28.8)
    ap.add_argument('--expert-mb',type=float,default=6.33)
    ap.add_argument('--fence-ms',type=float,default=10.,help='total policy-selection + event fence/commit overhead per exchange sweep; unmeasured')
    ap.add_argument('--sync-ms',type=float,default=1.,help='today all-device synchronization per sweep; unmeasured')
    ap.add_argument('--launch-ms',type=float,default=.08); ap.add_argument('--vram-gbs',type=float,default=600.)
    ap.add_argument('--ddr-gbs',type=float,default=60.); a=ap.parse_args()
    ids,new,init,layers,records=load_trace(a.run,a.cache)
    # Validate exact trace masks at N=104, then normalize to served N=96 identity tail.
    validation=replay(ids,new,init,Policy())
    initial=np.zeros_like(init); initial[:,-a.split:]=True
    base=replay(ids,new,initial,Policy()); alt=replay(ids,new,initial,Policy(interval=64,floor=4))
    # Calibrate measured A/B mean (same prompts): separate serial CPU saving from total overhead.
    def avg(r,k): return r['train'][k]  # Fit only on chronological first half.
    fullfreq=np.zeros_like(initial,dtype=int)
    for t in ids[:len(ids)//2]: np.add.at(fullfreq,(np.arange(len(layers))[:,None],t),1)
    fullstatic=np.zeros_like(initial)
    for l in range(len(layers)): fullstatic[l,np.argsort(fullfreq[l],kind='stable')[:a.split]]=True
    static_picks=float(fullstatic[np.arange(len(layers))[:,None],ids[:len(ids)//2]].sum()/(len(ids)//2))
    # Three conditional anchors: served A, more-frequent B, fitted-static C. The
    # profile is compute demand; only a fraction is exposed because GPU work overlaps it.
    matrix=np.array([[1.,avg(base,'cpu_picks'),avg(base,'swaps')],
                     [1.,avg(alt,'cpu_picks'),avg(alt,'swaps')], [1.,static_picks,0.]])
    target=np.array([1000/48.75-a.sync_ms*avg(base,'sweeps'),1000/39.1-a.sync_ms*avg(alt,'sweeps'),1000/63.2])
    fixed_ms,exposed_ms_per_pick,legacy_ms=np.linalg.solve(matrix,target)
    exposure=exposed_ms_per_pick/(.36/2.75)
    profile_legacy=((1000/39.1-1000/48.75)-(.36/2.75)*(avg(alt,'cpu_picks')-avg(base,'cpu_picks'))) / (avg(alt,'swaps')-avg(base,'swaps'))
    profile_fixed=1000/48.75-(.36/2.75)*avg(base,'cpu_picks')-profile_legacy*avg(base,'swaps')
    policies={Policy(),Policy(budget=0)}
    for interval,floor,hyst,scope,budget,cadence in itertools.product(
        (8,16,32,64,128),(0.,1.,2.,4.,8.),(1.2,2.),('global','layer'),(8,64,512),('served','exact')):
        if scope=='layer' and budget==512: continue
        policies.add(Policy(interval,floor,hyst,budget,scope,cadence))
    rows=[]
    for i,p in enumerate(sorted(policies,key=lambda p:tuple(asdict(p).values()))):
        r=replay(ids,new,initial,p)
        for mode in ('today','exchange','overlap'):
            for split in ('train','test'):
                ms=price(r[split], 'exchange' if mode=='overlap' else mode, fixed_ms, legacy_ms,
                         mode=='overlap',a.expert_mb,a.d2h_gbs,fence_ms=a.fence_ms,launch_ms=a.launch_ms,vram_gbs=a.vram_gbs,ddr_gbs=a.ddr_gbs,exposure=exposure,sync_ms=a.sync_ms)
                r[mode+'_'+split+'_tps']=1000/ms
                r['profile_'+mode+'_'+split+'_tps']=1000/price(r[split], 'exchange' if mode=='overlap' else mode,
                    profile_fixed,profile_legacy,mode=='overlap',a.expert_mb,a.d2h_gbs,fence_ms=a.fence_ms,launch_ms=a.launch_ms,vram_gbs=a.vram_gbs,ddr_gbs=a.ddr_gbs,sync_ms=0.)
        rows.append(r)
        if i%50==0:
            print(f'policies {i}',flush=True)
            Path(a.output+'.partial').write_text(json.dumps(rows))
    # Today's mechanism cannot execute exact cadence or a per-layer budget without modification.
    legal=lambda r: r['policy']['cadence']=='served' and r['policy']['scope']=='global'
    best={mode:max([r for r in rows if legal(r) or mode!='today'],key=lambda r:r[mode+'_train_tps'])
          for mode in ('today','exchange','overlap')}
    # Static trained counts is a useful out-of-sample comparator; no held-out route peek.
    fit=np.zeros_like(initial,dtype=int)
    for t in ids[:len(ids)//2]: np.add.at(fit,(np.arange(len(layers))[:,None],t),1)
    static=np.zeros_like(initial)
    for l in range(len(layers)): static[l,np.argsort(fit[l],kind='stable')[:a.split]]=True
    static_r={s:dict(cpu_picks=float(static[np.arange(len(layers))[:,None],v].sum()/len(v)),swaps=0.,sweeps=0.)
              for s,v in [('train',ids[:len(ids)//2]),('test',ids[len(ids)//2:])]}
    result=dict(steps=len(ids),records=records,layers=layers.tolist(),split=a.split,
                validation_trace_split=validation,calibration=dict(fixed_ms=fixed_ms,legacy_effective_ms_per_swap=legacy_ms,
                d2h_gbs_assumed=a.d2h_gbs,expert_mb=a.expert_mb, exposure=exposure,
                exposed_ms_per_pick=exposed_ms_per_pick,static_training_fit_picks=static_picks,
                profile_fixed_ms=profile_fixed,profile_legacy_ms=profile_legacy,sweep_overhead_ms_assumed=a.fence_ms,sync_ms_assumed=a.sync_ms,launch_ms_assumed=a.launch_ms,vram_gbs_assumed=a.vram_gbs,ddr_gbs=a.ddr_gbs),baseline=base,best_fit_first_half=best,
                static_train_fit=static_r,frontiers={m:frontier([r for r in rows if legal(r) or m!='today'],m)
                for m in ('today','exchange','overlap')},grid=rows)
    Path(a.output).write_text(json.dumps(result,indent=2)); Path(a.output+'.partial').unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ('grid','frontiers')},indent=2))

if __name__=='__main__': main()
