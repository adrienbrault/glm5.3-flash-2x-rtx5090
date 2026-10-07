#!/usr/bin/env python3
"""CPU-only replay. Raw traces -> call cache -> exhaustive grid -> shift diagnostics.
Run with Python + numpy: python out/sim_policy.py --extract, then --grid.
No GPU imports. All output is incremental under out/.
"""
import argparse, importlib.util, itertools, json, re, sys, time, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r2_cost',OUT/'sim_cost.py')
cost=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cost;spec.loader.exec_module(cost)
CAL=json.loads((OUT/'cost_calibration.json').read_text())

def log(s):
    print(s,flush=True)
    with (OUT/'simulation.log').open('a') as f: f.write(s+'\n');f.flush()

def extract(name,run):
    dest=OUT/(name+'.npz')
    if dest.exists(): return
    files=sorted(Path(run).glob('route-*.npz'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    calls={}; keys=set(); initial={}; actual=0
    for i,f in enumerate(files):
        with zipfile.ZipFile(f) as z:
            with z.open('meta.npy') as mf: m=json.loads(str(np.lib.format.read_array(mf,allow_pickle=False)))
            if m.get('component')!='text' or m.get('kind')!='actual': continue
            if m.get('expert_space')!='checkpoint': raise ValueError(m)
            layer=int(re.search(r'layers\.(\d+)\.',m['layer']).group(1));keys.add(layer)
            with z.open('ids.npy') as q: ids=np.lib.format.read_array(q,allow_pickle=False).astype(np.int16)
            with z.open('positions.npy') as q: pos=np.lib.format.read_array(q,allow_pickle=False)
            with z.open('request_keys.npy') as q: req=np.lib.format.read_array(q,allow_pickle=False)
            if 'cpu_mask.npy' in z.namelist() and layer not in initial:
                with z.open('cpu_mask.npy') as q: initial[layer]=np.lib.format.read_array(q,allow_pickle=False)
            c=calls.setdefault(m['call'],{'phase':m['phase'],'layers':{},'pos':pos,'req':req})
            if layer in c['layers']: raise ValueError('duplicate layer in call')
            c['layers'][layer]=ids;actual+=1
        if i%50000==0: log(f'{name}: {i}/{len(files)} records')
    layers=sorted(keys); seq=[]; prefills=[]; prefill_rows=[]; new=[]; rows=[]; positions=[]
    prompt=np.zeros((len(layers),288),dtype=np.float64); prompt_rows=np.zeros(len(layers),dtype=np.int64); prev=None; discarded=0; multi=0;prefill_calls=0;partial_prefill=0
    allcounts=np.zeros_like(prompt);deccounts=np.zeros_like(prompt); new_prompt=False
    for c in calls.values():
        request=str(c['req'][0]);p=int(c['pos'][0])
        if c['phase']=='prefill':
            prefill_calls+=1;partial_prefill+=int(sorted(c['layers'])!=layers)
            if not new_prompt:
                prompt.fill(0);prompt_rows.fill(0);new_prompt=True
            for l,key in enumerate(layers):
                if key not in c['layers']:continue
                xx=c['layers'][key]
                counts=np.bincount(xx.ravel(),minlength=288)
                prompt[l]+=counts;allcounts[l]+=counts;prompt_rows[l]+=len(xx)
            continue
        if sorted(c['layers'])!=layers: discarded+=1;continue
        x=np.stack([c['layers'][l] for l in layers])
        for l in range(len(layers)): allcounts[l]+=np.bincount(x[l].ravel(),minlength=288)
        if c['phase']!='decode': raise ValueError(c['phase'])
        start=new_prompt or prev is None or request!=prev[0] or p<=prev[1]
        if start:
            if not new_prompt: prompt.fill(0);prompt_rows.fill(0)
            prefills.append(prompt.copy()/np.maximum(prompt_rows[:,None],1));prefill_rows.append(prompt_rows.copy())
        new.append(start);seq.append(x[:,0,:]); rows.append(x.shape[1]);positions.append(p)
        multi+=int(x.shape[1]!=1)
        for l in range(len(layers)): deccounts[l]+=np.bincount(x[l].ravel(),minlength=288)
        prev=(request,p);new_prompt=False
    np.savez_compressed(dest,ids=np.asarray(seq),new=new,prompt=np.asarray(prefills),layers=layers,
                        prior=deccounts/deccounts.sum(axis=1,keepdims=True)*8,
                        allcounts=allcounts,decode_counts=deccounts,rows=rows,positions=positions,prefill_rows=prefill_rows)
    info=dict(files=len(files),actual_records=actual,complete_decode_calls=len(seq),layers=layers,
              generations=sum(new),discarded_decode_calls=discarded,prefill_calls=prefill_calls,partial_prefill_calls=partial_prefill,multirow_decode_calls=multi,
              decode_rows=sum(rows),initial_masks_available=bool(initial),prior='decode all rows, rate per row')
    (OUT/(name+'-trace.json')).write_text(json.dumps(info,indent=2)+'\n');log(str(info))

def load(name):
    with np.load(OUT/(name+'.npz')) as z: return {k:z[k] for k in z.files}

def static_mask(prior,n=96):
    # Matches load_cpu_split hot descending, ties ascending; cold tail reversed ties.
    order=np.argsort(-prior,axis=-1,kind='stable'); mask=np.ones_like(prior,dtype=bool)
    np.put_along_axis(mask,order[:,:288-n],False,axis=1);return mask

@dataclass(frozen=True)
class Policy:
    k:int=32
    H:int=128
    wp:float=0.
    wb:float=64.
    M:int=8
    rho:float=1.5
    init:str='static'


def layer_pairs(mask,score,budget,rho):
    cold=np.flatnonzero(~mask);hot=np.flatnonzero(mask)[::-1]
    cold=cold[np.argsort(score[cold],kind='stable')];hot=hot[np.argsort(-score[hot],kind='stable')]
    out=[]
    for a,b in zip(cold[:budget],hot[:budget]):
        if score[b]<=score[a] or score[b]<rho*score[a]: break
        out.append((int(a),int(b)))
    return out


def replay_group(data,prior,k,H,wp,wb,init):
    """Nine placements share exact per-token decayed counts. Global M, layer-order
    allocation matches r2 run_sweep; unvisited layers still decay at submission.
    Sweep before next call after k preceding decode calls.
    """
    ids=data['ids'];T,L,K=ids.shape
    variants=list(itertools.product((8,16,32),(1.2,1.5,2.)))
    cpu=np.broadcast_to(static_mask(prior) if init=='static' else np.tile(np.arange(288)>=192,(L,1)),(9,L,288)).copy()
    dec=np.zeros((L,288),dtype=np.float32);prompt=np.zeros((L,288))
    picks=np.zeros((9,T));swaps=np.zeros((9,T));sweeps=np.zeros(T);g=-1
    ri=np.arange(L)[:,None]
    for t in range(T):
        # A boundary changes prompt prior; no forced placement round / free TTFT swap.
        if data['new'][t]: g+=1;prompt=data['prompt'][g]
        if t and t%k==0:
            score=dec+wp*prompt+wb*prior
            # Unique integer ranks encode exact production tie order.
            cold_order=np.argsort(score,axis=1,kind='stable')
            hot_order=np.argsort(-score[:,::-1],axis=1,kind='stable');hot_order=287-hot_order
            cr=np.empty_like(cold_order);hr=np.empty_like(hot_order)
            np.put_along_axis(cr,cold_order,np.broadcast_to(np.arange(288),(L,288)),1)
            np.put_along_axis(hr,hot_order,np.broadcast_to(np.arange(288),(L,288)),1)
            a=np.argpartition(np.where(cpu,999,cr),31,axis=2)[:,:,:32]
            b=np.argpartition(np.where(cpu,hr,999),31,axis=2)[:,:,:32]
            a=np.take_along_axis(a,np.argsort(np.take_along_axis(np.broadcast_to(cr,cpu.shape),a,2),axis=2),2)
            b=np.take_along_axis(b,np.argsort(np.take_along_axis(np.broadcast_to(hr,cpu.shape),b,2),axis=2),2)
            cs=np.take_along_axis(np.broadcast_to(score,cpu.shape),a,2);hs=np.take_along_axis(np.broadcast_to(score,cpu.shape),b,2)
            rho=np.array([v[1] for v in variants])[:,None,None];M=np.array([v[0] for v in variants])[:,None,None]
            ok=(hs>cs)&(hs>=rho*cs)
            # Break on first rejection, as production does.
            ok=np.logical_and.accumulate(ok,axis=2)
            ok &= (np.cumsum(ok.reshape(9,-1),axis=1).reshape(ok.shape)<=M)
            v,l,j=np.nonzero(ok);cpu[v,l,a[v,l,j]]=True;cpu[v,l,b[v,l,j]]=False
            swaps[:,t]=ok.sum(axis=(1,2));sweeps[t]=1
        picks[:,t]=cpu[:,ri,ids[t]].sum(axis=(1,2))
        dec*=np.float32(2**(-1/H))
        np.add.at(dec,(ri,ids[t]),1)
    return [(Policy(k,H,wp,wb,M,rho,init),picks[v],swaps[v],sweeps) for v,(M,rho) in enumerate(variants)]


def predicted_ms(r,fence=None,d2h=None):
    return cost.price(r,'exchange',CAL['fixed_ms'],CAL['legacy_effective_ms_per_swap'],
                      exposure=CAL['exposure'],
                      fence_ms=CAL['sweep_overhead_ms_assumed'] if fence is None else fence,
                      d2h_gbs=CAL['d2h_gbs_assumed'] if d2h is None else d2h,
                      expert_mb=CAL['expert_mb'],launch_ms=CAL['launch_ms_assumed'],
                      vram_gbs=CAL['vram_gbs_assumed'],ddr_gbs=CAL['ddr_gbs'])


def metrics(picks,swaps,sweeps,L=42,fence=None,d2h=None):
    r=dict(cpu_picks=float(np.mean(picks)),cpu_share=float(np.mean(picks)/(L*8)),swaps=float(np.mean(swaps)),sweeps=float(np.mean(sweeps)))
    r['predicted_tps']=1000/predicted_ms(r,fence,d2h)
    return r


def histogram(data,prior,p,init='identity',prefill=False):
    ids=data['ids'];new=data['new'];cpu=static_mask(prior) if init=='static' else np.tile(np.arange(288)>=192,(len(prior),1))
    hist=np.zeros_like(prior);picks=[];swaps=[];sweeps=[];tick=0;pending=False;g=-1
    def sweep():
        nonlocal tick,pending
        total=0;b=p.budget
        for l in range(len(prior)):
            if p.scope=='layer': b=p.budget
            if b<=0:break
            h=hist[l];cold=np.flatnonzero(~cpu[l]);hot=np.flatnonzero(cpu[l])[::-1]
            cold=cold[np.argsort(h[cold],kind='stable')];hot=hot[np.argsort(-h[hot],kind='stable')]
            for a,c in zip(cold,hot):
                if b<=0 or h[c]<max(p.hyst*max(h[a],1),p.floor*h.sum()/288):break
                cpu[l,a]=True;cpu[l,c]=False;b-=1;total+=1
            h*=.5
        tick=0;pending=False;return total
    for t,x in enumerate(ids):
        ns=nq=0
        if new[t]:
            g+=1
            if pending and p.cadence=='served':ns+=sweep();nq+=1
        if p.cadence=='exact' and tick>=p.interval:ns+=sweep();nq+=1
        # Include prefill at its arrival as served histogram signal (first decode after prompt).
        if prefill and new[t]: hist+=data['prompt'][g]*data['prefill_rows'][g,:,None]
        tick+=1
        if p.cadence=='served' and tick>=p.interval:
            pending=True
            if tick>=4*p.interval:ns+=sweep();nq+=1
        picks.append(cpu[np.arange(len(prior))[:,None],x].sum())
        np.add.at(hist,(np.arange(len(prior))[:,None],x),1)
        swaps.append(ns);sweeps.append(nq)
    return np.array(picks),np.array(swaps),np.array(sweeps)


def worker_init():
    global _WORKER_A,_WORKER_B
    _WORKER_A,_WORKER_B=load('r869'),load('r860b')


def grid_group(group):
    k,H,wp,wb,init=group;a,b=_WORKER_A,_WORKER_B
    left=replay_group(b,a['prior'],k,H,wp,wb,init);right=replay_group(a,b['prior'],k,H,wp,wb,init)
    rows=[]
    for (p,x,s,q),(_,y,u,v) in zip(left,right):
        r={'policy':asdict(p),'r860b':metrics(x,s,q),'r869':metrics(y,u,v)}
        r['mean_tps']=.5*(r['r860b']['predicted_tps']+r['r869']['predicted_tps']);rows.append(r)
    return rows


def grid():
    a,b=load('r869'),load('r860b');rows=[]
    baselines={}
    for name,data,other in [('r860b',b,a),('r869',a,b)]:
        prior=other['prior'];mask=static_mask(prior);ids=data['ids'];L=len(prior)
        picks=mask[np.arange(L)[:,None],ids].sum(axis=(1,2))
        baselines[name]={'static':metrics(picks,np.zeros(len(ids)),np.zeros(len(ids)),L)}
        for label,p in [('B-safe',cost.Policy(64,4,2,64,'global','exact')),('B-fast',cost.Policy(16,2,1.2,64,'layer','served'))]:
            v=histogram(data,prior,p);baselines[name][label]=metrics(*v,L)
            v=histogram(data,prior,p,'static');baselines[name][label+'-static-init']=metrics(*v,L)
            v=histogram(data,prior,p,prefill=True);baselines[name][label+'-prefill-boundary-approx']=metrics(*v,L)
    (OUT/'baselines.json').write_text(json.dumps(baselines,indent=2)+'\n')
    from concurrent.futures import ProcessPoolExecutor
    result=OUT/'grid.jsonl'
    rows=[json.loads(v) for v in result.read_text().splitlines()] if result.exists() else []
    group_key=lambda p:(p['k'],p['H'],p['wp'],p['wb'],p['init'])
    complete={group_key(r['policy']) for r in rows if sum(group_key(x['policy'])==group_key(r['policy']) for x in rows)==9}
    rows=[r for r in rows if group_key(r['policy']) in complete]
    for r in rows:
        for n in ('r869','r860b'):r[n]['predicted_tps']=1000/predicted_ms(r[n])
        r['mean_tps']=.5*(r['r869']['predicted_tps']+r['r860b']['predicted_tps'])
    result.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    groups=[v for v in itertools.product((8,16,32),(128,512,2048),(0.,8.,32.),(0.,64.,256.),('identity','static')) if v not in complete]
    with ProcessPoolExecutor(max_workers=3,initializer=worker_init) as pool:
        for i,batch in enumerate(pool.map(grid_group,groups)):
            rows.extend(batch)
            with result.open('a') as f:
                for r in batch:f.write(json.dumps(r)+'\n')
                f.flush()
            if i%6==0:log(f'grid groups {len(complete)+i+1}/162; best {max(r["mean_tps"] for r in rows):.3f}')
    best=max(rows,key=lambda r:r['mean_tps']);(OUT/'selected.json').write_text(json.dumps(best,indent=2)+'\n');log('selected '+json.dumps(best))



def concatenate(a,b):
    return {'ids':np.concatenate((a['ids'],b['ids'])),
            'new':np.concatenate((a['new'],b['new'])),
            'prompt':np.concatenate((a['prompt'],b['prompt'])),
            'prefill_rows':np.concatenate((a['prefill_rows'],b['prefill_rows']))}


def recovery(picks,cut,L=42,window=128,steady_window=512):
    share=picks[cut:]/(L*8);steady=float(share[-steady_window:].mean())
    rolling=np.convolve(share,np.ones(window)/window,mode='valid')
    band=np.abs(rolling-steady)<=.05*steady
    # Require 128 successive rolling windows; report end of confirmation window.
    good=np.convolve(band.astype(int),np.ones(window,dtype=int),mode='valid')==window
    at=np.flatnonzero(good)
    upper=rolling<=1.05*steady; good_upper=np.convolve(upper.astype(int),np.ones(window,dtype=int),mode='valid')==window
    au=np.flatnonzero(good_upper)
    return {'terminal_512_share':steady,'first_128_share':float(share[:128].mean()),
            'within_5pct_confirmed_after_steps':int(at[0]+2*window-1) if len(at) else None,
            'no_more_than_5pct_above_confirmed_after_steps':int(au[0]+2*window-1) if len(au) else None,
            'rolling_window':window,'confirmation_windows':window,'lower_share_counts_as_recovered':False}


def diagnostics():
    selected=json.loads((OUT/'selected.json').read_text());p=Policy(**selected['policy'])
    a,b=load('r869'),load('r860b');shift={}
    for name,first,second in [('r869-to-r860b',a,b),('r860b-to-r869',b,a)]:
        data=concatenate(first,second);prior=first['prior'];cut=len(first['ids']);L=len(prior)
        sc=next(v for v in replay_group(data,prior,p.k,p.H,p.wp,p.wb,p.init) if v[0]==p)
        mask=static_mask(prior);static_p=mask[np.arange(L)[:,None],data['ids']].sum(axis=(1,2))
        cases={'score':sc[1:],'static':(static_p,np.zeros(len(static_p)),np.zeros(len(static_p)))}
        for label,policy in [('B-safe',cost.Policy(64,4,2,64,'global','exact')),('B-fast',cost.Policy(16,2,1.2,64,'layer','served'))]:
            cases[label]=histogram(data,prior,policy,'static')
        shift[name]={}
        for label,(x,s,q) in cases.items():
            shift[name][label]={'before':metrics(x[:cut],s[:cut],q[:cut],L),
                                'after':metrics(x[cut:],s[cut:],q[cut:],L),
                                'recovery':recovery(x,cut,L)}
        # Include replay time series so the recovery metric can be independently recomputed.
        np.savez_compressed(OUT/(name+'-series.npz'),cut=cut,**{label.replace('-','_')+'_picks':v[0] for label,v in cases.items()})
    (OUT/'shift.json').write_text(json.dumps(shift,indent=2)+'\n')
    small={}
    for name,data,other in [('r860b',b,a),('r869',a,b)]:
        small[name]={}
        for budget in (8,16,32):
            small[name]['static-small-'+str(budget)]=metrics(*histogram(data,other['prior'],cost.Policy(32,4,2,budget,'global','exact'),'static'))
    (OUT/'small-budget.json').write_text(json.dumps(small,indent=2)+'\n')
    rows=[json.loads(s) for s in (OUT/'grid.jsonl').read_text().splitlines()];sensitivity=[]
    for fence,d2h in itertools.product((.1,10.,50.),(10.,20.,28.8)):
        def reprice(r):
            return np.mean([1000/predicted_ms(r[n],fence,d2h) for n in ('r869','r860b')])
        winner=max(rows,key=reprice)
        sensitivity.append({'fence_ms':fence,'d2h_gbs':d2h,'selected_mean_tps':float(reprice(selected)),
                            'reselected_policy':winner['policy'],'reselected_mean_tps':float(reprice(winner))})
    (OUT/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    for name,data in [('r869',a),('r860b',b)]:
        counts={f'model.language_model.layers.{int(l)}.mlp':c.tolist() for l,c in zip(data['layers'],data['decode_counts'])}
        (OUT/(name+'-split-stats.json')).write_text(json.dumps(counts,indent=2)+'\n')
    # Preserve a real chronological slice and its opposite-trace prior with the packet tests.
    start=98;n=min(512,len(b['ids'])-start);g=int(b['new'][:start+1].sum())-1
    np.savez_compressed(OUT/'replay-slice.npz',ids=b['ids'][start:start+n],new=np.array([True]+[False]*(n-1)),
                        prompt=b['prompt'][g:g+1],prefill_rows=b['prefill_rows'][g:g+1],prior=a['prior'],layers=b['layers'],source_start=start)
    report={'selected':selected,'calibration':CAL,'grid_policies':len(rows),'shift':shift,
            'direction_winners':{n:max(rows,key=lambda r:r[n]['predicted_tps']) for n in ('r869','r860b')},
            'static_all_decode_row_share':{name:float((static_mask(other['prior'])*data['decode_counts']).sum()/data['decode_counts'].sum()) for name,data,other in [('r860b',b,a),('r869',a,b)]}}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');log('diagnostics complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');ap.add_argument('--grid',action='store_true');ap.add_argument('--extract-one');ap.add_argument('--run-r869',type=Path,default=ROOT/'trace-r869/run');ap.add_argument('--run-r860b',type=Path,default=ROOT/'trace-sample/run');ap.add_argument('--diagnostics',action='store_true');args=ap.parse_args()
    if args.extract:
        extract('r869',args.run_r869);extract('r860b',args.run_r860b)
    if args.extract_one:extract(args.extract_one,args.run_r869 if args.extract_one=='r869' else args.run_r860b)
    if args.grid:grid();diagnostics()
    elif args.diagnostics:diagnostics()
