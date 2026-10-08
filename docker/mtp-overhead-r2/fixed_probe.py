#!/usr/bin/env python3
"""A fixed CPU client workload; use original R892 probes too for comparable speed counters."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import urllib.request

PROMPTS={
'code':'Write a Python module implementing an LRU cache with get and put, a doubly linked list, and unit tests. Explain its invariants. Keep writing complete code and examples.',
'prose':'Describe how a city library works, from acquiring books to helping readers and preserving archives. Write a detailed essay with concrete examples.',
'chat':'Explain to a curious student how to plan a small vegetable garden through the seasons. Give detailed practical examples and discuss tradeoffs.',
'html':'Write a complete accessible HTML page with inline CSS for a library catalogue. Include navigation, search, book cards, responsive styling, and example content.',
'edit':'Refactor this Python loop into a documented function with tests and examples: result = []; for x in values: if x is not None: result.append(x * 2). Discuss behavior on empty inputs and mixed numeric types.'}
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8029');p.add_argument('--c',type=int,default=1);p.add_argument('--rounds',type=int,default=1);p.add_argument('--tokens',type=int,default=1024);p.add_argument('--prompts',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.prompts:PROMPTS=json.loads(a.prompts.read_text())
a.out.mkdir(parents=True,exist_ok=True)
key=os.environ.get('TABBY_API_KEY','')
headers={'Content-Type':'application/json'}
if key:headers['Authorization']='Bearer '+key

def request(kind,prompt,round_index,slot):
    body={'prompt':prompt,'max_tokens':a.tokens,'min_tokens':a.tokens,'temperature':0,'top_k':1,'top_p':1,
          'min_p':0,'repetition_penalty':1,'frequency_penalty':0,'presence_penalty':0,'token_healing':False,'stream':False}
    start=time.perf_counter()
    req=urllib.request.Request(a.url.rstrip('/')+'/v1/completions',json.dumps(body).encode(),headers)
    with urllib.request.urlopen(req,timeout=120) as response: result=json.load(response)
    record={'kind':kind,'round':round_index,'slot':slot,'c':a.c,'wall_s':time.perf_counter()-start,'request':body,'response':result}
    path=a.out/f'{kind}-r{round_index}-s{slot}.json';path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'kind':kind,'round':round_index,'slot':slot,'wall_s':record['wall_s'],'usage':result.get('usage')}),flush=True)
for rnd in range(a.rounds):
    for kind,prompt in PROMPTS.items():
        with ThreadPoolExecutor(max_workers=a.c) as pool:
            futures=[pool.submit(request,kind,prompt,rnd,slot) for slot in range(a.c)]
            for future in futures:future.result()
