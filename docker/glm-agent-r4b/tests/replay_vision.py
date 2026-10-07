#!/usr/bin/env python3
"""Replay R880's four unchanged vision requests; retain wire, check visible color."""
import argparse,json,re,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--url',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
failed=False
for name,expect in (('red','red'),('blue','blue'),('green','green'),('split','blue')):
    body=(Path(__file__).parent/'fixtures'/('vision-'+name+'.request.json')).read_bytes()
    req=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',data=body,headers={'Content-Type':'application/json'})
    content=''
    with urllib.request.urlopen(req,timeout=1800) as response, (a.out/(name+'.raw.jsonl')).open('w') as journal:
        for raw in response:
            line=raw.decode();journal.write(json.dumps(line)+'\n');journal.flush()
            if not line.startswith('data: {'):continue
            for c in json.loads(line[6:]).get('choices',[]):content+=c.get('delta',{}).get('content','')
    ok=re.findall(r'[a-z]+',content.lower())==[expect];failed|=not ok
    result=dict(name=name,expected=expect,content=content,passed=ok)
    (a.out/(name+'.result.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if failed:raise SystemExit(1)
