"""Exact HTTP probes; stdlib only. Run on each flag configuration and save JSON.

No claim about live model behavior is made until these probes run. History
recovery is best verified with the server's prompt dump in addition to responses.
"""
import argparse
import json
import os
from pathlib import Path
import urllib.request
import urllib.error

p=argparse.ArgumentParser()
p.add_argument('--url',default='http://127.0.0.1:5000/v1')
p.add_argument('--model',default='GLM-5.3-Flash')
p.add_argument('--save',type=Path,required=True)
a=p.parse_args();a.save.mkdir(parents=True,exist_ok=True)
tools=[dict(type='function',function=dict(name=n,description='Return the supplied text',
    parameters=dict(type='object',properties={'text':{'type':'string'},'count':{'type':['integer','null']}},required=['text'])))
    for n in ('echo','other')]
base=dict(model=a.model,messages=[dict(role='user',content='Call echo with text 123 and count null.')],
          tools=tools,temperature=0,max_tokens=512,reasoning_effort='low',enable_thinking=True)
headers={'Content-Type':'application/json'}
if os.getenv('TABBY_API_KEY'):headers['Authorization']='Bearer '+os.environ['TABBY_API_KEY']
def post(name,body):
    (a.save/(name+'.request.json')).write_text(json.dumps(body,ensure_ascii=False,indent=2))
    req=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),headers)
    try:
        with urllib.request.urlopen(req,timeout=180) as r:raw=r.read().decode()
    except urllib.error.HTTPError as exc:
        (a.save/(name+'.response.txt')).write_text(str(exc.code)+'\n'+exc.read().decode())
        raise
    (a.save/(name+'.response.txt')).write_text(raw)
    return raw
for choice in ('none',{'type':'function','function':{'name':'echo'}},'required'):
    body=dict(base,tool_choice=choice,parallel_tool_calls=False)
    name=choice if isinstance(choice,str) else 'named'
    for streaming in (False,True):post(name+('-sse' if streaming else ''),dict(body,stream=streaming))
first=json.loads(post('history-first',dict(base,tool_choice={'type':'function','function':{'name':'echo'}})))
message=first['choices'][0]['message'];calls=message.get('tool_calls') or []
assert calls,'First history probe did not call a tool'
assert message.get('reasoning_content'),'No reasoning generated; increase max_tokens and retry history probe'
for mode in ('preserved','omitted','renumbered','empty'):
    assistant=dict(message)
    if mode in ('omitted','renumbered'):assistant.pop('reasoning_content',None)
    if mode=='empty':assistant['reasoning_content']=''
    assistant=json.loads(json.dumps(assistant))
    if mode=='renumbered':
        for i,c in enumerate(assistant['tool_calls']):c['id']='client_'+str(i)
    results=[dict(role='tool',tool_call_id=c['id'],content='completed') for c in assistant['tool_calls']]
    body=dict(base,messages=base['messages']+[assistant]+results,tool_choice='auto')
    post('history-'+mode,body)
# SSE includes usage and strict append-only id/name/index fields.
post('auto-two-sse',dict(base,messages=[dict(role='user',content='Call echo and other in parallel, each with text 123.')],
                         tool_choice='auto',parallel_tool_calls=False,stream=True,stream_options={'include_usage':True}))
print('Saved exact requests and responses to',a.save)
