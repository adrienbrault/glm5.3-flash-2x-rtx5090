#!/usr/bin/env python3
"""Stdlib HTTP probes for tool_choice and parallel_tool_calls, both response modes."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def decode_sse(raw):
    calls={};finishes={};done=False;comments=0
    for line in raw.splitlines():
        if line.startswith(':'):comments+=1
        if not line.startswith('data:'):continue
        payload=line[5:].strip()
        if payload=='[DONE]':done=True;continue
        frame=json.loads(payload)
        if 'error' in frame:raise AssertionError(frame['error'])
        for choice in frame.get('choices',[]):
            if choice.get('finish_reason'):finishes[choice['index']]=choice['finish_reason']
            for d in choice.get('delta',{}).get('tool_calls',[]):
                target=calls.setdefault((choice['index'],d['index']),dict(function=dict(arguments='')))
                for k in ('id','type'):
                    if k in d:
                        assert k not in target,('repeated metadata',k)
                        target[k]=d[k]
                for k,v in d['function'].items():
                    if k=='arguments':target['function'][k]+=v
                    else:
                        assert k not in target['function'],('repeated metadata',k)
                        target['function'][k]=v
    assert done,'SSE ended without [DONE]'
    return list(calls.values()),list(finishes.values()),comments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default=os.getenv('TABBY_BASE_URL','http://127.0.0.1:5000').rstrip('/')+'/v1')
    p.add_argument('--model',default=os.getenv('TABBY_MODEL'))
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--expect-forcing',action='store_true',help='Require a call for named/required; only after grammar probe passes')
    a=p.parse_args();a.save.mkdir(parents=True,exist_ok=True)
    tools=[dict(type='function',function=dict(name=n,description='Return text',parameters=dict(
        type='object',properties={'text':{'type':'string'},'count':{'type':['integer','null']}},required=['text'])))
        for n in ('echo','other')]
    base=dict(messages=[dict(role='user',content='Call both echo and other, each with text 123 and count null.')],
        tools=tools,temperature=0,max_tokens=4096,reasoning_effort='low')
    if a.model:base['model']=a.model
    headers={'Content-Type':'application/json'}
    if os.getenv('TABBY_API_KEY'):headers['Authorization']='Bearer '+os.environ['TABBY_API_KEY']
    for name,choice,parallel in [('none','none',False),('named',{'type':'function','function':{'name':'echo'}},False),
            ('required','required',False),('auto-single','auto',False),('auto-parallel','auto',True),
            ('required-parallel','required',True)]:
        for streaming in (False,True):
            label=name+('-sse' if streaming else '')
            body=dict(base,tool_choice=choice,parallel_tool_calls=parallel,stream=streaming)
            if streaming:body['stream_options']={'include_usage':True}
            (a.save/(label+'.request.json')).write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
            req=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',json.dumps(body).encode(),headers)
            with urllib.request.urlopen(req,timeout=600) as response:raw=response.read().decode()
            (a.save/(label+'.response.txt')).write_text(raw)
            if streaming:calls,finishes,comments=decode_sse(raw)
            else:
                frame=json.loads(raw);assert 'error' not in frame,frame
                calls=frame['choices'][0]['message'].get('tool_calls') or []
                finishes=[frame['choices'][0]['finish_reason']];comments=0
            if name=='none':assert not calls,(label,calls)
            if name=='named':assert all(c['function']['name']=='echo' for c in calls),(label,calls)
            assert all(c['function']['name'] in ('echo','other') for c in calls),(label,calls)
            if not parallel:assert len(calls)<=1,(label,calls)
            for call in calls:
                assert call.get('id') and call.get('type')=='function',call
                args=json.loads(call['function']['arguments']);assert isinstance(args,dict),args
                if 'text' in args:assert isinstance(args['text'],str),args
            if a.expect_forcing and name in ('named','required','required-parallel'):
                assert calls and finishes==['tool_calls'],(label,calls,finishes)
            print(label,'names=',[c['function']['name'] for c in calls],'finish=',finishes,'comments=',comments,flush=True)
    bad=dict(base,tool_choice={'type':'function','function':{'name':'missing'}})
    try:
        urllib.request.urlopen(urllib.request.Request(a.url.rstrip('/')+'/chat/completions',json.dumps(bad).encode(),headers),timeout=60)
    except urllib.error.HTTPError as exc:
        assert exc.code==400,exc.code
        (a.save/'invalid-named.response.txt').write_text(str(exc.code)+'\n'+exc.read().decode())
    else:raise AssertionError('Invalid named tool_choice was accepted')
    print('Choice/parallel probes passed; saved requests and responses to',a.save)

if __name__=='__main__':main()
