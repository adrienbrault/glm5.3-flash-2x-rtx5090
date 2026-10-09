#!/usr/bin/env python3
"""Text-only HTTP prompts. Flush each request; measure throughput, never cross-run equality."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import urllib.request
import zlib


class Runner:
    def __init__(self,a):
        self.a=a;self.lock=threading.Lock();self.deadline=time.monotonic()+a.minutes*60
        self.output=a.out.open('w',buffering=1);self.prepared={}
        self.requests=a.out.with_suffix('.requests.jsonl').open('w',buffering=1)

    def record_request(self,tag,route,body):
        with self.lock:
            self.requests.write(json.dumps(dict(tag=tag,route=route,body=body),separators=(',',':'))+'\n')
            self.requests.flush()

    def post(self,route,body):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('workload deadline reached')
        headers={'Content-Type':'application/json'}
        if self.a.api_key:headers['Authorization']='Bearer '+self.a.api_key
        req=urllib.request.Request(self.a.base.rstrip('/')+route,json.dumps(body).encode(),headers)
        return urllib.request.urlopen(req,timeout=min(self.a.timeout,remaining))

    def json_post(self,route,body):
        with self.post(route,body) as r:return json.load(r)

    def save(self,row):
        with self.lock:
            self.output.write(json.dumps(row,separators=(',',':'))+'\n');self.output.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('text','message')}),flush=True)

    def prompt(self,label,tokens):
        key=(label,tokens)
        if key in self.prepared:return self.prepared[key]
        text=f'Task {label}: inspect this notebook, explain its invariants, and propose tests.\n'
        text+=''.join(f'def {label.replace("-","_")}_{i:05d}(x): return (x * {i+31}) % 65521\n' for i in range(max(200,tokens//8)))
        ids=self.json_post('/v1/token/encode',{'text':text,'add_bos_token':False})['tokens']
        if len(ids)<tokens:raise ValueError('generated corpus too short')
        # Completion prompts are ALWAYS text; token IDs are only used by the token endpoints.
        text=self.json_post('/v1/token/decode',{'tokens':ids[:tokens],'decode_special_tokens':False})['text']
        n=self.json_post('/v1/token/encode',{'text':text,'add_bos_token':False})['length']
        if abs(n-tokens)>16:raise ValueError(f'text roundtrip length {n}, target {tokens}')
        self.prepared[key]=text
        return text

    def complete(self,tag,prompt,tokens=256,barrier=None):
        if barrier:barrier.wait(timeout=30)
        start=time.monotonic();first=last=None;usage=None;pieces=[];done=False;finish=None
        body=dict(model=self.a.model,prompt=prompt,add_bos_token=False,max_tokens=tokens,min_tokens=tokens,
                  ban_eos_token=True,temperature=0,top_p=1,stream=True,
                  stream_options={'include_usage':True},loop_detect_window=0)
        self.record_request(tag,'/v1/completions',body)
        try:
            with self.post('/v1/completions',body) as response:
                for raw in response:
                    if time.monotonic()>self.deadline:raise TimeoutError('workload deadline reached during stream')
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    if line=='data: [DONE]':done=True;break
                    event=json.loads(line[5:]);usage=event.get('usage') or usage
                    if event.get('error'):raise ValueError(event['error'])
                    for choice in event.get('choices',[]):
                        text=choice.get('text') or ''
                        if text:
                            last=time.monotonic();first=first or last;pieces.append(text)
                        finish=choice.get('finish_reason') or finish
            if not done or not usage or not pieces:raise ValueError('missing DONE, usage, or output')
            if usage.get('completion_tokens')!=tokens:raise ValueError('truncated generation')
            row=dict(tag=tag,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),usage=usage,
                     wall_s=time.monotonic()-start,ttft_s=first-start,finish_reason=finish,text=''.join(pieces))
            row['decode_tps']=usage['completion_tokens']/max(last-first,1e-6)
            self.save(row);return row
        except Exception as e:
            self.save(dict(tag=tag,error=str(e)));raise

    def group(self,tag,lengths,tokens=512):
        prompts=[self.prompt(f'{tag}-row{i}',length) for i,length in enumerate(lengths)]
        barrier=threading.Barrier(len(prompts));start=time.monotonic()
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            rows=list(executor.map(lambda text:self.complete(tag,text,tokens,barrier),prompts))
        self.save(dict(tag=tag+'_group',concurrency=len(prompts),wall_s=time.monotonic()-start,
                       aggregate_tps=sum(r['usage']['completion_tokens'] for r in rows)/(time.monotonic()-start)))

    def chat(self,tag,messages,**extra):
        body=dict(model=self.a.model,messages=messages,max_tokens=16384,temperature=0,stream=False,**extra)  # 1024 truncated agentic_resume's thinking at reasoning effort high (R919 run 3)
        self.record_request(tag,'/v1/chat/completions',body)
        start=time.monotonic()
        try:
            response=self.json_post('/v1/chat/completions',body)
            if response.get('error') or not response.get('choices'):raise ValueError(response)
            choice=response['choices'][0];message=choice['message']
            if not (message.get('content') or message.get('tool_calls')):raise ValueError('empty chat reply')
            if choice.get('finish_reason')=='length':raise ValueError('chat truncated before completion')
            self.save(dict(tag=tag,message=message,usage=response.get('usage'),wall_s=time.monotonic()-start))
            return message
        except Exception as e:self.save(dict(tag=tag,error=str(e)));raise

    def vision(self):
        # Small, deterministic PNG fixture; no external image URL or dependencies.
        width=height=320
        raw=b''.join(b'\0'+b''.join(bytes((220,35,35)) if 80<x<240 and 80<y<240 else bytes((245,245,245))
                                  for x in range(width)) for y in range(height))
        def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
        png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
        url='data:image/png;base64,'+base64.b64encode(png).decode()
        self.chat('vision',[{'role':'user','content':[{'type':'text','text':'Describe the shape, color and position in this image.'},
                                                     {'type':'image_url','image_url':{'url':url}}]}])

    def agentic(self):
        tools=[dict(type='function',function=dict(name='lookup_note',description='Read one notebook entry',
                   parameters=dict(type='object',properties={'key':{'type':'string'}},required=['key'])))]
        messages=[{'role':'user','content':self.prompt('agentic-context',8192)+'\nUse lookup_note with key ring, then summarize the returned note.'}]
        reply=self.chat('agentic_tool',messages,tools=tools,tool_choice={'type':'function','function':{'name':'lookup_note'}})
        calls=reply.get('tool_calls') or []
        if not calls:raise ValueError('agentic probe produced no tool call')
        messages.append(reply)
        for call in calls:
            args=json.loads(call['function']['arguments'])
            if call['function']['name']!='lookup_note' or args.get('key')!='ring':raise ValueError('invalid tool arguments')
            messages.append(dict(role='tool',tool_call_id=call['id'],content='The ring preserves incomplete pools and rewind history.'))
        messages.append(self.chat('agentic_resume',messages,tools=tools,tool_choice='none'))
        messages.append({'role':'user','content':'Now give two test cases for that note.'})
        self.chat('agentic_followup',messages,tools=tools,tool_choice='none')

    def run(self):
        if self.a.mode=='warmup':
            self.complete('warmup',self.prompt('warmup1',8192),128)
            self.group('warmup4',[8192]*4,128);return
        if self.a.mode=='shadow':
            for length in (8192,16384,32768):self.complete('long'+str(length),self.prompt('long'+str(length),length),256)
            prefix=self.prompt('resume-prefix',8193)
            self.complete('resume_warm',prefix+'\nOriginal suffix: explain the notebook.',128)
            for tail in ('A','BB','CCC'):self.complete('resume',prefix+'\nChanged suffix '+tail+': identify three invariants.',128)
            self.group('c4',[8192,16384,32768,8192],512)
            self.group('mixed',[512,1024,32768,32768],1024)
            # Consume more than the pool in distinct prefixes so freed physical pages must be reassigned.
            for i in range(self.a.cache_tokens//32768+2):
                self.complete('churn',self.prompt(f'churn{i}',32768),32)
            self.vision();self.agentic();return
        tasks={'code':'Implement an LRU cache in Python with invariants.',
               'prose':'Write an explanatory report about a field trip.',
               'chat':'Help plan a week with coding, exercise, and meetings.',
               'html':'Create an accessible responsive notes page in HTML and CSS.',
               'edit':'Edit the following document for clarity and preserve its factual details.'}
        for kind,task in tasks.items():
            for repeat in range(2):
                self.complete(f'c1_{kind}',task+'\n'+self.prompt(f'{kind}-{repeat}',8192),512)
        for concurrency in (1,2,4):
            self.group(f'distinct_c{concurrency}',[8192+4096*i for i in range(concurrency)],512)
        self.vision()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('shadow','promotion','warmup'),required=True)
    p.add_argument('--base',default='http://127.0.0.1:8029');p.add_argument('--model',default='glm53-flash-exl3-2.05bpw-turboderp')
    p.add_argument('--api-key',default=os.environ.get('TABBY_API_KEY',''));p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--minutes',type=float,default=50);p.add_argument('--cache-tokens',type=int,default=262144)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();Runner(a).run()
