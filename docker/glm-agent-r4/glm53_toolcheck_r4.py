#!/usr/bin/env python3
"""GLM-5.3 tag probe: five repeats per case/mode at temperature 0 and server default.

Save exact requests and timestamped SSE lines, reconstruct calls, and check the
literal file body, including LF separators. No tools are executed. Use --dry-run to inspect the matrix.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

TOOLS = [{'type': 'function', 'function': {'name': 'write_file', 'description': 'Write a file',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                         'required': ['path', 'content']}}}]
# Preserve the original hard prompt byte for byte to compare with the captured failures.
CASES = {
    'simple': 'Write a 20-line Python file /tmp/x.py that prints the numbers 1 to 20, one per line, each with its own print '
              'statement. Use the write_file tool.',
    'hard': 'Use the write_file tool to write /tmp/y.py: 200 lines, line i is `row_i = "a\\\\b \\"q\\" </fake> <think>x</think>"` '
            'with i from 1 to 200. Write every line literally; do not use a loop.',
}
CASES['think-only'] = CASES['hard'].replace('<think>x</think>', '<think>')
CASES['close-only'] = CASES['hard'].replace('<think>x</think>', '</think>')
CASES['1000-line'] = CASES['hard'].replace('/tmp/y.py', '/tmp/long.py').replace('200', '1000')

def expected_lines(case):
    if case == 'simple':return [f'print({i})' for i in range(1,21)]
    line=CASES[case].split('`')[1]
    return [line.replace('row_i',f'row_{i}') for i in range(1,1001 if case=='1000-line' else 201)]

def expected_body(case):
    # Match the former line contract but now specify LF and one final LF.
    return '\n'.join(expected_lines(case)) + '\n'

def request_matrix(a):
    for case in a.cases:
        for mode in a.modes:
            for temperature in a.temperatures:
                for repeat in range(1,a.repeats+1):
                    tag=f'{case}-{mode}-{temperature}-r{repeat:02d}'
                    body={'model':a.model,'stream':mode=='stream',
                          'max_tokens':a.long_max_tokens if case=='1000-line' else a.max_tokens,
                          'reasoning_effort':'low','messages':[{'role':'user','content':CASES[case] + ' Use LF line separators and end the file with exactly one LF newline.'}], 'tools':TOOLS}
                    if mode=='stream':body['stream_options']={'include_usage':True}
                    if temperature=='zero':body['temperature']=0
                    yield tag,case,body

def summarize(raw,stream,case=None,tag=None):
    content=reasoning='';finish=eos_reason=stop_str=None;usage=None;calls={}
    for _,line in raw:
        if stream:
            if not line.startswith('data: {'):continue
            d=json.loads(line[6:])
        else:
            d=json.loads(line)
        usage=d.get('usage') or usage
        for ch in d.get('choices',[]):
            if ch.get('index',0)!=0:continue
            m=ch.get('delta') or ch.get('message') or {}
            content+=m.get('content') or '';reasoning+=m.get('reasoning_content') or ''
            finish=ch.get('finish_reason') or finish
            eos_reason=ch.get('eos_reason') or eos_reason;stop_str=ch.get('stop_str') or stop_str
            for pos,tc in enumerate(m.get('tool_calls') or []):
                call=calls.setdefault(tc.get('index',pos),{'name':None,'arguments':''})
                f=tc.get('function') or {};call['name']=f.get('name') or call['name']
                call['arguments']+=f.get('arguments') or ''
    results=[]
    for index,c in sorted(calls.items()):
        args=c['arguments'];parsed=None;error=None
        try:parsed=json.loads(args)
        except ValueError as exc:error=str(exc)
        object_ok=isinstance(parsed,dict)
        body=parsed.get('content') if object_ok else None
        lines=body.splitlines() if isinstance(body,str) else []
        want=expected_lines(case) if case else None
        mismatch=next((i for i,(got,expected) in enumerate(zip(lines,want or []),1) if got!=expected),None)
        want_bytes=expected_body(case).encode('utf-8') if case else None
        got_bytes=body.encode('utf-8') if isinstance(body,str) else None
        byte_offset=None
        if got_bytes is not None and want_bytes is not None and got_bytes!=want_bytes:
            byte_offset=next((i for i,(got,expected) in enumerate(zip(got_bytes,want_bytes)) if got!=expected),
                             min(len(got_bytes),len(want_bytes)))
        byte_diff=None if byte_offset is None else {
            'offset_zero_based':byte_offset,
            'actual_byte':got_bytes[byte_offset] if byte_offset<len(got_bytes) else None,
            'expected_byte':want_bytes[byte_offset] if byte_offset<len(want_bytes) else None,
            'actual_context_hex':got_bytes[max(0,byte_offset-16):byte_offset+48].hex(),
            'expected_context_hex':want_bytes[max(0,byte_offset-16):byte_offset+48].hex()}
        results.append({'index':index,'tool':c['name'],'args_len':len(args),'args_json_ok':object_ok,
            'args_json_error':error,'body_len':len(body) if isinstance(body,str) else None,
            'line_count':len(lines),'exact_body':got_bytes==want_bytes if want_bytes is not None else None, 'byte_diff':byte_diff,
            'first_mismatch_line':mismatch,'path':parsed.get('path') if object_ok else None,
            'think_count':body.count('<think>') if isinstance(body,str) else 0,
            'close_think_count':body.count('</think>') if isinstance(body,str) else 0,
            'args_head':args[:200],'args_tail':args[-200:]})
    passed=finish=='tool_calls' and len(results)==1 and results[0]['args_json_ok'] and results[0]['tool']=='write_file'
    if stream:passed=passed and any(line.strip()=='data: [DONE]' for _,line in raw)
    if case:
        path='/tmp/x.py' if case=='simple' else '/tmp/long.py' if case=='1000-line' else '/tmp/y.py'
        passed=passed and results[0]['exact_body'] and results[0]['path']==path
    return {'tag':tag,'case':case,'finish':finish,'eos_reason':eos_reason,'stop_str':stop_str,
            'passed':bool(passed),'calls':results,'content_len':len(content),'reasoning_len':len(reasoning),
            'completion_tokens':(usage or {}).get('completion_tokens'),'usage':usage,
            'reasoning_think_count':reasoning.count('<think>'),'reasoning_close_think_count':reasoning.count('</think>'),
            'reasoning_tail':reasoning[-500:],'content_tail':content[-300:],'raw_line_count':len(raw),
            'saw_done':any(line.strip()=='data: [DONE]' for _,line in raw) if stream else None}

def run(a,tag,case,body):
    destination=Path(a.out);request_path=destination/f'{tag}.request.json'
    request_path.write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
    endpoint=a.url.rstrip('/')
    endpoint+= '/chat/completions' if endpoint.endswith('/v1') else '/v1/chat/completions'
    headers={'Content-Type':'application/json'}
    if os.getenv('TABBY_API_KEY'):headers['Authorization']='Bearer '+os.environ['TABBY_API_KEY']
    req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers=headers)
    t0=time.monotonic();raw=[];error=None;error_response=None
    # JSONL preserves partial evidence during hangs/interruption; raw.json matches earlier captures.
    with (destination/f'{tag}.raw.jsonl').open('w') as journal:
        try:
            with urllib.request.urlopen(req,timeout=a.timeout) as response:
                source=response if body['stream'] else [response.read()]
                for line in source:
                    item=[round(time.monotonic()-t0,3),line.decode(errors='replace').rstrip('\n')]
                    raw.append(item);journal.write(json.dumps(item,ensure_ascii=False)+'\n');journal.flush()
        except urllib.error.HTTPError as exc:
            error=f'HTTPError: {exc.code} {exc.reason}'
            error_response=exc.read().decode(errors='replace')
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        finally:(destination/f'{tag}.raw.json').write_text(json.dumps(raw,ensure_ascii=False)+'\n')
    try:result=summarize(raw,body['stream'],case,tag)
    except (ValueError,TypeError,KeyError) as exc:result={'tag':tag,'case':case,'passed':False,'parse_error':str(exc)}
    result.update(temperature=body.get('temperature','server-default'),elapsed_seconds=round(time.monotonic()-t0,3))
    if error:result.update(transport_error=error,passed=False)
    if error_response is not None:result['error_response']=error_response
    (destination/f'{tag}.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RESULT '+json.dumps(result,ensure_ascii=False),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int,default=5);ap.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    ap.add_argument('--modes',nargs='+',choices=['stream','plain'],default=['stream','plain'])
    ap.add_argument('--temperatures',nargs='+',choices=['zero','default'],default=['zero','default'])
    ap.add_argument('--max-tokens',type=int,default=12000);ap.add_argument('--long-max-tokens',type=int,default=40000)
    ap.add_argument('--timeout',type=float,default=1800);ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.repeats<1 or min(a.max_tokens,a.long_max_tokens,a.timeout)<=0:ap.error('counts, budgets and timeout must be positive')
    destination=Path(a.out);destination.mkdir(parents=True,exist_ok=True);results=[]
    for tag,case,body in request_matrix(a):
        if a.dry_run:
            (destination/f'{tag}.request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
        else:
            result=run(a,tag,case,body);results.append(result)
            with (destination/'results.jsonl').open('a') as journal:journal.write(json.dumps(result,ensure_ascii=False)+'\n')
    if a.dry_run:
        print(f'Saved {sum(1 for _ in request_matrix(a))} exact requests; no HTTP calls.');return
    (destination/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)

if __name__=='__main__':main()
