#!/usr/bin/env python3
"""Summarize supplied wire traces and replay stock/r861 parsers without inference."""
import argparse
import importlib.util
import json
from pathlib import Path

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--stock',type=Path,required=True)
    ap.add_argument('--r861',type=Path,required=True);ap.add_argument('--r3',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    client=load('trace_client',Path(__file__).parents[1]/'glm53_toolcheck.py')
    report={'wire_traces':[],'parser_replays':[]}
    for path in sorted(a.evidence.glob('*/*.raw.json')):
        case='hard' if path.name.startswith('hard') else 'simple'
        result=client.summarize(json.loads(path.read_text()),'-stream.' in path.name,case,str(path))
        report['wire_traces'].append(result)
    for name,app in (('stock',a.stock),('r861',a.r861),('r3',a.r3)):
        parser=load('parser_'+name,app/'endpoints/OAI/utils/stream_parser.py')
        kwargs={'reasoning_start':'<think>','reasoning_end':'</think>',
                'tool_start':'<tool_call>','tool_end':'</tool_call>','start_in_reasoning':True}
        if name!='stock':kwargs['preserve_tool_reasoning_tags']=True
        if name=='r3':kwargs['preserve_reasoning_literals']=True
        for text in ('before <think>x</think> after</think>Answer',
                     'before `<think>x</think>` after</think>Answer',
                     '<tool_call>f<arg_key>body</arg_key><arg_value><think>x</think></arg_value></tool_call>'):
            p=parser.TagStreamParser(**kwargs);events=[]
            for ch in text:events+=p.feed(ch)
            events+=p.finish()
            channels={c:''.join(t for channel,t in events if channel==c) for c in ('reasoning','content','tool')}
            report['parser_replays'].append({'version':name,'input':text,'channels':channels,
                                            'generation_terminated_by_parser':False})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Saved nine trace summaries and nine real parser replays to',a.out)

if __name__=='__main__':main()
