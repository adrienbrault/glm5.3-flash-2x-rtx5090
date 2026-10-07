"""Execute production backend methods with only the tensor/job boundary replaced."""
import ast
import os
import asyncio
from pathlib import Path
import weakref
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
from harness import APP, LOG, glm_tag_safety

class Tensor:
    def __init__(self, ids):
        self.ids=list(ids);self.shape=(1,len(self.ids))
    def flatten(self):return self
    def tolist(self):return list(self.ids)
    def size(self,dim=-1):return self.shape[dim]
    def clone(self):return Tensor(self.ids)
    def new_tensor(self,rows):return Tensor(rows[0])

class Tokenizer:
    bos_token_id=1;bos_token='<s>';eos_token_id=99
    def __init__(self):
        self.calls=[]
        self.pieces={10:'<think>',11:'</think>',99:'<|observation|>'}
    def single_id(self,s):return {v:k for k,v in self.pieces.items()}.get(s)
    def get_id_to_piece_list(self,special=True):
        return [self.pieces.get(i,'') for i in range(100)]
    def encode(self,text,add_bos=False,encode_special_tokens=True,embeddings=None):
        self.calls.append((text,add_bos,encode_special_tokens))
        ids=[1] if add_bos else []
        while text:
            match=next(((tid,p) for tid,p in self.pieces.items() if text.startswith(p)),None)
            if encode_special_tokens and match:
                tid,p=match;ids.append(tid);text=text[len(p):]
            else:
                ids.append(1000+ord(text[0]));text=text[1:]
        return Tensor(ids)
    def decode(self,ids):return ''.join(self.pieces.get(i,chr(i-1000) if i>=1000 else '') for i in ids)
    def decode_(self,ids,special):return self.decode(ids)

def load_methods(app=APP):
    source=app/'backends/exllamav3/model.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExllamaV3Container')
    methods=[n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and
             n.name in {'generate_gen','_encode_prompt','validate_context_length'}]
    async def run_tokenize(size,fn,*a,**kw):return fn(*a,**kw)
    def ignore(*a,**kw):pass
    ns=dict(os=os,glm_tag_safety=glm_tag_safety,weakref=weakref,torch=NS(Tensor=Tensor),
        unwrap=lambda v,default:v if v is not None else default,
        encode_once_enabled=lambda:True,run_tokenize=run_tokenize,
        xlogger=LOG,validate_context_requirements=ignore,asyncio=asyncio,
        CancelledError=asyncio.CancelledError,ContextLengthExceededError=RuntimeError,
        log_prompt=ignore,log_request_start=ignore,log_generation_params=ignore,log_metrics=ignore,
        format_settings=lambda *a:'test',ExLlamaV3Grammar=lambda:NS(filters=[]),
        status_display=NS(add_job=lambda *a:NS(started=ignore,prefill=ignore,generated=ignore),remove_job=ignore),
        ExllamaV3SamplerBuilder=NS(from_params=lambda *a:NS(build=lambda greedy:NS(),settings=[])))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
    return ns

class Params(NS):
    def model_dump(self,**kw):return {'stop':list(self.stop),'temperature':self.temperature}
    def param_source(self,name):return 'default'
    def get_stop_on_loop(self):return None

def params(stops=()):
    return Params(stop=list(stops),temperature=0,add_bos_token=False,max_tokens=1000,min_tokens=0,
        banned_strings=[],json_schema=None,regex_pattern=None,grammar_string=None,
        token_healing=False,logprobs=0,top_logprobs=0,_prompt_ids=None)

class Disconnect:
    async def add_cleanup_task(self,*a):pass
    async def poll(self):pass
    async def finish(self,*a):pass

async def backend_replay(script,p,*,hf_eos=(99,),backend_eos=(),app=APP):
    ns=load_methods(app);tokenizer=Tokenizer();captured={}
    # Decode must match the scripted samples; r3 reused ID 20 for different
    # arbitrary text chunks, which is impossible for a real tokenizer.
    normalized=[];ordinary={}
    for tid,text in script:
        if tid not in (10,11,99):
            tid=ordinary.setdefault(text,200000+len(ordinary))
        tokenizer.pieces[tid]=text
        normalized.append((tid,text))
    script=normalized
    class Job:
        def __init__(self,*a,**kw):
            captured.update(kw);self.queue=asyncio.Queue();self.job=self;self.cancelled=False
        async def cancel(self):self.cancelled=True
        async def __aiter__(self):
            for tid,text in script:
                if tid in captured['stop_conditions'] or text in captured['stop_conditions']:
                    yield dict(stage='streaming',eos=True,eos_reason='stop_token',
                               eos_triggering_token_id=tid,eos_triggering_token_str=text,new_tokens=1)
                    return
                yield dict(stage='streaming',text=text,token_ids=Tensor([tid]),eos=False)
            raise AssertionError('Script must include a real EOS')
    ns['AsyncJob']=Job
    async def encode(*a,**kw):return Tensor([2,3])
    async def recover(ex,job):raise ex
    container=NS(tokenizer=tokenizer,tool_format='glm4_5',hf_model=NS(eos_tokens=lambda:list(hf_eos),add_bos_token=lambda:False),
        config=NS(eos_token_id_list=list(backend_eos)),max_seq_len=8192,
        cache=NS(max_num_tokens=8192),job_max_rq_tokens=lambda _:4096,
        generator=NS(generator=NS(recurrent_cache=None)),active_job_ids={},
        _encode_prompt=encode,_recover_from_generation_error=recover,
        handle_finish_chunk=lambda r,*a:dict(finish_reason='stop',eos_reason=r['eos_reason'],
                                           stop_str=r['eos_triggering_token_str']))
    trace=NS(render=lambda *a,**kw:None,attach=lambda *a:None,finished=lambda *a:None,encoded=lambda *a,**kw:None)
    exl=ModuleType('exllamav3');exl.cache_trace=trace
    with patch.dict('sys.modules',{'exllamav3':exl}):
        chunks=[c async for c in ns['generate_gen'](container,'script','<think>',p,Disconnect())]
    return chunks,captured
