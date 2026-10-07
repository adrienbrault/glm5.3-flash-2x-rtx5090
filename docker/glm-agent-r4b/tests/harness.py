"""CPU seam: real request models, template environment, parser and collector source.

Only GPU-bearing imports / model generation are replaced. No copied parser logic.
"""
import ast
import asyncio
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace as NS
from typing import *
from unittest.mock import patch
from pydantic import BaseModel
from jinja2 import TemplateError, nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

APP = Path(os.environ.get('TOOLFIX_APP', '/tmp/tabby-toolfix-work')).resolve()
sys.path.insert(0, str(APP))
LOG = NS(**{name: lambda *a, **kw: None for name in ('info','debug','warning','error')})
class Sampler(BaseModel):
    regex_pattern: Optional[str] = None
    grammar_string: Optional[str] = None
    max_tokens: Optional[int] = 100
    def get_stop_on_loop(self): return None
sampler = ModuleType('common.sampling')
sampler.BaseSamplerRequest = Sampler
sampler.get_default_sampler_value = lambda name, fallback: fallback
sys.modules['common.sampling'] = sampler
logger = ModuleType('common.logger'); logger.xlogger = LOG; sys.modules['common.logger'] = logger
from endpoints.OAI.types.chat_completion import ChatCompletionMessage, ChatCompletionRequest
from endpoints.OAI.types.tools import NamedToolChoice, ToolSpec, ToolCall, Tool
from endpoints.OAI.utils import glm_tool_fixes as fixes
from common import glm_tag_safety
from endpoints.OAI.utils.toolcall_formats import glm4_5 as glm
from endpoints.OAI.utils.toolcall_formats.glm4_5_stream import GLMToolCallStream, stream_toolcalls_enabled
from endpoints.OAI.utils.stream_parser import TagStreamParser, CONTENT, REASONING

class HTTPException(Exception):
    def __init__(self, status_code, detail): self.status_code, self.detail = status_code, detail

def functions(path, names, ns, classes=()):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))
                and node.name in set(names) | set(classes)]
    assert len(selected) == len(set(names) | set(classes))
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)] + selected,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns

UTIL = APP/'endpoints/OAI/utils'
choice = dict(json=json, os=os, re=re, dataclass=dataclass, glm_fixes=fixes, xlogger=LOG,
              HTTPException=HTTPException, NamedToolChoice=NamedToolChoice, qwen3_coder=object(),
              ALL_TOOLCALL_FORMATS={}, get_toolcall_tags=lambda f: ('<tool_call>','</tool_call>'),
              handle_request_error=lambda msg, **kw: NS(error=NS(message=msg)), _MAX_WS=8,
              _PARSEABLE_NAME=re.compile(r'[^>\s]+'), DEFAULT_CONTENT_MAX_TOKENS=1024,
              CONTENT_MAX_TOKENS_ENV='TABBY_TOOL_CHOICE_CONTENT_MAX_TOKENS',
              DEFAULT_REASONING_CAP=16384, REASONING_CAP_ENV='TABBY_TOOL_CHOICE_REASONING_CAP')
choice_names = {n.name for n in ast.parse((UTIL/'tool_choice.py').read_text()).body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
functions(UTIL/'tool_choice.py', choice_names, choice)

def environment(template):
    ns=dict(json=json, datetime=datetime, TemplateError=TemplateError, nodes=nodes,
            Extension=Extension, loopcontrols=loopcontrols, ImmutableSandboxedEnvironment=ImmutableSandboxedEnvironment)
    functions(APP/'common/templating.py', {'_tojson_compat','_strftime_now','_raise_exception','_create_environment'},
              ns, {'_GenerationTagExtension'})
    return ns['_create_environment']().from_string(Path(template).read_text())

class Model:
    harmony = muse_glimmer = use_vision = False
    reasoning = True
    reasoning_start_token, reasoning_end_token = '<think>', '</think>'
    tool_format = 'glm4_5'
    tool_calls_in_reasoning = True
    reasoning_budget_tokens = None
    reasoning_budget_message = None
    template_vars_default = {}; template_vars_force = {}
    hf_model = NS(add_bos_token=lambda: False)
    def __init__(self, chunks=(), eos='eos', template=None):
        self.chunks, self.eos = chunks, eos
        if template: self.prompt_template = NS(render=lambda kw: template.render_async(**kw))
    def get_special_tokens(self): return {}
    async def stream_generate(self, *a, **kw):
        for text in self.chunks: yield dict(text=text, token_ids=[1])
        yield dict(text='', finish_reason='length' if self.eos=='max_new_tokens' else 'stop', eos_reason=self.eos)

def namespace(mc, app=APP):
    from endpoints.OAI.utils.sse_keepalive import keepalive_seconds
    ns = dict(glm_tag_safety=glm_tag_safety, keepalive_seconds=keepalive_seconds, asyncio=asyncio, json=json, model=NS(container=mc), xlogger=LOG, glm_fixes=fixes,
              CONTENT=CONTENT, REASONING=REASONING, TagStreamParser=TagStreamParser,
              GLMToolCallStream=GLMToolCallStream, stream_toolcalls_enabled=stream_toolcalls_enabled,
              parse_toolcalls=lambda text, fmt: glm.parse_toolcalls(text),
              get_toolcall_tags=lambda f: ('<tool_call>','</tool_call>'),
              LoopDetector=None, ChatCompletionLogprobs=lambda **kw: NS(**kw),
              unwrap=lambda val, fallback: val if val is not None else fallback,
              MultimodalEmbeddingWrapper=lambda: None, HTTPException=HTTPException,
              TemplateError=TemplateError, handle_request_error=choice['handle_request_error'],
              time=lambda:1234, CONTINUE_FINAL_MESSAGE_TAG='CONTINUE_FINAL_MESSAGE_TAG ')
    ns.update({name: value for name,value in choice.items() if callable(value) and name not in ('get_toolcall_tags',)})
    names = {'_chat_stream_collector','_parse_tool_calls','_finish_with_tool_calls',
             '_resolve_reasoning_budget','_reasoning_budget_injection','_compose_serialize_stream_chunk',
             '_sort_tool_messages','format_messages_with_template','resolve_template_vars',
             'normalize_message_roles','apply_chat_template','_mark_continued_final_message',
             '_cut_prompt_at_continue_tag'}
    functions(app/'endpoints/OAI/utils/chat_completion.py',names,ns)
    return ns

def tool(name='f', props=None):
    return dict(type='function',function=dict(name=name,description='Test function',
                 parameters=dict(type='object',properties=props or {})))

def call(name='f', pairs=()):
    return '<tool_call>'+name+''.join('<arg_key>'+k+'</arg_key><arg_value>'+v+'</arg_value>' for k,v in pairs)+'</tool_call>'

async def collect(raw, data=None, mc=None, streaming=False, live=False, enabled=True, app=APP, width=1):
    mc=mc or Model([raw[i:i+width] for i in range(0,len(raw),width)])
    data=data or ChatCompletionRequest(messages=[dict(role='user',content='go')],tools=[tool('f')])
    queue=asyncio.Queue(); ns=namespace(mc,app)
    # Isolate generation, while separately testing the real forcing builder.
    async def replay(*a,**kw):
        async for chunk in mc.stream_generate(): yield chunk
    ns['forced_tool_generation']=replay
    with patch.dict(os.environ, {'TABBY_GLM_TOOL_FIXES':'1' if enabled else '0',
                                 'TABBY_STREAM_TOOLCALLS':'1' if live else '0'}):
        result=await ns['_chat_stream_collector'](0,queue,'req','<think>',data,True,
                                                 streaming_mode=streaming,label='test')
    frames=[]
    while not queue.empty():
        frame=queue.get_nowait()
        if isinstance(frame,Exception): raise frame
        frames.append(frame)
    if isinstance(result,Exception): raise result
    return result,frames

def assembled(frames):
    calls={}
    for frame in frames:
        for d in frame.get('delta_tool_calls') or []:
            target=calls.setdefault(d['index'],dict(function=dict(arguments='')))
            for k in ('id','index','type'):
                if k in d: target[k]=d[k]
            for k,v in d['function'].items():
                if k=='arguments':target['function']['arguments']+=v
                else:target['function'][k]=v
    return list(calls.values())
