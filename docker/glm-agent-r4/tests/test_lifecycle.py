"""Real SSE/AnyIO consuming the production chat/completion source collectors."""
import asyncio
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import test_overlay as h
from sse_starlette import sse

class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        watcher=getattr(sse,'_ensure_watcher_started_on_this_loop',None)
        if watcher:
            watcher()
            await asyncio.sleep(0)

    async def test_disconnect_cancels_and_joins_collectors(self):
        for endpoint in ('chat','completion'):
            for kind in ('comment','data'):
                with self.subTest(endpoint=endpoint,kind=kind):
                    baseline=asyncio.all_tasks()
                    closed=[];entered=[];sending=asyncio.Event()
                    class BlockingModel(h.Model):
                        async def stream_generate(self,request_id,*a,**kw):
                            entered.append(request_id)
                            try:
                                if kind=='data':
                                    yield dict(text='answer',token_ids=[1])
                                await asyncio.Event().wait()
                            finally:
                                closed.append(request_id)
                    funcs=h.collector(h.APP/h.UTIL/'chat_completion.py',BlockingModel([]))
                    funcs.update(CancelledError=asyncio.CancelledError,
                        ContextLengthExceededError=type('ContextError',(Exception,),{}),
                        _resolve_start_in_reasoning=lambda *a:False,
                        _gen_label=lambda *a:'r2',_parse_gen_request_id=lambda n,r,i:r+str(i),
                        request_tag=lambda r:'#r2',get_generator_error=lambda *a:'error',
                        UsageStats=h.Stats,CompletionTokensDetails=h.Stats,PromptTokensDetails=h.Stats)
                    h.source_functions(h.BASE/h.UTIL/'common_.py',{'get_usage_stats','aggregate_usage_stats'},funcs)
                    source_file='chat_completion.py' if endpoint=='chat' else 'completion.py'
                    names={'stream_generate_chat_completion','_compose_serialize_stream_usage_chunk'} if endpoint=='chat' else {
                        'stream_generate_completion','_stream_collector','_compose_serialize_stream_chunk','_compose_serialize_stream_usage_chunk'}
                    h.source_functions(h.APP/h.UTIL/source_file,names,funcs)
                    data=h.params();data.n=2;data.max_tokens=100;data.stream_options=NS(include_usage=False)
                    data.model_copy=lambda **kw:data;data.model_dump=lambda **kw:{}
                    dh=h.Disconnect(abort_event=asyncio.Event(),disconnected=False)
                    request=NS(state=NS(id='req'))
                    if endpoint=='chat':
                        source=funcs['stream_generate_chat_completion']('',None,data,request,Path('model'),dh)
                    else:
                        source=funcs['stream_generate_completion']('',data,request,Path('model'),dh)
                    async def send(message):
                        if message['type']=='http.response.body' and message.get('body'):
                            sending.set()
                            await asyncio.Event().wait()
                    async def receive():
                        await sending.wait()
                        dh.disconnected=True;dh.abort_event.set()
                        return {'type':'http.disconnect'}
                    with patch.dict(os.environ,{'TABBY_SSE_KEEPALIVE_S':'.001','TABBY_GLM_TOOL_FIXES':'0'}):
                        response=h.SSE.keepalive_response(source,dh,ping=123)
                        await asyncio.wait_for(response(dict(type='http',asgi={'version':'3.0','spec_version':'2.4'}),receive,send),2)
                    for _ in range(5):await asyncio.sleep(0)
                    self.assertEqual(len(entered),2)
                    self.assertCountEqual(closed,entered)
                    self.assertTrue(dh.cleaned)
                    self.assertEqual(asyncio.all_tasks()-baseline,set())

    async def test_send_error_and_external_cancellation_close_iterator(self):
        for fail in (False,True):
            baseline=asyncio.all_tasks();closed=asyncio.Event();sending=asyncio.Event()
            async def source():
                try:
                    yield 'payload'
                    await asyncio.Event().wait()
                finally:closed.set()
            dh=h.Disconnect(abort_event=asyncio.Event(),disconnected=False)
            async def send(message):
                if message['type']=='http.response.body' and message.get('body'):
                    sending.set()
                    if fail:raise RuntimeError('broken transport')
                    await asyncio.Event().wait()
            async def receive():await asyncio.Event().wait()
            with patch.dict(os.environ,{'TABBY_SSE_KEEPALIVE_S':'.001'}):
                response=h.SSE.keepalive_response(source(),dh,ping=123)
                task=asyncio.create_task(response(dict(type='http',asgi={'version':'3.0','spec_version':'2.4'}),receive,send))
                await asyncio.wait_for(sending.wait(),2)
                if not fail:task.cancel()
                try:await task
                except asyncio.CancelledError:pass
                except Exception:pass
            for _ in range(5):await asyncio.sleep(0)
            self.assertTrue(closed.is_set());self.assertTrue(dh.cleaned)
            self.assertEqual(asyncio.all_tasks()-baseline,set())

class NullKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_serialization_collision_order_and_extra_values(self):
        cases=[
            h.call('f',[('null','explicit')])[:-12]+'<arg_value>extra</arg_value></tool_call>',
            h.call('f',[('a','first'),('null','explicit')])[:-12]+'<arg_value>old</arg_value><arg_value>last</arg_value></tool_call>',
            h.call('f',[('null','explicit'),('a','last')])[:-12]+'<arg_value>{"nested":[1]}</arg_value></tool_call>',
            '<tool_call>f<arg_value>x</arg_value><arg_key>null</arg_key><arg_value>last</arg_value></tool_call>',
        ]
        for raw in cases:
            expected=h.parse(raw,'glm4_5')[0].function.arguments
            for width in (1,7,len(raw)):
                _,wire,_=await h.collect([raw[i:i+width] for i in range(0,len(raw),width)])
                actual=h.assemble(wire)[0]['function']['arguments']
                self.assertEqual(actual,expected)
                self.assertEqual(__import__('json').loads(actual),__import__('json').loads(expected))

if __name__=='__main__':unittest.main(verbosity=2)
