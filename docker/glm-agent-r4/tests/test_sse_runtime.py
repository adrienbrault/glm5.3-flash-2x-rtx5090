"""Landing test against the image's actual sse-starlette/Pydantic (no GPU imports)."""
import asyncio
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, os.environ.get('R861_APP', '/app'))
from sse_starlette import EventSourceResponse
from endpoints.OAI.utils.sse_keepalive import idle_comments, keepalive_response
from endpoints.OAI.utils.toolcall_formats.glm4_5_stream import GLMToolCallStream
from endpoints.OAI.utils.toolcall_formats.glm4_5 import parse_toolcalls


class Disconnect(NS):
    async def cleanup(self):
        await asyncio.sleep(0)
        self.cleaned = True


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()

    @classmethod
    def tearDownClass(cls):
        pending = asyncio.all_tasks(cls.loop)
        for task in pending:
            task.cancel()
        cls.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        cls.loop.close()

    def test_actual_sse_encoding_and_done(self):
        self.loop.run_until_complete(self._encoding_and_done())

    def test_real_disconnect_cancels_pending_generator(self):
        self.loop.run_until_complete(self._disconnect_cancels_generator())

    async def _encoding_and_done(self):
        async def source():
            yield '{"choices": [], "usage": {"total_tokens": 42}}'
            yield '[DONE]'
        event = asyncio.Event()
        dh = Disconnect(disconnected=False, abort_event=event)
        now = [0]
        waits = [0]
        async def fake_wait(tasks, timeout, return_when):
            waits[0] += 1
            if waits[0] == 1:
                now[0] += timeout
                return set(), tasks
            return await asyncio.wait(tasks, return_when=return_when)
        response = EventSourceResponse(
            idle_comments(source(), 5, dh, clock=lambda:now[0], wait=fake_wait),
            ping=sys.maxsize,
        )
        async def encode(response):
            sent = []
            async def send(message):
                sent.append(message)
            async def receive():
                await asyncio.Event().wait()
            await asyncio.wait_for(response(
                dict(type='http', asgi={'version':'3.0','spec_version':'2.4'}),receive,send), 2)
            return [m['body'] for m in sent if m['type'] == 'http.response.body']
        original = await encode(EventSourceResponse(source(), ping=sys.maxsize))
        bodies = await encode(response)
        self.assertEqual(len(original), 3)
        self.assertTrue(original[0].startswith(b'data: '))
        self.assertIn(b'data: [DONE]', original[1])
        self.assertEqual(original[2], b'')
        self.assertEqual(bodies, [b': keepalive\n\n'] + original)

    async def _disconnect_cancels_generator(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        async def source():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield '[DONE]'
            finally:
                closed.set()
        dh = Disconnect(disconnected=False,abort_event=asyncio.Event())
        with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':'0.01'}):
            response=keepalive_response(source(),dh,ping=sys.maxsize)
        async def receive():
            await entered.wait()
            return {'type':'http.disconnect'}
        sent=[]
        async def send(message): sent.append(message)
        await asyncio.wait_for(response(dict(type='http',asgi={'version':'3.0','spec_version':'2.4'}),receive,send),2)
        self.assertTrue(closed.is_set())
        self.assertTrue(dh.cleaned)
        self.assertFalse(any(m.get('body') == b': keepalive\n\n' for m in sent))

    def test_real_toolcall_models_unique_ids_and_exact_arguments(self):
        raw=('<tool_call>write<arg_key>body</arg_key><arg_value># héllo "\\\n</fake></arg_value></tool_call>'
             '<tool_call>empty</tool_call>')
        stream=GLMToolCallStream()
        assembled={}
        for char in raw:
            for d in stream.feed(char):
                target=assembled.setdefault(d['index'], {'function':{'arguments':''}})
                for k in ('id','type'):
                    if k in d: target[k]=d[k]
                f=d['function']
                if 'name' in f: target['function']['name']=f['name']
                target['function']['arguments']+=f.get('arguments','')
        self.assertEqual(len({a['id'] for a in assembled.values()}),2)
        for actual, expected in zip(assembled.values(),parse_toolcalls(raw)):
            self.assertEqual(actual['function'],expected.function.model_dump())
            self.assertRegex(actual['id'],r'^call_[0-9a-f]{24}$')


if __name__=='__main__': unittest.main(verbosity=2)
