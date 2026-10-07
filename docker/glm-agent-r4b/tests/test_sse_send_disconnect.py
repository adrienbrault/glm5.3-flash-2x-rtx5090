"""Operator probe: run with real sse-starlette/AnyIO in the patched server image.

R861_APP=/app python3 -B /path/to/test_sse_send_disconnect.py
Retains the response until assertions complete; cleanup cannot rely on GC.
"""
import asyncio
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, os.environ.get('R861_APP', '/app'))
from endpoints.OAI.utils.sse_keepalive import keepalive_response
import sse_starlette


class Disconnect(NS):
    async def cleanup(self):
        await asyncio.sleep(0)
        self.cleaned = True


class SendDisconnect(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_while_comment_or_data_is_being_sent(self):
        print('sse-starlette:', getattr(sse_starlette, '__version__', 'unknown'))
        from sse_starlette import sse
        start_watcher = getattr(sse, '_ensure_watcher_started_on_this_loop', None)
        if start_watcher:
            start_watcher()
            await asyncio.sleep(0)
        for frame_kind in ('comment', 'data', 'done'):
            with self.subTest(frame_kind=frame_kind):
                closed, sending = asyncio.Event(), asyncio.Event()
                async def source():
                    try:
                        if frame_kind == 'data':
                            yield '{"choices":[]}'
                        elif frame_kind == 'done':
                            yield '[DONE]'
                        await asyncio.Event().wait()
                        yield '[DONE]'
                    finally:
                        closed.set()
                dh = Disconnect(disconnected=False, abort_event=asyncio.Event())
                with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S': '.001'}):
                    response = keepalive_response(source(), dh, ping=sys.maxsize)
                async def send(message):
                    if message['type'] == 'http.response.body' and message.get('body'):
                        sending.set()
                        await asyncio.Event().wait()  # transport backpressure
                async def receive():
                    await sending.wait()
                    dh.disconnected = True
                    dh.abort_event.set()
                    return {'type': 'http.disconnect'}
                baseline = asyncio.all_tasks()
                try:
                    await asyncio.wait_for(response(
                        dict(type='http', asgi={'version':'3.0','spec_version':'2.4'}),
                        receive, send), 2)
                    for _ in range(5):
                        await asyncio.sleep(0)
                    self.assertTrue(closed.is_set(),
                        f'{frame_kind}: response returned with source still open')
                    self.assertTrue(getattr(dh, 'cleaned', False))
                    self.assertEqual(asyncio.all_tasks() - baseline, set())
                finally:
                    await response.body_iterator.aclose()


if __name__ == '__main__':
    unittest.main(verbosity=2)
