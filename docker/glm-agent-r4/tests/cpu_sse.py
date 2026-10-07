"""Dependency boundary for CPU lifecycle tests; real ASGI tests are separate.

A cancelled superclass sender deliberately leaves its body iterator open,
matching the reviewed dependency behavior. Executes the production subclass.
"""
import asyncio
from contextlib import nullcontext
import os
import sys
from types import ModuleType
from unittest.mock import patch

class Sender:
    def __init__(self, source, ping):
        self.body_iterator, self.ping = source, ping
    async def __call__(self, scope, receive, send):
        async for frame in self.body_iterator:
            await send({'type': 'http.response.body', 'body': frame})

def response(sse, source, dh):
    starlette, anyio = ModuleType('sse_starlette'), ModuleType('anyio')
    starlette.EventSourceResponse = Sender
    anyio.CancelScope = lambda **kw: nullcontext()
    with patch.dict(sys.modules, {'sse_starlette': starlette, 'anyio': anyio}), \
         patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S': '.001'}):
        return sse.keepalive_response(source, dh, ping=123)
