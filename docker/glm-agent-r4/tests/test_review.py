"""Independent CPU review contracts, updated for the revision-2 rejection policy.

Run via ./out/tests/run.sh; the pristine input remains untouched.
Reuses the delivered import boundary, but adds independent adversarial assertions.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_overlay as h


class ArgumentContracts(unittest.IsolatedAsyncioTestCase):
    async def consume(self, raw, *, cap=False, usage=False, enabled=True):
        mc = h.Model(list(raw), cap=cap)
        funcs = h.collector(h.APP / h.UTIL / 'chat_completion.py', mc)
        funcs.update(CancelledError=asyncio.CancelledError,
            ContextLengthExceededError=type('ContextError', (Exception,), {}),
            _resolve_start_in_reasoning=lambda *a: True,
            _gen_label=lambda *a: 'review', _parse_gen_request_id=lambda n, r, i: r,
            request_tag=lambda r: '#review',
            get_generator_error=lambda message: json.dumps({'error': message}),
            UsageStats=h.Stats, CompletionTokensDetails=h.Stats, PromptTokensDetails=h.Stats)
        h.source_functions(h.BASE / h.UTIL / 'common_.py',
                           {'get_usage_stats', 'aggregate_usage_stats'}, funcs)
        h.source_functions(h.APP / h.UTIL / 'chat_completion.py',
            {'stream_generate_chat_completion', '_compose_serialize_stream_usage_chunk'}, funcs)
        data = h.params()
        data.n = 1
        data.stream_options = NS(include_usage=usage)
        data.model_copy = lambda **k: data
        data.model_dump = lambda **k: {}
        dh = h.Disconnect(abort_event=asyncio.Event(), disconnected=False)
        with patch.dict(os.environ, {'TABBY_STREAM_TOOLCALLS': '1' if enabled else '0'}):
            wire = [frame async for frame in funcs['stream_generate_chat_completion'](
                '<think>', None, data, NS(state=NS(id='req')), Path('model'), dh)]
        self.assertTrue(dh.cleaned)
        self.assertEqual(wire[-1], '[DONE]')
        self.assertEqual(wire.count('[DONE]'), 1)
        return [json.loads(frame) for frame in wire[:-1]]

    async def test_usage_frame_does_not_turn_truncated_call_into_success(self):
        raw = h.call() + '<tool_call>write<arg_key>body</arg_key><arg_value># half a file'
        frames = await self.consume(raw, cap=True, usage=True)
        usage_frames = [frame for frame in frames if 'usage' in frame]
        self.assertEqual(len(usage_frames), 1)
        self.assertIs(frames[-1], usage_frames[0])
        self.assertEqual(usage_frames[0]['usage']['completion_tokens'], len(raw))
        self.assertEqual(usage_frames[0]['usage']['prompt_tokens'], 12)
        finishes = [c['finish_reason'] for f in frames for c in f['choices'] if c['finish_reason']]
        self.assertEqual(finishes, ['length'])

    async def test_null_key_collision_preserves_decoded_arguments(self):
        raw = ('<tool_call>f<arg_key>null</arg_key><arg_value>x</arg_value>'
               '<arg_value>y</arg_value></tool_call>')
        expected = json.loads(h.parse(raw, 'glm4_5')[0].function.arguments)
        for chunks in ([raw], list(raw)):
            _, wire, _ = await h.collect(chunks)
            actual = json.loads(h.assemble(wire)[0]['function']['arguments'])
            self.assertEqual(actual, expected)

    async def test_complete_then_truncated_call_must_not_signal_success(self):
        raw = h.call() + '<tool_call>write<arg_key>body</arg_key><arg_value># half a file'
        frames, wire, _ = await h.collect(list(raw), cap=True)
        calls = h.assemble(wire)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(calls[1]['function']['arguments'])
        # Sending partial deltas is legitimate; marking that turn successful is not.
        self.assertEqual(frames[-1]['eos_reason'], 'max_new_tokens')
        self.assertEqual(frames[-1]['finish_reason'], 'length')

    async def test_malformed_later_call_must_not_complete_an_unparsed_turn(self):
        raw = h.call() + '<tool_call>write<arg_key>body</arg_key></tool_call>'
        self.assertEqual(h.parse(raw, 'glm4_5'), [])
        frames, wire, _ = await h.collect(list(raw))
        self.assertEqual(len(h.assemble(wire)), 2)
        # Append-only streaming cannot retract a prefix. It can still fail the turn.
        self.assertNotEqual(frames[-1]['finish_reason'], 'tool_calls')

    async def test_duplicate_keys_invalidate_live_turn(self):
        # Review explicitly permits unique-key restriction with failed finish.
        raw = h.call(pairs=(('a', 'first'), ('a', 'last')))
        for chunks in ([raw], list(raw)):
            frames, wire, _ = await h.collect(chunks)
            self.assertNotEqual(frames[-1]['finish_reason'], 'tool_calls')
            self.assertEqual(frames[-1]['finish_reason'], 'stop')
            self.assertEqual(''.join(f.get('delta_content', '') for f in frames), raw)


    async def test_escaping_think_tags_and_reasoning_preserved(self):
        value = '# "\\\n\t\x00é😀 <literal>x</literal> </fake> '
        raw = 'explain' + h.call(pairs=(('body', value),)) + 'more</think>Answer'
        for size in (1, 2, 17, len(raw)):
            chunks = [raw[i:i+size] for i in range(0, len(raw), size)]
            on, wire, _ = await h.collect(chunks)
            off, old_wire, _ = await h.collect(chunks, enabled=False)
            self.assertEqual(h.assemble(wire), h.assemble(old_wire))
            for channel in ('delta_reasoning_content', 'delta_content'):
                self.assertEqual(''.join(f.get(channel, '') for f in on),
                                 ''.join(f.get(channel, '') for f in off))
            self.assertEqual(json.loads(h.assemble(wire)[0]['function']['arguments']),
                             {'body': value.strip()})

    async def test_none_and_nonstreamed_byte_equivalence_to_pristine(self):
        raw = 'explain</think>Before' + h.call() + h.call('second', [('x', '2')])
        for choice in ('none', 'auto', None):
            chunks = list(raw)
            base = await h.collect(chunks, enabled=False,
                                   path=h.BASE / h.UTIL / 'chat_completion.py', choice=choice)
            with patch.dict(os.environ, {}, clear=True):
                funcs = h.collector(h.APP / h.UTIL / 'chat_completion.py', h.Model(chunks))
                queue = asyncio.Queue()
                await funcs['_chat_stream_collector'](2, queue, 'req', '<think>',
                    h.params(choice), True, streaming_mode=True, label='test')
                frames = [queue.get_nowait() for _ in range(queue.qsize())]
                wire = [funcs['_compose_serialize_stream_chunk']('req', frame, 'test')[0]
                        for frame in frames
                        if not funcs['_compose_serialize_stream_chunk']('req', frame, 'test')[3]]
                self.assertEqual((frames, wire, None), base)
            base_nonstream = await h.collect(chunks, enabled=False, streaming=False,
                path=h.BASE / h.UTIL / 'chat_completion.py', choice=choice)
            for enabled in (False, True):
                actual = await h.collect(chunks, enabled=enabled, streaming=False, choice=choice)
                self.assertEqual(actual, base_nonstream)

    async def test_named_glm_choice_still_uses_actual_legacy_fallback(self):
        class NamedToolChoice:
            def __init__(self, name):
                self.function = NS(name=name)
        forcing_ns = dict(NamedToolChoice=NamedToolChoice, qwen3_coder=object(),
            ALL_TOOLCALL_FORMATS={'glm4_5': h.GLM}, Optional=object,
            xlogger=h.LOG, get_toolcall_tags=lambda f: ('<tool_call>', '</tool_call>'))
        h.source_functions(h.BASE / h.UTIL / 'tool_choice.py',
            {'forces_tool_call', 'resolve_tool_choice_forcing', 'supports_forcing',
             'prepare_tool_choice_forcing'}, forcing_ns)
        # Actual forcing functions run; only the Pydantic request class is stubbed.
        forcing_ns['ToolChoiceForcing'] = lambda **k: NS(**k)
        request = h.params(NamedToolChoice('wanted'))
        raw = 'explain</think>' + h.call('different')
        for enabled in (False, True):
            funcs = h.collector(h.APP / h.UTIL / 'chat_completion.py', h.Model(list(raw)))
            funcs.update({name: forcing_ns[name] for name in (
                'forces_tool_call', 'prepare_tool_choice_forcing')})
            queue = asyncio.Queue()
            with patch.dict(os.environ, {'TABBY_STREAM_TOOLCALLS': '1' if enabled else '0'}):
                await funcs['_chat_stream_collector'](0, queue, 'req', '<think>', request, True,
                    label='test')
            frames = [queue.get_nowait() for _ in range(queue.qsize())]
            self.assertFalse(any(isinstance(f, Exception) for f in frames))
            wire = [funcs['_compose_serialize_stream_chunk']('req', f)[0] for f in frames]
            self.assertEqual(h.assemble(wire)[0]['function']['name'], 'different')
            self.assertEqual(frames[-1]['finish_reason'], 'tool_calls')


class KeepaliveContracts(unittest.IsolatedAsyncioTestCase):
    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)

    async def test_done_closes_existing_handler_and_leaves_no_tasks(self):
        # Run the actual handler class, without FastAPI/logger/config imports.
        source = (h.APP / 'common/networking.py').read_text()
        import ast
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                    and n.name == 'DisconnectHandler')
        ns = dict(asyncio=asyncio, Request=object, logger=h.LOG, xlogger=h.LOG)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT/'app/common/networking.py'), 'exec'), ns)
        async def receive():
            await asyncio.Event().wait()
        baseline = asyncio.all_tasks()
        dh = ns['DisconnectHandler'](NS(receive=receive), 'review')
        cleanup_calls = []
        async def cleanup_job():
            await asyncio.sleep(0)
            cleanup_calls.append(1)
        await dh.add_cleanup_task('job', cleanup_job, ())
        closed = []
        async def source():
            try:
                yield 'payload'
                yield '[DONE]'
                yield 'must not send'
            finally:
                await dh.cleanup()
                closed.append(1)
        frames = [f async for f in h.SSE.idle_comments(source(), 5, dh)]
        await self.settle()
        self.assertEqual(frames, ['payload', '[DONE]'])
        self.assertEqual(cleanup_calls, [1])
        self.assertEqual(closed, [1])
        self.assertEqual(asyncio.all_tasks() - baseline, set())

    async def test_exhaustion_and_exception_leave_no_tasks(self):
        for fail in (False, True):
            baseline = asyncio.all_tasks()
            closed = []
            async def source():
                try:
                    yield 'payload'
                    if fail:
                        raise RuntimeError('backend failed')
                finally:
                    closed.append(1)
            dh = h.Disconnect(disconnected=False, abort_event=asyncio.Event())
            stream = h.SSE.idle_comments(source(), 5, dh)
            self.assertEqual(await anext(stream), 'payload')
            with self.assertRaises(RuntimeError if fail else StopAsyncIteration):
                await anext(stream)
            await self.settle()
            self.assertEqual(closed, [1])
            self.assertTrue(dh.cleaned)
            self.assertEqual(asyncio.all_tasks() - baseline, set())

    async def test_timer_resets_after_slow_send_and_done_wins(self):
        clock = [0]
        gate = asyncio.Event()
        async def source():
            yield '{"content":"a\\nb"}'
            await gate.wait()
            yield '[DONE]'
        observed_timeouts = []
        async def wait(tasks, timeout, return_when):
            observed_timeouts.append(timeout)
            if len(observed_timeouts) == 1:
                return await asyncio.wait(tasks, return_when=return_when)
            clock[0] += timeout
            gate.set()
            return await asyncio.wait(tasks, return_when=return_when)
        dh = h.Disconnect(disconnected=False, abort_event=asyncio.Event())
        stream = h.SSE.idle_comments(source(), 5, dh, clock=lambda: clock[0], wait=wait)
        self.assertEqual(await anext(stream), '{"content":"a\\nb"}')
        clock[0] = 100  # ASGI sender takes much longer than the idle interval.
        self.assertEqual(await anext(stream), '[DONE]')
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)
        self.assertEqual(observed_timeouts, [5, 5])

    async def test_disconnect_during_send_closes_wrapper_and_pending_generation(self):
        # The SSE sender awaits send() AFTER __anext__ returns. Cancel at that
        # boundary, which the delivery's cancellation tests never exercise.
        for frame_kind in ('comment', 'data', 'done'):
            with self.subTest(frame_kind=frame_kind):
                baseline = asyncio.all_tasks()
                closed, in_send = asyncio.Event(), asyncio.Event()
                dh = h.Disconnect(disconnected=False, abort_event=asyncio.Event())
                async def source():
                    try:
                        if frame_kind == 'data':
                            yield 'payload'
                        elif frame_kind == 'done':
                            yield '[DONE]'
                        await asyncio.Event().wait()
                        yield '[DONE]'
                    finally:
                        closed.set()
                from cpu_sse import response
                resp = response(h.SSE, source(), dh)
                stream = resp.body_iterator
                async def send_loop():
                    async def send(message):
                        in_send.set()
                        await asyncio.Event().wait()
                    async def receive():
                        await asyncio.Event().wait()
                    await resp({}, receive, send)
                sender = asyncio.create_task(send_loop())
                await asyncio.wait_for(in_send.wait(), 1)
                dh.disconnected = True
                dh.abort_event.set()
                sender.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await sender
                await self.settle()
                try:
                    self.assertTrue(closed.is_set(),
                        f'{frame_kind}: source still open; leftover tasks: '
                        f'{asyncio.all_tasks() - baseline}')
                    self.assertTrue(getattr(dh, 'cleaned', False))
                    self.assertEqual(asyncio.all_tasks() - baseline, set())
                finally:
                    # Explicit test teardown proves which missing operation is needed.
                    await stream.aclose()
                    await self.settle()
                    self.assertEqual(asyncio.all_tasks() - baseline, set())


class InheritedContracts(unittest.IsolatedAsyncioTestCase):
    async def test_inherited_usage_shape_preserved_off_gate(self):
        raw = 'explain</think>' + h.call()
        off = await ArgumentContracts.consume(self, raw, usage=True, enabled=False)
        on = await ArgumentContracts.consume(self, raw, usage=True)
        old_usage = [frame for frame in off if 'usage' in frame]
        new_usage = [frame for frame in on if 'usage' in frame]
        self.assertEqual(old_usage, new_usage)
        self.assertEqual(len(new_usage), 1)
        # This deviation is inherited; overlay preserves the base usage packet.
        self.assertEqual(new_usage[0]['choices'],
                         [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}])

    async def test_literal_think_in_value_should_survive_channel_parser(self):
        raw = h.call(pairs=(('body', '# <think>x</think>'),))
        base = await h.collect(list(raw), enabled=False,
                              path=h.BASE / h.UTIL / 'chat_completion.py')
        on = await h.collect(list(raw))
        self.assertEqual(h.assemble(base[1]), h.assemble(on[1]))
        # This failure is inherited, not attributable to the overlay.
        with patch.dict(os.environ, {'TABBY_GLM_TOOL_FIXES': '1'}):
            parser = h.TAGS.TagStreamParser(reasoning_start='<think>', reasoning_end='</think>',
                tool_start='<tool_call>', tool_end='</tool_call>', preserve_tool_reasoning_tags=True)
            text = ''.join(sub for ch, sub in parser.feed(raw) + parser.finish() if ch == 'tool')
            self.assertEqual(json.loads(h.parse(text, 'glm4_5')[0].function.arguments),
                             {'body': '# <think>x</think>'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
