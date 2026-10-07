"""CPU tests: execute served source with GPU/import dependencies replaced at the boundary."""
import ast
import asyncio
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

APP = Path(os.environ.get('R861_APP', Path(__file__).resolve().parents[1] / 'work'))
BASE = Path(os.environ.get('R861_BASE', Path(__file__).resolve().parents[2] / 'app'))
UTIL = Path('endpoints/OAI/utils')
LOG = NS(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)


class Disconnect(NS):
    async def cleanup(self):
        self.cleaned = True


class Tool:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class ToolCall:
    def __init__(self, function):
        self.function, self.id, self.type, self.index = function, 'call_test', 'function', None

    def model_dump(self, **kwargs):
        return dict(id=self.id, type=self.type, index=self.index,
                    function=dict(name=self.function.name, arguments=self.function.arguments))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_modules():
    # Only the logger and Pydantic call container are stubbed for pure parser tests.
    for name, attrs in [('common.logger', dict(xlogger=LOG)),
                        ('endpoints.OAI.types.tools', dict(Tool=Tool, ToolCall=ToolCall))]:
        m = ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
    sys.path.insert(0, str(APP))
    root = BASE if BASE.exists() else APP
    common = module('endpoints.OAI.utils.toolcall_formats.common', root / UTIL / 'toolcall_formats/common.py')
    glm = module('endpoints.OAI.utils.toolcall_formats.glm4_5', root / UTIL / 'toolcall_formats/glm4_5.py')
    tags = module('r861_tags', APP / UTIL / 'stream_parser.py')
    live = module('r861_live', APP / UTIL / 'toolcall_formats/glm4_5_stream.py')
    sse = module('r861_sse', APP / UTIL / 'sse_keepalive.py')
    return common, glm, tags, live, sse


COMMON, GLM, TAGS, LIVE, SSE = load_modules()


def parse(text, fmt):
    try:
        return GLM.parse_toolcalls(text)
    except Exception:
        return []  # same exception boundary as utils/tools.py


def source_functions(path, names, ns):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(nodes) == len(names), (path, names)
    # Resolve annotations without pulling GPU-bearing imports into this CPU process.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                            type_ignores=[])), str(path), 'exec'), ns)
    return ns


def collector(path, mc):
    from endpoints.OAI.utils import glm_tool_fixes
    from common import glm_tag_safety
    ns = dict(glm_tag_safety=glm_tag_safety, keepalive_seconds=SSE.keepalive_seconds, glm_fixes=glm_tool_fixes, asyncio=asyncio, model=NS(container=mc), xlogger=LOG, CONTENT=TAGS.CONTENT,
              REASONING=TAGS.REASONING, TagStreamParser=TAGS.TagStreamParser,
              GLMToolCallStream=LIVE.GLMToolCallStream, stream_toolcalls_enabled=LIVE.stream_toolcalls_enabled,
              parse_toolcalls=parse, get_toolcall_tags=lambda f: ('<tool_call>', '</tool_call>'),
              _resolve_reasoning_budget=lambda p,m: (None,''), forces_tool_call=lambda p: False,
              LoopDetector=None, ChatCompletionLogprobs=lambda **k: NS(model_dump=lambda: k),
              json=json, time=lambda: 1234)
    return source_functions(path, {'_chat_stream_collector', '_parse_tool_calls',
                                  '_finish_with_tool_calls', '_compose_serialize_stream_chunk'}, ns)


class Model:
    harmony = muse_glimmer = False
    reasoning = True
    reasoning_start_token, reasoning_end_token = '<think>', '</think>'
    tool_format = 'glm4_5'
    tool_calls_in_reasoning = True

    def __init__(self, chunks, cap=False):
        self.chunks, self.cap = chunks, cap

    async def stream_generate(self, *a, **k):
        for i, chunk in enumerate(self.chunks):
            yield dict(text=chunk, token_ids=[i], prompt_tokens=12, completion_tokens=i + 1, gen_tokens=i + 1,
                       finish_reason=None)
        yield dict(text='', token_ids=[], prompt_tokens=12, completion_tokens=len(self.chunks), gen_tokens=len(self.chunks), gen_time=2.0,
                   prompt_time=1.0, total_time=3.0,
                   finish_reason='length' if self.cap else 'stop',
                   eos_reason='max_new_tokens' if self.cap else 'eos')


def params(choice='auto'):
    return NS(tool_choice=choice, json_schema=None, regex_pattern=None, grammar_string=None,
              get_stop_on_loop=lambda: None)


async def collect(chunks, enabled=True, path=None, streaming=True, cap=False, start=True, choice='auto', mc=None):
    mc = mc or Model(chunks, cap)
    funcs = collector(path or APP / UTIL / 'chat_completion.py', mc)
    queue = asyncio.Queue()
    with patch.dict(os.environ, {'TABBY_STREAM_TOOLCALLS': '1' if enabled else '0'}):
        result = await funcs['_chat_stream_collector'](2, queue, 'req', '<think>', params(choice), start,
                                                     streaming_mode=streaming, label='test')
    frames = []
    while not queue.empty():
        value = queue.get_nowait()
        if isinstance(value, Exception):
            raise value
        frames.append(value)
    wire = [funcs['_compose_serialize_stream_chunk']('req', frame, 'test')[0] for frame in frames
            if not funcs['_compose_serialize_stream_chunk']('req', frame, 'test')[3]]
    return frames, wire, result


def assemble(wire):
    calls = {}
    for line in wire:
        for choice in json.loads(line)['choices']:
            for delta in choice['delta'].get('tool_calls', []):
                index = delta['index']
                target = calls.setdefault(index, dict(index=index, function=dict(arguments='')))
                for key in ('id', 'type'):
                    if key in delta:
                        assert key not in target, ('repeated metadata', key)
                        target[key] = delta[key]
                function = delta['function']
                if 'name' in function:
                    assert 'name' not in target['function'], 'repeated name'
                    target['function']['name'] = function['name']
                target['function']['arguments'] += function.get('arguments', '')
    return list(calls.values())


def call(name='write', pairs=(('body', 'hello'),)):
    return '<tool_call>' + name + ''.join('<arg_key>' + k + '</arg_key><arg_value>' + v + '</arg_value>'
                                         for k,v in pairs) + '</tool_call>'


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def assert_replay(self, raw, chunks, semantic=False):
        frames, wire, _ = await collect(chunks)
        actual = assemble(wire)
        for line in wire:
            entries = json.loads(line)['choices'][0]['delta'].get('tool_calls', [])
            self.assertEqual(len({d['index'] for d in entries}),len(entries))
        expected = [p.model_dump() for p in parse(raw, 'glm4_5')]
        for i,p in enumerate(expected):
            p['index'] = i
        if semantic:
            for calls in (actual, expected):
                for p in calls:
                    p['function']['arguments'] = json.loads(p['function']['arguments'])
        self.assertEqual(actual, expected)
        self.assertEqual(frames[-1]['finish_reason'], 'tool_calls')
        return frames

    async def test_frozen_fragment_trace(self):
        fixture_path = Path(os.environ.get('R861_TRACE', Path(__file__).parent / 'fixtures/glm-token-trace.json'))
        fixture = json.loads(fixture_path.read_text())
        raw = ''.join(fixture['chunks'])
        await self.assert_replay(raw, fixture['chunks'])

    async def test_plain_literal_prefix_becomes_live(self):
        raw = '<tool_call>f<arg_key>body</arg_key><arg_value>function example() {'
        frames, wire, _ = await collect(list(raw), cap=True)
        args = assemble(wire)[0]['function']['arguments']
        self.assertIn('function example()', args)

    async def test_malformed_turn_falls_back_to_raw_content(self):
        raw = call() + '<tool_call>f<arg_key>a</arg_key></tool_call>'
        self.assertEqual(parse(raw, 'glm4_5'), [])
        for cap in (True,False):
            off, _, _ = await collect(list(raw), enabled=False, cap=cap)
            on, _, _ = await collect(list(raw), enabled=True, cap=cap)
            self.assertEqual(on[-1]['finish_reason'], 'length' if cap else 'stop')
            self.assertEqual(''.join(f.get('delta_content', '') for f in on), raw)

    async def test_long_body_live_before_value_end(self):
        body = ('# file "quotes" \\ path\n</fake> </ partial é 😀 \t\x00\n' * 500)[:20480]
        raw = call(pairs=(('path', '/tmp/test.txt'), ('body', '\n  ' + body + '  \n')))
        text = 'thinking</think>Writing now.' + raw
        chunks = list(text)  # adversarial token trace: one character at a time
        frames = await self.assert_replay(raw, chunks)
        close_at = text.index('</arg_value>', text.index('<arg_key>body'))
        live = ''.join(d['function'].get('arguments','') for f in frames[:close_at]
                       for d in f.get('delta_tool_calls', []))
        self.assertIn('file \\"quotes\\"', live)
        self.assertGreater(len(live), 18000)
        self.assertEqual(''.join(f.get('delta_content','') for f in frames), 'Writing now.')
        self.assertEqual(''.join(f.get('delta_reasoning_content','') for f in frames), 'thinking')

    async def test_terminal_text_with_partial_tag(self):
        raw = '<tool_call>f<arg_key>body</arg_key><arg_value># code <'
        class TerminalModel(Model):
            async def stream_generate(self, *a, **k):
                yield dict(text=raw, finish_reason='length', eos_reason='max_new_tokens')
        frames,wire,_ = await collect([raw], mc=TerminalModel([raw]))
        deltas=json.loads(wire[0])['choices'][0]['delta']['tool_calls']
        self.assertEqual(len(deltas),1)
        self.assertTrue(assemble(wire)[0]['function']['arguments'].endswith('<'))
        self.assertEqual(frames[-1]['finish_reason'],'length')

    async def test_coercions_exact_and_multiple_calls(self):
        values = ['', ' \n ', '123', '-2.5e3', 'true', 'false', 'null', '[1, "x"]',
                  '{"a": 1, "b": [false]}', '"quoted\\nstring"', 'NaN', 'Infinity',
                  '-Infinity', 'falsehood', 'none', '{broken', '123x', 'héllo', 'a\x01b']
        raw = call('first', [(str(i), '\n' + v + '\n') for i,v in enumerate(values)]) + call('empty', [])
        for size in (1, 2, 7, 31, len(raw)):
            await self.assert_replay(raw, [raw[i:i+size] for i in range(0,len(raw),size)])

    async def test_random_chunk_boundaries(self):
        raw = call(pairs=((' x "\\ ', '# hi\nline \\ "'), ('number','3'))) + call('list', [('items','[1,2]')])
        rng = random.Random(861)
        for _ in range(50):
            chunks, pos = [], 0
            while pos < len(raw):
                size = rng.randrange(1,24)
                chunks.append(raw[pos:pos+size]); pos += size
            await self.assert_replay(raw, chunks)

    async def test_independent_key_value_lists_and_missing_key(self):
        raws = [
            '<tool_call>f<arg_key>a</arg_key><arg_key>b</arg_key><arg_value>x</arg_value><arg_value>2</arg_value></tool_call>',
            '<tool_call>f<arg_value>x</arg_value><arg_key>a</arg_key></tool_call>',
            '<tool_call>f<arg_key>a</arg_key><arg_value>x</arg_value><arg_value>2</arg_value></tool_call>',
            call(pairs=(('', 'x'),)),
            '<tool_call>  </tool_call>' + call(),
            call(pairs=(('a','<arg_key>b</arg_key>x'), ('c','y'))) + call('ok', []),
        ]
        for raw in raws[:5]:
            await self.assert_replay(raw, list(raw))
        # A key tag inside a value makes more keys than values: existing parse
        # fails the whole turn. Treat transmitted deltas as incomplete, not a successful JSON call.
        frames, wire, _ = await collect(list(raws[5]))
        self.assertEqual(frames[-1]['finish_reason'], 'stop')
        self.assertEqual(''.join(f.get('delta_content', '') for f in frames), raws[5])

    async def test_duplicate_keys_reject_live_turn(self):
        raw = call(pairs=(('a', 'first'), ('b', '2'), ('a', 'last')))
        frames, wire, _ = await collect(list(raw))
        self.assertEqual(frames[-1]['finish_reason'], 'stop')
        self.assertEqual(''.join(f.get('delta_content', '') for f in frames), raw)

    async def test_truncated_call_keeps_length_and_no_fake_close(self):
        raw = '<tool_call>write<arg_key>body</arg_key><arg_value># unfinished'
        frames, wire, _ = await collect(list(raw), cap=True)
        self.assertEqual(frames[-1]['finish_reason'], 'length')
        actual = assemble(wire)
        self.assertTrue(actual)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(actual[0]['function']['arguments'])

    async def test_calls_in_reasoning_setting_and_none(self):
        raw = call()
        frames, wire, _ = await collect(list(raw), mc=Model(list(raw)))
        self.assertTrue(assemble(wire))
        mc = Model(list(raw)); mc.tool_calls_in_reasoning = False
        frames, wire, _ = await collect(list(raw), mc=mc)
        self.assertEqual(assemble(wire), [])
        frames, wire, _ = await collect(list(raw), choice='none')
        self.assertEqual(assemble(wire), [])

    async def test_off_wire_equivalence_to_served_source(self):
        if not BASE.exists():
            self.skipTest('base source not available in container; checked on host before packaging')
        raws = ['a</think>Answer', 'a</think>' + call(), call('empty', []),
                'a</think>Before' + call() + call('f', [('a','[1]')]),
                '<tool_call>f<arg_key>a</arg_key>', call(pairs=(('a','true'),))]
        for raw in raws:
            for size in (1,7,len(raw)):
                chunks = [raw[i:i+size] for i in range(0,len(raw),size)]
                for cap in (True,False):
                    original = await collect(chunks, enabled=False, path=BASE / UTIL / 'chat_completion.py', cap=cap)
                    overlay = await collect(chunks, enabled=False, cap=cap)
                    self.assertEqual(overlay, original)

    async def test_nonstreamed_unchanged_when_on(self):
        raw = call()
        _,_,off = await collect(list(raw), enabled=False, streaming=False)
        _,_,on = await collect(list(raw), enabled=True, streaming=False)
        self.assertEqual(off,on)

    def test_gate_exact(self):
        for value in ('', '0', 'true', '01', '1'):
            with patch.dict(os.environ, {'TABBY_STREAM_TOOLCALLS':value}):
                self.assertEqual(LIVE.stream_toolcalls_enabled('glm4_5', True), value == '1')
                self.assertFalse(LIVE.stream_toolcalls_enabled('glm4_5', False))
                self.assertFalse(LIVE.stream_toolcalls_enabled('qwen3_coder', True))


class Stats(NS):
    def model_dump(self, **kwargs):
        return {key: value.model_dump() if hasattr(value, 'model_dump') else value
                for key, value in vars(self).items()}


class ConsumerTests(unittest.IsolatedAsyncioTestCase):
    async def run_stream(self, enabled, n, usage):
        raw = 'thinking</think>Ready.' + call(pairs=(('body', '# file \"quoted\"\n'), ('count','123')))
        mc = Model(list(raw))
        funcs = collector(APP / UTIL / 'chat_completion.py', mc)
        funcs.update(CancelledError=asyncio.CancelledError,
                     ContextLengthExceededError=type('ContextError',(Exception,),{}),
                     _resolve_start_in_reasoning=lambda *a: True,
                     _gen_label=lambda *a: 'test', _parse_gen_request_id=lambda n,r,i:r,
                     request_tag=lambda r: '#1', get_generator_error=lambda message: json.dumps({'error':message}),
                     UsageStats=Stats, CompletionTokensDetails=Stats, PromptTokensDetails=Stats)
        root = BASE if BASE.exists() else APP
        source_functions(root / UTIL / 'common_.py', {'get_usage_stats','aggregate_usage_stats'}, funcs)
        source_functions(APP / UTIL / 'chat_completion.py',
                         {'stream_generate_chat_completion', '_compose_serialize_stream_usage_chunk'}, funcs)
        data = params()
        data.n = n
        data.tools = [{'function': {'parameters': {'properties': {'count': {'type': 'string'}}}}}]
        data.stream_options = NS(include_usage=usage)
        data.model_copy = lambda **k: data
        data.model_dump = lambda **k: {}
        dh = Disconnect(abort_event=asyncio.Event(), disconnected=False)
        with patch.dict(os.environ, {'TABBY_STREAM_TOOLCALLS':'1' if enabled else '0'}):
            frames = [frame async for frame in funcs['stream_generate_chat_completion'](
                '<think>',None,data,NS(state=NS(id='req')),Path('model'),dh)]
        self.assertTrue(dh.cleaned)
        self.assertEqual(frames[-1],'[DONE]')
        return frames, len(raw)

    async def test_consumer_n_choices_usage_and_content(self):
        for n in (1,2):
            for usage in (False,True):
                off, token_count = await self.run_stream(False,n,usage)
                on, _ = await self.run_stream(True,n,usage)
                usage_off = [json.loads(f)['usage'] for f in off[:-1] if 'usage' in json.loads(f)]
                usage_on = [json.loads(f)['usage'] for f in on[:-1] if 'usage' in json.loads(f)]
                self.assertEqual(usage_on,usage_off)
                if usage:
                    self.assertEqual(usage_on[0]['completion_tokens'], n*token_count)
                    self.assertEqual(usage_on[0]['prompt_tokens'],12)
                for index in range(n):
                    def choice_frames(frames):
                        return [f for f in frames[:-1] if json.loads(f)['choices'][0]['index']==index]
                    self.assertEqual(assemble(choice_frames(on)),assemble(choice_frames(off)))
                    args = json.loads(assemble(choice_frames(on))[0]['function']['arguments'])
                    self.assertEqual(args['count'],123)  # served coercion ignores schema 'string'
                    for field in ('content','reasoning_content'):
                        def text(frames):
                            return ''.join(json.loads(f)['choices'][0]['delta'].get(field,'')
                                           for f in choice_frames(frames))
                        self.assertEqual(text(on),text(off))


class KeepaliveTests(unittest.IsolatedAsyncioTestCase):
    def test_env(self):
        for value in ('','0'):
            with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':value}):
                self.assertIsNone(SSE.keepalive_seconds())
        for value in ('5','.1'):
            with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':value}):
                self.assertEqual(SSE.keepalive_seconds(), float(value))
        for value in ('NaN','inf','-1','bad','-0'):
            with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':value}):
                with self.assertRaises(ValueError): SSE.keepalive_seconds()

    async def scenario(self, script):
        now = [0.0]
        source_queue = asyncio.Queue()
        dh = Disconnect(abort_event=asyncio.Event(), disconnected=False)
        cleaned = []
        async def source():
            try:
                while True:
                    yield await source_queue.get()
            finally:
                cleaned.append(True)
        timeouts = []
        async def wait(tasks, timeout, return_when):
            timeouts.append(timeout)
            advance, action = script.pop(0)
            now[0] += advance
            if action == 'disconnect':
                dh.disconnected=True; dh.abort_event.set()
            elif action is not None:
                await source_queue.put(action)
            # Execute ready tasks deterministically, with no real clock sleep.
            for _ in range(3): await asyncio.sleep(0)
            done = {t for t in tasks if t.done()}
            return done, tasks-done
        output = [frame async for frame in SSE.idle_comments(source(),5,dh,clock=lambda:now[0],wait=wait)]
        self.assertTrue(cleaned)
        self.assertFalse(script)
        return output,timeouts

    async def test_idle_and_done(self):
        output,timeouts = await self.scenario([(5,None),(5,None),(1,'payload'),(5,None),(0,'[DONE]')])
        self.assertEqual(output,[b': keepalive\n\n',b': keepalive\n\n','payload',b': keepalive\n\n','[DONE]'])
        self.assertEqual(timeouts,[5]*5)

    async def test_data_resets_timer_and_wins_at_deadline(self):
        output,timeouts = await self.scenario([(4,'a'),(4,'b'),(5,'[DONE]')])
        self.assertEqual(output,['a','b','[DONE]'])
        self.assertEqual(timeouts,[5]*3)

    async def test_disconnect_while_silent(self):
        output,_ = await self.scenario([(5,None),(1,'disconnect')])
        self.assertEqual(output,[b': keepalive\n\n'])

    async def test_already_disconnected_cleans_handler(self):
        async def source():
            yield 'must not send'
        dh = Disconnect(abort_event=asyncio.Event(), disconnected=True)
        output = [frame async for frame in SSE.idle_comments(source(),5,dh)]
        self.assertEqual(output,[])
        self.assertTrue(dh.cleaned)

    async def test_cancellation_closes_source(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        async def source():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield 'impossible'
            finally:
                closed.set()
        dh = Disconnect(abort_event=asyncio.Event(),disconnected=False)
        stream = SSE.idle_comments(source(),5,dh)
        task = asyncio.create_task(anext(stream))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertTrue(closed.is_set())

    def test_off_response_identity_and_on_ping_disabled(self):
        calls = []
        fake = ModuleType('sse_starlette')
        class FakeResponse:
            def __init__(self, source, **kwargs):
                self.body_iterator, self.kwargs = source, kwargs
                calls.append((source, kwargs))
        fake.EventSourceResponse = FakeResponse
        sentinel=object()
        fake_anyio = ModuleType('anyio')
        fake_anyio.CancelScope = lambda **kwargs: nullcontext()
        with patch.dict(sys.modules, {'sse_starlette':fake, 'anyio':fake_anyio}):
            with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':''}):
                response=SSE.keepalive_response(sentinel,None,17)
                self.assertIs(response.body_iterator,sentinel)
                self.assertEqual(response.kwargs,dict(ping=17))
            with patch.dict(os.environ, {'TABBY_SSE_KEEPALIVE_S':'5'}):
                response=SSE.keepalive_response(sentinel,None,17)
                self.assertEqual(response.kwargs,dict(ping=sys.maxsize))


if __name__ == '__main__':
    unittest.main(verbosity=2)
