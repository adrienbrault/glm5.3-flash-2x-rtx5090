#!/usr/bin/env python3
"""Stdlib live SSE probe. Requests a simulated file write; never executes the tool."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def request_body(model):
    data = {
        'stream': True, 'stream_options': {'include_usage': True},
        'reasoning_effort': 'low', 'max_tokens': 24000, 'temperature': 0.2,
        'messages': [
            {'role': 'system', 'content': 'Use write_file to fulfill the request. Generate all file contents as the body argument.'},
            {'role': 'user', 'content': (
                'Call write_file with path /tmp/r861-stream-probe.py. The body must start with # streaming probe\\n '
                'and contain 1000 distinct numbered Python assignments, row_0001 through row_1000. '
                'Each assignment contains a quoted string with a backslash, escaped double quotes, and the literal </fake> and <think>literal</think>. '
                'Generate all 1000 lines in the tool body; do not abbreviate or use a loop to generate them.'
            )},
        ],
        'tools': [{'type': 'function', 'function': {
            'name': 'write_file', 'description': 'Accept a file path and its complete text. This probe does not execute it.',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'body': {'type': 'string'}},
                           'required': ['path', 'body']},
        }}], 'tool_choice': 'auto',
    }
    if model:
        data['model'] = model
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=os.environ.get('TABBY_BASE_URL', 'http://127.0.0.1:5000'))
    parser.add_argument('--model', default=os.environ.get('TABBY_MODEL'))
    parser.add_argument('--expect-live', action='store_true')
    parser.add_argument('--expect-comments', action='store_true')
    parser.add_argument('--request-out', type=Path)
    parser.add_argument('--trace-out', type=Path, help='Save received SSE lines, times and reassembled calls')
    parser.add_argument('--capture-raw', type=Path, help='Use apply-template + raw completions to capture a replayable released-fragment trace')
    args = parser.parse_args()
    data = request_body(args.model)
    if args.request_out:
        args.request_out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    key = os.environ.get('TABBY_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    def post(route, body):
        return urllib.request.urlopen(urllib.request.Request(
            args.base.rstrip('/') + route, data=json.dumps(body).encode(), headers=headers), timeout=600)
    if args.capture_raw:
        with post('/v1/apply-template', data) as response:
            prompt = json.load(response)['prompt']
        data = {'prompt': prompt, 'stream': True, 'max_tokens': data['max_tokens'], 'temperature': data['temperature']}
        if args.model:
            data['model'] = args.model
        route = '/v1/completions'
    else:
        route = '/v1/chat/completions'
    start, previous = time.monotonic(), time.monotonic()
    max_gap, comments, done_at = 0.0, 0, None
    calls, raw_chunks, lines, timings, finishes = {}, [], [], {}, {}
    with post(route, data) as response:
        for raw_line in response:
            line = raw_line.decode('utf-8').rstrip('\r\n')
            if not line:
                continue
            now = time.monotonic() - start
            gap = time.monotonic() - previous
            previous = time.monotonic()
            max_gap = max(max_gap, gap)
            lines.append({'seconds': now, 'line': line})
            if line.startswith(':'):
                comments += 1
                print(f'{now:8.2f}s comment gap={gap:.2f}s {line}', flush=True)
                continue
            if not line.startswith('data:'):
                continue
            payload = line[5:].lstrip()
            if payload == '[DONE]':
                done_at = now
                print(f'{now:8.2f}s DONE maximum_line_gap={max_gap:.2f}s comments={comments}', flush=True)
                break
            chunk = json.loads(payload)
            if 'error' in chunk:
                raise RuntimeError(chunk['error'])
            if 'usage' in chunk:
                print(f'{now:8.2f}s usage={chunk["usage"]}', flush=True)
            for choice in chunk.get('choices', []):
                if choice.get('finish_reason'):
                    finishes[choice['index']] = choice['finish_reason']
                if args.capture_raw:
                    if choice.get('text'):
                        raw_chunks.append(choice['text'])
                    continue
                for delta in choice.get('delta', {}).get('tool_calls', []):
                    key = (choice['index'], delta['index'])
                    call = calls.setdefault(key, {'function': {'arguments': ''}})
                    for field in ('id', 'type'):
                        if field in delta:
                            assert field not in call, f'repeated {field}'
                            call[field] = delta[field]
                    function = delta['function']
                    if 'name' in function:
                        assert 'name' not in call['function'], 'repeated name'
                        call['function']['name'] = function['name']
                        print(f'{now:8.2f}s tool={key} name={function["name"]} id={call.get("id")}', flush=True)
                    fragment = function.get('arguments', '')
                    call['function']['arguments'] += fragment
                    if fragment:
                        timings.setdefault(key, []).append(now)
                        print(f'{now:8.2f}s tool={key} argument_fragment={len(fragment.encode())} bytes '
                              f'total_chars={len(call["function"]["arguments"])}', flush=True)
    assert done_at is not None, 'EOF before [DONE]'
    if args.expect_comments:
        assert comments > 0, 'no SSE comments observed; verify idle interval and proxy flushing'
    if args.capture_raw:
        assert raw_chunks, 'no raw completion text received'
        args.capture_raw.write_text(json.dumps({
            'provenance': 'Live /v1/completions released text fragments captured after /v1/apply-template',
            'start_in_reasoning': True, 'chunks': raw_chunks,
        }, ensure_ascii=False, indent=2) + '\n')
        print(f'Saved {len(raw_chunks)} raw fragments to {args.capture_raw}')
    else:
        assert calls, 'model did not call a tool; inspect response/server logs'
        assert finishes and all(f == 'tool_calls' for f in finishes.values()), finishes
        for key, call in calls.items():
            arguments = json.loads(call['function']['arguments'])
            print(f'tool={key} reassembled keys={list(arguments)} body_chars={len(arguments.get("body", ""))}')
            if args.expect_live:
                times = timings.get(key, [])
                assert len(times) >= 10, f'only {len(times)} argument fragments'
                assert times[-1] - times[0] >= 1, 'argument fragments arrived together'
                assert done_at - times[0] >= 2, 'no early argument fragments before DONE'
                assert len(arguments.get('body', '')) >= 20480, 'model wrote less than the requested long body'
                assert '<think>literal</think>' in arguments['body'], 'literal reasoning tags absent from file body'
    if args.trace_out:
        args.trace_out.write_text(json.dumps({'lines': lines, 'max_line_gap': max_gap,
            'calls': {str(k): v for k,v in calls.items()}}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
