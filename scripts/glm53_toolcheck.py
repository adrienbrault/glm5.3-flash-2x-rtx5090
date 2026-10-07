#!/usr/bin/env python3
"""R868 tool-call check for GLM-5.3 behind TabbyAPI (2026-10-07). R864's live probe on the r861 agent overlay got a
67-character write_file call and finish_reason "stop". This sends three requests and saves every raw SSE line:
  simple-stream   short write_file (20 print lines), streamed
  simple-plain    the same, non-streamed
  hard-stream     a 200-line body with quotes, backslashes and literal </fake> and <think>x</think>, streamed
For each: finish_reason, content and reasoning lengths, the reassembled arguments, whether they parse as JSON, and the
body length.
usage: glm53_toolcheck.py --url URL --model M --out DIR"""
import argparse, json, os, time, urllib.request

TOOLS = [{'type': 'function', 'function': {'name': 'write_file', 'description': 'Write a file',
          'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                         'required': ['path', 'content']}}}]
CASES = {
    'simple': 'Write a 20-line Python file /tmp/x.py that prints the numbers 1 to 20, one per line, each with its own print '
              'statement. Use the write_file tool.',
    'hard': 'Use the write_file tool to write /tmp/y.py: 200 lines, line i is `row_i = "a\\\\b \\"q\\" </fake> <think>x</think>"` '
            'with i from 1 to 200. Write every line literally; do not use a loop.',
}

def run(a, tag, prompt, stream):
    body = {'model': a.model, 'stream': stream, 'max_tokens': 12000, 'reasoning_effort': 'low',
            'messages': [{'role': 'user', 'content': prompt}], 'tools': TOOLS}
    if stream:
        body['stream_options'] = {'include_usage': True}
    req = urllib.request.Request(a.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); raw = []
    with urllib.request.urlopen(req, timeout=900) as r:
        if stream:
            for line in r:
                raw.append((round(time.time() - t0, 3), line.decode(errors='replace').rstrip('\n')))
        else:
            raw.append((round(time.time() - t0, 3), r.read().decode(errors='replace')))
    open(os.path.join(a.out, f'{tag}.raw.json'), 'w').write(json.dumps(raw, ensure_ascii=False))
    content = reasoning = args = ''; finish = None; name = None; usage = None
    if stream:
        for _, l in raw:
            if not l.startswith('data: {'):
                continue
            d = json.loads(l[6:]); usage = d.get('usage') or usage
            for ch in d.get('choices', []):
                dl = ch.get('delta') or {}
                content += dl.get('content') or ''; reasoning += dl.get('reasoning_content') or ''
                for t in dl.get('tool_calls') or []:
                    f = t.get('function') or {}; name = f.get('name') or name; args += f.get('arguments') or ''
                finish = ch.get('finish_reason') or finish
    else:
        d = json.loads(raw[0][1]); usage = d.get('usage'); ch = d['choices'][0]; m = ch['message']
        content = m.get('content') or ''; reasoning = m.get('reasoning_content') or ''; finish = ch.get('finish_reason')
        for t in m.get('tool_calls') or []:
            name = t['function']['name']; args += t['function']['arguments']
    try:
        parsed = json.loads(args) if args else None; ok = parsed is not None
    except ValueError:
        parsed = None; ok = False
    res = {'tag': tag, 'finish': finish, 'tool': name, 'args_len': len(args), 'args_json_ok': ok,
           'body_len': len((parsed or {}).get('content', '')) if ok else None, 'content_len': len(content),
           'reasoning_len': len(reasoning), 'completion_tokens': (usage or {}).get('completion_tokens'),
           'content_tail': content[-300:], 'args_head': args[:200], 'args_tail': args[-200:]}
    print('RESULT ' + json.dumps(res, ensure_ascii=False), flush=True)
    return res

ap = argparse.ArgumentParser()
ap.add_argument('--url', required=True); ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
run(a, 'simple-stream', CASES['simple'], True)
run(a, 'simple-plain', CASES['simple'], False)
run(a, 'hard-stream', CASES['hard'], True)
