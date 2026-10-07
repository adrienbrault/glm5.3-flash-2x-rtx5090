#!/usr/bin/env python3
"""R884: does keeping earlier reasoning keep the prefix cache across user turns? Synthetic agent session, two requests
per mode. A = [system, user1, assistant1(reasoning + tool call), tool]; B = A + [assistant2(reasoning), user2], i.e.
what an agent harness sends after the next user message. Mode keep = server default (KEEP_THINKING template);
mode clear = template_vars {"clear_thinking": true} (the stock behavior). Each mode gets its own nonce so caches never
cross. Prints prompt tokens, cached tokens and wall time of B per mode. Synthetic text only."""
import argparse, json, random, time, urllib.request

def words(rng, n):
    vocab = ('alpha beta gamma delta parse buffer index cache token layer router expert stream merge patch file line '
             'value check retry error result module config client server request handler table field').split()
    return ' '.join(rng.choice(vocab) for _ in range(n))

def post(url, body):
    t = time.time()
    req = urllib.request.Request(url + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d, time.time() - t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8029/v1')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    nonce = int(time.time() * 1000) % 100000
    rows = []
    for mode in ('clear', 'keep', 'clear2', 'keep2'):
        rng = random.Random(f'{nonce}-{mode}')
        system = f'[session {nonce}-{mode}] You are a coding agent. Repository notes: ' + words(rng, 3000)
        tool_call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'src/main.py'})}}
        msgs_a = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Fix the retry bug in src/main.py. ' + words(rng, 200)},
            {'role': 'assistant', 'content': '', 'reasoning_content': words(rng, 3000), 'tool_calls': [tool_call]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': words(rng, 1500)},
        ]
        msgs_b = msgs_a + [
            {'role': 'assistant', 'content': 'Fixed the retry loop.', 'reasoning_content': words(rng, 2000)},
            {'role': 'user', 'content': 'Now add a test for it.'},
        ]
        tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
                  'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
        extra = {'template_vars': {'clear_thinking': True}} if mode.startswith('clear') else {}
        res = {}
        for tag, msgs in (('A', msgs_a), ('B', msgs_b)):
            body = {'model': a.model, 'messages': msgs, 'tools': tools, 'max_tokens': 1, 'temperature': 0, **extra}
            d, wall = post(a.url, body)
            u = d.get('usage') or {}
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', u.get('cached_tokens'))
            res[tag] = {'prompt_tokens': u.get('prompt_tokens'), 'cached_tokens': cached, 'wall_s': round(wall, 2), 'usage': u}
        rows.append({'mode': mode, **res})
        print(f"{mode}: A prompt {res['A']['prompt_tokens']} ({res['A']['wall_s']} s); "
              f"B prompt {res['B']['prompt_tokens']} cached {res['B']['cached_tokens']} wall {res['B']['wall_s']} s", flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
