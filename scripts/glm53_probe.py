#!/usr/bin/env python3
"""Stdlib streaming sanity/decode/prefill; usage provides exact MTP acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request
import uuid
from pathlib import Path

LOCK = threading.Lock()

def post(base, route, body, timeout=900):
    req = urllib.request.Request(base + route, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def stream(a, tag, body, chat=True):
    body = dict(body, model=a.model, stream=True, stream_options={'include_usage': True}, loop_detect_window=0)
    stem = Path(a.out).parent / tag
    stem.with_suffix('.request.json').write_text(json.dumps(body, ensure_ascii=False) + '\n')
    start = time.monotonic()
    first = last = None
    usage = None
    content, reasoning, finish = '', '', None
    done = False
    with post(a.url, '/v1/chat/completions' if chat else '/v1/completions', body) as r, stem.with_suffix('.events.jsonl').open('w', buffering=1) as archive:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            if line == 'data: [DONE]':
                done = True
                break
            d = json.loads(line[5:])
            now = time.monotonic()
            archive.write(json.dumps({'elapsed_s': now-start, 'event': d}, ensure_ascii=False) + '\n')
            if d.get('error'):
                raise ValueError(d['error'])
            for ch in d.get('choices', []):
                dl = ch.get('delta') or {} if chat else {'content': ch.get('text')}
                c, th = dl.get('content') or '', dl.get('reasoning_content') or ''
                if c or th:
                    if first is None: first = now
                    last = now
                    content += c
                    reasoning += th
                    with LOCK:
                        print(json.dumps({'stream': tag, 'elapsed_s': round(now-start,4), 'content': c, 'reasoning': th}, ensure_ascii=False), flush=True)
                finish = ch.get('finish_reason') or finish
            usage = d.get('usage') or usage
    if not done or not usage or first is None or last is None:
        raise ValueError(f'{tag}: truncated/no-text/no-usage stream')
    detail = usage.get('completion_tokens_details') or {}
    result = {'tag': tag, 'first': first, 'last': last, 'ttft_s': first-start,
              'wall_s': time.monotonic()-start, 'content': content, 'reasoning': reasoning,
              'finish_reason': finish, 'usage': usage,
              'accepted': detail.get('accepted_prediction_tokens'),
              'rejected': detail.get('rejected_prediction_tokens')}
    return result

def emit(a, result):
    with Path(a.out).open('a', buffering=1) as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)

def sanity(a):
    tasks = [('multiply', 'What is 17*23? Answer with only the number.', '391'),
             ('capital', 'What is the capital of France? Answer with only one word.', 'paris'),
             ('repeat', 'Repeat exactly the word RED. Output only that word.', 'red')]
    thought = False
    always = os.environ.get('R858_ALWAYS_THINK') == '1'  # GLM-5.3: thinking cannot be disabled; arms = effort low/high
    for enabled in (False, True):
        for label, q, want in tasks:
            kw = {'reasoning_effort': 'high' if enabled else 'low'} if always else {'enable_thinking': enabled}
            r = stream(a, f'sanity-{int(enabled)}-{label}', {'messages':[{'role':'user','content':q}],
                       'max_tokens':8192 if always else 1024, 'temperature':0, 'chat_template_kwargs':kw})
            answer = re.sub(r'[\s.!*`]+', '', r['content']).lower()
            sane = answer == want and r['finish_reason'] != 'length'
            toggled = enabled or always or not r['reasoning'].strip()
            thought |= enabled and bool(r['reasoning'].strip())
            emit(a, dict(r, phase='sanity', thinking=enabled, sane=sane, toggle_ok=toggled))
            if not sane or not toggled:
                raise ValueError(f'sanity failed: thinking={enabled} {label}')
    if not thought:
        raise ValueError('thinking-on produced no reasoning across three prompts; do not claim toggle works')

def decode_round(a, c, run, tokens, phase):
    barrier = threading.Barrier(c)
    salt = uuid.uuid4().hex
    def one(i):
        body = {'messages':[{'role':'user','content':f'Run {salt} stream {i}. Write a long detailed Python tutorial with complete code and tests. Keep elaborating.'}],
                'max_tokens':tokens, 'min_tokens':tokens, 'ban_eos_token':True,
                'temperature':0, 'chat_template_kwargs':({'reasoning_effort':'low'} if os.environ.get('R858_ALWAYS_THINK') == '1' else {'enable_thinking':False})}
        barrier.wait(timeout=30)
        return stream(a, f'{phase}-c{c}-r{run}-s{i}', body)
    with ThreadPoolExecutor(max_workers=c) as pool:
        rs = list(pool.map(one, range(c)))
    for r in rs:
        if r['usage']['completion_tokens'] != tokens:
            raise ValueError(f'forced generation ended early: {r["usage"]}')
        emit(a, dict(r, phase=phase+'-stream', c=c, run=run))
    overlap = min(r['last'] for r in rs) - max(r['first'] for r in rs)
    if overlap <= 0 or any(r['last'] <= r['first'] for r in rs):
        raise ValueError('no shared steady decode window')
    # Same client estimator as repo/probes/decode_ss.py --client, ending at last TEXT,
    # not delayed final usage. Chunk arrival is not an exact per-token counter.
    rates = [(tokens-1)/(r['last']-r['first']) for r in rs]
    ac = [r['accepted'] for r in rs]; rej = [r['rejected'] for r in rs]
    observed = all(x is not None for x in ac+rej)
    accepted = sum(ac) if observed else None
    drafted = sum(ac+rej) if observed else None
    result = {'phase':phase, 'c':c, 'run':run, 'tokens':tokens,
              'ss_agg_tps':sum(rates), 'ss_per_stream_tps':sum(rates)/c,
              'ss_window_s':overlap, 'ttft_s_median':statistics.median(r['ttft_s'] for r in rs),
              'estimator':'decode_ss client proxy: sum full-stream decode rates with confirmed common overlap',
              'accepted_tokens':accepted, 'drafted_tokens':drafted,
              'accept_rate':accepted/drafted if drafted else None,
              'mtp_status':'accepted' if accepted else ('no accepted drafts' if observed else 'unobservable: inspect engine log')}
    emit(a, result)
    return result

def decode(a):
    summaries = []
    for c in (1,2,4):
        rs = [decode_round(a,c,run,1024,'decode') for run in range(a.runs)]
        summaries.append({'c':c, 'ss_agg_tps_median':statistics.median(r['ss_agg_tps'] for r in rs),
                          'ss_per_stream_tps_median':statistics.median(r['ss_per_stream_tps'] for r in rs),
                          'common_windows_s':[r['ss_window_s'] for r in rs]})
    emit(a, {'phase':'decode-summary', 'summaries':summaries})

def prefill(a):
    words = 'river mountain harbor lantern copper meadow orbit velvet canyon ember alpha bravo charlie delta echo foxtrot'.split()
    for target in (8192,32768):
        rng = random.Random(uuid.uuid4().hex)
        prefix = f'Unique document {uuid.uuid4().hex}. '
        filler = [rng.choice(words) for _ in range(target*2)]
        n = target
        count = 0
        for _ in range(12):
            prompt = prefix + ' '.join(filler[:n]) + '\nSummarize the document in one sentence.'
            with post(a.url, '/v1/token/encode', {'text':prompt, 'add_bos_token':False}) as r:
                d = json.load(r)
            count = d.get('length', len(d.get('tokens', [])))
            if count <= 0:
                raise ValueError(f'unrecognized tokenize response {d.keys()}')
            if abs(count-target) <= target*0.01: break
            n = max(1,min(len(filler),int(n*target/count)))
        if abs(count-target) > target*0.02:
            raise ValueError('could not calibrate prefill prompt within 2%')
        r = stream(a, f'prefill-{target}', {'prompt':prompt, 'max_tokens':32, 'min_tokens':32,
                   'ban_eos_token':True, 'temperature':0, 'add_bos_token':False}, chat=False)
        u = r['usage']; cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        seconds = u.get('prompt_time')
        if cached != 0 or not isinstance(seconds,(int,float)) or seconds <= 0:
            raise ValueError(f'prefill is cached or engine timing unavailable: {u}')
        if abs(u['prompt_tokens']-target) > target*0.02:
            raise ValueError('server prompt count outside 2% target')
        emit(a, dict(r, phase='prefill', target=target, calibrated_tokens=count,
                     engine_prefill_s=seconds, engine_prefill_tps=u['prompt_tokens']/seconds))

PROMPTS = {  # R859 c1 decode content kinds (decode rate is content-dependent; MTP acceptance moves with the text)
    'code': 'Write a long detailed Python tutorial with complete code and tests. Keep elaborating.',
    'prose': 'Write a long, detailed essay on the history of the printing press in Europe. Keep elaborating.',
    'chat': 'Explain step by step, in a friendly conversational tone, how to plan a two-week trip to Japan on a budget. Keep elaborating.',
    # R860: long single-file code generation, the shape of a real agent "write the file" turn (2026-10-07 pagoda session)
    'html': ('Write a complete single-file HTML page with inline JavaScript that uses three.js from a CDN to render a detailed '
             'voxel-art scene of a Japanese pagoda in a garden with cherry blossom trees, a koi pond, a wooden bridge, stone '
             'lanterns and falling petals, with orbit controls and soft lighting. Output only the code.'),
}
KIND_TOKENS = {'html': 2048}
# R864: the agent "edit a file" shape, where prompt lookup can copy spans from the prompt. The source is this
# directory's own helper modules (~1.7k tokens), read lazily so other kinds do not depend on them.
EDIT_SOURCES = ('glm53_verify.py', 'r860_score.py', 'glm53_resources.py', 'glm53_follow.py')
def edit_prompt():
    here = os.path.dirname(os.path.abspath(__file__))
    src = '\n\n'.join(open(os.path.join(here, f)).read() for f in EDIT_SOURCES)
    return ('Here is a Python module:\n```python\n' + src + '\n```\nReturn the complete module unchanged except: rename '
            'the function `score` to `score_kinds` everywhere it is used, and add a one-line docstring to every function '
            'that lacks one. Output only the code.')

def c1_decode(a, kind, run, tokens=1024):
    salt = uuid.uuid4().hex
    text = edit_prompt() if kind == 'edit' else PROMPTS[kind]
    body = {'messages':[{'role':'user','content':f'Run {salt}. {text}'}],
            'max_tokens':tokens, 'min_tokens':tokens, 'ban_eos_token':True, 'temperature':0,
            'chat_template_kwargs':{'reasoning_effort':'low'}}
    r = stream(a, f'c1-{kind}-r{run}', body)
    if r['usage']['completion_tokens'] != tokens:
        raise ValueError(f'forced generation ended early: {r["usage"]}')
    ac, rej = r['accepted'], r['rejected']
    res = {'phase':'c1-decode', 'kind':kind, 'run':run, 'tokens':tokens,
           'tps':(tokens-1)/(r['last']-r['first']), 'ttft_s':r['ttft_s'], 'finish_reason':r['finish_reason'],
           'accepted':ac, 'rejected':rej, 'accept_rate':(ac/(ac+rej) if ac is not None and rej is not None and ac+rej else None)}
    emit(a, res)
    return res

def calibrated_prompt(a, target):
    words = 'river mountain harbor lantern copper meadow orbit velvet canyon ember alpha bravo charlie delta echo foxtrot'.split()
    rng = random.Random(uuid.uuid4().hex)
    prefix = f'Unique document {uuid.uuid4().hex}. '
    filler = [rng.choice(words) for _ in range(target*2)]
    n, count = target, 0
    for _ in range(12):
        prompt = prefix + ' '.join(filler[:n]) + '\nSummarize the document in one sentence.'
        with post(a.url, '/v1/token/encode', {'text':prompt, 'add_bos_token':False}) as r:
            d = json.load(r)
        count = d.get('length', len(d.get('tokens', [])))
        if abs(count-target) <= target*0.01: return prompt, count
        n = max(1,min(len(filler),int(n*target/count)))
    if abs(count-target) > target*0.02:
        raise ValueError('could not calibrate prompt within 2%')
    return prompt, count

def depth(a, target, gen=256):
    """Cold prefill of `target` prompt tokens, then `gen` forced decode tokens at that depth (c1)."""
    prompt, count = calibrated_prompt(a, target)
    r = stream(a, f'depth-{target}', {'prompt':prompt, 'max_tokens':gen, 'min_tokens':gen,
               'ban_eos_token':True, 'temperature':0, 'add_bos_token':False}, chat=False)
    u = r['usage']; cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
    seconds = u.get('prompt_time')
    if cached != 0 or not isinstance(seconds,(int,float)) or seconds <= 0:
        raise ValueError(f'prefill is cached or engine timing unavailable: {u}')
    ac, rej = r['accepted'], r['rejected']
    emit(a, {'phase':'depth', 'target':target, 'calibrated_tokens':count, 'prompt_tokens':u['prompt_tokens'],
             'engine_prefill_s':seconds, 'engine_prefill_tps':u['prompt_tokens']/seconds, 'ttft_s':r['ttft_s'],
             'decode_tokens':u['completion_tokens'], 'decode_tps':(u['completion_tokens']-1)/(r['last']-r['first']),
             'accept_rate':(ac/(ac+rej) if ac is not None and rej is not None and ac+rej else None),
             'finish_reason':r['finish_reason']})

def png_rgb(pixel, w=256, h=256):
    """PNG from pixel(x, y) -> (r, g, b)."""
    import struct, zlib
    rows = b''.join(b'\x00' + b''.join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))

RED, GREEN, BLUE = (220, 20, 20), (20, 170, 40), (20, 40, 220)
VISION_CASES = [  # name, pixel fn, question, accepted answers (lowercase substrings)
    ('red', lambda x, y: RED, 'What single colour fills this image? Answer with one word.', ('red',)),
    ('blue', lambda x, y: BLUE, 'What single colour fills this image? Answer with one word.', ('blue',)),
    ('green', lambda x, y: GREEN, 'What single colour fills this image? Answer with one word.', ('green',)),
    ('split', lambda x, y: RED if x < 128 else BLUE, 'This image has two halves. What colour is the RIGHT half? Answer with one word.', ('blue',)),
]

def vision(a):
    """Serving check (user 2026-10-07: "Need vision"). R864-R871 used 64 px solid squares (a handful of image tokens) and
    saw blue answered as "White" twice; 256 px images, three colours and a left/right split are a sturdier check."""
    import base64
    bad = []
    for name, pixel, question, ok_words in VISION_CASES:
        url = 'data:image/png;base64,' + base64.b64encode(png_rgb(pixel)).decode()
        body = {'messages':[{'role':'user','content':[{'type':'text','text':question},
                                                       {'type':'image_url','image_url':{'url':url}}]}],
                'max_tokens':512, 'temperature':0, 'chat_template_kwargs':{'reasoning_effort':'low'}}
        r = stream(a, f'vision-{name}', body)
        text = (r.get('content') or '').lower()
        ok = any(w in text for w in ok_words)
        emit(a, {'phase':'vision', 'expect':name, 'ok':ok, 'text':(r.get('content') or '')[-200:], 'ttft_s':r['ttft_s'],
                 'prompt_tokens':(r.get('usage') or {}).get('prompt_tokens')})
        if not ok:
            bad.append(f'{name}: {(r.get("content") or "")[-80:]!r}')
    if bad:
        raise ValueError('vision check failed: ' + '; '.join(bad))

def c1(a):
    kinds = [k for k in a.kinds.split(',') if k]
    for run in range(a.runs):
        for kind in kinds:
            c1_decode(a, kind, run, KIND_TOKENS.get(kind, 1024))
    for t in [int(x) for x in a.depths.split(',') if x]:
        depth(a, t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--max-batch', type=int, default=4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', choices=['warmup','sanity','measure','c1','vision'], required=True)
    ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--kinds', default='code,prose,chat')
    ap.add_argument('--depths', default='')
    a = ap.parse_args()
    if a.phase == 'sanity': sanity(a)
    elif a.phase == 'c1': c1(a)
    elif a.phase == 'vision': vision(a)
    elif a.phase == 'warmup':
        # R865: c4 cannot share a decode window when max_batch_size < 4 (MAX_BATCH=2 booted fine, then failed here)
        for c in (c for c in (1,2,4) if c <= a.max_batch): decode_round(a,c,0,32,'warmup')
    else:
        decode(a)
        prefill(a)

if __name__ == '__main__':
    main()
