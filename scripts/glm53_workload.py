#!/usr/bin/env python3
"""R869 broad routing workload for GLM-5.3 (2026-10-07). Thirty-two prompts that share no text with glm53_probe.py's
c1 kinds, spread over agent work (code, edits, refactors, debugging, shell, tool calls with an agent-like system
prompt), docs, chat, French, JSON and maths. Each is generated once, streamed, reasoning_effort low, up to
--max-tokens. It exists to collect a route trace whose expert counts are then scored out of sample on the probe kinds.
usage: glm53_workload.py --url URL --model M [--max-tokens 512]"""
import argparse, json, time, urllib.request

SYSTEM = ('You are a coding agent working in a repository. You can read files, run shell commands and write files '
          'using the provided tools. Think briefly, then act.')
TOOLS = [{'type': 'function', 'function': {'name': n, 'description': d, 'parameters': {'type': 'object',
          'properties': p, 'required': list(p)}}} for n, d, p in (
    ('read_file', 'Read a file', {'path': {'type': 'string'}}),
    ('bash', 'Run a shell command', {'command': {'type': 'string'}}),
    ('write_file', 'Write a file', {'path': {'type': 'string'}, 'content': {'type': 'string'}}))]
AGENT = [
    'The test suite fails with "KeyError: user_id" in services/session.py line 88. Investigate and fix it.',
    'Add pagination (limit/offset query params) to the GET /api/orders endpoint in a FastAPI app and update its tests.',
    'Refactor utils/dates.py: replace the three near-identical parse_* functions with one function and a format table.',
    'Our Dockerfile builds take 9 minutes. Look at it and make the layer caching effective.',
    'Write a Rust function that merges overlapping intervals and add unit tests for edge cases.',
    'Port this Bash deploy script to a Makefile with the same targets: build, test, push, deploy.',
    'Find why the React component re-renders on every keystroke in the search box and fix it.',
    'Write a SQL migration adding a nullable "archived_at" timestamp to the projects table with an index, and a rollback.',
    'Implement an LRU cache class in TypeScript with get/put in O(1) and a test file.',
    'The CI job "lint" fails with ruff E501 errors across 40 files. Fix them without changing behavior.',
    'Write a Go HTTP middleware that adds request IDs and logs latency with slog.',
    'Add retry with exponential backoff and jitter to the fetch_prices() function that calls an external API.',
]
PLAIN = [
    'Explain the difference between optimistic and pessimistic locking in databases, with examples.',
    'Write a short story (about 400 words) about a lighthouse keeper who receives letters from the future.',
    'Summarize the causes and consequences of the 1929 stock market crash for a high-school audience.',
    'Write the README for a small CLI tool that converts CSV files to Parquet, including install and usage.',
    'Explique en français comment fonctionne une pompe à chaleur et quand elle est rentable.',
    'Rédige un e-mail professionnel pour décaler une réunion de projet à la semaine prochaine.',
    'Produce a JSON object describing a fictional company with 5 employees, departments and salaries.',
    'Solve step by step: a train leaves at 14:10 at 92 km/h, another at 14:40 at 118 km/h on the same track. When does the second catch up?',
    'Compare PostgreSQL, MySQL and SQLite for a small SaaS backend in a table, then recommend one.',
    'Write a haiku sequence (five haiku) about debugging at night.',
    'Explain how transformers use attention, without equations, to a product manager.',
    'Draft a privacy policy section about cookies for a small e-commerce site.',
    'What are the trade-offs of microservices vs a modular monolith for a 6-person team?',
    'Write a regex that validates ISO 8601 dates with optional time and timezone, and explain each part.',
    'Translate into English and keep the tone: "On s\'est bien marré hier soir, faudrait remettre ça vite."',
    'Write a YAML GitHub Actions workflow that runs pytest on Python 3.11 and 3.12 with caching.',
    'List ten interview questions for a senior backend engineer and what a strong answer covers.',
    'Describe the water cycle in detail, including evaporation, condensation, precipitation and runoff.',
    'Write a limerick about a cat who learns to use git.',
    'Explain what a Kubernetes readiness probe is and how it differs from a liveness probe.',
]

def gen(a, messages, tools=None):
    body = {'model': a.model, 'messages': messages, 'max_tokens': a.max_tokens, 'temperature': 0.6, 'stream': True,
            'reasoning_effort': 'low'}
    if tools:
        body['tools'] = tools
    req = urllib.request.Request(a.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time(); n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            n += line.startswith(b'data: {')
    return time.time() - t0, n

ap = argparse.ArgumentParser()
ap.add_argument('--url', required=True); ap.add_argument('--model', required=True)
ap.add_argument('--max-tokens', type=int, default=512)
a = ap.parse_args()
for i, p in enumerate(AGENT):
    dt, n = gen(a, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}], TOOLS)
    print(json.dumps({'i': i, 'kind': 'agent', 'seconds': round(dt, 1), 'chunks': n}), flush=True)
for i, p in enumerate(PLAIN):
    dt, n = gen(a, [{'role': 'user', 'content': p}])
    print(json.dumps({'i': i, 'kind': 'plain', 'seconds': round(dt, 1), 'chunks': n}), flush=True)
