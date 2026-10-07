# r4b verification

Inherited 100 CPU/SSE methods retained. Six new methods in test_r4b.py:

1. Real Rust BPE merge across the space and `</fake>` is retained (red on r4).
2. Surviving added tokens retain the model's original IDs after Rust rebuilding.
3. Identical messages twice through the real template, preflight, request cache,
   fresh encode and verification flag. IDs match; cache returns a clone.
4. Actual A4 hard zero-r01 SSE and selective tag trace replay: real closer IDs,
   two recorded standalone-space deltas, all tool-body bytes preserved through
   real collector/parser in plain, buffered and live modes. XML is explicitly
   reconstructed from captured JSON arguments; original pre-parser IDs are absent.
5. A literal control also reachable through base BPE merges falls back to
   checked subdivision without rejection.
6. runs/registered/legacy arms preserve decoded text and literal/control distinction.

Fresh zero-fuzz stock and r2 layouts each pass **106**, including seven actual
SSE/ASGI/disconnect methods. Logs: fresh-stock-tests.txt / fresh-r2-tests.txt.
Local environment: CPython 3.12, tokenizers 0.23.2, Pydantic 2.13.5, Jinja2
3.1.6, AnyIO 4.15.1, sse-starlette 3.4.11, Starlette 1.6.0. CPU dependencies
were copied from existing local caches/environments, no install/download.
Build reruns strict checks in the actual parent image. Earlier limited/applied
logs are retained; the fresh logs are authoritative.

The real model tokenizer and installed ExLlama tokenizer are not in the brief's
local inputs. probe_prompt_cpu.py is supplied for that whole-prompt seam and
is unexecuted locally. GPU page dedup/recurrent snapshots and inference are
not simulated or claimed tested. All GPU commands are for the operator.
