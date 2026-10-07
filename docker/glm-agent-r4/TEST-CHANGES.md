# CPU test changes

Retained all six inherited CPU suites and the three strict SSE suites.
Added `test_r4.py` with 12 methods. Fresh stock and r2 applications each pass
93 methods; seven real SSE/ASGI methods are mandatory in Docker but unexecuted
locally because sse-starlette is unavailable. Rust tokenizers 0.23.2, Pydantic 2,
Jinja2 and Python 3.12 were available in an existing local CPU environment.

Three inherited seam adjustments are intentional:

- `backend_seam.Tokenizer` now has `decode_`; scripted ordinary chunks receive
  distinct synthetic IDs for distinct text. r3 reused one ID for unrelated
  arbitrary strings, which cannot support an honest ID-to-text alignment test.
- The unsafe-tokenizer test now requires exact ordinary-piece round trip,
  instead of expecting r3's forbidden-request exception.
- Real closing-ID tests now expect a phase transition even inside backticks.
  The scripted model supplies explicit control spans; quote/nesting heuristics
  are still tested only for legacy producers without provenance.

Trace log assertions use the r4 event prefix. No failure was weakened into a
skip. The r4 runner requires tokenizers; the strict runner also requires all
SSE dependencies. The clearly named `R4_CPU_ONLY=1` option omits and announces
only the three inherited SSE suites.

`red-tests.txt` is an intentional failing r3 comparison (12 methods, 1 failure,
8 errors); `fresh-stock-tests.txt` and `fresh-r2-tests.txt` are passing r4 runs.
These are scripted CPU proofs, not GPU/model-quality results. The 12 A2 SSE
fixtures are losslessly gzip-compressed and retain original timestamps/lines.
Original pre-parser model text and IDs are absent, so tool XML replay is
explicitly reconstructed from the recorded JSON arguments.
