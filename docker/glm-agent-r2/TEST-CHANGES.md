# Merged tests and intentional contract updates

No tests are skipped or marked expected-failure. CPU seams execute actual
served functions with model/GPU imports replaced at the boundary; runtime
suites use real SSE/AnyIO and Pydantic models. Each suite runs in a separate
process because the older harness replaces modules in `sys.modules`.

- Original stream suite: all 22 methods retained. Malformed-success and
  duplicate-semantic tests now assert raw-content fallback and stop/length;
  those old expectations contradicted the review requirements. The SSE
  constructor stub is now a class because the production response subclasses
  EventSourceResponse. The channel parser is loaded from patched source, while
  the legacy argument parser and pristine collector remain comparison sources.
- Independent review: all 14 methods retained. Duplicate equivalence follows
  the review's explicitly permitted unique-key restriction and failed turn.
  The send-disconnect test invokes the production response lifecycle instead
  of cancelling an unrelated iterator consumer; an iterator suspended at yield
  cannot own cleanup of its consumer. Its superclass is stubbed in this CPU
  test; unchanged real send-disconnect cases are also run in the runtime suite.
  Literal-think assertions check the tool-fixes gate separately from off-gate
  identity. The inherited usage-only OpenAI-shape failure is changed to assert
  the unchanged pristine shape: fixing it would violate the brief's byte
  identity requirement, and the brief does not request that inherited fix.
- Tool-fixes suite: all 21 methods retained. Grammar tests explicitly enable
  the new forcing sub-flag. Mixed valid/malformed endpoint turns now reject the
  whole turn, matching the live fallback. Off-gate comparisons use pristine
  app with live streaming disabled; empty internal delta containers are
  normalized, IDs are normalized, and the original stream suite separately
  compares serialized bytes. New tests expand pristine off-gate cases.
- Runtime send-disconnect: all three subcases (comment/data/DONE) retained.
  For sse-starlette versions with a persistent per-loop shutdown watcher, the
  watcher starts before the request's task baseline; the test still rejects
  every new leftover request task. The older runtime suite drains persistent
  dependency tasks when closing its own event loop.
- New revision-2 tests: 7 methods cover exact literal reasoning tags, unchanged
  outer thinking state, all response-mode fallbacks, partial opener after valid
  call, no invalid reasoning-cache entry, default-off equivalence and the
  independent forcing gate.
- New lifecycle/collision tests: 3 methods cover real response cancellation
  of both chat and ordinary completion collectors with two choices, send
  exceptions/external cancellation, and null collisions in both key orders,
  multiple extra values, and nested JSON values.

Total: 22 + 14 + 21 + 7 + 3 original runtime + 1 send-disconnect + 3 new
lifecycle methods = **71**, with additional loops/subcases. The upstream suites
and hostile review reproduce nine assertion failures before these repairs;
`evidence/reproduction-review.txt` preserves that initial run. Intermediate
logs remain labelled in evidence; only `combined-tests.txt` is the final run.
