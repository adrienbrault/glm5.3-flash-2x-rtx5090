# Combined revision-2 implementation audit

The combined patch changes ten Python files against pristine `app/`. It starts
with the supplied stream overlay and toolfix overlay, fixes the four ranked
review findings and inherited literal-think deletion, and adds the forcing
sub-flag. Template bytes and TensorFold credit files are preserved.

| Finding | Cause and repair | Regression coverage |
| --- | --- | --- |
| High 1: send disconnect leaks | Wrapper is suspended at yield, so cancellation of a sender cannot enter its finally. The enabled response subclasses EventSourceResponse and closes its iterator in a shielded response-lifecycle finally after superclass tasks stop. Both endpoint source generators cancel/join their collector tasks when keepalive is enabled. | CPU lifecycle test; real ASGI comment/data/DONE cancellation; two-choice chat/completion collector cleanup; external cancellation and send error |
| High 2: partial/malformed successful finish | Completed-call list was used without accounting for invalid/open calls. Live parser now reports unfinished state; terminal collector refuses tool success for invalid/open turns and emits raw content with stop or length. With tool fixes, all response modes validate the whole turn. | Complete then truncated; usage finish; malformed later; partial opener; raw fallback across modes and chunk widths |
| Medium 3: null collision | Decoding canonical JSON collapses Python None and explicit string null into the same JSON key. Close now appends the independently paired extra-value member, using last extra value, without collapsing the top-level member sequence. | Exact string equality, reverse key order, multiple extras, nested JSON |
| Medium 4: duplicate key divergence | Append-only live JSON cannot replace an earlier value. Live scanner restricts accepted calls to unique real keys and invalidates the entire turn on duplicate. No successful duplicate-key call is advertised. | Both merged original/review assertions and whole-turn fallback tests |
| Inherited literal think deletion | Reasoning tokens were recognized but dropped inside TOOL. An explicit parser option preserves these tokens without changing reasoning state; enabled only by GLM tool fixes. | File body with whitespace, controls, quotes, Unicode, repeated tags; all response modes; off-gate corruption retained |

## Gates and production semantics

`TABBY_STREAM_TOOLCALLS=1` enables GLM live streaming only for streaming requests.
`TABBY_SSE_KEEPALIVE_S>0` enables serialized idle comments and response/collector
cleanup. `TABBY_GLM_TOOL_FIXES=1` enables GLM prompt/history/name/count/typing
fixes and literal-tag preservation. `TABBY_GLM_FORCING=1` additionally enables
GLM required/named forcing, and requires tool fixes. All default off. Existing
Qwen forcing remains independent of these GLM gates.

The safe invalid-turn terminal is raw tool-channel text in content, no accepted
call list, stop on ordinary stop and length on a token cap. Earlier transmitted
prefixes cannot be erased. For a stream with earlier content, that content
remains and raw tool text is appended at termination. Reasoning stays in its
own channel. The raw fallback is an explicit rejection contract, not a claim
that client adapters can retract earlier displayed calls. Forcing postconditions
may instead send an explicit generation error when the request demanded a call.
Successful completed calls keep the existing finish behavior.

Unique keys are required for successful live calls. Legacy malformed extra
values retain independent key/value behavior when tool fixes are off; Python
None and string null are different internal keys and serialize in canonical
order even though strict duplicate-rejecting JSON readers may reject that
inherited collision. Tool fixes reject extra/unpaired values. Literal XML
`</arg_value>` and `</tool_call>` still delimit values/calls; file text containing
those protocol delimiters cannot be represented literally by this format.
The requested `<think>`/`</think>` case is lossless with tool fixes on.

## Retained toolfix overlay work and limits

- History preprocessing retains explicit client reasoning, accepts a string
  `reasoning` alias when `reasoning_content` is absent, normalizes empty historical
  arguments, preserves unmatched tool results, honors clear_thinking and closes
  the initial think block for explicit thinking-off.
- Per-model reasoning memory uses hashes of full visible history and canonical
  calls; it is bounded to 256 entries / 8 MiB UTF-8. It stores only successful
  call turns with reasoning. Identical visible histories remain ambiguous;
  worker changes, compaction, edits, eviction and restart can cause misses.
  It is not a durable session/tenant-isolation mechanism.
- `none` removes advertised tools; named filters advertised/accepted names;
  parallel false limits exposed calls before allocating further live headers.
  Required/named engine grammar is independently gated pending a real tokenizer
  probe. With grammar off those choices can still produce no call.
- Whole-schema typing preserves exact declared strings, handles nullable and
  anyOf/oneOf types, avoids treating bool as integer and keeps nonfinite values
  as text when fixes are on. It is coercion, not full JSON Schema validation.
  `$ref`, intersections, enum membership and nested constraints are not fully
  enforced. The grammar constrains tool names/count, not schema values.

These history/typing adaptations credit TensorFold Spark patch 0620 under
Apache-2.0. `LICENSE.tensorfold`, `NOTICE.tensorfold` and `THIRD_PARTY.md` retain
credits; no engine/CUDA implementation or performance claim is included.

## Verification

The final fresh zero-fuzz patch application, base/landed hashes and landing
assertions passed. All 71 merged test methods pass using actual request models,
real sse-starlette 3.4.11 / AnyIO, and replayed backend output. Default-off
pristine prompt/response comparisons include malformed calls, duplicates,
null collisions, reasoning and usage; UUIDs are controlled/normalized by the
harness. Off-gate response uses the original EventSourceResponse constructor.
The original inherited usage-frame shape remains unchanged.

The live long-file probe, choice/parallel probe, history probe and real-tokenizer
compilation/mask probe are shipped. No Docker, Git, SSH, GPU, production
tokenizer or live model endpoint was used here. The Docker build runs the
entire suite on the actual parent-image dependency versions. Grammar and
client-adapter adoption require the operator checks in HOW-TO-RUN.md.
