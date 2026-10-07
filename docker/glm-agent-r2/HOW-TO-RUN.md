# TabbyAPI r861 GLM agent revision 2

One combined `app.patch` applies directly to pristine `app/` / the image
`tabbyapi:r828-prompt-lookup-r3`. It includes both earlier overlays and the
review repairs. No intermediate overlay image or patch is required.

## Verify and build

From this directory, with Python providing Pydantic 2, Jinja2, sse-starlette,
AnyIO and the baseline logger dependencies:

```sh
python3 -B tests/check_hashes.py SHA256SUMS .
R2_PYTHON=python3 bash tests/run.sh /path/to/pristine/app
./build.sh
```

`build.sh` tags `tabbyapi:r861-glm-agent-r2`, uses the locally present parent
image and disables network/pulls. Docker verifies delivery hashes, pristine
file hashes, zero-fuzz dry-run/application, landed hashes, both landing asserts
and all 71 merged tests. The image build requires no model or GPU. `tests/run.sh`
uses a temporary full source copy and leaves the supplied baseline untouched.

The image sets all controls to off. For GLM-5.3 production, add these to your
existing container invocation, preserving your model/config mounts and ports:

```sh
-e TABBY_STREAM_TOOLCALLS=1 \
-e TABBY_SSE_KEEPALIVE_S=5 \
-e TABBY_GLM_TOOL_FIXES=1 \
-e TABBY_GLM_FORCING=0
```

For example, replace only the existing image argument with
`tabbyapi:r861-glm-agent-r2` and add those four `-e` arguments. Keep the supplied
`glm53-chat_template.jinja` installed as the model's selected chat template;
the image does not silently select a different template. The model's tool
format must be `glm4_5`. Boolean flags require exactly `1`. Keepalive accepts a
finite positive interval in seconds; `0` or unset disables this wrapper and
retains the parent's original configured SSE ping behavior.

## Live long file-write and keepalive probe

```sh
export TABBY_BASE_URL=http://127.0.0.1:5000
export TABBY_MODEL=YOUR_LOADED_MODEL_NAME
# If authentication is configured, export TABBY_API_KEY in your shell.
python3 tests/probe_live.py --expect-live --expect-comments \
  --request-out /tmp/r861-long-write.request.json \
  --trace-out /tmp/r861-direct-sse.json
```

This asks `write_file` for 1,000 numbered source lines with quotes, backslashes,
Unicode-compatible text, `</fake>` and literal `<think>literal</think>` in the
file body. It prints receive timestamps, argument-fragment sizes and running
length, `: keepalive` comments, usage and `[DONE]`. It reconstructs the exact
argument JSON and asserts successful `tool_calls`, at least ten deltas spread
over time, an early delta before `[DONE]`, at least 20 KiB of body, and the
literal reasoning tags. It never executes the file-write tool.

Comments occur only during idle intervals, so a continuously flowing stream
may legitimately have none. For an explicit comment test, temporarily use
`TABBY_SSE_KEEPALIVE_S=0.1`, restart your usual container, and rerun during an
idle prefill/reasoning interval. Restore `5` afterward. A failed assertion is
evidence to investigate, not proof the deployment passed. Repeat through your
Olla/proxy URL and compare receive-time traces; proxies must flush SSE comments
and data without buffering. A comment reaching the socket does not prove an
application's parsed-event watchdog resets on comments.

For a released-fragment trace suitable for deterministic CPU replay:

```sh
python3 tests/probe_live.py --capture-raw /tmp/r861-real-fragments.json
R861_TRACE=/tmp/r861-real-fragments.json bash tests/run.sh /path/to/pristine/app
```

The delivered fixture is synthetic, explicitly labelled as such. Raw capture
uses `/v1/apply-template` followed by `/v1/completions`; inspect the trace for
complete calls before treating it as the successful-call fixture.

## Tool choice and parallel calls

With tool fixes enabled and forcing still off:

```sh
python3 tests/probe_choices.py --save /tmp/r861-choice-probes
```

The probe sends `none`, named `echo`, `required`, auto with parallel false/true,
and required with parallel true, each streamed and non-streamed. It saves exact
requests/responses, reconstructs SSE calls, prints names/finishes/comments,
asserts no calls for `none`, only `echo` for named, at most one call when parallel
is false, valid JSON argument objects, string typing and rejection of an
unadvertised named choice. Without grammar forcing, named/required requests
can end without a tool call; the probe reports that rather than claiming forced
generation. It does not require two calls with parallel true: model choice is
nondeterministic. It checks exposed calls, not underlying extra model tokens.

Reasoning-history recovery has a separate exact-request probe:

```sh
python3 tests/probe_history.py --url "$TABBY_BASE_URL/v1" \
  --model "$TABBY_MODEL" --save /tmp/r861-history-probes
```

The first response must have a tool call and nonempty reasoning. Increase its
output budget if needed. Compare preserved, omitted, renumbered and explicitly
empty reasoning histories using server prompt dumps as well as responses.
The per-model reasoning memory is bounded and best-effort; explicit reasoning
from the client is authoritative. See `AUDIT.md` for limits.

## Enable grammar forcing only after the production tokenizer probe

The GLM grammar has a separate sub-flag, `TABBY_GLM_FORCING=1`; it also requires
`TABBY_GLM_TOOL_FIXES=1`. Leave it off until this probe passes against the actual
loaded model's tokenizer JSON and its real EOS ID:

```sh
TOOLFIX_APP=/app python3 /opt/r861/tests/probe_grammar.py \
  /actual/model/tokenizer.json --eos-id ACTUAL_EOS_ID
```

Run in the server Python environment with `tokenizers` and `llguidance`, without
loading model weights. The probe prints tag token IDs, requires a single-token
reasoning end, compiles both full and continuation grammars, checks masks for
EOS/name/count restrictions and exercises malformed calls and literal text.
It sets both flags only within the probe process. A failure means keep forcing
off. A pass is a prerequisite, followed by live model and log verification:

```sh
# After adding TABBY_GLM_FORCING=1 to the existing container env and restarting:
python3 tests/probe_choices.py --expect-forcing --save /tmp/r861-forced-probes
```

Check server logs for filter-construction failures; the backend can fail open
on grammar initialization. The forced-call postcondition remains in place.
Passing the tokenizer probe alone does not certify GPU generation or client
adapters.

## Invalid tool-turn contract and rollback

Live deltas are append-only. A duplicate argument key, malformed accepted call,
unclosed block/value, or distinct partial tool opener invalidates the whole
live turn, including any earlier complete calls. The terminal delta emits the
raw tool-channel text as content and finishes `stop` on natural termination,
or `length` on a token cap. It does not invent closing argument JSON. Earlier
tool prefixes remain on the wire; **execute calls only after a successful
`tool_calls` finish with valid complete arguments**. The reasoning memory does
not record an invalid turn. Client adapters that execute prefixes despite the
terminal failure require their own fix before adoption.

With tool fixes enabled, buffered SSE/non-streamed collectors apply the same
whole-turn validation and raw-content fallback. With only live streaming on,
legacy independent key/value pairing is retained, including extra values
serialized under `"null"`; explicit string key `"null"` and Python `None` members
both survive in their canonical order. Duplicate real argument keys are
rejected in live mode. See `TEST-CHANGES.md` for the deliberate updated tests.

Turn all four controls to `0` (or remove them) and restart to retain the parent's
prompt/response behavior, including its inherited usage-frame shape and
literal-tag deletion. This byte identity is covered by pristine comparisons.
SSE keepalive cleanup is independently gated. Off-gate deployment does not
provide the tool fixes.

Local verification: 71 test methods pass, including real sse-starlette 3.4.11 /
AnyIO ASGI send cancellation and actual production collectors. No Docker build,
GPU run, production tokenizer probe or live HTTP generation was performed in
this implementation environment. `evidence/combined-tests.txt` is the full
fresh-application test log. Run the image build's tests against the actual
parent dependency versions before deployment.
