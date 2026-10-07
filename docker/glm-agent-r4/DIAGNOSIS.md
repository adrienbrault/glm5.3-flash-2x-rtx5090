# R877 diagnosis and r4 changes

r2 fails all 12 recorded checker cases. Five end early; seven return valid
`write_file` calls with the correct path and altered bodies. r3 is worse for
these prompts: every toolcheck request raises before generation. The r4 CPU
package repairs that rejection and the demonstrable text/control confusion;
**live r4 inference has not been run**.

## 1. The r3 exception and prompt provenance

The exact exception is the added line at **`r3/app.patch:182`**, landed as
**`/app/common/glm_tag_safety.py:74`**. The engine traceback identifies the
caller as `/app/backends/exllamav3/model.py:995`, in
`validate_context_length`, reached from `common/model.py:340`,
`common/tokenize_offloop.py:102`, and `endpoints/OAI/router.py:163`.

The intended protection was correct: after masking message data before Jinja,
restoring its exact spelling and recording literal spans, r3 requested
`tokenizer.encode(literal, encode_special_tokens=False)`. It then refused any
span whose IDs still included `tokenizer.single_id(literal)`. The refusal makes
quoted code/log/transcript tags unusable as ordinary agent inputs. Removing the
exception without repairing encoding would quietly reintroduce control IDs.

A user literal reaches this path intentionally, through recorded data spans.
Jinja did not turn the masked literal into a delimiter: restoration recovered
it and the **prompt-encode call's effective added-token matching** returned its
registered ID. The exception is in the preflight context check, so no hard-case
sampling or output-parser decision occurred in A3. A3's tag-trace tail is a
filtered, incomplete log; the full engine log contains successful nonliteral
warmups and vision generation. “A3 never generated” applies to toolcheck, not
to every request during this boot.

Stock `prev/app/endpoints/OAI/utils/chat_completion.py:376` inserts messages into
template variables and renders them. The supplied upstream template in
`evidence/r877/A2/template.json` inserts user/tool text verbatim. Its assistant
history fallback splits `content` on `</think>` when no explicit reasoning is
provided, introducing a separate history-loss bug for quoted closers. Stock
`backends/exllamav3/model.py:994,1046` encodes the rendered prompt with
`encode_special_tokens=True`; no user-role-specific escaping exists. Therefore
native added-token matching encodes a user `</think>` as its registered ID, not
ordinary BPE text pieces. R877 reports `<think>` = **154841**, `</think>` =
**154842**. r3's failures demonstrate that its false flag also failed to prevent
those IDs on these inputs.

The model's structural assistant prefix/outer reasoning delimiters should still
use the registered IDs. Literal message data should use ordinary pieces under
the requested agent contract. Native upstream rendering/tokenization supplies
added IDs even in message data; that is not evidence that the model was trained
to require ordinary pieces for quoted tags, or that it will always copy them
correctly after this intervention. Training expectations and live adherence
cannot be established from the supplied SSE.

The exact model tokenizer JSON and installed ExLlama tokenizer source are absent.
Two mechanisms can explain the ineffective false flag: a wrapper that merely
suppresses automatic BOS/EOS insertion, or an added tag registered as
`special=False` (still matched even in split-special mode). The current
[ExLlama tokenizer source](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/tokenizer/tokenizer.py)
separates those operations and also explicitly preserves unspecial added-token
matching. It is a guide, not the source hash of flan's fork. r4 reproduces the
second mechanism with the real Rust `tokenizers` library and repairs either
mechanism by checking the emitted ID and subdividing that data span when needed.
`tests/probe_literals.py` records the **actual model's** added-token flags,
native IDs, r4 ordinary IDs and byte round trips without loading weights.

## 2. The r2 early stop is upstream of the parser; the terminal ID is missing

Raw files are `evidence/r877/A2/toolcheck/*.raw.json` (timestamp, SSE line pairs).
`out/a2-analysis.json` summarizes each capture directly, not the prose brief.

| Case | zero r01 / r02 | default r01 / r02 |
|---|---|---|
| hard | stop at 26 / 26 tokens | tool_calls / stop at 19 tokens |
| think-only | stop at 25 / 27 tokens | tool_calls / tool_calls |
| close-only | tool_calls / tool_calls | tool_calls / tool_calls |

All tool calls fail body validation. Thus the brief's “close-only stops at
zero” and “all defaults return calls” statements are inaccurate. The two
hard/zero reasoning tails are exactly the same and stop after the visible
`</fake> ` in the backtick example. `</fake>` is not a parser reasoning or
termination tag.

In landed r2, `TagStreamParser._handle_tag` at
`endpoints/OAI/utils/stream_parser.py:160` can consume think spellings/change
channels; it does not produce a finish, cancel a job, or stop iteration. The
collector at `chat_completion.py:1015` breaks on a backend `finish_reason`.
`backends/exllamav3/model.py:1649` handles engine `result['eos']`, emits a finish,
and breaks. `handle_finish_chunk:1258` maps every reason other than token limit
to `stop`. The raw wire includes the terminal finish/usage/DONE, and A2's engine
log independently records 26 decode steps/tokens for each hard/zero request
(`#14/#15`, log lines 169–178). This is not a parser merely hiding a continuing
long generation. Parser tag deletion is a separate reproducible corruption.

A3 stop logs help **exclude a previous hypothesis** for this model boot:
HF EOS = `[154827,154820,154829]`, backend EOS =
`[154820,154827,154829]`, think IDs = `154841/154842`, `removed=[]`.
Think IDs are not in that runtime stop set. The same model/config and unchanged
r2 backend stop assembly make a think-stop misconfiguration unlikely; these
are not direct A2 terminal-event logs.

The A2 captures cannot identify the sampled terminal ID or prove that prompt
control IDs caused the model to stop. `include_usage=True` suppresses the
original finish choice and substitutes a usage finish, losing `eos_reason` and
`stop_str`. A real EOS/observation stop, a string stop or another engine
termination remains distinguishable only with additional evidence. Do not
infer the hidden token from the next intended copied character.

The exact settling probes are packaged:

1. `operator-run.sh` repeats the original r3 prompts against r2 in **plain**
   mode at zero, two repeats per case. Plain responses expose `eos_reason` and
   `stop_str`, without changing the historical message text.
2. If that does not expose enough detail, `r2-eos-probe.patch` adds this one
   opt-in console JSON event immediately inside r2's `if result.get('eos')`
   branch, before `handle_finish_chunk`:

   ```text
   [GLM-TAG-R4-A2-EOS] {request_id, effective_stops, eos_reason,
     eos_triggering_token_id, eos_triggering_token_str,
     eos_triggering_string, new_tokens}
   ```

   `Dockerfile.r2-trace` builds a separately named diagnostic r2 image with
   exactly that logging addition. The real terminating ID/reason settles the
   engine mechanism. Establishing *causation* by prompt encoding also requires
   a controlled native-IDs versus ordinary-pieces repeat with the same sampling
   and parser, after recording the actual tokenizer metadata; the current
   r2/r4 comparison changes both input encoding and parsing.

## 3. Default-temperature failures are body failures, not path failures

All seven captured calls are valid JSON named `write_file`, path `/tmp/y.py`.
For every returned call, the first difference is on line 1. Offsets below are
zero-based UTF-8 byte offsets in the **JSON-decoded file content**.

| Capture | Offset | Actual / expected byte | Additional visible alteration |
|---|---:|---|---|
| hard default r01 | 11 | `0x62` (b) / `0x5c` (backslash) | tag pair replaced by `ǭ`; closing quote absent |
| think-only default r01 | 28 | `0x5c` / `0x3c` (<) | `<think>` replaced by two backslashes plus `mock` |
| think-only default r02 | 11 | `0x62` / `0x5c` | `<think>` replaced by space + `vas` |
| close-only default r01 | 11 | `0x62` / `0x5c` | tag replaced by row number; only 199 lines |
| close-only default r02 | 11 | `0x62` / `0x5c` | tag replaced by `...` |
| close-only zero r01 | 28 | `0x20` / `0x3c` | closer replaced by a space |
| close-only zero r02 | 28 | `0x20` / `0x3c` | closer and final quote absent |

For hard/default r01, JSON string representations of the first-line contexts:

```json
{
  "expected": "row_1 = \"a\\\\b \\\"q\\\" </fake> <think>x</think>\"\n",
  "actual":   "row_1 = \"a\\b \\\"q\\\" </fake> ǭ\n"
}
```

The 7,528-character argument JSON is valid; the body is 6,092 characters,
6,292 UTF-8 bytes versus the expected 9,492 bytes including final LF. This is
not merely a transport JSON-escaping difference. Reasoning already discusses
the substituted glyphs/words or missing closer. A parser cannot restore content
the model did not emit. Raw SSE lacks token IDs, so these captures do not prove
whether every missing tag was avoided by the model or consumed before SSE.

## 4. Agent contract, implementation and verification boundaries

- User/tool/assistant message data, reasoning history, tool argument history
  and schemas preserve literal tag spelling but encode it as ordinary pieces.
  Structural tags introduced by the template remain control IDs. Masking before
  Jinja also prevents its assistant-content split from interpreting a literal.
- Ordinary-piece `<think>` / `</think>` in generated reasoning/content are text,
  including unquoted/unbalanced examples. Only actual reasoning control IDs
  change the reasoning phase; a real closer does so even inside backticks.
  Think tags inside tool arguments remain argument data.
- For GLM protocol delimiters, only real control IDs end reasoning or the
  turn; a visible tag spelling has no terminating effect. Normal explicit API
  stops, limits and cancellation still apply. Native EOS / observation stops remain. r4 keeps r3's narrowly scoped think-stop hygiene,
  which removes nothing on the supplied R877 stop set.
- Parsed tool string values preserve exact bytes. Transport JSON may escape
  those bytes; JSON-decoded values must match. r4 never substitutes glyphs,
  “repairs” file contents to the expected answer, or retries generation.

r4 changes, with **landed r4** line numbers:

- `common/glm_tag_safety.py:60,109`: replace the refusal with checked ordinary
  pieces, preserving BOS, request-local spans, context/generation encoding and
  cached IDs. Literal angle tags/role tags/mask tags are protected too.
- `glm_tag_safety.py:123`, `backends/exllamav3/model.py:1670`: carry ID-derived
  reasoning-control spans separately from engine text across merged chunks and
  multibyte prefixes. Mismatched decoding is logged, not guessed into a control.
- `stream_parser.py:159`, `chat_completion.py:915`: use those spans on the
  production ExLlama GLM path; text spelling alone cannot close reasoning.
  Legacy producers without provenance retain the inherited parser interface;
  their tests do not establish the new ID-aware contract.
- Prompt masking excludes actual multimodal aliases; positional embeddings
  retain their global offsets when segmentation is necessary. No vision source
  or configuration was changed. The R877 red/blue/green/split vision passes are
  inherited observations, not a new r4 GPU test.
- `TABBY_GLM_TAG_TRACE=1` emits `[GLM-TAG-R4]` prompt, stop-set, generator,
  control-span and parser diagnostics. `decode_mismatch=true` is an explicit
  reason to retain the installed tokenizer/job sources and inspect alignment.
  Neither temperature defaults nor sampler construction were changed.

An actual `</think>` ID cannot simultaneously mean a literal and a delimiter
in the same reasoning state. The contract assigns literal text to ordinary
pieces. If the model itself emits the control ID while intending a quotation,
CPU code cannot infer that intent; r4 follows the ID. Live exact-body success
and the actual tokenizer's ordinary-piece round trip remain operator checks.

The old checker was correct to reject these A2 calls, but its `exact_body`
comparison used `splitlines()`, accepting CRLF/missing/extra final LF as equal.
`out/glm53_toolcheck_r4.py` checks UTF-8 bytes and reports the first differing
byte/context hex. It adds an explicit LF/one-final-LF instruction to remove the
old prompt's newline ambiguity. All other literal patterns, budgets and
zero/default sampling choices remain; run this same checker on both images.
The original checker is retained as `r4/glm53_toolcheck.py` for exact historical
prompt probes.

Fresh zero-fuzz runs from both stock and r2 pass **93 CPU methods each**
(81 inherited + 12 r4). Fixtures preserve all 12 original A2 SSE captures.
Replays preserve their emitted reasoning/content and recorded argument bytes;
reconstructed GLM tool grammar is labeled as reconstruction, since original
pre-parser XML/IDs are absent. Tests also cover actual Rust added-token
matching, message roles/schemas with the supplied upstream template, real vs
literal closers, merged chunks, UTF-8 boundaries, 1,000-line exact arguments,
BOS/cache/context encoding and vision-alias preservation. Red tests against r3
reproduce its refusal and channel loss (`1 failure, 8 errors` in 12 methods).

Seven inherited real SSE/ASGI tests could not run locally because sse-starlette
is absent and network package installation is unavailable. They remain
mandatory in the Docker build, bringing the strict total to 100. Docker, GPU,
SSH and live model inference were not run. See the packaged fresh test logs,
`TEST-CHANGES.md` and `HOW-TO-RUN.md`. Confidence is high in the demonstrated
refusal/input/parser/byte-contract defects; the exact historical A2 stop token
and live r4 model behavior remain unproven, with exact probes provided.
