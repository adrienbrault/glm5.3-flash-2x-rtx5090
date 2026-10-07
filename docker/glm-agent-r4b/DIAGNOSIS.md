# R880 / r4b diagnosis (incremental)

R880 A4 has 16/16 valid completed write_file calls and only 2/16 exact bodies.
The leading hypothesis is r4 segmentation at nonregistered literal `</fake>`
and its whitespace neighbors. This is not yet a demonstrated tokenizer byte
mutation: the capture lacks whole-prompt IDs/round trips and ordinary output IDs.
No output whitespace normalization is justified.

All four 1000-line requests report `cached_tokens: 0` and engine `none cached`.
The equal prompt hash/token count is not an ID comparison. Page dedup and
recurrent snapshot reuse are not logged. A cached pre-r4 ID claim is unsupported.

Evidence extraction is in r880-analysis.json / r880-tag-events.json, generated
by analyze_r880.py. Exact SSE windows and final findings will follow.

## Trace localization: hard-stream-zero-r01

Request ID `10eb18db8d574710a254d9d067e6be47`.
`A4/toolcheck/hard-stream-zero-r01.raw.jsonl` lines are 1-based:

| Channel / location | First space delta | Extra space delta | Next spelling delta |
|---|---|---|---|
| First reasoning example | 37 / 1.770 s | none | 39 / 1.820 s `</fake` |
| Later reasoning, before tag | 165 / 4.916 s | 167 / 4.960 s ` ` | 169 / 5.004 s `</fake` |
| First tool body, before tag | 975 / 30.227 s | 977 / 30.307 s ` ` | 979 / 30.377 s `</fake` |
| First tool body, after tag | 983 / 30.512 s | 985 / 30.581 s ` ` | 987 / 30.660 s `<think` |

The first body closer finishes at line 981 / 30.442 s (`>`). These are actual
SSE argument fragments, not a reserialization of an expected file. The model's
reasoning first spells single spaces, then switches to doubled spaces and says
it will copy them. The two space deltas precede tool-argument completion. This
rules out corruption solely in client JSON assembly. Replay of the captured
arguments through the real parser/collector preserves those same wrong bytes;
it neither inserts nor repairs spaces. It cannot establish the pre-parser bytes.

The A4 engine trace is **selective**. For this request it has just one non-EOS
`generator` ID list: `[154842]` at engine line 427. Line 433 verifies the real
closer's `[0,8,"</think>"]` span with `decode_mismatch=false`; parser changes
reasoning phase at line 437. Line 443 opens tool parsing. Line 453 records
`154829`, `<|observation|>`, at the finish. No ordinary `</fake` or space IDs
are recorded, because those spellings arrive split over engine chunks and the
r4 logger logs only chunks containing a complete `<think>`/`</think>` spelling
or EOS. **The IDs producing deltas 167, 977 and 985 are absent.** Assigning
IDs (including a guessed standalone-space ID) would be fabrication.

Thus `decode_mismatch=false` only verifies the real control boundary. It says
nothing about spaces in prompt ordinary runs or tool-body output. R880's CPU
literal probe proves isolated ordinary-piece tags decode exactly: `</fake>` =
`[522,30450,29]`, `<think>` = `[13699,766,29]`, `</think>` =
`[522,339,766,29]`. Those are **probe input IDs**, not sampled output IDs.
The probe's scope explicitly excludes ExLlama imports. It supplies no full
prompt, no neighbors, and no `spaces_between_special_tokens` setting. Metadata
has `lstrip=false`, `rstrip=false` for the actual added think tokens; they are
`special=false`, explaining why r4 subdivides them. No captured evidence shows
special-token padding causing this defect.

Ranked remaining hypotheses:

1. Segmentation changes the BPE topology around ordinary tags/whitespace, and
   the model copies its noncanonical interpretation. Prediction: whole ordinary
   runs with disabled literal added-token matching improve exact copying even
   if both old and new prompt ID sequences decode byte-exactly.
2. The installed prompt encoder inserts/changes bytes at segment boundaries.
   Prediction: the legacy whole-prompt decode differs from rendered prompt at
   those spaces; contiguous runs restore exact decoding.
3. Generation/output decoding adds spaces (or the model deliberation chooses
   them despite an exact prompt). Prediction: prompt IDs/decode are exact in
   all arms; sampled ID-window decoding distinguishes model emission from
   engine text padding. Parser assembly alone is contradicted by replay.

r4b defaults to `TABBY_GLM_LITERAL_ENCODING=runs`; `registered` retains only
registered-tag segmentation, `legacy` reproduces r4. All three are shipped;
no file-body trimming, substitutions, checker-specific rewriting or retries.

## 1000-line repeated requests and cache provenance

| Case | Request ID | Prompt tokens / cached | Prompt time | Result |
|---|---|---|---|---|
| zero r01 | a7dd1d82109f4484b57af70cf813d86c | 246 / 0 | 0.82 s | PASS |
| zero r02 | 00111895d645492eb0f9a1be29ad8263 | 246 / 0 | 0.85 s | FAIL at byte 20 |
| default r01 | d53347a441094133a811dc8a6910a10b | 246 / 0 | 0.83 s | PASS |
| default r02 | dca6a99e55034edf9ec44b85588ce7e6 | 246 / 0 | 0.83 s | FAIL at byte 11 |

Full request JSONs are equal within each temperature pair. All four
`prompt_literals` events have SHA256
`b680676ecc5931b3c9f067cab57a372f2ef5d7ff90dc6676a743af536ba9d297`,
3 spans, 246 tokens. The engine summaries explicitly say `none cached`
(lines 1089, 1141, 1193, 1245); usage agrees. None of these events contains
prompt IDs or their hash. R880 disabled `EXL3_CACHE_TRACE`, so page digests,
page dedup, allocation selection and recurrent-state snapshots are unobserved.
No evidence establishes reuse of a GPU prefix/snapshot, let alone old native
IDs in one. The 246-token prompt is shorter than the stock backend's 256-token
allocation unit for nonrecurrent caching; the recurrent path may use another
boundary. This supports the zero-cache observation but isn't an installed
ExLlama allocation trace.

The separate app cache is `_prompt_ids`, **request-local**, not a cross-request
prefix cache. In the hash-verified r4 patch, `validate_context_length` calls
`glm_tag_safety.encode_prompt`, stores those IDs under `(prompt,add_bos)` with
a tokenizer weakref, and `_encode_prompt` returns a clone when that weakref
matches. Fresh generation encoding calls the very same r4 encoder. There is
no code path here storing pre-r4 native IDs after that validation. This source
inspection plus CPU cache regression is evidence about the app cache; it
cannot prove unrecorded runtime IDs or external GPU-cache correctness.
Neither “cache stores pre-r4 IDs” nor “r4 loses information on reuse” is
established. Even greedy output divergence can arise outside that app cache;
recurrent/kernel numerical variation versus ID variation needs the probe.

Exact JSON string representations of the first-line differences (file bytes,
not transport escaping):

```json
{
  "expected": "row_1 = \"a\\\\b \\\"q\\\" </fake> <think>x</think>\"\n",
  "zero_r02_actual": "row_1 = \"a\\\\b \\\"q\\\"  </fake>  <think>x</think>\"\n",
  "default_r02_actual": "row_1 = \"a\\b \\\"q\\\" </fake> <think>x</think>\"\n"
}
```

Zero r02: byte 20 actual `0x20`, expected `0x3c`. Default r02: byte 11
actual `0x62`, expected `0x5c`, i.e. one of the two backslashes was omitted.
Both still contain 1000 lines and 1000 intact tag pairs. Close-only default
r02 also omits a closing quote on line 1; a space-only fix cannot guarantee all
captured byte defects vanish.

## r2 control stop

The plain probe confirms **three** `<|user|>` stop-token terminations:
hard zero r01/r02 and think-only zero r01. Think-only zero r02 finishes with
`<|observation|>` instead. In the three stopped responses the visible
reasoning ends immediately after `</fake> ` while quoting the line, before
visible `<think>`. `eos_reason=stop_token`, `stop_str=<|user|>` names the
terminating control; the literal probe maps it to ID **154827**. This confirms
a control-token termination at that visible boundary, not just a parser
hiding the rest. The plain response does not include an independent sampled
integer ID or an unparsed full token sequence. In particular, it does **not**
show a literal `<think>` ID being sampled immediately before 154827.
The requested stronger “ID right after literal tags” ordering is not in the
trace. It supports control/literal confusion but is not controlled causal
proof. Use the retained r2-eos-probe.patch/Dockerfile.r2-trace from r4 for a
numeric r2 terminal trace if needed; r4's parser and encoder changed together.

## Settling probes (no guessing)

The r4b installed-tokenizer CPU probe runs all four rendered prompts twice
through real app template/preflight/cache/fresh-encode methods in each of
`legacy`, `registered`, `runs`. It records full prompt IDs, fresh IDs, rendered
and decoded prompts, exact equality and tokenizer source hash. No weights.
It requires the model tokenizer files in the operator container and is
unexecuted here; the model files are absent locally.

The GPU runner keeps `TABBY_GLM_TAG_TRACE=1`, enables existing
`EXL3_CACHE_TRACE=1`, and captures request-correlated `job_input` full IDs /
ID hash / decode equality. New `literal_window` events include split ordinary
spelling IDs and 16 following IDs, so both neighboring spaces can be checked.
The native sampler and output decoder are unchanged. Run the same matrix
with `R4B_LITERAL_MODE=runs` (default) and then `registered` and `legacy`
(separate clean boots). Only the encoding mode differs. If IDs/decode agree
but model output differs, do not label this a prompt byte mutation.

`R4B_CACHE_VERIFY=0` retains request-local ID reuse; `=1` compares that stored
ID tensor against a fresh encoding, logs `cache_compare.equal` and hands the
fresh tensor to the job. Run `legacy` at both values to isolate the app-cache
hypothesis without changing prompt encoding. This flag does not disable GPU
page/snapshot reuse. Compare `[R823-cache]` allocation/page/saved events at
matching job IDs to settle those separate paths. If the engine reports zero
selected prefix/snapshot, repeated prompt failures cannot be assigned to that
reuse mechanism. Do not “fix” recurrent cache on the current evidence.

## r4b implementation and validation

The complete stock patch retains r4's masking/history safety, no-refusal
ordinary literal encoding, control-ID reasoning spans, tool argument byte
preservation, stop hygiene, scoped flags and SSE lifecycle behavior. New
changes are confined to the prompt encoding helper and backend diagnostics /
optional fresh-ID comparison. Sampler construction, output detokenizer,
vision source/config and tool parser are unchanged from r4.

Default encoding uses a private Rust tokenizer clone with literal added-token
matching removed, preserving BPE across ordinary tags and neighboring spaces.
Structural occurrences still go through the native encoder. ExLlama embedding
and BOS handling remain in its wrapper; global alias positions retain offsets.
Rust renumbers surviving added tokens after deletion, so a raw-encode adapter
maps those IDs back to the original model IDs before wrapper embedding or
missing-special insertion. A regression specifically protects that boundary.
The native tokenizer is not modified. Wrappers without a Rust codec retain a
registered-tag segmentation fallback; tags also reachable through base BPE
merges use the same checked, nonrejecting fallback. Runtime `prompt_ids.path` reports it.

Fresh application from both hash-verified stock and r2 passes **106 CPU/SSE
methods each**, with zero-fuzz dry-run/application and landed hashes/assertions.
The real Rust BPE topology test is red on r4, green on r4b. The A4 SSE replay
preserves its captured **wrong** file bytes; it does not turn a failing captured
model output into a pass. Identical messages twice through real template,
preflight/cache/fresh-encode methods produce identical IDs. These tests do not
simulate GPU page allocation, recurrent snapshots or sampled output.

Verdict: **CPU-verified candidate with discriminating probes; GPU exact-body
acceptance pending.** No source evidence justifies silently repairing file
contents or asserting that this delivers 16/16 inference success. The supplied
brief explicitly permits probe arms where evidence cannot discriminate; all
three prompt arms and the app-cache comparison are packaged with exact commands.
The structural cause that would prevent recurrence is preserving contiguous
ordinary tokenization boundaries and asserting original model ID identity,
with request-correlated IDs at the prompt/cache/output seams.

Literal-window events label `token_ids_sources`: only `engine` arrays are
sampled-ID evidence. If `reencoded_text` appears, retain the capture but do not
use reconstructed IDs to conclude what the model sampled.
