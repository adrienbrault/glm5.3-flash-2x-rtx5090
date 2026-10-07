# GLM agent r4b

Candidate overlay, CPU-verified; GPU byte-exact acceptance is pending. The
supplied R880 trace does not record ordinary space/tag IDs or GPU cache
allocation. See `DIAGNOSIS.md` for the observed deltas and explicit limits.

`app.patch` is complete **stock -> r4b**. Base remains
`FROM tabbyapi:r861-glm-agent-r2`. Docker verifies r2 hashes, reverses the
bundled parent patch, verifies stock hashes, applies with zero fuzz, checks
landed hashes/assertions and runs all **106** strict CPU/SSE tests without
weights. No downloads or sampler/config changes. Do not apply this complete
patch straight onto r2 or r4.

## Build on flan

From the transferred workspace root:

```sh
python3 -B out/r4b/tests/check_hashes.py out/r4b/SHA256SUMS out/r4b
bash out/r4b/build.sh
```

Produces `tabbyapi:r861-glm-agent-r4b`, `--pull=false --network=none`. No Docker,
SSH, GPU or git was used to prepare this packet. Fresh local stock and r2
verification passed all 106 tests using cached CPU dependencies. The Docker
build repeats them against the actual parent-image dependencies.

Source-only verification (copies the source, does not mutate it):

```sh
R4_PYTHON=python3 bash out/r4b/tests/run.sh /path/to/stock/app stock
R4_PYTHON=python3 bash out/r4b/tests/run.sh /path/to/r2/app r2
```

Only if SSE dependencies are unavailable: `R4_CPU_ONLY=1` executes 99 methods
and explicitly reports the seven omitted SSE/ASGI methods. Strict is required
for delivery. Tests use Pydantic 2, Jinja2, tokenizers, AnyIO, sse-starlette and
the normal logger dependencies from the parent image.

## Main operator matrix

With the existing GPU lock held, using a new result directory:

```sh
bash out/r4b/operator-run.sh /srv/qwen5090/results/glm-agent-r4b-runs
```

The runner boots existing r4, then r4b, on :8029. It temporarily stops the
running `glm53`, restores it on exit, and refuses an existing probe container
or results directory. Model/split/cache paths are the R880 paths. The config
is byte-identical to R880 A4: static N=96, MTP off, vision on, sampling omitted
except where the captured requests explicitly set temperature 0. It enables
existing `EXL3_CACHE_TRACE=1` for both arms and keeps `TABBY_GLM_TAG_TRACE=1`.
The baseline r4 ignores the two new flags.

Per arm it executes the unchanged 16-case checker:

```sh
python3 out/r4b/glm53_toolcheck_r4.py --url http://127.0.0.1:8029/v1 \
  --model glm53-flash-exl3-2.05bpw-turboderp --out /absolute/arm/toolcheck \
  --cases hard think-only close-only 1000-line --modes stream \
  --temperatures zero default --repeats 2 --max-tokens 12000 --long-max-tokens 40000
```

Then three repeats of the same 1000-line prompt at **each** temperature:

```sh
python3 out/r4b/glm53_toolcheck_r4.py --url http://127.0.0.1:8029/v1 \
  --model glm53-flash-exl3-2.05bpw-turboderp --out /absolute/arm/cache-repeat \
  --cases 1000-line --modes stream --temperatures zero default --repeats 3 \
  --max-tokens 12000 --long-max-tokens 40000
```

It replays the four unchanged R880 vision requests, captures JSON/SSE/logs,
and runs the r4b actual-model CPU tokenizer/cache probe. The checker does not
execute tools. Baseline failures do not prevent r4b execution. The runner
returns failure for any r4b matrix, repeat, vision or CPU-probe failure.
Warmup traffic is not reproduced; each arm starts fresh and receives the same
22 checker requests before the four vision requests.

## Distinguishing prompt hypotheses

These are concrete implementations shipped in the same image; no output
byte normalization occurs in any arm:

| Flag value | Prompt path |
|---|---|
| `TABBY_GLM_LITERAL_ENCODING=runs` | Default/best guess. Private Rust codec disables matching for literal added tags; encode contiguous runs around data tags and their whitespace. Preserve structural tag IDs and original added-token IDs. |
| `registered` | Split only registered literal tags; leave ordinary `</fake>` and neighbors together. Tests the fake-tag split independently from think-tag subdivision. |
| `legacy` | Original r4 encoder, including subdivision and splitting every recorded tag. Control/input behavior matches r4. |

Run additional clean r4b boots, one variable at a time:

```sh
R4B_ARMS=r4b R4B_LITERAL_MODE=registered \
  bash out/r4b/operator-run.sh /srv/qwen5090/results/glm-agent-r4b-registered
R4B_ARMS=r4b R4B_LITERAL_MODE=legacy \
  bash out/r4b/operator-run.sh /srv/qwen5090/results/glm-agent-r4b-legacy
R4B_ARMS=r4b R4B_LITERAL_MODE=legacy R4B_CACHE_VERIFY=1 \
  bash out/r4b/operator-run.sh /srv/qwen5090/results/glm-agent-r4b-legacy-fresh
```

`R4B_CACHE_VERIFY=0` (default) reuses the request-local preflight tensor.
`=1` compares it against a fresh encode, logs equality, and sends fresh IDs to
the job. This isolates that app cache; GPU prefix/page/recurrent caches stay
on and their reuse is measured by `[R823-cache]` events.

Actual tokenizer CPU probe, inside r4b with model files mounted (no weights):

```sh
python3 /opt/r4b/tests/probe_prompt_cpu.py \
  /models/glm53-flash-exl3-2.05bpw-turboderp --app /app --out /tmp/prompt-cpu.json
```

The runner executes this automatically and copies its result. It renders the
four checker prompts twice in all three encoding modes, compares preflight
cached IDs with fresh IDs and emits full prompt/decoded text, ID arrays/hashes
and the installed tokenizer source hash. It uses landed app methods and the
installed ExLlama tokenizer. GPU page allocation is outside this CPU seam.
The probe must report `decode_exact=true` for runs and cache/repeat equality
for all modes. Inspect literal-span ID windows to verify no control IDs enter
data. The older isolated `probe_literals.py` is retained as supplemental data.

## Trace interpretation and acceptance

`[GLM-TAG-R4]` remains the prefix. New events:

- `prompt_ids`: full IDs/hash, render/decode equality, mode, spans; `job_input`
  includes the generation request ID, even when preflight has no ID.
- `cache_compare`: fresh versus request-cache equality (`=1` only).
- `literal_window`: first four split `</fake>` windows, including 16 following
  sampled IDs; decode through the unchanged installed output tokenizer.

Join `[R823-cache]` events by job request ID: page digests, allocation prefix,
selected cached pages and recurrent snapshot candidates/saves. Equal rendered
prompt hashes alone do not prove equal IDs. Usage `cached_tokens=0` is not a
complete page-dedup/snapshot log. The R880 IDs behind doubled spaces are absent;
the replay does not invent them.

Accept only r4b **16/16 + 6/6 byte-exact** checker results, valid tool calls,
intact tags, four vision passes, CPU-probe equality and no prompt rejection or
lost turn. If these fail, keep the captures and select the hypothesis supported
by prompt IDs and literal windows. A CPU pass is not a GPU exact-body claim.

Literal-window events label `token_ids_sources`: only `engine` arrays are
sampled-ID evidence. If `reencoded_text` appears, retain the capture but do not
use reconstructed IDs to conclude what the model sampled.
