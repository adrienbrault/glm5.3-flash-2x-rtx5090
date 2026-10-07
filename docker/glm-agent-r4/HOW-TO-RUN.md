# GLM agent r4

`app.patch` is a complete **stock -> r4** patch against the exact stock tree
verified by r3's `SHA256SUMS.stock`. The base is unchanged:
`FROM tabbyapi:r861-glm-agent-r2`. The Dockerfile verifies the r2 hashes, reverses
its exact bundled `parent-r861.patch`, verifies stock hashes, then applies
`app.patch` with `--fuzz=0`. It checks landed hashes/assertions and runs all
100 strict CPU/SSE methods without loading weights. It does not pull/install
anything or change sampler defaults. Do not apply this complete patch directly
to r2 without reversing its parent patch first.

## Build on flan

From the transferred workspace root:

```sh
python3 -B out/r4/tests/check_hashes.py out/r4/SHA256SUMS out/r4
bash out/r4/build.sh
```

This builds `tabbyapi:r861-glm-agent-r4` from the existing r2 image with
`--pull=false --network=none`. No Docker/GPU build was performed in this
workspace. The build must pass the seven locally unexecuted SSE/ASGI methods.

Optional source-only verification (does not modify the input tree):

```sh
R4_PYTHON=python3 bash out/r4/tests/run.sh /path/to/stock/app stock
R4_PYTHON=python3 bash out/r4/tests/run.sh /path/to/r2/app r2
# Only when sse-starlette is unavailable; prints omissions:
R4_CPU_ONLY=1 R4_PYTHON=python3 bash out/r4/tests/run.sh /path/to/stock/app stock
```

Dependencies are the parent image's Pydantic 2, Jinja2, tokenizers, and for the
strict suites AnyIO/sse-starlette. The limited run has 93 methods; strict has
100. Fresh limited runs from both layouts passed locally.

## Run the exact r2/r4 matrix

With the existing operator GPU lock held, from the workspace root on flan:

```sh
bash out/r4/operator-run.sh /srv/qwen5090/results/glm-agent-r4
```

The script is concrete startup/checker commands derived from the captured A2
Docker inspection, with these existing paths:

- model `/storage/data/models/glm53-flash-exl3-2.05bpw-turboderp`;
- split statistics `/srv/qwen5090/r877-tools/split-stats-broad-r869.json`;
- cache `/srv/qwen5090/.exl3cache-r858-glm53`;
- the bundled, unchanged R877 A2 config: static N=96, MTP disabled, vision on.

It stops the existing running `glm53` container, boots one arm at a time on
localhost :8029, then restores the original container on exit. No container is
replaced. It refuses to overwrite an existing probe container. Use a fresh
results directory. Each arm sends 16 streaming requests: hard, think-only,
close-only, 1000-line; zero and omitted/server-default temperature; two repeats.
A failing r2 checker exit is recorded and does not prevent the r4 arm. Requests,
partial SSE journals, results, engine logs and Docker inspection are retained.

The checker command used for **each** arm is:

```sh
python3 out/glm53_toolcheck_r4.py --url http://127.0.0.1:8029/v1 \
  --model glm53-flash-exl3-2.05bpw-turboderp --out /absolute/arm-results/toolcheck \
  --cases hard think-only close-only 1000-line --modes stream \
  --temperatures zero default --repeats 2 --max-tokens 12000 --long-max-tokens 40000
```

That command assumes the chosen arm has already been started by the script or
your equivalent existing launcher. No tool execution occurs. The r4 checker
specifies LF/one final LF and compares UTF-8 bytes; its other literal patterns
and sampling matrix are inherited. Both images receive identical requests.
`glm53_toolcheck.py` is the unchanged r3 checker, retained only for probes using
the exact old message text.

Keep `TABBY_STREAM_TOOLCALLS=1 TABBY_SSE_KEEPALIVE_S=5
TABBY_GLM_TOOL_FIXES=1 TABBY_GLM_FORCING=0 TABBY_GLM_TAG_TRACE=1` for this run.
Flags are default-off in the image as before. The original upstream model
chat template works because masking happens before rendering; the bundled
`glm53-chat_template.jinja` is also compatible and used by inherited tests.
No template/multimodal/model config change is needed for this fix.

## Resolve the remaining evidence gaps

The runner adds original-prompt **plain** r2 probes (zero, two repeats per
hard/think-only/close-only) to expose `eos_reason`/`stop_str`, lost by the old
usage-inclusive SSE serializer. If an actual ID is still needed, build the
separate, logging-only baseline and repeat:

```sh
docker build --pull=false --network=none -f out/r4/Dockerfile.r2-trace \
  -t tabbyapi:r861-glm-agent-r2-eos-trace out/r4
R4_A2_IMAGE=tabbyapi:r861-glm-agent-r2-eos-trace \
  bash out/r4/operator-run.sh /srv/qwen5090/results/glm-agent-r4-eos-trace
```

Its only change is the opt-in `[GLM-TAG-R4-A2-EOS]` JSON line at the actual
backend EOS boundary. The main matrix normally uses the unmodified r2 image.

The r4 arm also runs the actual-model CPU tokenizer probe automatically:

```sh
# Inside the r4 image/server Python environment; no weights/GPU allocation:
python3 /opt/r4/tests/probe_literals.py /models/glm53-flash-exl3-2.05bpw-turboderp \
  --app /app --out /tmp/glm-literals.json
```

It records model-file hashes, registered added-token flags/IDs, native false-
flag encoding, r4 ordinary IDs and decoded byte equality. This proves the Rust
model tokenizer seam, not every detail of flan's ExLlama fork. Retain installed
`exllamav3/tokenizer/tokenizer.py` and `generator/job.py` if runtime
`control_spans` logs report `decode_mismatch=true`. r4 will preserve text rather
than invent a control position in that case; investigate alignment before
claiming the live contract passed.

With `TABBY_GLM_TAG_TRACE=1`, `[GLM-TAG-R4]` events carry prompt-span counts and
hashes, effective stop sets, real engine EOS reason/trigger, ID-derived control
spans and parser states. An actual EOS remains a real stop; do not remove it
because the preceding text looks truncated. Ordinary think spellings remain
text, while an actual closing ID changes reasoning even in quoted prose.
No CPU test can establish which ID an unlogged past model sample emitted or
guarantee that a future model will copy all file bytes. Read `DIAGNOSIS.md` for
the demonstrated defects and those limits.
