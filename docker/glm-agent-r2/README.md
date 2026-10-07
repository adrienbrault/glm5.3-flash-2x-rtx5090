# TabbyAPI GLM agent overlay, revision 2 (served)

Image `tabbyapi:r861-glm-agent-r2`, built `FROM tabbyapi:r828-prompt-lookup-r3`; the served image `tabbyapi:cheapswap-r3-agent-r2` applies the same package on `tabbyapi:cheapswap-r3` (`../glm-agent-r2-on-cheapswap-r3/`). It patches ten Python files of TabbyAPI under `/app`; ExLlamaV3 is untouched.

## What it changes

Every change is off unless its flag is `1`; the launcher's `AGENT=1` sets the served values.

- `TABBY_STREAM_TOOLCALLS=1`: tool-call names and argument fragments stream as `tool_calls` deltas while the model writes them, instead of arriving in one piece at the end of the turn.
- `TABBY_SSE_KEEPALIVE_S=5`: an SSE comment every 5 s while the stream is idle (long prefill, long reasoning), so clients and proxies keep the connection.
- `TABBY_GLM_TOOL_FIXES=1`: GLM `glm4_5` tool-call parsing with exact argument bytes, whole-schema argument typing, reasoning-history recovery across turns, `tool_choice` handling, and literal `<think>` / `</think>` text in reasoning or arguments kept as text. R868 measured the effect: with the flags off, a 200-line `write_file` whose body quotes `</fake>` and `<think>x</think>` ended with `finish_reason: stop` and no tool call; with the flags on it returned a valid call of 8,528 argument characters (`bench/results/r868-glm53-agent-debug.md`).
- `TABBY_GLM_FORCING=0`: grammar forcing of tool calls stays off; `tests/probe_grammar.py` has not passed on the real tokenizer.

Known limit, measured in R877 and R880: when a prompt quotes GLM's own control tags (`<|user|>`, `<|observation|>`, `</think>`), the stock tokenizer encodes them as control ids, and at temperature 0 the turn can end on them. Revisions 4 and 4b (`../glm-agent-r4/`, `../glm-agent-r4b/`) address that and are under test.

## Build

`Dockerfile` checks `SHA256SUMS` (this package), `SHA256SUMS.base` (the stock `/app` of the parent image), applies `app.patch` with `--fuzz=0`, checks `SHA256SUMS.patched`, and runs `tests/run_applied.sh`. Keep the Dockerfile inside the build context: with `-f` pointing outside it, the legacy builder rewrites `.dockerignore` and the checksum gate fails (`docs/GOTCHAS.md`). `HOW-TO-RUN.md` is the runbook as delivered; `AUDIT.md` and `TEST-CHANGES.md` record the review.

## Differences from the package on the box

The `evidence/` directory (test logs from the authoring workspace, which print its local paths) and `last.txt` are left out. In `tests/harness.py` and `tests/test_fixes.py` the default scratch paths `tabby-toolfix-*`, which pointed into the authoring machine's temporary directory, now point to `/tmp/tabby-toolfix-*`. `SHA256SUMS` is regenerated for the published files; the image on the box was built from the unedited package. `SHA256SUMS.base` and `SHA256SUMS.patched` fingerprint `/app` and are unchanged.

## Credit

The reasoning-history recovery and whole-schema argument typing are derived from `glm53-tensorfold-spark` by Jay Leaton (Apache-2.0), which modifies TensorFold (MIT). `LICENSE.tensorfold`, `NOTICE.tensorfold` and `THIRD_PARTY.md` are kept as delivered. The patch modifies TabbyAPI (AGPL-3.0) and stays under that licence. Written by an OpenAI Codex agent.
