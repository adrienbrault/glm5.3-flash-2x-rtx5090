# TabbyAPI GLM agent overlay, revision 4 (candidate, not served)

Image `tabbyapi:r861-glm-agent-r4`, built `FROM tabbyapi:r861-glm-agent-r2`. The Dockerfile reverses revision 2 (`parent-r861.patch`, checked against `SHA256SUMS.stock`), then applies `app.patch` to the stock `/app`.

## What it changes against revision 2

- Literal GLM tags in message text (`<|user|>`, `<|observation|>`, `<think>`, `</think>`) are encoded as ordinary text pieces; only the template's own tags become control ids (`common/glm_tag_safety.py`). Revision 3 refused such prompts with `ValueError: Tokenizer encoded a literal GLM tag as a control ID` (R877); revision 2 lets them through as control ids, and the turn can end on them.
- Reasoning control ids are carried separately from text, and tool-argument bytes stay exact.
- `TABBY_GLM_TAG_TRACE=1` logs stop sources, the EOS trigger id and the parser state.

`DIAGNOSIS.md` traces the R877 failures to these causes. `glm53_toolcheck_r4.py` is the checker (UTF-8 exact body, path, argument JSON); the same file is in `scripts/`.

## Measured

R880, 2026-10-07: 2 of 16 checker cases passed (revision 2: 0 of 16). Every case ended in a valid `write_file` call with all 200 `<think>` / `</think>` pairs intact; 12 cases failed on a doubled space on both sides of `</fake>`, and one 1000-line repeat failed after a likely prefix-cache reuse (`bench/results/r880-glm53-agent-r4.md`). Revision 4b (`../glm-agent-r4b/`) follows from it.

## Differences from the package on the box

The test logs `fresh-r2-tests.txt`, `fresh-stock-tests.txt` and `red-tests.txt` (they print the authoring workspace's local paths) are left out. In `tests/harness.py` and `tests/test_fixes.py` the default scratch paths `tabby-toolfix-*`, which pointed into the authoring machine's temporary directory, now point to `/tmp/tabby-toolfix-*`. `SHA256SUMS` is regenerated for the published files; the image on the box was built from the unedited package. `tests/fixtures/a2-sse.json.gz` is the recorded SSE stream of the R877 toolcheck cases (synthetic prompts).

Credit as for revision 2 (`../glm-agent-r2/README.md`): derived in part from `glm53-tensorfold-spark` (Apache-2.0); licence files kept. Written by an OpenAI Codex agent.
