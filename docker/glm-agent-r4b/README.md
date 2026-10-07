# TabbyAPI GLM agent overlay, revision 4b (candidate, not served)

Image `tabbyapi:r861-glm-agent-r4b`, built `FROM tabbyapi:r861-glm-agent-r2` like revision 4 (`../glm-agent-r4/README.md`): reverse revision 2, apply `app.patch` to the stock `/app`.

## What it changes against revision 4

- `TABBY_GLM_LITERAL_ENCODING=runs` (image default): message text is encoded as whole ordinary runs with literal added-token matching disabled, aimed at the doubled space around `</fake>` seen in R880. `DIAGNOSIS.md` ranks this as the first of three hypotheses; `registered` and `legacy` (revision 4's encoding) are kept as options. Not measured live yet.
- `TABBY_GLM_CACHE_VERIFY` (default `0`): with `1`, the request-local stored prompt ids are compared with a fresh encoding and the fresh ids are used, to test whether the 1000-line repeat that failed in R880 comes from that reuse.
- Vision request fixtures (`tests/fixtures/vision-*.request.json`) and CPU prompt probes.

## Status

R881d, 2026-10-07: the boot failed. `_encode_prompt` calls `os.getenv` and the module does not import `os`, so every chat request raised `NameError` and the launcher's warmup aborted (`bench/results/r881d-glm53-agent-r4b.md`). Revision 4c (`../glm-agent-r4c/`) adds the import; R881e measures it.

## Differences from the package on the box

The test logs `applied-tests.txt`, `fresh-r2-tests.txt`, `fresh-stock-tests.txt`, `green-tests.txt`, `red-tests.txt` and `strict-applied-tests.txt` are left out (they print the authoring workspace's local paths). In `tests/harness.py` and `tests/test_fixes.py` the default scratch paths `tabby-toolfix-*`, which pointed into the authoring machine's temporary directory, now point to `/tmp/tabby-toolfix-*`. `SHA256SUMS` is regenerated for the published files; the image on the box was built from the unedited package. `tests/fixtures/a2-sse.json.gz` and `a4-hard.json.gz` are recorded SSE streams of toolcheck cases (synthetic prompts).

Credit as for revision 2: derived in part from `glm53-tensorfold-spark` (Apache-2.0); licence files kept. Written by an OpenAI Codex agent.
