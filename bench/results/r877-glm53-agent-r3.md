# R877: agent overlay r3 rejects every prompt that quotes a GLM tag; r2 passes none of the 12 strict cases

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r877-glm53-agent-r3-095516/`. Driver: `scripts/r877-glm53-agent-r3.sh`. Raw records: `results/2026-10-07-r877-glm53-agent-r3/` (per arm `toolcheck/results.jsonl`, `toolcheck/summary.json`, one `*.result.json` per case, `vision.jsonl`, `A3/tag-trace-tail.txt`, configs, `summary.txt`). The raw SSE captures (up to 1.3 MB per case) are not copied.

## Configuration

Base configuration, 96 experts per layer on the CPU, static placement from `scripts/split-stats-broad-r869.json`, MTP off, `AGENT=1`. Arm A2: `tabbyapi:r861-glm-agent-r2` (served then). Arm A3: `tabbyapi:r861-glm-agent-r3` with `TAG_TRACE=1`. Checker `scripts/glm53_toolcheck_r3.py`: argument JSON, exact body bytes and path of a `write_file` call; cases hard (200 lines quoting `</fake>` and `<think>x</think>`), think-only, close-only, streamed, temperature 0 and default, 2 repeats (12 requests); A3 also ran the 1,000-line case. Vision check: 256-pixel images, red, blue, green and a left/right split.

## Results

- A3 (r3): every toolcheck request raised `ValueError: Tokenizer encoded a literal GLM tag as a control ID` before generation; nothing was generated for the 12 cases or the 1,000-line case. Not served.
- A2 (r2): 0 of 12 passed. 5 requests ended with `finish_reason: stop` and no call (at temperature 0, the hard case stopped after about 26 reasoning tokens, at the literal `</fake>`); 7 returned a `write_file` call with the right path and an altered body.
- Vision: pass on both arms.

The r4 overlay (`docker/glm-agent-r4/`, `DIAGNOSIS.md`) follows from this round. Serving stayed on r2.
