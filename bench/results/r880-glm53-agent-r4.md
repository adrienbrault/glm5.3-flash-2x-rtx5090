# R880: agent overlay r4 ends every case in a valid tool call with the tags intact; 2 of 16 strict cases pass, r2 0 of 16

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r880-glm53-agent-r4-161049/`. Driver: `scripts/r880-glm53-agent-r4.sh`. Raw records: `results/2026-10-07-r880-glm53-agent-r4/` (per arm `toolcheck/results.jsonl`, `toolcheck/summary.json`, one `*.result.json` per case, `vision.jsonl`, configs; `A2/original-plain-probe/`; `summary.txt`). The raw SSE captures (up to 4.8 MB per case) are not copied.

## Configuration

Base configuration, 96 experts per layer on the CPU, static placement from `scripts/split-stats-broad-r869.json`, MTP off, `AGENT=1`. A2: `tabbyapi:r861-glm-agent-r2`. A4: `tabbyapi:r861-glm-agent-r4` (`docker/glm-agent-r4/`) with `TAG_TRACE=1`. Checker `scripts/glm53_toolcheck_r4.py`, the same on both arms: hard, think-only, close-only and 1,000-line cases, streamed, temperature 0 and default, 2 repeats (16 requests).

| arm | passed | failures |
|---|---|---|
| A2 (r2) | 0 of 16 | 9 ended with `stop` and no call; 7 returned a call with an altered body |
| A4 (r4) | 2 of 16 | every request returned a valid `write_file` call; the 12 hard, think-only and close-only cases have a doubled space on both sides of `</fake>` on every line, at both temperatures; the second repeat of the 1,000-line case failed at both temperatures |

- In A4 all 200 `<think>` / `</think>` pairs of the 200-line bodies arrived intact, and no turn was lost. The 1,000-line case passed byte-exact on its first repeat at both temperatures; the second repeat of the same prompt, plausibly a prefix-cache reuse, failed.
- A2 plain (non-streamed) probe with the original prompts: at temperature 0 the turns ended on the model's own `<|user|>` (hard, think-only) or `<|observation|>` stop strings, consistent with literal prompt tags encoded as control ids.
- Vision: pass on both arms.

Serving stayed on r2. Revision 4b (`docker/glm-agent-r4b/`) targets the doubled space and the repeat.
