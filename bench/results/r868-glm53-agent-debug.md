# R868: with its flags off, TabbyAPI ends a GLM tool call when the reasoning quotes a think tag; the agent overlay r2 returns the call

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r868-glm53-agent-debug-031155/`. Driver: `scripts/r868-glm53-agent-debug.sh`. Raw records: `results/2026-10-07-r868-glm53-agent-debug/` (per arm `toolcheck.log` with one `RESULT` line per case, configs, `summary.txt`). The raw SSE captures (up to 1.5 MB per case) are not copied.

## Configuration

Image `tabbyapi:r861-glm-agent-r2` (`docker/glm-agent-r2/`), base configuration, 96 experts per layer on the CPU, dynamic placement, MTP off. Two boots: `agent0` with every overlay flag off (the stock TabbyAPI code path), `agent1` with `AGENT=1`. Checker `scripts/glm53_toolcheck.py`, temperature 0.

| case | flags off | flags on |
|---|---|---|
| simple, streamed | `tool_calls`, 245 argument characters, valid JSON | `tool_calls`, 247, valid JSON |
| simple, not streamed | `tool_calls`, 245, valid JSON | `tool_calls`, 247, valid JSON |
| hard, streamed: `write_file` of 200 lines quoting `</fake>` and `<think>x</think>` | `stop` after 107 completion tokens, no tool call | `tool_calls`, 8,528 argument characters, valid JSON, 6,692-character body |

With the flags off, the reasoning stopped where the model wrote `</think>` while planning how to escape the body. This checker validated JSON and length only; R877's stricter checker found that the bodies were not byte-exact (`r877-glm53-agent-r3.md`). The round ended serving the plain image.
