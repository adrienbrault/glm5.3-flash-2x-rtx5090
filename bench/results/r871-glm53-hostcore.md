# R871: reserving a host core gives nothing on this box; confining the server to one core is −31 %

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r871-glm53-hostcore-071136/`. Driver: `scripts/r871-glm53-hostcore.sh`. Raw records: `results/2026-10-07-r871-glm53-hostcore/` (per arm `c1.jsonl`, `host-confine.txt`, configs; `S/toolcheck.log`, `S/vision.jsonl`; `summary.txt`).

## Configuration

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, 96 experts per layer on the CPU, static placement from `scripts/split-stats-broad-r869.json`, MTP off. The idea under test is ExLlamaV3 `dev` commit `fa822cf` ("CPU MoE: reserve a host core"), emulated without a rebuild by `HOST_CONFINE=1` in the launcher, which runs `taskset` on every thread of the TabbyAPI process. c1 score as in `bench/RESULTS.md`, five kinds.

| arm | CPU worker threads | host process confined to | c1 score (tok/s) | per kind |
|---|---|---|---|---|
| t8 | 8 | not confined | 57.5 | code 57.8, prose 63.2, chat 61.5, html 54.3, edit 50.8 |
| t7h | 7 | core 7 and its sibling 15 | 39.4 | code 39.4, prose 40.7, chat 40.7, html 40.0, edit 36.2 |
| t7 | 7 | not confined | 54.1 | code 55.2, prose 62.1, chat 60.4, html 46.8, edit 46.3 |
| t8s | 8 | SMT siblings 8 to 15 | 57.8 | code 57.5, prose 63.7, chat 62.6, html 53.6, edit 51.7 |
| t8b | 8 | not confined (control) | 58.5 | code 59.9, prose 63.6, chat 61.6, html 53.2, edit 54.4 |

- t7h puts all of the server's threads (event loop, generator, CUDA launches and the handoff path) on one physical core; the thread that issues GPU work then competes with its siblings. The upstream commit keeps the host off the worker's cores and does not confine it to one core.
- t8s is within the t8 / t8b spread (57.5 to 58.5): the worker already has its 8 physical cores and the host thread mostly waits on the GPU.
- The serving arm (static plus agent overlay r2): simple tool calls returned `tool_calls` streamed and not streamed; the hard case ended with `stop` and no call; the 64-pixel vision check answered `White` for blue.
