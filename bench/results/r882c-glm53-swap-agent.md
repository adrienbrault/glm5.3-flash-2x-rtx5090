# R882c: the swap and agent-overlay image passes vision and the simple tool calls and is served

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r882c-glm53-swap-agent-213745/`. Driver: `scripts/r882c-glm53-swap-agent.sh`. Raw records: `results/2026-10-07-r882c-glm53-swap-agent/` (`XA/vision.jsonl`, `XA/toolcheck.log`, `XA/toolcheck/results.jsonl` and `summary.json`, `XA/toolcheck-hard/`, configs, `summary.txt`).

Re-gate of R882b's arm XA, same image and environment (`r882b-glm53-swap-agent.md`), GPU lock taken 2026-10-07 21:47 UTC. Decode was not measured again; the gate used R882b's c1 score of 59.9 tok/s against a threshold of 59.0.

- Vision, 256-pixel red, blue, green and a left/right split (`XA/vision.jsonl`): pass.
- Simple toolcheck (`scripts/glm53_toolcheck_r3.py`, streamed, temperature 0 and default, 2 repeats): 4 of 4 returned `tool_calls` with valid argument JSON and the exact 191-byte, 20-line body.
- The r4 checker's hard case, reported only: `stop` with no call, as expected for the r2 overlay (`r880-glm53-agent-r4.md`).

The round ended about 22:13 UTC serving XA: `tabbyapi:cheapswap-r3-agent-r2`, exchange swaps from the broad hot set, agent overlay r2, 96 experts per layer on the CPU, MTP off, vision on, 256k cache. R881d (22:13 UTC) restored the same environment after its own arm.
