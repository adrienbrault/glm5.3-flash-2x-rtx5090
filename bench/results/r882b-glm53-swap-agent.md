# R882b: r3 swaps with the agent overlay on one image: 59.9 tok/s at c1, 109.9 tok/s summed over four streams, prefill 2.0k to 2.15k tok/s

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r882b-glm53-swap-agent-203933/`. Driver: `scripts/r882b-glm53-swap-agent.sh` (`scripts/r882-glm53-swap-agent.sh` is the first attempt, whose image build failed on the checksum gate, `docs/GOTCHAS.md`). Raw records: `results/2026-10-07-r882b-glm53-swap-agent/` (`XA/c1.jsonl`, `XA/measure.jsonl` one line per stream, round and prefill request, `XA/vision.jsonl`, `XA/toolcheck.log`, configs, `summary.txt`).

## Configuration (arm XA)

Image `tabbyapi:cheapswap-r3-agent-r2`, built in this round from `docker/glm-agent-r2-on-cheapswap-r3/` (the agent overlay r2 applied on `tabbyapi:cheapswap-r3`). Base configuration of `bench/RESULTS.md`, 96 experts per layer on the CPU, MTP off, `AGENT=1`, `PINNED_ARENA=1`, exchange swaps: `SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0`, initial placement from `scripts/split-stats-broad-r869.json`.

## c1 by content kind

Method of `bench/RESULTS.md`: one chat request per run forced with `min_tokens` to 1,024 tokens (html 2,048), temperature 0, `reasoning_effort: low`, 2 runs.

| kind | runs (tok/s) | median | time to first token (s) |
|---|---|---|---|
| code | 58.8, 57.3 | 58.1 | 0.37 |
| prose | 60.5, 64.6 | 62.5 | 0.39 |
| chat | 66.2, 62.0 | 64.1 | 0.39 to 0.41 |
| html | 62.6, 63.5 | 63.0 | 0.50 to 0.53 |
| edit | 51.0, 52.1 | 51.6 | 1.67 to 1.75 |

c1 score (mean of the medians): 59.9 tok/s.

## Concurrency and prefill

`glm53_probe.py --phase measure --runs 3`: each stream one chat request with the code-tutorial prompt forced to 1,024 tokens, temperature 0, the template's default reasoning effort; the streams start together; rate per stream = (tokens − 1) / time between its first and last text frame.

| concurrency | per-stream decode, median of 3 (tok/s) | sum of the stream rates, median of 3 (tok/s) | time to first token, median (s) |
|---|---|---|---|
| 1 | 59.9 | 59.9 | 0.38 |
| 2 | 41.1 | 82.2 | 0.67 to 0.77 |
| 4 | 27.5 | 109.9 | 1.27 to 1.32 |

Cold prefill, one completion request each, random word-list prompt, engine timing with `cached_tokens == 0`: 8,156 prompt tokens at 1,999 tok/s; 32,811 at 2,146 tok/s.

## Checks and outcome

Vision: pass on the 64-pixel red and blue check (the round's tools copy carried the older probe; R882c ran the 256-pixel check). The simple toolcheck did not run: `glm53_toolcheck_r3.py` was missing from the round's tools copy (`XA/toolcheck.log`), so the gate (`OK=0`) did not promote XA and the round served static placement with the agent overlay r2. R882c repeated the checks.

Same configuration before this round (R873, static placement with the agent overlay r2): 57.4 at c1, 73.3 at c2 and 86.3 at c4 summed, prefill 1,697 at 8,198 tokens and 1,797 at 32,804 tokens.
