# R876: at c1 the GPUs run kernels for 11 ms per token; the rest of the step waits on the CPU expert worker

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r876-glm53-timeline-090330/`. Driver: `scripts/r876-glm53-timeline.sh`. Raw records: `results/2026-10-07-r876-glm53-timeline/` (per run `TABLE.md`, `analysis.json`, `categories.csv`, `kernels.csv`, `layers.csv`, `per-token.csv`, `wall-clock.csv`, `handoff-aggregate.csv`). The torch-profiler traces and the event and gap tables (up to 53 MB per run) are not copied.

## Method

Standalone model loads (no server) on `tabbyapi:r861-glm-agent-r2`, static placement from `scripts/split-stats-broad-r869.json`, 96 experts per layer on the CPU, MTP off, torch profiler; 32 decode steps per run on a fixed probe prompt. Runs: `base-c1` (no ranges), `graphs-c1` (CUDA graphs, ranges), `waits-c1` (stream-wait events), `graphs-c4` (4 rows), `graphs-32k-c1` (32k context). The profiler code was written by an OpenAI Codex agent and is not in this repository.

## Per decode step

| run | wall mean (ms) | GPU0 kernels (ms) | GPU1 kernels (ms) |
|---|---|---|---|
| base-c1 | 25.29 | | |
| graphs-c1 | 26.71 | 5.58 | 5.42 |
| graphs-32k-c1 | 24.07 | | |
| graphs-c4 (4 rows per step) | 60.54 | | |

Kernel time per device per step at c1 (`graphs-c1/categories.csv`): KDA attention 1.31 to 1.43 ms, routed experts 1.26 to 1.28, DSA attention 0.73 to 0.85, shared expert 0.90 to 0.93, hyper-connection and norm 0.47 to 0.52, DSA indexer 0.16 to 0.19, router 0.15, handoff 0.11 to 0.12.

## Analysis

The analysis of these files (done by the same agent, document not copied) separates, per token at c1: 11.0 ms of GPU kernels, 12.1 ms of exposed wait on the CPU expert worker, and a native step of 23.8 ms (42.0 tok/s) without the profiler. This prompt routes 3.63 CPU picks per layer, against about 2.06 in the R860b sample, which accounts for the gap to the 57.4 tok/s measured on the server.

- One extra CPU pick on a layer that already waits costs 0.10 to 0.11 ms (linear fit, R² 0.955 to 0.998).
- With all eight picks computed on the GPU the step would be about 13 to 15 ms (67 to 77 tok/s), an estimate; an instantaneous worker bounds it at about 11.1 ms.
- The GPU0 to GPU1 boundary costs 0.025 ms per token.
- At c4, every fourth round the multi-row pooled-DSA index update falls out of CUDA graphs: 660 extra launches and 44 scalar-copy synchronisations, about 5.2 ms per round under the profiler.
- At 32k, context adds almost no GPU time.
