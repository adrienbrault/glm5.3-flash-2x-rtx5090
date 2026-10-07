# R860b: routing trace and CPU worker profile; dynamic placement leaves the CPU share of expert picks at 0.347

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r860b-glm53-trace-013158/`. Driver: `scripts/r860b-glm53-trace.sh`. Raw records: `results/2026-10-07-r860b-glm53-trace/` (`predict/c1.jsonl`, `routes-analysis*.txt`, `sim-placement*.txt`, `summary.txt`). The 2.1 GB of trace files (456,639 records) are not copied.

## Configuration

Image `tabbyapi:r859-predict-trace-r2` (the route-trace overlay of `docker/route-trace-r1/` plus a predictor probe; the probe is not in this repository), base configuration, 104 experts per layer on the CPU, dynamic placement, MTP off, `PROFILE=1` (CPU worker and handoff profilers on). c1 four kinds, 1 run each, then a 32k-token depth request.

## Measured

- Instrumented c1 score 23.4 tok/s (code 23.5, prose 23.6, chat 22.9, html 23.8): the trace writer and the profilers roughly halve the rate, so these are not speed samples. 32k prefill 1,812 tok/s.
- 5,474 decode steps × 42 MoE layers traced over 9 generations.
- Handoff profile per layer job: GPU-side gap 0.5 ms, CPU expert compute 0.36 ms, spin 0.06 ms, 2.75 CPU picks per token per layer. CPU job phases: prep 47 µs, gate/up GEMV 239 µs, down GEMV 117 µs; about 17 MB per job in 0.36 ms, about 47 GB/s.

## Routing analysis (`scripts/analyze_routes.py`)

Share of the 8 picks per token that land on the 104 CPU-resident experts, mean over layers:

| placement | CPU share |
|---|---|
| uniform (104 / 288) | 0.361 |
| observed, dynamic placement | 0.347 |
| static hot set fitted on the same trace | 0.131 |
| static hot set, fit on even steps, scored on odd steps | 0.134 |
| static hot set, fit on the first half, scored on the second half | 0.292 |

- 30 % of a token's experts were used by the previous token; an 8-slot LRU per layer catches 38 to 48 % of the remaining CPU picks (`routes-analysis.txt`, columns `reuse(t-1)` and `LRU8`).
- The hot set depends on the content: the split-half estimate (0.292) is the one that predicts a held-out gain.

## Placement simulator (`scripts/sim_placement.py`)

Replaying the dynamic placement policy on the trace reproduces the observed share at the served defaults (0.345 at a sweep every 128 tokens, floor 8, budget 64 swaps). `sim-placement-grid.txt` lists the grid: a sweep every 64 tokens at floor 4 gives 0.288 at 186 swaps per 1,000 steps; every 16 tokens at floor 4, 0.209 at 597 swaps. Each swap of the served placement rereads two experts from the checkpoint during a sync, a cost the simulator does not price; R866 measured it.
