# R866: a static hot set fitted on the probe prompts reaches 63.2 tok/s in sample and gains nothing on the held-out kind; faster dynamic sweeps lose 20 %

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r866-glm53-placement-030039/`. Driver: `scripts/r866-glm53-placement.sh`. Raw records: `results/2026-10-07-r866-glm53-placement/` (per arm `c1.jsonl`, `config.yml`, `resolved.json`; `S/vision.jsonl`; `summary.txt`).

## Configuration and results

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, 96 experts per layer on the CPU, MTP off. c1 score as in `bench/RESULTS.md`, five kinds, 2 runs each.

| arm | placement | c1 score (tok/s) | code | prose | chat | html | edit |
|---|---|---|---|---|---|---|---|
| A | dynamic, served defaults (sweep every 128, floor 8) | 48.9 | 47.3 | 48.7 | 48.9 | 49.6 | 50.2 |
| B | dynamic, sweep every 64, floor 4 | 39.1 | 34.4 | 39.8 | 41.3 | 38.3 | 41.8 |
| C | static, counts from the R860b trace (`scripts/split-stats-r860b.json`) | 63.2 | 61.2 | 69.3 | 69.1 | 67.0 | 49.3 |
| A2 | A again | 48.6 | 46.9 | 48.6 | 48.9 | 48.3 | 50.0 |

- B: the lower CPU share the simulator predicted for faster sweeps is outweighed by the swap cost (two checkpoint reads plus an all-device sync per swap, and an inline sweep when a sweep is four intervals overdue, which lands mid-generation).
- C: the counts were fitted on traces of the code, prose, chat and html prompts themselves. Those four kinds gain 30 to 40 %; edit, which was not in the trace, gains nothing. The arm shows the rate a hot set matched to the traffic reaches (61 to 69 tok/s), not a general gain. R869 refits on a different prompt set.
- The serving arm (static C) answered `White` for the blue 64-pixel square in the vision check (red was right); R877 replaced that check with 256-pixel images.
