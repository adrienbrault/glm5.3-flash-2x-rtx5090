# R869: a static hot set fitted on 32 different prompts is +18.2 % at c1 out of sample

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r869-glm53-hotset-055219/`. Driver: `scripts/r869-glm53-hotset.sh`. Raw records: `results/2026-10-07-r869-glm53-hotset/` (`split-stats-broad.json`, per arm `c1.jsonl`, configs, `S/vision.jsonl`, `summary.txt`). The trace files are not copied.

## Method

1. Trace boot (`tabbyapi:r859-route-trace-r1`, `docker/route-trace-r1/`): 32 prompts from `scripts/glm53_workload.py`, none of them the probe prompts; 3,177,888 expert selections over 42 layers.
2. `scripts/build_split_stats.py` turns the decode selections into per-layer counts: `split-stats-broad.json`, the same file as `scripts/split-stats-broad-r869.json`.
3. Two boots on `tabbyapi:r828-prompt-lookup-r3`, base configuration, 96 experts per layer on the CPU, MTP off; c1 score as in `bench/RESULTS.md`, five kinds, 2 runs each.

| arm | placement | c1 score (tok/s) | code | prose | chat | html | edit |
|---|---|---|---|---|---|---|---|
| SB | static, broad counts | 57.5 | 55.6 | 63.4 | 62.8 | 49.8 | 55.8 |
| A | dynamic, served defaults | 48.6 | 47.0 | 48.1 | 48.6 | 49.3 | 50.2 |

SB is 18.2 % above A; every kind is at or above A (html +1 %, prose +32 %). The probe prompts were not in the counts, unlike R866's arm C. The round ended serving SB; the 64-pixel vision check answered `White` for blue (R877 replaced the check).
