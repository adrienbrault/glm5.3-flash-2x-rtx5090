# R916 and R916b: greedy output identical within and across boots with three opt-in flags; c1 cost a few percent; not served by default

Results directories on the box: `/srv/qwen5090/results/2026-10-08-r916-glm53-determinism-fix-145555/` (R916) and `/srv/qwen5090/results/2026-10-08-r916-glm53-determinism-fix-201416/` (R916b). Driver: `scripts/r916-glm53-determinism-fix.sh` (the R916b revision; R916 ran an earlier one with the other image). Raw records: `results/2026-10-08-r916-glm53-determinism-fix/` and `results/2026-10-08-r916b-glm53-determinism-fix/` (`summary.txt`, `first-divergence.jsonl`, `daily-cost.json`, `winner.env`, per arm `settings.json` and `speed.json`). The per-request outputs, the hash traces and the engine logs are not published. The overlays (`determinism-r3` for R916, `determinism-r4` for R916b, written by OpenAI Codex agents) are not in this repository.

## Question

Greedy output was not reproducible on this engine: the same temperature-0 request diverged after a few to a few hundred characters, across boots and within one process (R910b, `docs/GOTCHAS.md`). This round locates the first divergence and tries switches that remove it.

## Images and arms

- R916 (from 14:55 UTC): `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_determ3`, built on the image served before the per-device split, so its arms ran one CPU expert count (104) on every layer at `GPU_SPLIT=31.8,31`. Its Daily arm ran the served image.
- R916b (from 20:14 UTC): `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_determ4`, the same flags ported onto the served image, so every arm ran the served per-device counts.

Each check sends the unit's five fixed requests (code, prose, chat, html, edit) twice in one process and compares the outputs byte for byte; a cross-boot check compares two boots' outputs.

| arm | boot | flags | R916 | R916b |
|---|---|---|---|---|
| T0 | static placement, MTP off (S0) | none, module-boundary hash trace on | 3 of 5 differ | 4 of 5 differ |
| C0 | S0 | none | 4 of 5 differ | 3 of 5 differ |
| C1 | S0 | `EXL3_ORDERED_MOE=1` | identical | identical |
| S1, then S1 against C1 | second static boot | `EXL3_ORDERED_MOE=1` | identical; across boots edit differs | identical; across boots edit differs |
| retry-C2 | S1 | `EXL3_FIXED_STREAM_T=8` | 4 of 5 differ | 4 of 5 differ |
| retry-C3 | S1 | both | identical | identical |
| retry-S3, then against retry-C3 | third static boot | both | identical; identical across boots | identical; identical across boots |
| D0 | served configuration (MTP, exchange swaps) | both | 5 of 5 differ | 5 of 5 differ |
| Dfreeze | served configuration | both and `EXL3_FREEZE_EXCHANGE=1` | identical | identical |
| D1, then against Dfreeze | second served-configuration boot | the three flags | identical; identical across boots | identical; identical across boots |

## Where the first divergence is

`first-divergence.jsonl` compares the hash traces of T0's two repeats. In both runs, four of the five requests first differ at `MoE.cpu_partial`, the CPU expert worker's partial sums, at prefill step 0, with identical input hashes; the fifth request's two repeats took different prefill and decode schedules, so the trace could not compare module by module.

## Cost

`speed.json` is the unit's own c1 figure over its five fixed requests (the mean of the per-kind rates), not the c1 score of `bench/RESULTS.md`.

| arm | R916 (tok/s) | R916b (tok/s) |
|---|---|---|
| Daily (served image, no flags) | 63.7 | 65.2 |
| Dfreeze (three flags) | 62.1 | 62.1 |
| D1 (three flags, second boot) | 63.1 | 64.0 |

Dfreeze reads 2.5 % (R916) and 4.7 % (R916b) below Daily; the unit's `daily-cost.json` states the same gap relative to Dfreeze, 2.54 % and 4.97 %. D1 reads 1.0 % and 1.8 % below Daily. Each is one arm, inside the run-to-run spread of c1 measures (`docs/GOTCHAS.md`), so the cost reads as a few percent.

## Reading

- The CPU expert worker's partial sums depend on the order in which its threads finish and on how the work is partitioned. `EXL3_ORDERED_MOE=1` (fixed reduction order) makes output identical within one boot; adding `EXL3_FIXED_STREAM_T=8` (a fixed thread partition) makes it identical across boots with static placement.
- On the served configuration, exchange swaps move experts between the GPU and the CPU at points that depend on timing, so the two flags are not enough there; `EXL3_FREEZE_EXCHANGE=1` keeps the placement of the boot fixed, and with the three flags output is identical within and across boots on all five requests.
- The flags stay off in the served configuration. Boots that compare two patches byte for byte, or measure KL between boots, set `EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_FREEZE_EXCHANGE=1`.
