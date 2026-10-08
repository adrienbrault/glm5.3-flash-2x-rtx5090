# R925: 98 against 100 CPU experts per layer for GPU0's layers, alternated twice in one session: 100 stays served

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r925-glm53-splitdev-abab-144311/`. Driver: `scripts/r925-glm53-splitdev-abab.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r925-glm53-splitdev-abab/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, `<arm>/server-steps.txt`, free VRAM, configs, `summary.txt`). Engine logs are not published.

## Question and rule

At the served `EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104` GPU0 keeps 847 MiB free after warmup; at 98,104 it keeps 587 MiB, above the 450 MiB floor the drivers require. R915c measured 98 once and lower on every measure. The rule written into the driver before the run: serve 98 if its mean 4-stream sum is above 100's and its c1 mean is not lower by more than 1 %.

## Arms

One session on 2026-10-08 from 14:43 UTC, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2`, `scripts/glm-daily.env` with the CPU count for GPU0's layers set per arm (GPU1's layers 104), in the order A1, B1, A2, B2. A first run of the unit failed before measuring: the daily's own `EXL3_MOE_CPU_SPLIT_BY_DEVICE` stayed in `EXL3_EXTRA` next to the arm's value, the engine applied the last one and the launcher's check read the first; the driver now removes the daily's value and `scripts/glm53_verify.py` reads the last.

| arm | GPU0's layers | free VRAM after warmup, GPU0 / GPU1 | after the matrix |
|---|---|---|---|
| A1, A2 | 100 | 847 / 365 MiB | 839 / 363 MiB |
| B1, B2 | 98 | 587 / 365 MiB | 579 / 363 MiB |

## Results

c1 score: method of `bench/RESULTS.md` (five kinds, 2 runs each, 1,024 forced tokens, html 2,048, temperature 0, `reasoning_effort: low`). Concurrency: `--phase decode --concurrency 1,2,4 --runs 2 --distinct`, sum of the stream rates, median of 2 rounds. Every request ended with `finish_reason: length`; vision 4 of 4 in every arm.

| arm | c1 score (tok/s) | code / prose / chat / html / edit (tok/s, median of 2) | distinct c1 (tok/s) | c2 sum (tok/s) | c4 sum (tok/s) |
|---|---|---|---|---|---|
| A1 (100) | 65.3 | 62.3 / 69.4 / 72.8 / 67.2 / 54.7 | 63.9 | 77.0 | 92.0 |
| B1 (98) | 65.8 | 64.6 / 71.0 / 73.1 / 66.1 / 54.2 | 66.4 | 75.9 | 90.5 |
| A2 (100) | 65.0 | 61.8 / 69.0 / 72.9 / 66.8 / 54.5 | 64.5 | 77.5 | 90.9 |
| B2 (98) | 66.2 | 62.0 / 70.6 / 72.9 / 67.9 / 57.7 | 65.4 | 78.0 | 89.6 |
| mean 100 / 98 | 65.2 / 66.0 | | 64.2 / 65.9 | 77.3 / 77.0 | 91.5 / 90.1 |

Time to the first token in the distinct rounds: 0.42 to 0.48 s at c1, 0.73 to 0.84 s at c2, 1.55 to 1.85 s at c4.

`server-steps.txt` counts 15 requests per arm, including requests that draft nothing (1.51 to 1.56 tokens per step); its 17.05 to 17.35 ms per token are not comparable with the 10-request c1 figures of R914 and R927.

## Reading

- Both 98 arms read below both 100 arms at c4: mean 90.1 against 91.5 tok/s (−1.5 %). By the rule, 100,104 stays served.
- At c1, 98 reads +1.3 % on the c1 score and +2.6 % on the distinct c1 round, inside the run-to-run spread of the c1 measures (`docs/GOTCHAS.md`).
- At 98, the 260 MiB that 100 leaves free hold two more GPU experts on each of GPU0's 20 trunk layers and on the MTP layer; this round measured no gain from them.
