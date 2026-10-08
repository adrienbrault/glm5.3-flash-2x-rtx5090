# R911: MTP depth 1 with the batch cap on the exchange-swap placement against the configuration served before it

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r911-glm53-mtpcap-dynamic-095005/`. Driver: `scripts/r911-glm53-mtpcap-dynamic.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r911-glm53-mtpcap-dynamic/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, configs, VRAM at boot for the arms that booted, `summary.txt`). Engine logs are not published; the server step figures are `scripts/mtp_steps.py` over each arm's engine log, as `summary.txt` records them.

## Arms

One session on 2026-10-08, in this order. All arms use exchange swaps from the broad hot set (interval 64, histogram policy), agent overlay r2 and the keep-thinking template, with the base settings of `bench/RESULTS.md`.

| arm | image | MTP | vision tower | routed experts per layer on the CPU | boot |
|---|---|---|---|---|---|
| D0 | `tabbyapi:cheapswap-r3-agent-r2` | off | GPU | 96 | yes |
| M-n96, M-n98, M-n100, M-n102 | `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2` | depth 1, `MTP_FAST=1`, `EXL3_MTP_MAX_BATCH=1` | host RAM (`VISION_OFFLOAD=1`) | 96, 98, 100, 102 | no: `Insufficient VRAM in split for model and cache` |
| M-n104 | as above | as above | host RAM | 104 | yes |
| D1 | as D0 | off | GPU | 96 | yes |

D0 and D1 are the configuration served from R885c until this round's successor was promoted. M-n104 is the served configuration of R914 without the four MTP host-overhead flags. VRAM in use after boot: D0 and D1 31,662 and 31,032 MiB (489 and 1,119 MiB free); M-n104 30,780 and 31,786 MiB (1,371 and 365 MiB free).

## c1 by content kind

Method of `bench/RESULTS.md`, 2 runs, every run ended with `finish_reason: length`. Draft acceptance per run in parentheses for M-n104.

| kind | D0 runs (tok/s) | M-n104 runs (tok/s) | D1 runs (tok/s) |
|---|---|---|---|
| code | 53.1, 56.1 | 59.6, 63.0 (0.84, 0.86) | 55.6, 57.9 |
| prose | 61.0, 61.8 | 69.0, 68.9 (0.70, 0.62) | 61.6, 62.4 |
| chat | 62.6, 56.6 | 70.2, 71.9 (0.68, 0.68) | 64.1, 62.4 |
| html | 60.4, 60.6 | 65.5, 68.6 (0.61, 0.59) | 61.6, 62.1 |
| edit | 48.3, 50.3 | 52.7, 54.3 (0.94, 0.95) | 48.5, 53.1 |
| c1 score | 57.1 | 64.4 | 58.9 |
| server ms per token (ms per step, tokens per step) | 17.50 (17.50, 1.000) | 15.65 (26.82, 1.714) | 16.97 (16.97, 1.000) |

## Decode, distinct prompts

`glm53_probe.py --phase decode --concurrency 1,4 --runs 2 --distinct` (the method of R914's write-up). Median of 2 rounds.

| arm | c1 (tok/s) | c4, sum / per stream (tok/s) | time to first token c4 (s, median) |
|---|---|---|---|
| D0 | 57.3 | 98.1 / 24.5 | 1.51, 1.53 |
| M-n104 | 66.6 | 88.9 / 22.2 | 1.68, 1.70 |
| D1 | 58.8 | 96.6 / 24.1 | 1.54, 1.62 |

M-n104's c4 rounds record `no accepted drafts`. Vision: 4 of 4 correct in every arm that booted.

## Reading

- M-n104 measures a c1 score of 64.4 against 58.0 for the mean of D0 and D1 (+11 %) and 66.6 against 58.1 tok/s on the distinct-prompt c1 round (+15 %); the server time per token is 15.65 ms against 17.50 and 16.97 ms.
- At c4 M-n104 sums 88.9 tok/s against 97.4 for the mean of D0 and D1 (−9 %), with no drafts at c4. The MTP head's VRAM is the reason the arm needs 104 experts per layer on the CPU; 96 to 102 did not boot.
- After this round the configuration of M-n104 plus the four MTP host-overhead flags of `docker/mtp-overhead-r2/` was promoted; R914 measures it (`r914-glm53-promote-combo.md`).
