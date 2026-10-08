# R914: the served configuration (MTP depth 1 while one request is active, 104 experts per layer on the CPU, vision weights in host RAM) measured in three arms

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r914-glm53-promote-combo-111753/`. Driver: `scripts/r914-glm53-promote-combo.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r914-glm53-promote-combo/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, `CF/sanity.jsonl`, configs, VRAM at boot, `summary.txt`). Engine logs are not published; the server step figures below are `scripts/mtp_steps.py` over each arm's engine log, as `summary.txt` records them.

## Arms

All arms in one session on 2026-10-08, in this order, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2` (`docker/mtp-overhead-r2/`), with the settings of `scripts/glm-daily.env` on top of the base settings of `bench/RESULTS.md`: MTP draft depth 1 with `MTP_FAST=1` (`docker/mtp-fast-r1/`), `EXL3_MTP_MAX_BATCH=1`, `VISION_OFFLOAD=1`, 104 routed experts per MoE layer on the CPU, exchange swaps from the broad hot set (interval 64, histogram policy), agent overlay r2, keep-thinking template.

| arm | MTP host-overhead flags (`EXL3_DRAFT_PINNED_STAGING`, `EXL3_MTP_GPU_DRAFT`, `EXL3_MTP_GREEDY_ACCEPT`, `EXL3_MTP_CACHED_REWIND`) | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|
| C0 | off | 1,347 / 365 MiB |
| CF | on: `scripts/glm-daily.env` as served | 1,347 / 365 MiB |
| C1 | off (C0 repeated) | 1,349 / 365 MiB |

## c1 by content kind

Method of `bench/RESULTS.md`: one chat request per run forced with `min_tokens` to 1,024 tokens (html 2,048), temperature 0, `reasoning_effort: low`, 2 runs; every run ended with `finish_reason: length`. Draft acceptance per run in parentheses.

| kind | C0 runs (tok/s) | CF runs (tok/s) | C1 runs (tok/s) |
|---|---|---|---|
| code | 61.4, 66.2 (0.83, 0.82) | 58.6, 60.3 (0.88, 0.90) | 62.4, 60.8 (0.83, 0.90) |
| prose | 69.4, 68.7 (0.66, 0.64) | 68.1, 72.4 (0.63, 0.68) | 67.5, 68.2 (0.64, 0.63) |
| chat | 72.1, 74.1 (0.66, 0.69) | 70.2, 69.3 (0.66, 0.69) | 70.5, 73.3 (0.68, 0.67) |
| html | 65.2, 67.0 (0.60, 0.56) | 65.3, 66.8 (0.62, 0.84) | 65.3, 65.9 (0.60, 0.86) |
| edit | 53.9, 56.0 (0.92, 0.93) | 53.5, 57.4 (0.95, 0.88) | 53.5, 54.3 (0.93, 0.97) |
| c1 score (mean of the kind medians) | 65.4 | 64.2 | 64.2 |

Time to the first token: 0.42 to 0.49 s for code, prose and chat (about 60 prompt tokens), 0.55 to 0.59 s for html, 1.80 to 1.87 s for edit (its prompt carries a 1.7k-token module).

## Server step at c1

`scripts/mtp_steps.py` over the 10 c1 requests of each arm (the engine's per-request step counters).

| arm | ms per MTP step | tokens per step | ms per token |
|---|---|---|---|
| C0 | 26.21 | 1.696 | 15.46 |
| CF | 27.56 | 1.756 | 15.69 |
| C1 | 27.58 | 1.753 | 15.73 |

## Concurrency, distinct prompts

`glm53_probe.py --phase decode --concurrency 1,2,4 --runs 2 --distinct`: c streams start together behind a barrier, stream i gets the code, prose, chat or html prompt in that order, each forced to 1,024 tokens at temperature 0. Sum of the stream rates over the confirmed common decode window, median of 2 rounds; per-stream figure from the same rounds. Every stream ended with `finish_reason: length`. The c2 and c4 rounds of every arm record `no accepted drafts`: with two or more active requests `EXL3_MTP_MAX_BATCH=1` decodes without the draft.

| arm | c1 (tok/s) | c2, sum / per stream (tok/s) | c4, sum / per stream (tok/s) | time to first token c1 / c2 / c4 (s, median) |
|---|---|---|---|---|
| C0 | 65.9 | 75.8 / 37.9 | 83.5 / 20.9 | 0.43 to 0.47 / 0.76 to 0.80 / 1.66 to 1.89 |
| CF | 65.3 | 75.5 / 37.7 | 87.7 / 21.9 | 0.45 to 0.47 / 0.76 to 0.81 / 1.70 to 1.73 |
| C1 | 64.2 | 75.5 / 37.7 | 88.6 / 22.1 | 0.42 to 0.48 / 0.77 to 0.80 / 1.67 to 1.72 |

The c1 round is the code prompt alone; its draft acceptance was 0.84 and 0.86 in CF.

## Vision and short answers

`--phase vision` (256-pixel red, blue and green squares and a left/right split): 4 of 4 correct in every arm, with the vision tower's linear weights in pinned host RAM. CF's six short-answer requests (`--phase sanity`, a product, a capital, a repeated word, with and without reasoning) all returned the expected answer.

## Reading

- The four host-overhead flags measure inside the arm-to-arm spread: CF's 15.69 ms per token lies between C0's 15.46 and its repeat C1's 15.73, its c1 score equals C1's, and its c4 sum lies between the two (83.5, 88.6). They stay in the served configuration.
- The served configuration measures a c1 score of 64.2 tok/s and 15.69 ms per token on the server. R911 measured the previous configuration (MTP off, 96 experts per layer on the CPU, vision on the GPU) twice in its own session at 57.1 and 58.9, 17.50 and 16.97 ms per token (`r911-glm53-mtpcap-dynamic.md`).
- At c4 the sum of the stream rates is 83.5 to 88.6 tok/s against 96.6 and 98.1 for the previous configuration in R911, with the same distinct-prompt method, earlier the same day in another session. With the batch cap the MTP head drafts nothing at c4, so the difference is attributed to running 104 instead of 96 experts per layer on the CPU, which the MTP head's VRAM requires (96 to 102 did not boot in R911): R902 (`2026-10-08-r902-glm53-mtp-overhead-071847`, static placement) measured MTP with the cap at 74.6 tok/s against 74.5 for plain decode at c4 when both ran at 104 experts per layer.
- GPU0 keeps 1,347 MiB free after warmup while GPU1 keeps 365 MiB: the layer split gives every layer the same CPU expert count, so the card with room cannot take more experts. Per-device counts are the next round (R915).
- This round measured no prefill.
