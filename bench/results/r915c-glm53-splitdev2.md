# R915c: per-device CPU expert count, revision 2: 100 experts per layer on the CPU for GPU0's layers and 104 for GPU1's boots with 847 MiB free on GPU0; served

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r915c-splitdev2-131018/`. Driver: `scripts/r915c-glm53-splitdev2.sh` (helpers `scripts/glm_arms.sh`, probes `scripts/glm53_probe.py`). Raw records: `results/2026-10-08-r915c-glm53-splitdev2/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, `<arm>/server-steps.txt`, free VRAM after warmup and after the matrix, configs, `summary.txt`). Engine logs are not published. Overlay: `docker/glm-splitdev-r2/`.

## Before this round

R914 left GPU0 with 1,347 MiB unused after warmup and GPU1 with 365 MiB at 104 routed experts per MoE layer on the CPU: the layer split gives every layer the same CPU count, so the card with room cannot take more experts. Revision 1 of `EXL3_MOE_CPU_SPLIT_BY_DEVICE` (one override of the count by the layer's CUDA device) ran in two rounds, `scripts/r915-glm53-split-by-device.sh` and `scripts/r915b-glm53-split-by-device.sh`:

- R915 (`2026-10-08-r915-glm53-split-by-device-115217`), `GPU_SPLIT=31,31`: 94 and 96 for GPU0's layers did not boot (`Insufficient VRAM`); 98 and 100 loaded and failed the launcher's registration check, which then accepted one count only (`scripts/glm53_verify.py` now reads both). Its control arm D, the configuration served at the time (104 on every layer, `GPU_SPLIT=31,31`), measured a c1 score of 65.0 tok/s, 15.52 ms per token on the server, and 63.2, 75.5 and 86.8 tok/s summed at 1, 2 and 4 streams with distinct prompts.
- R915b (`2026-10-08-r915b-glm53-split-by-device-120518`), `GPU_SPLIT=31.8,31`: 94 and 96 did not boot; 98 loaded, and its first long warmup request (5,590 prompt tokens) ended in a device-side `index out of bounds` assert and exit 139. The diagnosis is in `docker/glm-splitdev-r2/DIAGNOSIS.md`: the autosplit moved layer 23 from GPU0 to GPU1 after registering it with GPU0's count, and the CPU worker kept the first registration.

## Arms

All arms in one session on 2026-10-08 from 13:10 UTC, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2`, `GPU_SPLIT=31.8,31`, the other settings of the configuration served at the time (MTP depth 1 with `EXL3_MTP_MAX_BATCH=1` and the four host flags, `VISION_OFFLOAD=1`, exchange swaps, agent overlay r2, keep-thinking template). The value in the arm name is the CPU count for GPU0's layers; GPU1's layers run 104 in every arm.

| arm | `EXL3_MOE_CPU_SPLIT_BY_DEVICE` | `EXL3_MOE_CPU_SPLIT_CHECK` | outcome | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|---|---|
| C-n98 | 98,104 | 1 | boot and warmup (including the 5,590-token request) with no check raised; not measured | 587 / 365 MiB |
| B-n98 | 98,104 | 0 | measured | 587 / 365 MiB |
| B-n96 | 96,104 | 0 | no boot | |
| B-n100 | 100,104 | 0 | measured | 847 / 365 MiB |

## c1 by content kind

Method of `bench/RESULTS.md`: one chat request per run forced with `min_tokens` to 1,024 tokens (html 2,048), temperature 0, `reasoning_effort: low`, 2 runs; every run ended with `finish_reason: length`. Draft acceptance per run in parentheses.

| kind | B-n98 runs (tok/s) | B-n100 runs (tok/s) |
|---|---|---|
| code | 55.9, 60.2 (0.83, 0.86) | 60.1, 61.9 (0.82, 0.87) |
| prose | 65.9, 65.1 (0.61, 0.60) | 66.1, 69.7 (0.59, 0.65) |
| chat | 67.7, 67.7 (0.66, 0.65) | 71.6, 71.7 (0.67, 0.65) |
| html | 64.0, 64.8 (0.59, 0.57) | 65.3, 68.8 (0.58, 0.61) |
| edit | 51.2, 53.9 (0.96, 0.94) | 56.5, 57.6 (0.90, 0.77) |
| c1 score (mean of the kind medians) | 61.6 | 64.9 |

## Concurrency, distinct prompts

`glm53_probe.py --phase decode --concurrency 1,2,4 --runs 2 --distinct`, forced to 1,024 tokens, temperature 0; sum of the stream rates over the common decode window, median of 2 rounds. Every stream ended with `finish_reason: length`; at 2 and 4 streams no token is drafted.

| arm | c1 (tok/s) | c2, sum / per stream (tok/s) | c4, sum / per stream (tok/s) | time to first token c1 / c2 / c4 (s, round medians) |
|---|---|---|---|---|
| B-n98 | 60.6 | 73.6 / 36.8 | 85.5 / 21.4 | 0.43 to 0.44 / 0.78 to 0.84 / 1.63 to 1.77 |
| B-n100 | 66.6 | 76.9 / 38.4 | 90.0 / 22.5 | 0.43 to 0.44 / 0.74 / 1.66 to 1.69 |

Vision (`--phase vision`, 256-pixel squares and a left/right split): 4 of 4 in both arms.

## Server step

`server-steps.txt` is `scripts/mtp_steps.py` over 15 requests of each arm's engine log: 26.25 ms per step at 1.516 tokens per step for B-n100, 27.83 ms at 1.520 for B-n98. The 15 requests include requests that draft nothing, which lowers the tokens per step; R914 and R927 count the 10 c1 requests only (1.71 to 1.76 tokens per step), so these figures are not compared with theirs.

## Reading

- With revision 2 the per-device count boots and serves at 98 and 100 for GPU0's layers; 96 does not fit at `GPU_SPLIT=31.8,31`. The check mode raised nothing on the request that failed in R915b.
- At 100,104 GPU0 keeps 847 MiB free after warmup and GPU1 365 MiB; at 98,104 GPU0 keeps 587 MiB.
- B-n100 measured a c1 score of 64.9 tok/s and a 4-stream sum of 90.0 tok/s. The configuration served before it measured 65.0 and 86.8 in R915's arm D, about 80 minutes earlier in another unit, and 83.5 to 88.6 at 4 streams in R914's three arms; no arm of this session ran the previous configuration, so the 4-stream difference is a comparison across sessions.
- B-n98 read lower than B-n100 on every measure in this session (c1 score 61.6, 4-stream sum 85.5). R925 repeated the pair in alternation.
- Served from 2026-10-08 about 13:37 UTC: `scripts/glm-daily.env` with image `..._splitdev2`, `GPU_SPLIT=31.8,31` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104`.
