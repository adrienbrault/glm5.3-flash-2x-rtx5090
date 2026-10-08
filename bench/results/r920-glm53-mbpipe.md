# R920: two-lane micro-batch pipeline at 2 to 4 streams: output identical, 4-stream sum −10 %; not served

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r920-glm53-mbpipe-133444/`. Driver: `scripts/r920-glm53-mbpipe.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r920-glm53-mbpipe/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, free VRAM, configs, `P1/http-functional/summary.json`, the quality gate's report as `Q/report-without-ids.json`, `summary.txt`). The gate's report and one line of `summary.txt` carry generated token ids; the published copies replace those ids with a marker and keep every other field. The overlay (`mbpipe-r1`, written by an OpenAI Codex agent, with a native worker change) is not in this repository.

## What the overlay does

With `EXL3_MB_PIPELINE=1` and two to four active requests, the batch splits into two lanes: GPU0 runs one lane's first stage (its layers) while GPU1 runs the other lane's second stage, each lane samples and re-enters on its own, and the one 8-thread CPU expert worker serves both lanes in the order they become ready. At c1, during prefill, and when requests join, finish or are cancelled, it drains to the normal path. The design model predicted 110 to 129 tok/s summed at c4.

## Quality gate

One process, the served model at 104 routed experts per layer on the CPU with the exchange placement frozen, 4 streams with prompts of 4,096 to 4,147 tokens, 128 generated tokens, plain decode (no MTP). Greedy: the pipeline's tokens equal the normal path's and a repeat of it. Teacher-forced on the same reference tokens (512 positions), against the normal path:

| arm | mean KL | p99 KL | top-1 agreement |
|---|---|---|---|
| normal path repeated (the floor) | 7.1e-7 | 8.6e-6 | 1.000 |
| pipeline | 1.6e-6 | 2.0e-5 | 1.000 |
| normal path, after the pipeline | 8.8e-7 | 1.5e-5 | 1.000 |

The gate's limits were three times the larger of the two normal-path figures for KL (2.6e-6 mean, 4.5e-5 p99, from the arm after the pipeline) and the normal path's top-1 minus 0.002 (0.998); `summary.txt` prints those limits after `limits=`. PASS.

## Served arms

One session on 2026-10-08 from 13:34 UTC, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_mbpipe1` (built from `..._overhead-r2`, the image served at the time), with the configuration served at the time: 104 routed experts per layer on the CPU on both cards, `GPU_SPLIT=31,31`, MTP depth 1 with `EXL3_MTP_MAX_BATCH=1` and the four host flags, `VISION_OFFLOAD=1`, exchange swaps. Order F0, P1, F0b. Method of `bench/RESULTS.md`; every request ended with `finish_reason: length`; vision 4 of 4 in every arm.

| arm | `EXL3_MB_PIPELINE` | c1 score (tok/s) | server ms per token (c1, 10 requests) | distinct c1 (tok/s) | c2 sum / per stream (tok/s) | c4 sum / per stream (tok/s) | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|---|---|---|---|---|
| F0 | 0 | 62.8 | 16.10 | 63.3 | 74.7 / 37.4 | 86.0 / 21.5 | 1,349 / 365 MiB |
| P1 | 1 | 63.6 | 15.86 | 63.7 | 76.4 / 38.2 | 78.5 / 19.6 | 1,347 / 365 MiB |
| F0b | 0 | 64.5 | 15.64 | 64.6 | 75.6 / 37.8 | 88.7 / 22.2 | 1,347 / 365 MiB |

In P1 the engine log counts 51 pipeline layouts and 51 drains during the decode rounds and none during the c1 kinds. P1's functional checks (`functional_http.py`: ragged c2 to c4, a disconnect, a long prefill during decode, a prefix repeat, vision, a forced tool call) passed, 20 of 20 requests.

## Reading

- At 4 streams the pipeline measured 78.5 tok/s summed against 86.0 and 88.7 for the two arms without it (−10.1 % against their mean); at 2 streams 76.4 against 74.7 and 75.6 (+1.7 %). At 1 stream the pipeline does not engage, and the c1 figures lie between those of the two control arms.
- The design assumed the CPU worker serves one lane while the GPUs run the other. The measured sum fell instead; this round did not trace where the lanes wait, so the cause is not established.
- Not served; reopening it starts from a trace of one 4-stream round with the lanes on.
