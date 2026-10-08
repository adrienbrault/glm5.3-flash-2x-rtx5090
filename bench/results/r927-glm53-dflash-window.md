# R927: DFlash2 drafting with a window-sized drafter cache, MTP off: c1 54.3 against 63.2 tok/s and c4 58.3 against 93.5 for the served configuration in the same session; not served

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r927-glm53-dflash-window-204220/`. Driver: `scripts/r927-glm53-dflash-window.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r927-glm53-dflash-window/` (`<arm>/c1.jsonl`, `<arm>/dec12.jsonl` for 1 and 2 streams, `<arm>/dec4.jsonl` for 4 streams, `<arm>/vision.jsonl`, free VRAM after warmup and at the end, configs, `summary.txt`). Engine logs are not published; the server step figures are `scripts/mtp_steps.py` over each arm's engine log as `summary.txt` records them. The draft model's weights are not redistributed (`THIRD_PARTY.md`). The overlays (`dflash-adaptive-r1` and `dflash-swa-r1`, written by OpenAI Codex agents) are not in this repository.

## Setup

DFlash2 is a block-diffusion draft model for GLM-5.3-Flash (incoai) with five sliding-window layers (window 2,048 tokens); this round used r0b0tlab's EXL3 3.00 bpw quantisation of it (610 MiB) at `draft_num_tokens: 5`, on GPU0, with the model's own MTP head off. R921b had found that TabbyAPI sizes the drafter's KV cache for the full 262,144-token window, about 2.6 GB at 8 bits, which did not fit. `EXL3_DRAFT_WINDOW_CACHE=1` (`dflash-swa-r1`) keeps the drafter's KV on the GPU for the window only: 4 slots × 10 pages, 106.25 MiB, with a 2,720 MiB host backing for prefix reuse.

## Arms

One session on 2026-10-08 from 20:42 UTC. The DFlash image, `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_dflash1_swa3`, is built on the image served before the per-device split and runs one CPU expert count on every layer; its arms used `GPU_SPLIT=30.8,31` for the model and 1.0 GiB on GPU0 for the drafter, `MTP_FAST=0`, the served settings otherwise. The count was raised from 104 until both GPUs kept the 450 / 300 MiB the drivers require after warmup.

| arm | image | routed experts per layer on the CPU | outcome | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|---|---|
| C-n104 | DFlash | 104 on every layer | no boot (`Insufficient VRAM in split for model and cache`) | |
| C-n108 | DFlash | 108 | no boot (same error) | |
| C-n112 | DFlash | 112 | boots, below the floor | 437 / 1,765 MiB |
| C-n116 | DFlash | 116 | measured | 915 / 2,223 MiB |
| B-D | `..._splitdev2` (served) | 100 for GPU0's layers, 104 for GPU1's | measured: `scripts/glm-daily.env` unchanged | 847 / 365 MiB |

## c1 by content kind

Method of `bench/RESULTS.md`: one chat request per run forced with `min_tokens` to 1,024 tokens (html 2,048), temperature 0, `reasoning_effort: low`, 2 runs; every run ended with `finish_reason: length`. Draft acceptance per run in parentheses (C-n116: of 5 drafted tokens per step; B-D: of 1).

| kind | C-n116 runs (tok/s) | B-D runs (tok/s) |
|---|---|---|
| code | 53.0, 57.9 (0.56, 0.61) | 59.2, 60.8 (0.89, 0.88) |
| prose | 42.6, 44.6 (0.26, 0.26) | 67.6, 70.9 (0.64, 0.66) |
| chat | 45.5, 48.7 (0.27, 0.27) | 72.8, 72.6 (0.66, 0.70) |
| html | 47.1, 43.0 (0.44, 0.24) | 65.3, 67.9 (0.57, 0.60) |
| edit | 48.7, 47.8 (0.63, 0.62) | 53.7, 56.1 (0.94, 0.97) |
| c1 score (mean of the kind medians) | 47.9 | 64.7 |

Server step over the 10 c1 requests: C-n116 60.67 ms per step at 2.827 tokens per step, 21.46 ms per token; B-D 26.74 ms per step at 1.714 tokens per step, 15.60 ms per token.

## Concurrency, distinct prompts

`glm53_probe.py --phase decode --distinct --runs 2`, run as two invocations per arm: 4 streams first (`dec4.jsonl`), then 1 and 2 streams (`dec12.jsonl`). Forced to 1,024 tokens, temperature 0; sum of the stream rates over the common decode window, median of 2 rounds; every stream ended with `finish_reason: length`. The served configuration drafts nothing at 2 and 4 streams (`EXL3_MTP_MAX_BATCH=1`); the DFlash arm drafted at every concurrency (acceptance 0.32 to 0.37 per round at 2 and 4 streams).

| arm | c1 (tok/s) | c2, sum / per stream (tok/s) | c4, sum / per stream (tok/s) | time to first token c1 / c2 / c4 (s, round medians) |
|---|---|---|---|---|
| C-n116 | 54.3 | 53.1 / 26.6 | 58.3 / 14.6 | 0.58 to 0.61 / 1.21 to 1.29 / 2.47 to 2.59 |
| B-D | 63.2 | 78.9 / 39.5 | 93.5 / 23.4 | 0.44 / 0.74 to 0.78 / 1.55 to 1.74 |

Vision: 4 of 4 in both arms.

## Reading

- The draft model and its window cost about 12 routed experts per layer of VRAM: the DFlash arm needs 116 experts per layer on the CPU on every layer against the served 100 and 104.
- At c1 a DFlash step yields 2.83 tokens in 60.7 ms and an MTP step 1.71 tokens in 26.7 ms: 21.5 against 15.6 ms per token. Acceptance is 0.56 to 0.63 on code and edits and 0.24 to 0.27 on prose, chat and one html run.
- At 2 and 4 streams the DFlash sum stays at 53 to 58 tok/s while the served configuration reaches 78.9 and 93.5.
- What makes a DFlash step cost 2.3 times an MTP step is not established. Its verify step checks 6 rows instead of 2, and those rows can route to more distinct experts, part of them on the CPU; the drafter's own forward and the larger CPU count (116) are other candidates. This round has no measurement that separates them.
- Not served.
