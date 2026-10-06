# GLM-5.3-Flash on 2× RTX 5090

GLM-5.3-Flash (zai-org, 321 B parameters, about 18 B active per token) served by TabbyAPI on ExLlamaV3 on one desktop box: two RTX 5090 cards (32 GB each, PCIe 5.0 ×8 each), a Ryzen 7 9800X3D and 60 GB of dual-channel DDR5-6000. The EXL3 checkpoint is turboderp's 2.05 bpw quantisation (76 GB on disk). It does not fit the 64 GB of VRAM, so the tail of every MoE layer's 288 routed experts runs on the CPU from host RAM. No configuration is promoted yet; this repository records the audition and the work on c1 decode.

## Numbers

Measured 2026-10-06 in round R858 (`bench/results/r858-glm53-audition.md`, raw records in `bench/results/2026-10-06-r858-glm53/`) on image `tabbyapi:r828-prompt-lookup-r3` (ExLlamaV3 `dev` at `5783a93` with TabbyAPI), configuration: 104 of 288 routed experts per MoE layer on the CPU (`cpu_moe_split_experts: 104`, 8 CPU threads), layer split 31 + 31 GiB, MTP draft depth 1, cache 65,536 tokens at 8-bit K and V. Each sample is one chat request forced to 1,024 tokens with `min_tokens`, greedy, `reasoning_effort: low`, the R858 code-tutorial prompt; rate is (tokens − 1) divided by the time between the first and the last streamed text frame.

| concurrency | per-stream decode, median of 3 | sum of the stream rates | MTP draft acceptance |
|---|---|---|---|
| 1 | 49.5 tok/s (runs 44.6, 49.5, 50.5) | 49.5 tok/s | 0.80 to 0.90 |
| 2 | 30.0 tok/s | 60.1 tok/s | 0.85 to 0.90 |
| 4 | 17.1 tok/s | 68.6 tok/s | 0.87 |

- Card memory at that configuration: 30.7 GB and 28.7 GB in use; host RAM 46 GB in use.
- GPU utilisation reads 11 to 12 % during c1 decode at about 110 W per card: the CPU-side experts set the step time.
- Prefill was not measured in R858: the first cold prefill raised an engine error (`docs/GOTCHAS.md`). Round R859 measures c1 decode by content kind and cold prefill plus decode at 8k, 32k, 128k and 240k tokens with a 256k cache.

## What the stack is

- Image: `tabbyapi:r828-prompt-lookup-r3`, built from the Qwen3.8-Flash-Next stack's image chain; the GLM-5.3 architecture (`glm5_next.py`, `glm5_next_mtp.py`) is upstream ExLlamaV3 code. The launcher removes every `EXL3_*` variable the image sets and passes only the CPU-offload settings (`scripts/FLAG-DECISIONS.txt` lists each one).
- Offload: TabbyAPI `cpu_moe_split_experts` runs the tail N routed experts of every MoE layer in a worker process on the CPU from compressed EXL3 trellis bytes; dynamic placement swaps frequently routed experts into the GPU slots between generations.
- Chat template: GLM-5.3's template always opens a `<think>` block; `reasoning_effort` (`low`, `high`, default `max`) is its only reasoning control.

## Work plan

`docs/PLAN.md` lists the expert-offload work for c1 decode, each item with the published work it comes from: a routing trace and a cache simulator first, then a GPU expert cache with next-layer router prediction and asynchronous host-to-device copies.
