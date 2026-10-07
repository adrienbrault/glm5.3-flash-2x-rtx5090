# GLM-5.3-Flash on 2× RTX 5090

GLM-5.3-Flash (zai-org, 321 B parameters, about 18 B active per token, 42 MoE layers of 288 routed experts, 8 per token) served by TabbyAPI on ExLlamaV3 on one desktop box. The EXL3 checkpoint is turboderp's 2.05 bpw quantisation (76 GB on disk). It does not fit the 64 GB of VRAM: 96 of the 288 routed experts of every MoE layer run on the CPU from host RAM, and the experts the traffic uses most are swapped onto the GPUs while it runs.

## What is served

On port 8029 since R882c (2026-10-07, about 22:13 UTC); experiment rounds take the GPUs in between and restore this configuration when they end:

- Image `tabbyapi:cheapswap-r3-agent-r2`: ExLlamaV3 with the checkpoint-free expert exchange (`docker/cheapswap-r3/`) and TabbyAPI with the GLM agent overlay r2 (`docker/glm-agent-r2/`), built by `docker/glm-agent-r2-on-cheapswap-r3/`.
- Placement: 96 experts per MoE layer on the CPU worker (8 threads, one per physical core), initialised from `scripts/split-stats-broad-r869.json` (routing counts of 32 prompts) and then adapted by exchange swaps: histogram policy, a sweep every 64 decode tokens exactly, at most 64 swaps per sweep over all layers, floor 4, hysteresis 2.0. A swap copies the two experts between the pinned host arena and the GPU slot (0.46 ms per pair) instead of rereading them from the checkpoint.
- Agent overlay r2 on (`AGENT=1`): streamed tool-call arguments, SSE keepalive every 5 s, GLM `glm4_5` tool-call fixes; grammar forcing off.
- MTP draft off, vision on, cache 262,144 tokens at 8-bit K and V, `max_batch_size` 4, chunk 2,048.

## Numbers

Measured 2026-10-07 in R882b on the served configuration (`bench/results/r882b-glm53-swap-agent.md`, results directory on the box `/srv/qwen5090/results/2026-10-07-r882b-glm53-swap-agent-203933`). Decode samples are chat requests forced with `min_tokens`, temperature 0; a stream's rate is (tokens − 1) divided by the time between its first and last streamed text frame (`bench/RESULTS.md`, Method).

c1 decode by content kind, `reasoning_effort: low`, 1,024 forced tokens (html 2,048), median of 2 runs:

| code | prose | chat | html | edit | mean of the kinds |
|---|---|---|---|---|---|
| 58.1 tok/s | 62.5 tok/s | 64.1 tok/s | 63.0 tok/s | 51.6 tok/s | 59.9 tok/s |

Concurrent decode, code-tutorial prompt, 1,024 forced tokens per stream, streams started together, median of 3 rounds:

| concurrency | per-stream decode | sum of the stream rates | time to first token |
|---|---|---|---|
| 1 | 59.9 tok/s | 59.9 tok/s | 0.38 s |
| 2 | 41.1 tok/s | 82.2 tok/s | 0.67 to 0.77 s |
| 4 | 27.5 tok/s | 109.9 tok/s | 1.27 to 1.32 s |

Cold prefill, engine-timed, `cached_tokens` 0: 8,156 prompt tokens at 1,999 tok/s; 32,811 prompt tokens at 2,146 tok/s.

- R882c re-ran the checks on the same image and environment: 256-pixel vision check (red, blue, green, left/right split) passed; 4 of 4 simple streamed `write_file` calls returned valid JSON with the exact body (`bench/results/r882c-glm53-swap-agent.md`).
- R883's same-session control of this configuration measured a c1 mean of 61.1 tok/s (code 58.9, prose 64.5, chat 65.7, html 64.7, edit 51.8); the c1 mean moves by about 1 to 2 % from boot to boot (`docs/GOTCHAS.md`).
- The first requests after a boot decode slower until the placement has adapted to the traffic (R870, R872).
- Card memory at boot: 31,250 and 30,484 MiB in use, 901 and 1,667 MiB free.

Pending on 2026-10-08, not reflected above: R885 (agent overlay r4c on the strict tool-call cases, and the keep-thinking template described below). R883 measured MTP depth 1 and 2 on this configuration and found no gain (`bench/results/r883-glm53-swap-mtp.md`).

## How to run it

`scripts/launch-glm53.sh` writes the TabbyAPI config from `scripts/glm53_plan.py`, removes the image's own `EXL3_*` variables, checks that the requested offload registered, and starts the container on :8029. It expects the box's paths (`/storage/data/models/<pack>`, `/srv/qwen5090/`); `scripts/FLAG-DECISIONS.txt` lists each engine flag and why it is set or dropped. The served configuration:

```sh
PACK=A OFFLOAD_MODE=split OFFLOAD_N=96 CACHE_TOKENS=262144 MAX_SEQ=262144 CACHE_MODE=8,8 \
GPU_SPLIT=31,31 CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024 DRAFT=0 \
GLM_IMG=tabbyapi:cheapswap-r3-agent-r2 PINNED_ARENA=1 AGENT=1 \
SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$PWD/scripts/split-stats-broad-r869.json \
SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0 \
bash scripts/launch-glm53.sh
```

`KEEP_THINKING` defaults to 1 since 2026-10-08: the launcher then mounts `scripts/templates/glm53-keep-thinking.jinja`, the model's template with `clear_thinking` defaulting to false, so a new user message in an agent session does not rewrite the previous tool loop and the prefix cache keeps matching (`docs/GOTCHAS.md`). The container serving at the time of writing was started before that change and runs the model's own template; R885 measures the cached tokens with the new default. `KEEP_THINKING=0` serves the model's own template.

The chat template always opens a `<think>` block; `reasoning_effort` (`low`, `high`, default `max`) is its only reasoning control. Tool calls need `tool_format: glm4_5`, which the launcher sets.

## Hardware

- Two NVIDIA RTX 5090 (32 GB each), each on PCIe 5.0 ×8; layer split 31 + 31 GiB; stock power limits 600 and 575 W, memory clock offset +4,500 MHz, core offset 0.
- AMD Ryzen 7 9800X3D (8 cores, one CCD, 96 MB L3); the CPU expert worker reads cold experts at about 61 GB/s of a 63 GB/s read ceiling (R874).
- 60 GB usable of 2 × 32 GB dual-channel DDR5-6000.

## More

- `bench/RESULTS.md` indexes every round, newest first, with the shared method; `bench/results/` holds one write-up per round and the raw records.
- `docs/HISTORY.md` lists how the served configuration got here, from R858's 49.5 tok/s (MTP on, code-tutorial prompt) and R860's 46.2 (MTP off, four kinds).
- `docs/GOTCHAS.md` lists the traps met on the way.
- `docs/PLAN.md` lists the next levers and the parked ones, each with its source.
- `docker/` holds the overlays; `THIRD_PARTY.md` credits the code and ideas this builds on.
