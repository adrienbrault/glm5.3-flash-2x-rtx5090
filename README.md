# GLM-5.3-Flash on 2× RTX 5090

[GLM-5.3-Flash][model] by [zai-org][zai] served by [TabbyAPI][tabby] on [ExLlamaV3][exl3] from one desktop box.

**Goal**: serve coding agents, one to four sessions at once, with long prompts that grow with every tool call. Single-stream decode comes first because an agent waits on its own stream; the 4-stream sum, prefill and tool-call correctness follow.

- **Model**: 321 B parameters, about 18 B active per token; [turboderp][turboderp]'s [2.05 bpw EXL3 quantisation][ckpt], 76 GB.
- **Box**: two RTX 5090 (64 GB of VRAM) and 64 GB of DDR5 ([Hardware](#hardware)).
- **Offload**: the checkpoint does not fit the VRAM, so part of the 288 routed experts of every MoE layer run on the CPU ([ExLlamaV3's CPU MoE offload][exl3]): 96 per layer for the layers on GPU0 (layers 3 to 22 and the MTP layer), 102 per layer for the layers on GPU1 (layers 23 to 44).
- **Swaps**: the experts the traffic uses most move onto the GPUs while it runs.
- **Draft**: the model's MTP head drafts one token per step while one request is active; with two to four active requests every stream decodes without a draft.
- **Exact output**: no expert is skipped; every drafted token is verified by the full model.
- **Patches written here**: nine overlays. [`docker/cheapswap-r3/`](docker/cheapswap-r3/) patches ExLlamaV3 for the expert swaps; [`docker/glm-agent-r2/`](docker/glm-agent-r2/) patches TabbyAPI for GLM tool calls in agent harnesses (streamed arguments, tool-call parsing fixes, reasoning-history recovery, keepalive); [`docker/mtp-fast-r1/`](docker/mtp-fast-r1/) runs the MTP head's CPU experts on the trunk's CPU worker; [`docker/mtp-overhead-r2/`](docker/mtp-overhead-r2/) adds the one-request draft cap and four host-side MTP step changes; [`docker/glm-splitdev-r2/`](docker/glm-splitdev-r2/) sets the CPU expert count per GPU; [`docker/glm-index-ring-r3/`](docker/glm-index-ring-r3/) keeps the DSA indexer's per-token rows in a fixed ring instead of the KV pool; [`docker/tabby-livemetrics-r3/`](docker/tabby-livemetrics-r3/) adds live token counters to TabbyAPI's `/metrics`; [`docker/draftchunk-r1/`](docker/draftchunk-r1/) loads the MTP draft model with the prefill chunk size, so chunks above 2,048 tokens boot; [`docker/textclock-r1/`](docker/textclock-r1/) gives the expert-exchange clock to the first text MoE layer, so swaps also run while two to four requests are active.
- **Base image**: the served image of [qwen3.8-flash-next-2x-rtx5090][fn], ExLlamaV3 `dev` `5783a93` with that repository's 19 patch sets and TabbyAPI `53da7919`. The launcher clears that image's `EXL3_*` selectors and passes only the GLM offload, swap and MTP settings ([`scripts/FLAG-DECISIONS.txt`](scripts/FLAG-DECISIONS.txt)).
- **Window and KV pool**: 262,144 tokens per request; one 8-bit KV page pool of 262,144 tokens shared by up to 4 concurrent requests, about 1.6 GiB (608 bytes per token on each of the 11 attention layers: latent 512, scales 32, pooled indexer keys 64), plus an 8.5 MiB ring per DSA layer for the indexer's per-token rows; the other 31 layers keep a fixed-size recurrent state. Vision on, with the vision tower's linear weights in pinned host RAM.

How it got here: [`docs/HISTORY.md`](docs/HISTORY.md). Credits: [`THIRD_PARTY.md`](THIRD_PARTY.md).

[model]: https://huggingface.co/zai-org/GLM-5.3-Flash
[zai]: https://huggingface.co/zai-org
[tabby]: https://github.com/theroyallab/tabbyAPI
[exl3]: https://github.com/turboderp-org/exllamav3
[turboderp]: https://huggingface.co/turboderp
[ckpt]: https://huggingface.co/turboderp/GLM-5.3-Flash-exl3
[fn]: https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090

## Speed

Decode measured 2026-10-10 from 17:08 UTC on the served configuration ([R969](bench/results/r969-glm53-textclock.md), arms B1, B2 and B3, three fresh boots, results directory `2026-10-10-r969-glm53-textclock-170158`). On each boot the requests ran in the order 1, 2, 3, 4 streams, starting right after the launcher's boot warmup; each figure is the median of the three boots. 1 stream: the ten c1-score requests (code, prose, chat, html, edit, 2 runs each). 2 to 4 streams: a different prompt per stream (code, prose, chat, html, in that order), sum of the stream rates over the common decode window, median of 2 rounds per boot; with two or more streams no token is drafted. Every decode stream is a chat request forced to 1,024 tokens (html 2,048), temperature 0. Prefill and the server-side counters are from R959d (2026-10-10 from 06:19 UTC, arm C4a, results directory `2026-10-10-r959d-glm53-chunk-confirm-HoWhmg`), on the image before the text clock with the same settings otherwise; R969 did not measure them.

![Decode rate at 1 to 4 concurrent streams: per stream 64.8, 37.2, 30.7 and 24.5 tok/s; sum of the streams 64.8, 74.4, 92.0 and 98.0 tok/s](docs/img/decode-concurrency.svg)

![Single-stream decode by content kind with the MTP draft: code 59.9, prose 66.7, chat 68.5, html 64.8, edit 53.9 tok/s](docs/img/c1-by-kind.svg)

![Cold prefill rate on the image before the text clock: 1,973 tok/s at 32,738 prompt tokens, 2,023 tok/s at 131,051](docs/img/prefill.svg)

<details><summary>The same numbers as tables</summary>

| R969, served configuration | median of B1, B2, B3 | range over the three boots |
|---|---|---|
| decode, 1 stream, median of the ten c1-score requests | 64.8 tok/s | 63.7 to 65.6 tok/s |
| c1 score (mean of the five kinds' medians) | 62.9 tok/s | 62.2 to 63.2 tok/s |
| decode, 2 streams, per stream / sum | 37.2 / 74.4 tok/s | 37.2 to 37.7 / 74.3 to 75.5 tok/s |
| decode, 3 streams, per stream / sum | 30.7 / 92.0 tok/s | 30.1 to 30.8 / 90.3 to 92.5 tok/s |
| decode, 4 streams, per stream / sum | 24.5 / 98.0 tok/s | 23.9 to 24.7 / 95.6 to 98.8 tok/s |
| time to the first token, 1 stream (median per boot) | 0.48 s | 0.47 to 0.49 s |
| time to the first token, 2 / 3 / 4 streams (median per round) | | 0.75 to 0.82 / 1.11 to 1.21 / 1.57 to 1.70 s |

| R969, 1 stream by kind | code | prose | chat | html | edit |
|---|---|---|---|---|---|
| median of B1, B2, B3 | 59.9 tok/s | 66.7 tok/s | 68.5 tok/s | 64.8 tok/s | 53.9 tok/s |

| R959d arm C4a, image before the text clock | |
|---|---|
| cold prefill, 32,738 prompt tokens | 1,973 tok/s |
| cold prefill, 131,051 prompt tokens | 2,023 tok/s |

</details>

- Draft acceptance at 1 stream 0.59 to 0.97 per request over the three boots, highest on edits and code (R969).
- Single stream, server side (R959d arm C4a, the engine's per-request counters over its 10 c1-score requests): 15.61 ms per token, 27.19 ms per MTP step at 1.741 tokens per step. Four streams: 11.01 ms per generated token, the longest stream's generation time over the 4 streams' tokens, median of 2 rounds.
- Prefill: engine-timed (`usage.prompt_time`) on a cold prompt (`cached_tokens` 0) of real text, in chunks of 4,096 tokens.
- Spread: the three boots differ by up to 3.0 % on the 1-stream median and 3.4 % on the 4-stream sum. The comparison with the clock served before, on the same image in the same session, is in the [R969 write-up](bench/results/r969-glm53-textclock.md) and [`docs/HISTORY.md`](docs/HISTORY.md).

Method and run-to-run spread: [`bench/RESULTS.md`](bench/RESULTS.md). The charts are drawn from the raw records by [`bench/plot.py`](bench/plot.py) (`uv run bench/plot.py`).

## Memory

The served configuration uses 31,600 MiB of GPU0 and 31,864 MiB of GPU1 after warmup, leaving 551 and 287 MiB free, and 267 MiB on GPU1 after a 131k-token prefill (`nvidia-smi`, R959d arm C4a); the text clock changes no allocation (R968: 563 and 297 MiB free after the launcher's boot warmup with and without it). The chart below is the configuration served before the per-device count and the ring (104 experts per layer on the CPU on every layer), measured 2026-10-08 idle after warmup ([R900](bench/results/r900-glm53-memory-layout.md)); the served configuration moves 8 experts per layer of GPU0's 21 MoE layers and 2 per layer of GPU1's 22 from host RAM to the GPUs, removes the 1.4 GiB per-token indexer plane from the KV pool, prefills in 4,096-token chunks, and has no category breakdown yet. Weights come from the checkpoint's tensor sizes and the served config; the VRAM total comes from `nvidia-smi` and the host total from the container's memory cgroup (anonymous plus shared memory, without the page cache); "other" is the measured total minus the listed items.

![VRAM: 61.1 of 63.7 GiB used, of which routed experts 46.6, other weights 4.6, KV pool 3.0, other 6.8; host DRAM: 32.9 of 60.4 GiB used, of which routed experts 26.4, embedding table 1.2, vision tower 0.5, other 4.9](docs/img/memory.svg)

## What is served

The whole configuration is one line, [`scripts/glm-daily.env`](scripts/glm-daily.env), installed on the box by [`scripts/install-glm-daily.sh`](scripts/install-glm-daily.sh).

- Image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1` on port 8029: [`docker/cheapswap-r3/`](docker/cheapswap-r3/) and [`docker/glm-agent-r2/`](docker/glm-agent-r2/) combined ([`docker/glm-agent-r2-on-cheapswap-r3/`](docker/glm-agent-r2-on-cheapswap-r3/)), then [`docker/mtp-fast-r1/`](docker/mtp-fast-r1/), then [`docker/mtp-overhead-r2/`](docker/mtp-overhead-r2/), then [`docker/glm-splitdev-r2/`](docker/glm-splitdev-r2/), then [`docker/glm-index-ring-r3/`](docker/glm-index-ring-r3/), then [`docker/tabby-livemetrics-r3/`](docker/tabby-livemetrics-r3/), then [`docker/draftchunk-r1/`](docker/draftchunk-r1/), then [`docker/textclock-r1/`](docker/textclock-r1/).
- `GPU_SPLIT=31.8,31` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102`: 96 experts per MoE layer on the CPU for the layers on GPU0, 102 for the layers on GPU1 (8 threads). Placement starts from [`scripts/split-stats-broad-r869.json`](scripts/split-stats-broad-r869.json) and adapts every 64 decode tokens by swapping hot CPU experts with cold GPU ones.
- MTP draft depth 1 with `MTP_FAST=1` and `EXL3_MTP_MAX_BATCH=1` (drafting only while one request is active), plus `EXL3_DRAFT_PINNED_STAGING`, `EXL3_MTP_GPU_DRAFT`, `EXL3_MTP_GREEDY_ACCEPT` (greedy requests only) and `EXL3_MTP_CACHED_REWIND`.
- `CHUNK=4096`: prompts are prefilled in chunks of 4,096 tokens (TabbyAPI's `chunk_size`), and the MTP draft model is loaded with the same bound ([`docker/draftchunk-r1/`](docker/draftchunk-r1/)). At 96,100 this chunk size does not boot (`Insufficient VRAM`), hence 102 for GPU1's layers.
- `EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1`: the exchange decides its swaps every 64 invocations of the first text MoE layer (layer 3) instead of the MTP layer, which does not run while two or more requests are active, so 2 to 4 concurrent streams also move experts onto the GPUs ([`docker/textclock-r1/`](docker/textclock-r1/)).
- Live token counters in `/metrics` ([`docker/tabby-livemetrics-r3/`](docker/tabby-livemetrics-r3/)), so a scrape-to-scrape rate is current during long generations.
- `INDEX_RING=1`: the DSA indexer's per-token rows in a ring of 4,352 rows per request slot on each DSA layer (one 4,096-token chunk plus the 256 rows a speculative rewind needs); the 4-token pooled indexer keys stay in the KV pool.
- `VISION_OFFLOAD=1`: the vision tower's linear weights in pinned host RAM. 262,144-token cache at 8-bit K and V, up to 4 concurrent requests.
- Agent overlay r2 (`AGENT=1`) and the keep-thinking chat template (`KEEP_THINKING=1`); a missing `reasoning_effort` means `high`; tool calls in TabbyAPI's `glm4_5` format.
- Sampler fallbacks [`scripts/sampler_overrides/glm53.yml`](scripts/sampler_overrides/glm53.yml): temperature 1.0 and top_p 0.95, the values of the model's `generation_config.json`, for requests that send none; a value the request sends is kept.

## Run it

```sh
PACK=A OFFLOAD_MODE=split CACHE_TOKENS=262144 MAX_SEQ=262144 CACHE_MODE=8,8 GPU_SPLIT=31,31 CHUNK=2048 MAX_BATCH=4 SYSMEM_RC_MB=1024 \
OFFLOAD_N=104 DRAFT=1 DRAFT_N=1 MTP_FAST=1 VISION_OFFLOAD=1 GPU_SPLIT=31.8,31 CHUNK=4096 \
GLM_IMG=tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1_textclock1 INDEX_RING=1 PINNED_ARENA=1 AGENT=1 KEEP_THINKING=1 \
EXL3_EXTRA='EXL3_MTP_MAX_BATCH=1;EXL3_DRAFT_PINNED_STAGING=1;EXL3_MTP_GPU_DRAFT=1;EXL3_MTP_GREEDY_ACCEPT=1;EXL3_MTP_CACHED_REWIND=1;EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102;EXL3_MOE_CPU_SWAP_TEXT_CLOCK=1' \
SWAP_MODE=exchange SWAP_POLICY=histogram SWAP_INIT_STATS=$PWD/scripts/split-stats-broad-r869.json \
SWAP_CADENCE=exact SWAP_INTERVAL=64 SWAP_MAX=64 SWAP_SCOPE=global SWAP_FLOOR=4 SWAP_HYST=2.0 \
bash scripts/launch-glm53.sh
```

The first line is the base settings every round uses ([`scripts/glm_arms.sh`](scripts/glm_arms.sh)); the rest is [`scripts/glm-daily.env`](scripts/glm-daily.env), whose `GPU_SPLIT` and `CHUNK` replace the base's. `OFFLOAD_N=104` is the configured count that `EXL3_MOE_CPU_SPLIT_BY_DEVICE` overrides per GPU; the override applies only when it is non-zero. [`scripts/launch-glm53.sh`](scripts/launch-glm53.sh) writes the TabbyAPI config, mounts the sampler preset from `scripts/sampler_overrides/`, removes the image's own `EXL3_*` variables, passes the `EXL3_EXTRA` flags to overlay images only, checks that the offload registered and starts the container. It expects the box's paths (`/srv/qwen5090/models/<pack>` on NVMe, falling back to `/storage/data/models/<pack>`; `/srv/qwen5090/`); [`scripts/FLAG-DECISIONS.txt`](scripts/FLAG-DECISIONS.txt) gives the reason for each engine flag.

## Notes

- Agent sessions: the keep-thinking template ([`scripts/templates/glm53-keep-thinking.jinja`](scripts/templates/glm53-keep-thinking.jinja)) is the model's template with `clear_thinking` defaulting to false, so earlier reasoning stays in the prompt and a new user message does not invalidate the prefix cache ([`docs/GOTCHAS.md`](docs/GOTCHAS.md)). In R885c a follow-up agent request reused 7,680 of its 9,915 prompt tokens from the cache with it, and 0 with the model's own template. `KEEP_THINKING=0` restores the model's template.
- Tool calls were last checked on overlay r2 in R882c (4 of 4 streamed `write_file` calls) and R885c (simple tool calls); R914 checked vision (4 of 4) and short answers, and R915c to R929c checked vision on the image served at the time, not tool calls. Strict tool-call cases with literal GLM tags in the arguments fail on overlay r2; candidates r4 to r4c are in [`docker/`](docker/), and r4c returned 16 of 16 of them as valid calls in R885 ([`bench/RESULTS.md`](bench/RESULTS.md)).
- Greedy output differs from one run to the next, even within one process, unless `EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_FREEZE_EXCHANGE=1` are added to `EXL3_EXTRA` (R916, a c1 cost of a few percent; off in the served configuration; [`docs/GOTCHAS.md`](docs/GOTCHAS.md)).
- The first requests after a boot decode slower until the placement has adapted to the traffic (R870, R872); before R968, traffic with two or more concurrent requests did not adapt it (R966b, R967).
- The chat template always opens a `<think>` block; `reasoning_effort` (`low`, `high`, `max`) is its only reasoning control. Both served templates default a missing or unknown value to `high`; the model's own template defaults to `max`, which in our runs often reasoned until the length limit. An explicit `max` is kept.
- How the configuration got here: [`docs/HISTORY.md`](docs/HISTORY.md). What is being tried next: [`docs/PLAN.md`](docs/PLAN.md). Traps: [`docs/GOTCHAS.md`](docs/GOTCHAS.md).

## Hardware

- Two NVIDIA RTX 5090 (32 GB each), each on PCIe 5.0 ×8; layer split 31.8 + 31 GiB; stock power limits 600 and 575 W, memory clock offset +4,500 MHz, core offset 0.
- AMD Ryzen 7 9800X3D (8 cores, one CCD, 96 MB L3); the CPU expert worker reads cold experts at about 61 GB/s of a 63 GB/s read ceiling (R874).
- 60 GB usable of 2 × 32 GB dual-channel DDR5-6000.

## More

- `bench/RESULTS.md` indexes every round, newest first, with the shared method; `bench/results/` holds one write-up per round and the raw records.
- `docs/HISTORY.md` lists how the served configuration got here, from R858's 49.5 tok/s (MTP on, code-tutorial prompt) and R860's 46.2 (MTP off, four kinds).
- `docs/GOTCHAS.md` lists the traps met on the way.
- `docs/PLAN.md` lists the next levers and the parked ones, each with its source.
- `docker/` holds the overlays; `THIRD_PARTY.md` credits the code and ideas this builds on.
