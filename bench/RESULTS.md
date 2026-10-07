# Results index (newest first)

Each write-up's first line names the results directory on the box and the driver in `scripts/`; raw records are in `results/<date>-<round>/`. The method shared by the rounds is at the end of this file.

- Pending, 2026-10-08: R885 (agent overlay r4c on the 16 strict tool-call cases, then the served configuration with the keep-thinking template: cached prompt tokens of a follow-up agent request with and without the earlier reasoning). Driver `scripts/r885-glm53-r4c-keepthink.sh`, which replaced the queued R881e and R884 before they started.
- `results/r883-glm53-swap-mtp.md` — R883, 2026-10-07/08: MTP depth 1 at 104 experts per layer on the CPU measures 59.8 tok/s at c1 against a 61.1 control and 92.5 tok/s summed at c4 against 109.9; depth 2 at 112 is lower on every measure; MTP stays off.
- `results/r882c-glm53-swap-agent.md` — R882c, 2026-10-07: the swap and agent-overlay image passes vision and 4 of 4 simple tool calls and is served.
- `results/r882b-glm53-swap-agent.md` — R882b, 2026-10-07: exchange swaps with the agent overlay r2 on one image: c1 59.9 tok/s (code 58.1, prose 62.5, chat 64.1, html 63.0, edit 51.6), 82.2 and 109.9 tok/s summed over 2 and 4 streams, cold prefill 1,999 tok/s at 8,156 prompt tokens and 2,146 at 32,811.
- `results/r881d-glm53-agent-r4b.md` — R881d, 2026-10-07: agent overlay r4b does not boot (`NameError`, missing `import os`).
- `results/r880-glm53-agent-r4.md` — R880, 2026-10-07: agent overlay r4 passes 2 of 16 strict tool-call cases, every case a valid call with the tags intact; r2 passes 0 of 16.
- `results/r875c-glm53-cheapswap-r3.md` — R875b and R875c, 2026-10-07: cheapswap r3 gates pass; exchange swaps from the broad hot set 60.6 tok/s at c1 against 56.7 static (+6.9 %).
- `results/r874-glm53-cpuworker.md` — R874 and R874c, 2026-10-07: the CPU expert GEMV reads cold experts at 61 to 62 GB/s of a 63 GB/s ceiling; affinity, spin, prefetch and page-size variants are within 1 %; waking a parked worker pool costs about 210 µs.
- `results/r877-glm53-agent-r3.md` — R877, 2026-10-07: agent overlay r3 rejects every prompt quoting a GLM tag; r2 passes 0 of 12 strict cases.
- `results/r876-glm53-timeline.md` — R876, 2026-10-07: decode timeline: 11 ms of GPU kernels per token at c1, the rest of the step waiting on the CPU worker; one extra CPU pick costs 0.10 to 0.11 ms.
- `results/r873-glm53-c4.md` — R873, 2026-10-07: static placement with the agent overlay: 57.4 tok/s at c1, 86.3 summed over 4 streams; cold prefill 1,697 and 1,797 tok/s at 8,198 and 32,804 tokens.
- `results/r872-glm53-cheapswap.md` — R872, 2026-10-07: cheapswap r2 gates pass, 0.46 ms per exchange pair; fast exchange from the identity start 60.1 tok/s at c1 against 57.4 static.
- `results/r871-glm53-hostcore.md` — R871, 2026-10-07: no gain from reserving a host core; confining the server to one core is −31 %.
- `results/r870-glm53-depth.md` — R870, 2026-10-07: decode at 131k context runs at the short-context rate; the slow requests are the first ones after a boot.
- `results/r869-glm53-hotset.md` — R869, 2026-10-07: static hot set from 32 other prompts: 57.5 tok/s at c1 against 48.6 dynamic (+18.2 %).
- `results/r868-glm53-agent-debug.md` — R868, 2026-10-07: stock TabbyAPI ends a GLM tool call when the reasoning quotes a think tag; the agent overlay r2 returns the call.
- `results/r867-glm53-lowern.md` — R867, 2026-10-07: 92 experts per layer on the CPU: +2 % at half the batch, unreliable boots, not used; decode after a 32k prompt 22.7 tok/s.
- `results/r866-glm53-placement.md` — R866, 2026-10-07: in-sample static hot set 63.2 tok/s at c1 with no gain on the held-out kind; faster dynamic sweeps −20 %.
- `results/r865-glm53-vram.md` — R865, 2026-10-07: 96 experts per layer on the CPU reproduces 48.2 tok/s; 92 does not fit.
- `results/r864-glm53-levers.md` — R864, 2026-10-07: 96 on the CPU +4.8 % at c1; MTP depth 1 helps code and edits only; prompt lookup adds nothing; boot-to-boot drift 1.3 %.
- `results/r860b-glm53-trace.md` — R860b, 2026-10-07: routing trace and CPU profile; dynamic placement leaves the CPU share of picks at 0.347 against 0.361 uniform.
- `results/r860-glm53-chain.md` — R860, 2026-10-07: MTP off 46.2 tok/s against depth 1 45.7 at c1; depth 2 and 3 do not fit; 16 CPU threads halve decode.
- `results/r859-glm53-c1.md` — R859, 2026-10-06: c1 by content kind at a 256k cache with MTP depth 1 (code 46 to 51 tok/s); cold prefill 1,383 to 1,758 tok/s from 8,200 to 130,998 tokens.
- `results/r858-glm53-audition.md` — R858, 2026-10-06: first boot of GLM-5.3-Flash 2.05 bpw on two RTX 5090 with 104 of 288 experts per MoE layer on the CPU; c1 decode 49.5 tok/s (code-tutorial prompt, 1,024 forced tokens, MTP on).

## Method

**Base configuration** (every round from R859 on unless its write-up says otherwise): checkpoint `turboderp/GLM-5.3-Flash-exl3` revision `2.05bpw`; cache and `max_seq_len` 262,144 tokens, `cache_mode: 8,8`; `gpu_split` 31, 31 GiB; `chunk_size` 2,048; `max_batch_size` 4; `sysmem_recurrent_cache` 1,024 MB; vision tower on the GPU; `tool_format: glm4_5`; 8 CPU expert worker threads, one per physical core of the Ryzen 7 9800X3D; power limits 600 and 575 W (stock), memory clock offset +4,500 MHz, core clock offset 0. The launcher writes the full `config.yml`, which each arm's raw records include.

**c1 score** (`scripts/glm53_probe.py --phase c1`, scored by `scripts/r860_score.py`): for each content kind, one chat request per run, the prompt prefixed with a per-request nonce, forced with `min_tokens` to 1,024 tokens (html: 2,048), temperature 0, `reasoning_effort: low`. A run's rate is (tokens − 1) divided by the time between its first and last streamed text frame; time to the first token is recorded separately. Per kind, the median over the runs; the score is the mean of the per-kind medians. Kinds: `code` (a Python tutorial with tests), `prose` (an essay on the printing press), `chat` (a two-week trip plan), `html` (a single-file three.js voxel scene), `edit` (return a 1.7k-token Python module with a rename and docstrings added; the module is the probe's own helper files, so this prompt changes when those files change). With the MTP draft on, the rate moves with the text; the write-ups give the draft acceptance per kind.

**Concurrency** (`--phase measure --runs 3`): c streams start together behind a barrier, each one chat request with the code-tutorial prompt forced to 1,024 tokens, temperature 0, the template's default reasoning effort. Each stream's rate is computed as above; a round's figure is the sum of the stream rates over a confirmed common decode window, and the per-stream figure is that sum divided by c. The median of 3 rounds is reported.

**Cold prefill**: one completion request per size with a calibrated random word-list prompt and a fresh prefix, 32 forced tokens; the rate is `usage.prompt_tokens` divided by `usage.prompt_time`, and a sample whose `cached_tokens` is not 0 is rejected.
