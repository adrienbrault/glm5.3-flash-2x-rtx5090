# Third-party code and attribution

This repository is MIT-licensed (see [LICENSE](LICENSE)) for the original work: documentation, launcher, drivers, probes and measurements. Everything below is derived from, redistributes, or was made possible by someone else's work, and stays under its own licence. The overlays in `docker/` modify third-party code and carry that code's licence, as listed.

## The model and the checkpoints

| item | origin | licence |
|---|---|---|
| GLM-5.3-Flash (the model) and its chat template | zai-org — https://huggingface.co/zai-org/GLM-5.3-Flash | MIT |
| `glm53-flash-exl3-2.05bpw-turboderp` (served) | EXL3 quantisation by turboderp — https://huggingface.co/turboderp/GLM-5.3-Flash-exl3, revision `2.05bpw`; its model card's KL chart is quoted in `docs/PLAN.md` | the model's licence |
| `glm53-flash-exl3-2.25bpw-r0b0tlab` (downloaded, not run) | EXL3 quantisation by r0b0tlab — https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121 | the model's licence |

This repository ships no weights. `scripts/templates/glm53-keep-thinking.jinja` and `docker/glm-agent-r*/glm53-chat_template.jinja` are the model's chat template; the keep-thinking copy changes one condition so that an undefined `clear_thinking` counts as false.

## The engine and the server

| item | origin | licence | how it is used |
|---|---|---|---|
| ExLlamaV3 (`dev` `5783a93` with the patch stack of https://github.com/adrienbrault/qwen3.8-flash-next-2x-rtx5090, whose served image `tabbyapi:r828-prompt-lookup-r3` is the base of every image here) | turboderp — https://github.com/turboderp-org/exllamav3 | MIT | the inference engine, including the GLM-5.3 architecture code, the CPU MoE offload (`cpu_moe_split_experts`) and the dynamic placement that `docker/cheapswap-r3/` extends; `docker/route-trace-r1/`, `docker/cheapswap-r3/`, `docker/mtp-fast-r1/` and `docker/mtp-overhead-r2/` patch its Python files and are derivatives under MIT. Commit `fa822cf` on its `dev` branch ("CPU MoE: reserve a host core") is the idea R871 tested |
| TabbyAPI (commit `53da791`) | theroyallab — https://github.com/theroyallab/tabbyAPI | AGPL-3.0 | the OpenAI-compatible server in the image; the agent overlays `docker/glm-agent-r2/`, `glm-agent-r4/`, `glm-agent-r4b/` and `glm-agent-r4c/` patch its Python files and stay under AGPL-3.0; `docker/mtp-overhead-r2/` changes its `common/config_models.py` (a draft depth of 0 per batch size), and that part stays under AGPL-3.0 |

## Code adapted from other projects

| item | origin | licence | where |
|---|---|---|---|
| Reasoning-history recovery and whole-schema tool-argument typing for GLM tool calls | glm53-tensorfold-spark by Jay Leaton — https://github.com/jayleaton/glm53-tensorfold-spark, `patches/0620-glm-tool-calling.patch` (including its `glm5_next/cuda/toolfix.py` additions) and `docs/TOOL-CALLING.md` | Apache-2.0; the TensorFold code it modifies is MIT (TensorFold before v0.6.0, as its `NOTICE` states) | `docker/glm-agent-r2/` and the r4 revisions built on it; each keeps `LICENSE.tensorfold`, `NOTICE.tensorfold` and its own `THIRD_PARTY.md`. No TensorFold engine or CUDA code is included |

## Ideas and measurements this work builds on

| item | origin | used for |
|---|---|---|
| `clear_thinking` defaulting to false; capturable pooled-indexer update; fused KDA kernel; radix-select DSA top-k; MLA value-expansion grid; RMSNorm in the hyper-connection finish; recurrent checkpoints at message boundaries; cost-priced MTP depth; KDA rollback by replay | TensorFold by ashhart — https://github.com/ashhart/TensorFold, Python engine v0.6.6, read 2026-10-08 (Apache-2.0 since v0.6.0, commit `e3ac0ea`, 2026-09-30; MIT before) | the launcher's `KEEP_THINKING=1` default and `scripts/templates/glm53-keep-thinking.jinja`; `docs/PLAN.md` items 1 to 3 and 5 to 7 |
| Tiered expert cache, next-layer router prediction and its input definition, cache simulator, low-weight expert skipping with KL figures | strata-glm — https://github.com/sergqwer/strata-glm and its Linux fork https://github.com/Grigory-Rylov/strata-glm-3090, built on Strata by Niko1221 — https://github.com/Niko1221/Strata (MIT) | the routing trace and placement simulator (`scripts/analyze_routes.py`, `scripts/sim_placement.py`), the predictor recall measurement and the lossy skip mode in `docs/PLAN.md` |
| Measurements of a CPU expert tier against PCIe copies on dual-channel DDR5, next-layer predictor recall and k sweep, MTP verify cost on a copy-bound engine, demand copies queued behind prefetches | dabeljo — https://huggingface.co/spaces/dabeljo/glm53-flash-on-one-5090 (MIT) | `docs/PLAN.md` constraints, the parked predictor and GPU-cache items |
| CPU expert tier for GLM-5.3-Flash on an 8-channel EPYC | 0xSero — https://github.com/0xSero, as described by dabeljo and in glm53-flash-offload | `docs/PLAN.md` |
| Cost model splitting expert misses between zero-copy reads and a CPU tier; `fast` (lossy CPU tier), `exact` and `nvme` modes with their measurements | glm53-flash-offload by sybil-solutions (formerly 0xSero) — https://github.com/sybil-solutions/glm53-flash-offload, commit `7f1ee89`, read 2026-10-07 (MIT) | the comparison of R876's per-pick cost with its CPU-job model, the lossy skip mode and the 3.05 bpw estimate in `docs/PLAN.md` |
| Lookahead predictor: layer l+1's gate on layer l's normalised MoE input to issue expert reads early | mlx-stream by David Tai — https://github.com/davidtai/mlx-stream, `docs/lookahead-predictor.md` at commit `ad82f6f`, read 2026-10-08 (MIT) | the parked L3 pre-warm item in `docs/PLAN.md` |
| GLM-5.3-Flash engine with hot-expert placement on 1 or 2 GPUs | Project Maya by mw00 — https://github.com/mw00/project-maya (MIT), read 2026-10-07 | a reference point for hot-expert engines in `docs/PLAN.md` |

## Written by agents

| item | origin | where |
|---|---|---|
| Route-trace overlay and simulator design | an OpenAI Codex agent, from the ExLlamaV3 sources | `docker/route-trace-r1/` |
| Checkpoint-free expert exchange (cheapswap r2 and r3), its simulator and cost model, its gates | an OpenAI Codex agent, from the ExLlamaV3 sources | `docker/cheapswap-r3/` |
| TabbyAPI agent overlays r2, r4 and r4b with their diagnoses and tests | an OpenAI Codex agent, from the TabbyAPI sources and glm53-tensorfold-spark | `docker/glm-agent-r2/`, `docker/glm-agent-r4/`, `docker/glm-agent-r4b/` |
| MTP fast mode (the MTP head's CPU-resident experts join the trunk's CPU worker), its diagnosis and test | a Claude agent, from the ExLlamaV3 sources and the R883 and R886 boot logs | `docker/mtp-fast-r1/` |
| MTP overhead overlay r2: batch cap, GPU draft ids, greedy batched acceptance, cached rewind descriptors, phase profiler, its tests and probes | an OpenAI Codex agent, from the ExLlamaV3 and TabbyAPI sources | `docker/mtp-overhead-r2/` |
| CPU worker microbenchmark (`cpuworker-r1`) and the decode-timeline profiler and its analysis | an OpenAI Codex agent | R874, R874c and R876; the packages are not in this repository |
| Code survey of TensorFold, survey of GLM engines with hot-expert placement | a Claude agent | `docs/PLAN.md` |
