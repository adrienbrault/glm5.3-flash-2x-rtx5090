# Third-party code and attribution

This repository is MIT-licensed (see [LICENSE](LICENSE)) for the original work: documentation, launcher, drivers, probes, overlays and measurements. Everything below is derived from, redistributes, or was made possible by someone else's work, and stays under its own licence.

## The model and the checkpoints

| item | origin | licence |
|---|---|---|
| GLM-5.3-Flash (the model) | zai-org — https://huggingface.co/zai-org/GLM-5.3-Flash | MIT |
| `glm53-flash-exl3-2.05bpw-turboderp` | EXL3 quantisation by turboderp — https://huggingface.co/turboderp/GLM-5.3-Flash-exl3, revision `2.05bpw` | the model's licence |
| `glm53-flash-exl3-2.25bpw-r0b0tlab` (downloaded, not yet run) | EXL3 quantisation by r0b0tlab — https://huggingface.co/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121 | the model's licence |

This repository ships no weights.

## The engine and the server

| item | origin | licence | how it is used |
|---|---|---|---|
| ExLlamaV3 | turboderp — https://github.com/turboderp-org/exllamav3 | MIT | the inference engine, including the GLM-5.3 architecture code and the CPU MoE offload (`cpu_moe_split_experts`); `docker/route-trace-r1/` patches five of its Python files and is a derivative under MIT |
| TabbyAPI | theroyallab — https://github.com/theroyallab/tabbyAPI | AGPL-3.0 | the OpenAI-compatible server in the image |

## Ideas and measurements this work builds on

| item | origin | used for |
|---|---|---|
| Tiered expert cache, next-layer router prediction, cache simulator | strata-glm — https://github.com/sergqwer/strata-glm and its Linux fork https://github.com/Grigory-Rylov/strata-glm-3090, built on Strata https://github.com/Niko1221/Strata (MIT) | `docs/PLAN.md` items 2, 3 and 5 |
| Measurements of a CPU expert tier against PCIe copies on dual-channel DDR5, next-layer predictor k sweep, MTP verify cost on a copy-bound engine, demand copies queued behind prefetches | dabeljo — https://huggingface.co/spaces/dabeljo/glm53-flash-on-one-5090 (MIT) | `docs/PLAN.md` constraints and items 3, 5 and 6 |
| CPU expert tier for GLM-5.3-Flash on an 8-channel EPYC | 0xSero, as described by dabeljo | `docs/PLAN.md` constraints |
| Route-trace overlay and simulator design | written for this repository by an OpenAI Codex agent from the ExLlamaV3 sources | `docker/route-trace-r1/` |
