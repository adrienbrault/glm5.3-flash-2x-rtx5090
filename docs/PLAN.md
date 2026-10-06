# Expert-offload work plan for c1 decode

Status 2026-10-07. Each item names the source of the idea. Nothing below is served.

## Constraints of this box

- Two RTX 5090 cards, each on PCIe 5.0 ×8 (link width read with `nvidia-smi` on 2026-10-07); the two links serve different layers under a layer split, so one token's copies do not use both links at once.
- Dual-channel DDR5-6000, 2 × 32 GB. CPU expert compute and host-to-device copies read the same memory. dabeljo measured on a comparable dual-channel DDR5 desktop that a CPU tier taking 30 GB/s cut pinned PCIe copies from 57 to 40 GB/s, and estimated its net c1 gain at about 3 % (source 2 below).
- 60 GB of RAM: the 2.05 bpw checkpoint's routed experts are 76.5 GB; about 25 to 35 GB of them can live in RAM next to the server.

## Items, in order

1. **Routing trace** (`docker/route-trace-r1/`, built in round R859). Records the selected expert ids and router weights of every MoE layer, token row and request, off the forward path. It feeds the simulator and sets the gates of the later items.
2. **Cache simulator.** Replays a trace against policies: today's split mode, a static hot set chosen from the prompt, static plus LRU, next-layer prediction with prefetch, and Belady's optimal replacement as the upper bound, under a cost model with explicit PCIe, CPU and GPU parameters. The policies follow strata-glm's tiered cache (source 1).
3. **Next-layer router prediction, measured before any build.** Apply layer l+1's router (after its hyper-connection mix and norm) to the hidden state available at layer l and compare the predicted experts with the experts layer l+1 selects. strata-glm reports top-1/2/3 accuracy of 95/86/77 % (source 1); dabeljo reports recall 0.66 for the top 6 with no training and a +14.5 % decode gain at 117k context (source 2).
4. **GPU expert cache with demand copies.** Fixed VRAM slots holding EXL3 trellis bytes verbatim, a logical-to-slot pointer table consumed by the existing fused MoE kernels, LRU replacement, misses served by a synchronous copy and the same GPU kernel as a hit, so the output stays bitwise identical to a fully resident model. Build only if the simulator projects at least 20 % lower c1 step time than the measured split mode.
5. **Asynchronous prefetch of predicted experts** on a copy stream with ready events, a cap on outstanding prefetch bytes, and demand copies ahead of queued prefetches (source 2 measured a demand miss waiting 0.71 ms behind other copies instead of 0.14 ms). Prefetch about 6 experts per layer: dabeljo's k sweep was flat from 4 to 6 and lower at 8 (source 2).
6. **MTP on or off by measurement.** Verifying a draft token needs the experts of two positions. dabeljo measured the naive verifier slower than plain decode on a copy-bound engine (source 2); R858 measured draft acceptance of 0.80 to 0.90 with CPU expert compute. Round R859 measures both at c1.

Not planned: dropping low-weight experts (strata-glm `--skip-disk`, source 1) changes the output; static hot-expert pinning beyond the LRU (source 2 measured an oracle bound of about +1.5 % on GLM-5.3, whose cache turns over about every ten tokens).

## Sources

1. strata-glm, GLM-5.3-Flash NVFP4 on one RTX 5090 with 128 GB RAM and two NVMe drives, built on the Strata engine: https://github.com/sergqwer/strata-glm, its Linux fork https://github.com/Grigory-Rylov/strata-glm-3090, and Strata https://github.com/Niko1221/Strata (MIT).
2. dabeljo, "GLM-5.3-Flash 320B on one 5090": https://huggingface.co/spaces/dabeljo/glm53-flash-on-one-5090 (MIT), read 2026-10-07.
3. 0xSero's CPU expert tier for GLM-5.3-Flash on an RTX 3090 with an 8-channel EPYC, as described in source 2.
