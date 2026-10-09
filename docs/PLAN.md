# Work plan for c1 and c4 decode

Status 2026-10-08, after R914. Each item names the measurement it starts from and the source of the idea (numbered list at the end). Nothing below is served unless it says so; the served configuration is in `README.md`.

## Where a c1 token goes

R876 (`bench/results/r876-glm53-timeline.md`) timed a decode step on static placement: 11.0 ms of GPU kernels per token over both cards and 12.1 ms of exposed wait on the CPU expert worker, on a prompt that routed 3.63 CPU picks per layer. One extra CPU pick on a layer that already waits costs 0.10 to 0.11 ms. The CPU expert GEMV reads cold experts at 61 to 62 GB/s of a 63 GB/s read ceiling (R874), so the CPU side gets faster only through fewer picks or fewer bytes per pick, and the GPU side through shorter serial kernels.

## Constraints of this box

- Two RTX 5090 cards, each on PCIe 5.0 ×8; one 6.49 MB expert crosses a link in 0.22 ms, 57 GB/s across both cards (R860, R872).
- Ryzen 7 9800X3D, one CCD, dual-channel DDR5-6000, 2 × 32 GB. CPU expert compute and host-to-device copies read the same memory. 8 worker threads on the physical cores; SMT siblings halve the rate (R860, R874).
- 60 GB of RAM: at 96 experts per layer on the CPU the CPU-resident experts take about 26 GB, at 104 about 28 GB.
- Host-to-GPU copies and the CPU expert tier share the DDR5 bandwidth: R897 (`2026-10-08-r897-glm53-cpu-dma-contention-055621`, no model loaded) measured cold expert reads at 62.4 GB/s alone, 53.8 GB/s during a pinned copy loop into one GPU and 11.2 GB/s during copies into both.

## Levers, in order

1. **MTP step cost (served since 2026-10-08, R914).** MTP depth 1 drafts while one request is active (`docker/mtp-fast-r1/`, `docker/mtp-overhead-r2/`). R896 (`2026-10-08-r896-glm53-mtp-cpu-ceiling-054903`, 104 experts per layer on the CPU, before `MTP_FAST`) split the step: a plain step 17.13 ms with 4.50 ms of exposed CPU wait, an MTP step 28.87 ms at 1.715 tokens with 12.39 ms of exposed CPU wait; the GPU side of MTP added 3.85 ms. The second verify row's CPU experts are most of the difference; the two rows already share one CPU job per layer and read each common expert once. Candidates being built (unmeasured): a fused two-row finish of the CPU job, the draft layer's experts on the GPU, a hot set weighted for verify rows.
2. **CPU expert count per GPU (served since 2026-10-08, R915c).** At 104 experts per layer on the CPU GPU0 kept 1,347 MiB unused after warmup and GPU1 365 MiB (R914), because every layer got the same count. `EXL3_MOE_CPU_SPLIT_BY_DEVICE=100,104` (`docker/glm-splitdev-r2/`) gives GPU0's layers 100; GPU0 keeps 847 MiB. R925 measured 98 for GPU0's layers against 100 in one session: no gain at c4 (`bench/results/r925-glm53-splitdev-abab.md`).
3. **Serial-path GPU kernels (an OpenAI Codex round, in progress).** From R876's kernel table: KDA attention 1.3 to 1.4 ms per card per token, routed experts 1.3, DSA attention 0.75 to 0.85, shared expert 0.9, hyper-connection and norm 0.5. Candidates from source 5: one fused KDA kernel for conv, gates, delta rule and gated RMSNorm (about +1 % c1); DSA top-k by radix select (about +1 % if it lands near 5 µs against today's 21.5 µs); a 16 times wider grid for the MLA value expansion (about +0.5 %); RMSNorm folded into the hyper-connection finish kernel (about +0.6 to +0.8 %). Together about 0.5 to 0.6 ms per token, +3 to +4 % c1, estimated from R876 kernel times, not measured. Each must match the current kernels within tolerance at batch 1 to 4 and stay capturable in CUDA graphs.
4. **c4: keep the pooled-DSA index update inside CUDA graphs.** R876: every fourth round at c4, the multi-row pooled-index update falls out of graphs (660 extra launches, 44 scalar-copy synchronisations, about 5.2 ms per round under the profiler). Source 5's capturable design writes index keys and completed 4-token pools with device-side positions and a fixed grid; ours needs a per-row position array. Estimated c4 +5 to +10 %, c1 unchanged; being built.
5. **c4: two micro-batches whose CPU and GPU halves overlap.** R899 (`2026-10-08-r899-glm53-c4-distinct-061545`) measured a 4-stream round at 41.8 ms with distinct prompts on the R882b configuration and 21.6 ms with every CPU expert skipped (a timing-only diagnostic), so about 20 ms per round is exposed CPU time. Overlapping one micro-batch's CPU experts with the other's GPU work needs per-micro-batch CPU job rings and item 4 first. R920 measured a two-lane pipeline (`EXL3_MB_PIPELINE`) on the configuration before the per-device count: greedy output identical, 4-stream sum 78.5 against 86.0 and 88.7 tok/s without it (`bench/results/r920-glm53-mbpipe.md`). Closed as built; reopening starts from a trace of one 4-stream round with the lanes on.
6. **VRAM for more GPU experts.** The DSA indexer keeps a per-token key plane that decode does not read once the 4-token pooled keys exist; a per-slot ring for the per-token rows would free about 1.4 GiB at the 262,144-token pool (sources 10 and 11). R919 (`bench/results/r919-glm53-index-ring.md`) compared the ring with the old indexer on 446,959,710 rows under served traffic, 0 mismatches; with one CPU count on every layer the pool boots at 99 instead of 104. R929c (`bench/results/r929-glm53-ring-splitdev.md`) repeated the comparison on the per-device image (428,445,793 rows, 0 mismatches) and measured the ring at 96,100 against the served 100,104 in one session: c1 score 66.3 against 64.4 and 64.6. Served since 2026-10-09 (`docker/glm-index-ring-r3/`).
7. **Reproducible greedy output (opt-in since R916).** Two runs of the same temperature-0 request diverged, even in one process. R916 and R916b located the first divergence in the CPU expert worker's partial sums; `EXL3_ORDERED_MOE=1;EXL3_FIXED_STREAM_T=8;EXL3_FREEZE_EXCHANGE=1` make output identical within and across boots at a c1 cost of a few percent (`bench/results/r916-glm53-determinism-fix.md`). Exactness and KL gates set them; the served configuration does not.
8. **Recurrent-state checkpoints at message boundaries.** ExLlamaV3 checkpoints KDA state every 32,768 rows plus the prompt end. Placing checkpoints at the start of the shared system block and the last assistant turn (source 5, item 5) would let a new agent session that shares a 10k to 20k-token system and tools prefix skip that prefill. Decode unchanged; unmeasured.
9. **KDA rollback by replay with two state planes (only for MTP depth 2 or more).** Source 5, item 9: writing the new state to the other plane and replaying kept rows on partial acceptance would free about 0.5 GiB at depth 2 and 1.1 GiB at depth 3 over both cards, about 2 to 4 experts per layer back on the GPU, or the difference between a boot and no boot. R883 measured depth 2 below depth 1 on every measure, so this waits until the depth-1 step is cheaper.

## Parked

- **L2 persistence windows on the decode path (R923).** c1 score −17.6 % and 4-stream sum −10.4 % against the two arms without them (`bench/results/r923-glm53-l2.md`).
- **MXFP8 attention output projection for prefill (R924).** No prefill change at 8,200 and 32,700 prompt tokens; quality gate FAIL (`bench/results/r924-glm53-lowp.md`).
- **DFlash2 drafting instead of MTP (R927).** The drafter needs 116 CPU experts per layer; 54.3 and 58.3 tok/s at 1 and 4 streams against 63.2 and 93.5 for the served configuration in the same session (`bench/results/r927-glm53-dflash-window.md`).
- **Next-layer router predictor to pre-warm CPU experts in L3.** Applying layer l+1's router (after its hyper-connection mix and norm, source 1's input definition) to the residual at layer l, offline on the R860b decode traces with 96 experts per layer on the CPU: 229,600 rows, 2.77 actual CPU picks per row.

  | top-k predicted | precision on CPU picks | recall of CPU picks |
  |---|---|---|
  | 1 | 95.4 % | 11.6 % |
  | 2 | 92.2 % | 22.8 % |
  | 4 | 84.3 % | 42.0 % |
  | 6 | 75.3 % | 56.6 % |
  | 8 | 65.9 % | 66.3 % |

  The build gate was 70 % recall at useful precision. The window is also short: the CPU worker is busy with layer l until the serial GPU part of layer l+1 (about 0.13 ms at c1, about 8 MB of DRAM reads, about one expert), so about one expert per layer could be pre-warmed. A cold GEMV takes 281 µs against about 205 µs L3-hot (R874c); with top-1 that is about 0.4 to 0.5 ms per token, +2 to +3 % c1, below the drift of one session. Sources 1, 2 and 7.
- **A GPU expert cache fed over PCIe.** Copies read the same DDR5 as the CPU worker and one expert takes 0.22 ms to cross a link against about 0.10 ms of CPU time per pick (R860, R872, R876); within this RAM budget the cache would hold one LRU slot per layer beside the hot set. Exchange swaps already hold the hot experts on the GPU and follow the locality within one generation that source 9 measured. Reopen with a larger RAM budget. Sources 1, 2 and 9.
- **Fewer experts per layer on the CPU.** Without MTP, 92 boots only at half the batch and fails intermittently, 88 does not boot (R864, R865, R867); with MTP, 96 to 102 do not boot (R911).
- **A 524,288-token KV pool.** Does not boot at 104, 106 or 108 experts per layer on the CPU (R889, `2026-10-08-r889-glm53-pool512k-022206`); it waits for the VRAM of item 6.
- **Eight concurrent requests.** Boots at 100 experts per layer on the CPU without MTP and measured 126.3 tok/s summed at c8 against 105.4 at c4, one prompt in every stream (R888, `2026-10-08-r888-glm53-c8-020151`). The agent traffic this box serves is at most 4 streams.

## Other checkpoints: estimates, not measured

Computed from the measured 6,485,606 bytes per routed expert at 2.05 bpw (R860), the 192 GPU expert slots per layer of the split without MTP (96 on the CPU; 184 with the MTP head at 104), and the measured costs above, on 2026-10-07.

- **r0b0tlab 2.25 bpw (88 GB, downloaded, not run).** About 7.1 MB per expert, so the same VRAM holds about 175 experts per layer and at least 113 go to the CPU (more if the non-expert tensors also grew). CPU-resident experts would take about 34 GB of RAM against about 26 GB now. With more picks on the CPU and about 10 % more bytes per pick, c1 is estimated 10 to 20 % below the served rate. Its accuracy against turboderp's 2.05 bpw is not measured; the two packs come from different quantisers.
- **turboderp 3.05 bpw.** About 9.6 MB per expert, about 116 GB of routed experts. About 130 experts per layer would fit on the GPUs, and the remaining 158 per layer (about 64 GB) exceed the 60 GB of RAM, so it needs a disk tier. Source 6 measured 14.9 tok/s at c1 for a 3.05 bpw pack in that shape (one RTX 3090, 55 GiB RAM cap, four NVMe drives in RAID 0). turboderp's model card charts KL against the original model of 0.194 at 2.05 bpw and 0.068 at 3.05 bpw.

## Closed by measurement

Skipping low-weight CPU expert picks (the lossy fast mode, formerly item 4): R893 (`2026-10-08-r893-glm53-cpu-skip-quality-050418`, teacher-forced over 6 kinds × 3,500 tokens) measured a mean KL of 0.0013 at a 5 % weight threshold, 0.0027 at 8 % and 0.0133 with one CPU pick per token and layer, against 0.00048 between two runs of the exact path, while R886 measured the 5 % and 8 % settings inside the c1 noise; not served. Streaming CPU-tier experts to the GPUs over PCIe: R897's bandwidth figures above. Faster sweeps of the stock dynamic placement (−20 %, R866); reserving a host core as in ExLlamaV3 `fa822cf` (no gain, −31 % when confined to one core, R871; source 8); CPU kernel affinity, spinning, software prefetch and page size (within 1 %, R874c); prompt lookup beside MTP (no gain on code or edits, R864).

## Sources

1. strata-glm, GLM-5.3-Flash NVFP4 on one RTX 5090 with 128 GB RAM and two NVMe drives, built on the Strata engine: https://github.com/sergqwer/strata-glm, its Linux fork https://github.com/Grigory-Rylov/strata-glm-3090, and Strata by Niko1221, https://github.com/Niko1221/Strata (MIT): tiered expert cache, next-layer router prediction and its input definition, low-weight expert skipping and its KL figures.
2. dabeljo, "GLM-5.3-Flash 320B on one 5090": https://huggingface.co/spaces/dabeljo/glm53-flash-on-one-5090 (MIT), read 2026-10-07: CPU tier against PCIe copies on dual-channel DDR5, predictor recall, MTP verify cost on a copy-bound engine.
3. 0xSero's CPU expert tier for GLM-5.3-Flash on an RTX 3090 with an 8-channel EPYC, as described in source 2 and in source 6.
4. turboderp, ExLlamaV3 and the GLM-5.3-Flash EXL3 quantisations: https://github.com/turboderp-org/exllamav3, https://huggingface.co/turboderp/GLM-5.3-Flash-exl3 (KL chart on the model card, read 2026-10-07).
5. TensorFold by ashhart, https://github.com/ashhart/TensorFold (MIT), its Python engine at v0.6.6, read 2026-10-08: the `clear_thinking=false` default, the capturable pooled-indexer update, the fused KDA kernel, radix-select top-k, the MLA expansion grid, RMSNorm in the HC finish, message-boundary recurrent checkpoints, cost-priced MTP depth, KDA rollback by replay.
6. glm53-flash-offload by sybil-solutions (formerly 0xSero), MIT, commit 7f1ee89, read 2026-10-07: single-GPU ExLlamaV3 fork with pinned host experts, a GPU expert cache, a cost-model split between zero-copy reads and an AVX2 CPU tier (CPU job 0.11 + 0.104 ms per expert, the slope R876 measured here), `fast`, `exact` and `nvme` modes.
7. mlx-stream's `lookahead-predictor.md`: layer l+1's gate on layer l's normalised MoE input to issue expert reads early, measured on DeepSeek-V4.1-Flash.
8. ExLlamaV3 `dev` commit `fa822cf`, "CPU MoE: reserve a host core", by turboderp.
9. Project Maya by mw00, https://github.com/mw00/project-maya (MIT): a Strata-derived GLM-5.3-Flash engine with hot-expert placement on 1 or 2 GPUs (2 × V100 32 GB, 45.9 to 50.5 tok/s c1 on its own lossy quantisation), read 2026-10-07.
10. glm53-tensorfold-spark by Jay Leaton, https://github.com/jayleaton/glm53-tensorfold-spark (Apache-2.0), `patches/0065-glm-1m-memory-indexer.patch` and `docs/MEMORY-1M.md` at commit `f79bf9c`, read 2026-10-08: per-token indexer keys and gates kept in a ring buffer, pooled keys in full.
11. MiaAI-Lab, https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold (Apache-2.0), `patches/0038-glm-kv-fp8.patch` at commit `33b50fd`, read 2026-10-08: the same per-slot ring layout for the indexer rows.
