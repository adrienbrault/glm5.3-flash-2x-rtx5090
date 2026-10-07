# Work plan for c1 and c4 decode

Status 2026-10-08. Each item names the measurement it starts from and the source of the idea (numbered list at the end). Nothing below is served unless it says so; the served configuration is in `README.md`.

## Where a c1 token goes

R876 (`bench/results/r876-glm53-timeline.md`) timed a decode step on static placement: 11.0 ms of GPU kernels per token over both cards and 12.1 ms of exposed wait on the CPU expert worker, on a prompt that routed 3.63 CPU picks per layer. One extra CPU pick on a layer that already waits costs 0.10 to 0.11 ms. The CPU expert GEMV reads cold experts at 61 to 62 GB/s of a 63 GB/s read ceiling (R874), so the CPU side gets faster only through fewer picks or fewer bytes per pick, and the GPU side through shorter serial kernels.

## Constraints of this box

- Two RTX 5090 cards, each on PCIe 5.0 ×8; one 6.49 MB expert crosses a link in 0.22 ms, 57 GB/s across both cards (R860, R872).
- Ryzen 7 9800X3D, one CCD, dual-channel DDR5-6000, 2 × 32 GB. CPU expert compute and host-to-device copies read the same memory. 8 worker threads on the physical cores; SMT siblings halve the rate (R860, R874).
- 60 GB of RAM: at 96 experts per layer on the CPU the CPU-resident experts take about 26 GB.

## Levers, in order

1. **MTP on the served configuration: measured, no gain (R883, `bench/results/r883-glm53-swap-mtp.md`).** Depth 1 at 104 experts per layer on the CPU measures 59.8 tok/s at c1 against a 61.1 control (acceptance 0.81 to 0.96 on code and edits) and 92.5 against 109.9 tok/s summed at c4; depth 2 is lower. A verify row routes to other experts, a share of them on the CPU, so a verify step costs close to two plain steps. MTP returns to this list only with a way to make the draft row's CPU experts nearly free, for example the CPU-skip mode applied to verify rows only.
2. **Serial-path GPU kernels (an OpenAI Codex round, in progress).** From R876's kernel table: KDA attention 1.3 to 1.4 ms per card per token, routed experts 1.3, DSA attention 0.75 to 0.85, shared expert 0.9, hyper-connection and norm 0.5. Candidates from source 5: one fused KDA kernel for conv, gates, delta rule and gated RMSNorm (about +1 % c1); DSA top-k by radix select (about +1 % if it lands near 5 µs against today's 21.5 µs); a 16 times wider grid for the MLA value expansion (about +0.5 %); RMSNorm folded into the hyper-connection finish kernel (about +0.6 to +0.8 %). Together about 0.5 to 0.6 ms per token, +3 to +4 % c1, estimated from R876 kernel times, not measured. Each must match the current kernels within tolerance at batch 1 to 4 and stay capturable in CUDA graphs.
3. **c4: keep the pooled-DSA index update inside CUDA graphs.** R876: every fourth round at c4, the multi-row pooled-index update falls out of graphs (660 extra launches, 44 scalar-copy synchronisations, about 5.2 ms per round under the profiler). Source 5's capturable design writes index keys and completed 4-token pools with device-side positions and a fixed grid; ours needs a per-row position array. Estimated c4 +5 to +10 % (from 109.9 tok/s summed), c1 unchanged.
4. **Lossy CPU-skip fast mode (being implemented, off by default).** Skip CPU-resident picks whose router weight is below a share of the layer's total. Priced on the R860b decode routing with the broad hot set's top 192 on the GPU: skipping below 3 % of the layer weight drops 1 % of CPU picks (0.03 % of routed weight, no gain); below 5 %, 6 % of picks (0.47 % of weight, about +2 % c1); below 8 %, 26 % of picks (3.2 % of weight, about +9 % c1). It changes the output on top of a 2.05 bpw quantisation, so it ships as an opt-in mode and the served configuration stays exact; any setting gets a KL measurement against the exact path before use. Source 1 skips low-weight experts in the same way and reports KL 0.018 to 0.044 at the 8 % point; source 6's `fast` mode serves a lossy CPU tier at KL 0.005.
5. **Recurrent-state checkpoints at message boundaries.** ExLlamaV3 checkpoints KDA state every 32,768 rows plus the prompt end. Placing checkpoints at the start of the shared system block and the last assistant turn (source 5, item 5) would let a new agent session that shares a 10k to 20k-token system and tools prefix skip that prefill. Decode unchanged; unmeasured.
6. **KDA rollback by replay with two state planes (only for MTP depth 2 or more).** Source 5, item 9: writing the new state to the other plane and replaying kept rows on partial acceptance would free about 0.5 GiB at depth 2 and 1.1 GiB at depth 3 over both cards, about 2 to 4 experts per layer back on the GPU, or the difference between a boot and no boot. Worth building only if R883 shows depth 2 paying.
7. **`clear_thinking` defaulting to false (in the launcher, R885 pending).** Source 5, item 1; `docs/GOTCHAS.md` describes the cache effect. R885 measures the cached prompt tokens of a follow-up request with and without it.

## Parked

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
- **Fewer experts per layer on the CPU.** 92 boots only at half the batch and fails intermittently, 88 does not boot (R864, R865, R867).

## Other checkpoints: estimates, not measured

Computed from the measured 6,485,606 bytes per routed expert at 2.05 bpw (R860), the 192 GPU expert slots per layer of the served split, and the measured costs above.

- **r0b0tlab 2.25 bpw (88 GB, downloaded, not run).** About 7.1 MB per expert, so the same VRAM holds about 175 experts per layer and at least 113 go to the CPU (more if the non-expert tensors also grew). CPU-resident experts would take about 34 GB of RAM against about 26 GB now. With more picks on the CPU and about 10 % more bytes per pick, c1 is estimated 10 to 20 % below the served rate. Its accuracy against turboderp's 2.05 bpw is not measured; the two packs come from different quantisers.
- **turboderp 3.05 bpw.** About 9.6 MB per expert, about 116 GB of routed experts. About 130 experts per layer would fit on the GPUs, and the remaining 158 per layer (about 64 GB) exceed the 60 GB of RAM, so it needs a disk tier. Source 6 measured 14.9 tok/s at c1 for a 3.05 bpw pack in that shape (one RTX 3090, 55 GiB RAM cap, four NVMe drives in RAID 0). turboderp's model card charts KL against the original model of 0.194 at 2.05 bpw and 0.068 at 3.05 bpw.

## Closed by measurement

Faster sweeps of the stock dynamic placement (−20 %, R866); reserving a host core as in ExLlamaV3 `fa822cf` (no gain, −31 % when confined to one core, R871; source 8); CPU kernel affinity, spinning, software prefetch and page size (within 1 %, R874c); prompt lookup beside MTP (no gain on code or edits, R864).

## Sources

1. strata-glm, GLM-5.3-Flash NVFP4 on one RTX 5090 with 128 GB RAM and two NVMe drives, built on the Strata engine: https://github.com/sergqwer/strata-glm, its Linux fork https://github.com/Grigory-Rylov/strata-glm-3090, and Strata by Niko1221, https://github.com/Niko1221/Strata (MIT): tiered expert cache, next-layer router prediction and its input definition, low-weight expert skipping and its KL figures.
2. dabeljo, "GLM-5.3-Flash 320B on one 5090": https://huggingface.co/spaces/dabeljo/glm53-flash-on-one-5090 (MIT), read 2026-10-07: CPU tier against PCIe copies on dual-channel DDR5, predictor recall, MTP verify cost on a copy-bound engine.
3. 0xSero's CPU expert tier for GLM-5.3-Flash on an RTX 3090 with an 8-channel EPYC, as described in source 2 and in source 6.
4. turboderp, ExLlamaV3 and the GLM-5.3-Flash EXL3 quantisations: https://github.com/turboderp-org/exllamav3, https://huggingface.co/turboderp/GLM-5.3-Flash-exl3 (KL chart on the model card, read 2026-10-07).
5. TensorFold by ashhart, https://github.com/ashhart/TensorFold (MIT), its Python engine at v0.6.6, read 2026-10-08: the `clear_thinking=false` default, the capturable pooled-indexer update, the fused KDA kernel, radix-select top-k, the MLA expansion grid, RMSNorm in the HC finish, message-boundary recurrent checkpoints, cost-priced MTP depth, KDA rollback by replay.
6. glm53-flash-offload by sybil-solutions (formerly 0xSero), MIT, commit 7f1ee89, read 2026-10-07: single-GPU ExLlamaV3 fork with pinned host experts, a GPU expert cache, a cost-model split between zero-copy reads and an AVX2 CPU tier (CPU job 0.11 + 0.104 ms per expert, the slope R876 measured here), `fast`, `exact` and `nvme` modes.
7. mlx-stream's `lookahead-predictor.md`: layer l+1's gate on layer l's normalised MoE input to issue expert reads early, measured on DeepSeek-V4.1-Flash.
8. ExLlamaV3 `dev` commit `fa822cf`, "CPU MoE: reserve a host core", by turboderp.
9. Project Maya by mw00, https://github.com/mw00/project-maya (MIT): a Strata-derived GLM-5.3-Flash engine with hot-expert placement on 1 or 2 GPUs (2 × V100 32 GB, 45.9 to 50.5 tok/s c1 on its own lossy quantisation), read 2026-10-07; and PeasantSmith's 82 % VRAM hit rate within one long generation on a custom Strata build with one V100 (2026-10-06), the reference for single-request expert locality.
