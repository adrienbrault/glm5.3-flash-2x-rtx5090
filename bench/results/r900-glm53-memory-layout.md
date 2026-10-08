# R900: memory of the served configuration (2026-10-08)

Results directory on the box: `2026-10-08-r900-glm53-memory-layout-122331`; driver `scripts/memory_layout.py` (run read-only against the serving container once the daily had been healthy and the GPU queue empty for three minutes). Raw record: [`2026-10-08-r900-glm53-memory-layout/memory.json`](2026-10-08-r900-glm53-memory-layout/memory.json), printed summary in `summary.txt`, image in `image.txt`.

Configuration: `scripts/glm-daily.env` (image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2`, 104 of 288 routed experts per MoE layer on the CPU, MTP depth 1, `vision_offload`, 262,144-token 8-bit KV pool, 1,024 MB host recurrent-state cache).

Method:

- Weights by category are the checkpoint's tensor bytes from the safetensors headers. Routed experts, the MTP layer's included, are split between GPU and host in the served proportion 184 : 104 of 288.
- The KV pool is computed from the cache layout: 1,120 bytes per token on each of the 11 full-attention layers at 8-bit K and V. That covers latent 512, scales 32, fp16 indexer plane 512 and pooled indexer keys 64.
- The VRAM used is `nvidia-smi` per GPU after warmup. GPU "other" is that total minus the listed items: CUDA contexts, graph pools, scratch, recurrent state, the MTP layer's own cache and allocator reserve.
- Host used is the container's memory cgroup, anonymous plus shared memory; the page cache of the checkpoint reads is left out. Host "other" is that total minus the listed items: runtime, staging buffers and pinned arenas.

| | VRAM (2 × RTX 5090) | host DRAM |
|---|---|---|
| routed experts | 46.63 GiB | 26.36 GiB |
| attention, norms and other weights | 3.65 GiB | |
| shared experts | 0.49 GiB | |
| lm_head | 0.37 GiB | |
| MTP layer (attention, shared expert, norms) | 0.11 GiB | |
| KV pool | 3.01 GiB | |
| embedding table | | 1.18 GiB |
| vision tower | | 0.49 GiB |
| recurrent-state cache | | 1.00 GiB |
| other | 6.84 GiB | 3.85 GiB |
| used | 61.1 GiB (GPU0 30.06, GPU1 31.04) | 32.9 GiB |
| capacity | 63.7 GiB | 60.4 GiB (MemTotal) |
