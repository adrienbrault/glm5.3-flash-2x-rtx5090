# Gotchas

**A cold prefill fails with `KeyError: 'dictionary is empty'` in `recurrent.py` `put()`.** GLM-5.3 has linear-attention (KDA) layers, so ExLlamaV3 builds a recurrent-state checkpoint cache even when TabbyAPI's `sysmem_recurrent_cache` is 0; the first checkpoint stash then finds a zero-size cache and pops from an empty dict. Decode does not stash, so short chats work and the first long prompt fails. Set `sysmem_recurrent_cache` above the size of one checkpoint (the launcher defaults to 1,024 MB). Seen in R858 on the first 8,192-token prefill.

**The chat template has no thinking switch.** `enable_thinking` is not read by GLM-5.3's template; the generation prompt always ends in `<think>`. `reasoning_effort` (`low`, `high`, anything else means `max`) is the only control.

**The fit estimate from tensor sizes is short by several GB.** In R858, 80 experts per layer on the CPU (estimated to fit 31 + 31 GiB) and 12 whole MoE layers on the CPU both filled both cards at module 45 of 50 and failed with `Insufficient VRAM in split for model and cache`; 104 experts per layer on the CPU booted with about 2 GB free on one card. Plan the ladder against a smaller budget than the load budget (`PLAN_SPLIT` in `scripts/r858-glm53-audition.sh`).

**Concurrency adds little aggregate throughput with CPU-side experts.** Four streams route to about four times as many distinct experts per layer, and the CPU side does not batch them: R858 measured 49.5 tok/s at c1 and 68.6 tok/s summed over four streams.
