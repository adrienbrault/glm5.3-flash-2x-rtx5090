# MTP fast overlay, revision 1

`tabbyapi:cheapswap-r3-agent-r2_mtpfast1` is `tabbyapi:cheapswap-r3-agent-r2` (`../glm-agent-r2-on-cheapswap-r3/`) with `mtp-fast.patch` applied to ExLlamaV3's `modules/block_sparse_mlp_cpu.py`. It is the base of `../mtp-overhead-r2/`, whose image is served.

Without the flag, the MTP head's MoE layer (module 45), which has CPU-resident experts like the trunk's layers, starts a second CPU worker whose thread pool is pinned to the same 8 physical cores as the trunk's worker, and every forward (draft, verify, accept-prefill) re-arms the spin-wait of both pools. With `EXL3_MTP_FAST=1` that layer joins the trunk's worker before it starts: one worker serves 43 layers. Expert kernels, thread count and work partition are unchanged.

`apply.py` checks the file's SHA256 before and after the patch (`--fuzz=0`), compiles it and runs `test_join_host.py`. A landed boot logs one `CPU MoE worker started: 43 layers` line and no `1 layers` line.

R892 (2026-10-08, results directory `2026-10-08-r892-glm53-mtp-fast-041528`, MTP depth 1, 104 routed experts per layer on the CPU, dynamic exchange placement, server step time over 15 c1 requests): 27.74 and 27.06 ms per MTP step with the flag against 30.02 and 28.81 ms without it, two boots each; c1 score 62.6 and 63.9 against 58.9 and 60.2 tok/s.

```sh
sha256sum -c SHA256SUMS
docker build -t tabbyapi:cheapswap-r3-agent-r2_mtpfast1 .
```
