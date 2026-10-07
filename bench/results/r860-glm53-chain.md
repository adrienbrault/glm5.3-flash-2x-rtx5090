# R860: MTP off ties MTP depth 1 at c1; depth 2 and 3 do not fit; 16 CPU threads halve decode

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r860-glm53-chain-010840/`. Driver: `scripts/r860-glm53-chain.sh`. Raw records: `results/2026-10-07-r860-glm53-chain/` (per arm `c1.jsonl`, `config.yml`, `resolved.json`, `vram-boot.csv`; `h2d/h2d.jsonl`; `summary.txt`).

## Configuration

Image `tabbyapi:r828-prompt-lookup-r3`, base configuration, 104 experts per layer on the CPU, dynamic placement. One boot per arm. c1 score as in `bench/RESULTS.md`, kinds code, prose, chat and html, 2 runs each.

| arm | change | c1 score (tok/s) | code | prose | chat | html |
|---|---|---|---|---|---|---|
| d1 | MTP depth 1 | 45.7 | 48.5 (acc 0.84) | 43.7 (acc 0.61) | 46.2 (acc 0.65) | 44.2 (acc 0.59) |
| off | MTP off | 46.2 | 45.4 | 45.4 | 47.2 | 47.0 |
| t7 | MTP off, 7 CPU threads | 45.0 | 43.2 | 45.0 | 45.6 | 45.9 |
| t16 | MTP off, 16 CPU threads (SMT) | 20.2 | 19.6 | 20.2 | 20.4 | 20.7 |
| pinned | MTP off, pinned CPU arena | 45.6 | 43.8 | 45.8 | 47.5 | 45.3 |

- MTP depth 2, depth 3 and depth 3 with dynamic draft length did not boot: `Insufficient VRAM` at 104 experts per layer on the CPU.
- With MTP on, the verify row routes to other experts than the decode row, so a step reads close to twice the CPU expert bytes; at c1 that cost cancels the gain of 1.6 to 1.8 tokens per step (1 plus the acceptance rate) except on code.
- Two hardware threads per core (16 threads on the 8-core 9800X3D) halve the rate; 7 threads cost 2.6 %.
- The `predict` arm (route trace with CPU swaps disabled) did not boot: the launcher's offload check requires the dynamic split registrations. R860b reran it with placement on.
- The serving boot at the end measured one code run at 38.5 tok/s against 45.4 in the `off` arm with the same configuration; R864 added a repeated control arm to measure boot-to-boot drift.

## Host-to-device copy probe

`h2d/h2d.jsonl`: pinned host buffers, 6,485,606-byte experts (one 2.05 bpw routed expert). One expert takes 28.7 to 28.9 GB/s per PCIe 5.0 x8 link (0.23 ms per expert); 8 experts on both cards at once reach 55.4 GB/s in aggregate. Pinned copies read the same dual-channel DDR5 as the CPU expert worker.
