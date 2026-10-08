# R919: DSA indexer ring, revision 2: 446,959,710 indexer rows compared under served traffic, 0 mismatches; frees about 5 CPU experts per layer of VRAM; no speed claim

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r919-glm53-index-ring-190234/`. Driver: `scripts/r919-glm53-index-ring.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r919-glm53-index-ring/` (`SH-n108/verdict.json`, the shadow run's verdict and coverage; `P-n99/c1.jsonl`, `dec.jsonl`, `vision.jsonl`; free VRAM of the arms that booted; configs; `summary.txt`). The shadow workload's requests, the engine logs and the per-request traffic log are not published. The overlay (`index-ring-r2`, written by an OpenAI Codex agent from the idea credited in `THIRD_PARTY.md`, native extension rebuilt) is not in this repository.

## What the overlay does

GLM-5.3's DSA attention layers keep a per-token indexer plane (512 bytes per token per layer at 8,8) beside the KV cache. With `INDEX_RING=1` the per-token indexer rows live in a per-slot ring and the pooled keys stay in full, which shrinks the 262,144-token pool's VRAM. `EXL3_DSA_INDEX_RING_SHADOW=1` runs the old indexer beside the ring and compares their outputs row by row.

## Arms

One session on 2026-10-08 from 19:02 UTC, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_ring2`, built on the image served before the per-device split: one CPU expert count on every layer, `GPU_SPLIT=31.8,31`, the served settings otherwise.

- Kernel tests on both GPUs passed. The graph-shadow test's clean probe compared 8,046 rows with 0 mismatches; its final summary then reports 252 mismatches, the bytes the test corrupts on purpose to check that its gate detects them (`PASS graph/oracle/read-only/negative-control`).
- SH-n104 (shadow on, 104 on every layer) did not boot: the shadow keeps both indexers in VRAM.
- SH-n108 (shadow on, 108 on every layer): 50 minutes of a mixed workload, 31 requests, 0 errors. 59,608 indexer calls, 446,959,710 rows compared, 0 mismatches, on all 12 DSA attention layers (the trunk's 11 and the MTP layer's), covering 1 to 4 streams, prefill, MTP draft, verify and rewind, CUDA graphs, recurrent checkpoint stash and restore, prefix hits, page reuse and vision.
- P-n96 and P-n97 (ring on, shadow off) did not boot; P-n98 booted with 1,315 / 291 MiB free after warmup, below the 300 MiB floor on GPU1; P-n99 booted with 1,453 / 429 MiB and was measured.
- P-n104-524k (a 524,288-token pool at 104) did not boot.

## P-n99

Method of `bench/RESULTS.md`; every request ended with `finish_reason: length`; vision 4 of 4. No arm without the ring ran in this session.

| measure | value |
|---|---|
| c1 score | 64.1 tok/s (code 57.9, prose 69.0, chat 72.0, html 65.8, edit 55.6) |
| server step, 10 c1 requests | 26.84 ms per step, 1.702 tokens per step, 15.77 ms per token |
| distinct prompts, sum at c1 / c2 / c4 | 69.9 / 76.7 / 87.4 tok/s |

## Reading

- The ring matched the old indexer on every compared row under served traffic.
- With one count on every layer, the 262,144-token pool boots with the ring at 99 CPU experts per layer and keeps the drivers' VRAM floors; without it, 104 was the lowest single count that booted with MTP (R911, at `GPU_SPLIT=31,31`). The ring is worth about 5 experts per layer. It does not make room for a 524,288-token pool at 104.
- P-n99 measured a c1 score of 64.1 and a 4-stream sum of 87.4 tok/s at 99 on every layer; the served configuration measured 64.7 to 65.3 and 90.0 to 93.5 that day in other sessions (R915c, R925, R927) at its per-device counts. With no same-session control, this round makes no speed claim. The next step rebases the ring onto the per-device image and measures it against the served configuration in one session.
