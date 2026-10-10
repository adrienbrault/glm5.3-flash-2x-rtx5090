# R959 to R959e: prefill chunk 4,096 with the MTP draft's ring bound passed through: cold prefill 1,973 / 1,977 against 1,443 / 1,449 tok/s at about 32,700 prompt tokens (+37 %), server time per token at 4 streams +4.0 / +4.9 %, index ring exact at 3,072 and 4,096; served

Results directories on the box: `/srv/qwen5090/results/2026-10-10-r959d-glm53-chunk-confirm-HoWhmg/` (R959d, speed, stress) and `/srv/qwen5090/results/2026-10-10-r959e-glm53-chunk-shadow-2ymbQh/` (R959e, exactness). Earlier attempts: `2026-10-09-r959-glm53-prefill-chunk-cEQZ71`, `2026-10-10-r959c-glm53-prefill-chunk-EUKCux`, `2026-10-10-r959c-glm53-prefill-chunk-5ftf1J`. Drivers: `r959d-glm53-chunk-confirm.sh` and `r959e-glm53-chunk-shadow.sh` with `r959d.py` and `r959e.py`, written by an OpenAI Codex agent; they are not in this repository. Raw records: `results/2026-10-10-r959d-glm53-chunk-confirm/` (`<arm>/c1.jsonl`, `c1distinct.jsonl`, `dec.jsonl`, `prefill.jsonl` with the `content` field removed, `house-c1-score.txt`, `timing-windows.jsonl`, `boot-settings.json`, `effective-env.json`, `vram-warmup.csv`, `vram-after-128k.csv` for C3a and C4a; `C4b/stress.jsonl` with `content` removed, `stress-status.json`, `stress-health.txt`; `summary.txt`, `summary.json` with the engine's per-request counters, `c4-free-check.json`, `boot-attempts.jsonl`) and `results/2026-10-10-r959e-glm53-chunk-shadow/` (`summary.txt`, `summary.json`: shadow verdict and coverage per chunk size). The prefill prompts are real text from a private notes file and are not published; their SHA-256 is in each record. Engine and boot logs are not published. Overlay: `docker/draftchunk-r1/`.

## Question and rule

TabbyAPI prefills a prompt in chunks of `chunk_size` tokens (2,048 in every earlier round). R959 asks whether 3,072 (C3) or 4,096 (C4) raises cold prefill and what it costs in decode and VRAM on the served configuration (`scripts/glm-daily.env` before this round: image `..._ring3_livemetrics2`, `INDEX_RING=1`, `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,100`). R959d's rule, registered before the run: a candidate passes when, in both of its pairs with the adjacent 2,048 arm, cold prefill at 32k is at least 1.15 times the control's and the server's time per token at 1 and at 4 streams is at most 1.02 times the control's; C4 also needs the stress test; promotion also needs R959e's exactness result. The unit does not promote.

## Runs

- R959 (2026-10-10 00:44 to 01:12 UTC, `2026-10-09-r959-glm53-prefill-chunk-cEQZ71`): every C3 and C4 boot (4 attempts) stopped with `Generator chunk/verify exceeds model.load ring bound`. TabbyAPI's `load_model_sync` loads the MTP draft model without `max_chunk_size`, so the draft's index rings keep the 2,048 default while the target's follow `chunk_size`. The 2,048 arms measured cold prefill 1,317 / 1,441 / 1,427 and 1,344 / 1,461 / 1,422 tok/s at 8k / 32k / 128k. No raw records published.
- `docker/draftchunk-r1/` passes `max_chunk_size=self.chunk_size` to the draft load. Each later unit builds it on the daily image as `<daily image>_draftchunk1` and runs every arm, the 2,048 controls included, on that image.
- R959c (02:58 to 03:46 UTC, `2026-10-10-r959c-glm53-prefill-chunk-EUKCux`): every chunk size booted; C4 did not boot at 96,100 (`Insufficient VRAM`) and ran at 96,102. Cold prefill 32k / 128k: C3 1,714 / 1,746, C4 1,992 / 2,016 tok/s. An analysis process ran on the box's CPU from 03:08 to 03:27, inside three arms; the controls drifted 7 % at c1 and 12 % at c4, so its decode figures are not used. No raw records published.
- R959c repeat (03:53 to 05:10 UTC, `2026-10-10-r959c-glm53-prefill-chunk-5ftf1J`, no other CPU work on the box): cold prefill 32k / 128k D 1,416 / 1,416, D2 1,440 / 1,420, C3 1,709 / 1,746, C4 (96,102; 96,100 again `Insufficient VRAM`) 1,965 / 2,018 tok/s. Decode c1 score / distinct c4 sum: D 59.7 / 89.1, D2 62.5 / 92.0, C3 63.2 / 95.4, C4 62.9 / 100.0 tok/s. C4 kept 551 / 285 MiB free after warmup and 265 MiB on GPU1 after a 128k prefill. No raw records published.
- R959d (06:19 to 07:15 UTC): this write-up's speed and stress. A first R959d attempt was void: its boot check matched a line the daily logs on every boot (`gpu_embedding declined: insufficient VRAM for full exact mirror`) as a VRAM failure; the line is now excluded before matching.
- R959e (05:50 to 06:12 UTC): exactness.

## R959d arms

One session on 2026-10-10 from 06:19 UTC, order D1 C3a C4a C4b C3b D2, one boot per arm, all on the image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1` (same image ID in every arm), `scripts/glm-daily.env` as served before the round otherwise. C4's split was chosen before the timed arms from the free VRAM of the served daily (827 / 323 MiB free against the 768 / 512 MiB the driver requires for 96,100, `c4-free-check.json`), so C4 ran at 96,102. A CPU guard sampled other processes' CPU use in every arm; no arm was flagged.

| arm | chunk | CPU experts per layer, GPU0's / GPU1's layers | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|---|
| D1, D2 | 2,048 | 96 / 100 | 1,073 / 549 MiB |
| C3a, C3b | 3,072 | 96 / 100 | 835 / 329 MiB |
| C4a, C4b | 4,096 | 96 / 102 | 551 / 287 MiB (C4a after a 131k prefill: 551 / 267) |

## Speed

Cold prefill: engine-timed (`usage.prompt_time`), `cached_tokens` 0, a real-text prompt per request with a per-invocation seed, 32 forced output tokens. c1 score: method of `bench/RESULTS.md` (five kinds, 2 runs each, 1,024 forced tokens, html 2,048, temperature 0, `reasoning_effort: low`). Distinct c1 and c4: `--phase decode --distinct --runs 2`, sum of the stream rates over the common decode window, median of 2 rounds. Server time per token: the engine's per-request counters; at c1 the summed generation time over the summed tokens of the 10 c1-score requests; at c4 the longest stream's generation time over the 4 streams' tokens, median of 2 rounds. Every decode request ended with `finish_reason: length`.

| arm | prefill 32k (prompt tokens) | prefill 128k (prompt tokens) | c1 score | distinct c1 | distinct c4 sum (per stream) | server ms per token, c1 / c4 | ms per MTP step / tokens per step, c1 |
|---|---|---|---|---|---|---|---|
| D1 | 1,443 (32,766) | | 64.7 | 68.7 | 95.7 (23.9) | 15.58 / 10.59 | 27.99 / 1.797 |
| C3a | 1,699 (32,631) | 1,745 (130,455) | 63.2 | 66.8 | 94.4 (23.6) | 15.96 / 10.76 | 28.00 / 1.754 |
| C4a | 1,973 (32,738) | 2,023 (131,051) | 64.6 | 65.7 | 92.7 (23.2) | 15.61 / 11.01 | 27.19 / 1.741 |
| C4b | 1,977 (32,683) | | 64.5 | 67.3 | 92.1 (23.0) | 15.66 / 11.04 | 26.59 / 1.697 |
| C3b | 1,684 (32,765) | | 65.3 | 65.9 | 92.6 (23.1) | 15.51 / 10.97 | 27.21 / 1.754 |
| D2 | 1,449 (32,768) | | 65.6 | 71.8 | 96.7 (24.2) | 15.40 / 10.52 | 26.38 / 1.713 |

c1 by kind, tok/s, median of 2 runs (`house-c1-score.txt`), code / prose / chat / html / edit: D1 62.9 / 68.3 / 70.8 / 66.2 / 55.3; C3a 61.0 / 66.8 / 69.0 / 64.3 / 54.7; C4a 62.9 / 68.3 / 70.5 / 66.0 / 55.1; C4b 62.3 / 65.8 / 73.5 / 65.5 / 55.1; C3b 62.2 / 69.3 / 73.1 / 65.3 / 56.3; D2 62.6 / 69.5 / 73.6 / 66.9 / 55.3. Draft acceptance per c1 run 0.55 to 0.97 over all arms. Time to the first token, median per round: c1 0.42 to 0.49 s, c4 1.55 to 1.68 s in every arm.

Pairs with the adjacent control (`summary.json`, `verdict`):

| pair | prefill 32k ratio | server ms per token ratio, c1 | c4 | rule |
|---|---|---|---|---|
| C3a / D1 | 1.177 | 1.025 | 1.016 | FAIL (c1) |
| C3b / D2 | 1.162 | 1.007 | 1.043 | FAIL (c4) |
| C4a / D1 | 1.368 | 1.002 | 1.040 | FAIL (c4) |
| C4b / D2 | 1.365 | 1.017 | 1.049 | FAIL (c4) |

Outcome by the registered rule: NO PASS for both candidates.

## Stress (C4b's boot, 96,102)

After C4b's timed phases, 4 completion requests with distinct prompts of 66,003 to 66,047 tokens (`cached_tokens` 0) and 256 forced output tokens each, started together so that prefill and decode overlapped: all 4 ended with `finish_reason: length` within 153.3 s, the health endpoint answered `healthy` after them, and the GPUs kept 547 / 261 MiB free (`C4b/stress.jsonl`, `stress-status.json`, `stress-health.txt`; the free-VRAM reading is the operator's from the box's `vram-stress.csv`).

## Exactness (R959e)

R929c's shadow method on the draftchunk image: the ring and the paged indexer side by side, the full fixed served workload (1 to 4 streams, ragged batches, prefill and mixed prefill and decode, MTP draft, verify, accept and rewind, CUDA graphs and eager calls, checkpoint stash and restore, prefix hits, page reuse, vision), all 12 DSA layers, boot log `max_chunk` equal to the chunk size on every ring. The shadow arms ran at 116,124 CPU experts per layer: the shadow holds both indexers in VRAM.

| chunk | indexer calls | rows compared | mismatched bytes |
|---|---|---|---|
| 3,072 | 52,514 | 429,559,085 | 0 |
| 4,096 | 44,415 | 407,295,558 | 0 |

## Reading

- Chunk 4,096 raises cold prefill at about 32,700 prompt tokens by 36.8 and 36.5 % in the two pairs and reaches 2,023 tok/s at 131,051; chunk 3,072 by 17.7 and 16.2 %. The repeat of R959c measured the same steps (+38 to 42 % and +20 to 23 %).
- At 4 streams the server's time per token at chunk 4,096 is 4.0 and 4.9 % above the adjacent control; the distinct c4 sums are 3.0 and 4.7 % lower. At 1 stream the server step is 0.2 and 1.7 % above the control, and the c1 score 0.2 and 1.7 % lower. A likely cause, not measured: chunk 4,096 needs 2 more CPU experts per layer on GPU1's layers and leaves less free VRAM for the exchange swaps.
- The index ring at chunk 4,096 has 4,352 rows per slot, 8,912,896 bytes per DSA layer (boot log, not published; 2,304 rows and 4,718,592 bytes at 2,048), and matched the paged indexer on every compared byte.
- Served from 2026-10-10 about 08:10 UTC by the operator's decision, against the registered rule, for the prefill gain at the measured c4 cost: `scripts/glm-daily.env` with image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3_livemetrics2_draftchunk1`, `CHUNK=4096` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,102`. The README's figures are arm C4a's, with C4b beside them.
