# R929c: DSA indexer ring on the per-device image: 428,445,793 indexer rows compared under served traffic, 0 mismatches; at 96,100 CPU experts per layer c1 score 66.3 against 64.4 and 64.6 for the configuration served before it; served

Results directory on the box: `/srv/qwen5090/results/2026-10-09-r929-glm53-ring-splitdev-061402/`. Driver: `scripts/r929-glm53-ring-splitdev.sh` (helpers `scripts/glm_arms.sh` and `docker/glm-index-ring-r3/r929_helpers.py`; shadow workload and checker `docker/glm-index-ring-r3/tests/serve_workload.py`, `check_shadow.py`). Raw records: `results/2026-10-09-r929-glm53-ring-splitdev/` (`<arm>/c1.jsonl`, `c1-score.txt`, `dec.jsonl`, `vision.jsonl`, `server-step.txt`, free VRAM after warmup and after the matrix; `aba.json`, `arms.tsv`, `capacity.txt`, `summary.txt`, `daily-extra.txt`; `SH/verdict.json`, the shadow run's verdict and coverage). The engine logs, the shadow workload's requests and traffic log, the kernel test logs and the arms' configs are not published. Overlay: `docker/glm-index-ring-r3/`.

## Question and rule

R919 showed the ring exact under served traffic and worth about 5 CPU experts per layer of VRAM on the image with one CPU count on every layer, with no same-session control. R929 ports the ring onto the per-device image (`docker/glm-index-ring-r3/`), checks it again under served traffic on that image, and measures it at the lowest per-device pair that keeps the drivers' VRAM floors against the served configuration in one session. The driver selects, without a speed threshold, the first of `96,100`, `97,101`, `98,102`, `99,103` that boots with at least 450 MiB free on GPU0 and 300 MiB on GPU1 after warmup; the promotion was decided after the round from the shadow verdict, the VRAM floors, vision, the c1 score and server step, and the distinct-prompt sums against both controls.

## Runs

- Run 1 (2026-10-08): the image built and the kernel tests passed on both GPUs; the first boot failed because `systemd-run` sets no `HOME` (the driver now exports it).
- Run 2: the shadow arm at the served `100,104` did not boot (`Insufficient VRAM`): the shadow keeps the paged indexer and the ring in VRAM. The shadow arm moved to `108,112`.
- Run 3 (2026-10-09): `108,112` warmed with 61 MiB free on GPU1, and the shadow traffic hit a CUDA out-of-memory error after 2 minutes. The driver now walks the shadow arm over `108,112`, `108,116` and `112,120` until the warm headroom is 450 / 300 MiB.
- R929b (`2026-10-09-r929-glm53-ring-splitdev-041651`): kernel tests passed again; `108,112` warmed at 1,703 / 59 MiB and was rejected, and the helper refused `108,116` and `112,120` before boot (its pair list did not include them). The helper now lists them.
- R929c (`2026-10-09-r929-glm53-ring-splitdev-061402`, from 06:14 UTC): this write-up.

## Arms

One session on 2026-10-09 from 06:14 UTC, `scripts/glm-daily.env` as served before R929 (`GPU_SPLIT=31.8,31`, 262,144-token pool and window, MTP depth 1 while one request is active, exchange swaps, agent overlay r2, vision weights in host RAM), with the image, `INDEX_RING` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE` set per arm. Each arm was one boot.

| arm | image | ring | CPU experts per layer, GPU0's / GPU1's layers | free VRAM after warmup, GPU0 / GPU1 | after the matrix |
|---|---|---|---|---|---|
| K | `..._splitdev2_ring3` | kernel tests | | | |
| SH (`108,112`) | `..._splitdev2_ring3` | on, shadow on | 108 / 112 | 1,703 / 61 MiB, rejected | |
| SH | `..._splitdev2_ring3` | on, shadow on | 108 / 116 | 1,701 / 579 MiB | |
| A1 | `..._splitdev2` | off | 100 / 104 | 847 / 365 MiB | 839 / 363 MiB |
| B-96-100 | `..._splitdev2_ring3` | on | 96 / 100 | 1,073 / 549 MiB | 1,061 / 545 MiB |
| A2 | `..._splitdev2` | off | 100 / 104 | 847 / 365 MiB | 833 / 357 MiB |

B's first listed pair passed the floors, so `97,101` to `99,103` did not run (`capacity.txt`).

## Exactness

- K: `test_kernels` (1, 4 and 8 slots, chunk 2,048, 2,304 ring rows), `test_graph_shadow` and `test_dispatch_shadow` passed on both GPUs. The graph-shadow test's clean probe compared 8,046 rows with 0 mismatches; its final summary then reports 252 mismatches, the bytes the test corrupts on purpose to check that its gate detects them.
- SH: 15 minutes of a mixed workload with the ring and the paged indexer side by side: 52,628 indexer calls, 428,445,793 rows compared, 0 mismatched bytes, on all 12 DSA attention layers (layers 3, 7, ..., 43 and the MTP layer 45). Covered: 1 to 4 streams and ragged batches, prefill and mixed prefill and decode, MTP draft, verify, accept and rewind, CUDA graphs and eager calls, recurrent checkpoint stash and restore, prefix hits, page reuse, vision (`SH/verdict.json`).

## Speed

c1 score: method of `bench/RESULTS.md` (five kinds, 2 runs each, 1,024 forced tokens, html 2,048, temperature 0, `reasoning_effort: low`). Server step: `scripts/mtp_steps.py` over the arm's 10 c1 requests. Concurrency: `--phase decode --concurrency 1,2,4 --runs 2 --distinct`, sum of the stream rates, median of 2 rounds. Every request ended with `finish_reason: length`; vision 4 of 4 in every arm.

| arm | c1 score (tok/s) | code / prose / chat / html / edit (tok/s, median of 2) | draft acceptance per run | server: ms per step / tokens per step / ms per token |
|---|---|---|---|---|
| A1 | 64.4 | 60.0 / 68.9 / 72.2 / 66.0 / 54.7 | 0.62 to 0.95 | 27.57 / 1.758 / 15.68 |
| B-96-100 | 66.3 | 63.8 / 70.4 / 73.4 / 66.4 / 57.6 | 0.57 to 0.96 | 26.28 / 1.723 / 15.25 |
| A2 | 64.6 | 61.3 / 68.3 / 72.5 / 65.3 / 55.7 | 0.56 to 0.97 | 26.80 / 1.715 / 15.63 |

| arm | distinct c1 (tok/s) | c2 sum (tok/s) | c4 sum (tok/s) | time to the first token, median per round, c1 / c2 / c4 |
|---|---|---|---|---|
| A1 | 65.6 | 77.8 | 90.2 | 0.45 to 0.47 / 0.74 to 0.78 / 1.69 to 1.72 s |
| B-96-100 | 68.1 | 77.6 | 91.4 | 0.41 to 0.45 / 0.76 to 0.77 / 1.52 to 1.61 s |
| A2 | 62.7 | 77.5 | 85.8 | 0.47 to 0.48 / 0.77 to 0.78 / 1.62 to 1.66 s |

B against the mean of A1 and A2 (`aba.json`): c1 score +2.8 %, server ms per token −2.6 %, distinct c1 +6.2 %, c2 −0.1 %, c4 +3.9 %. A2 against A1: c1 score +0.3 %, distinct c1 −4.3 %, c2 −0.3 %, c4 −4.9 %.

## Reading

- The ring matched the paged indexer on every compared byte under served traffic on the per-device image, as it did on the earlier image in R919.
- The ring frees 134,217,728 bytes of paged indexer rows per DSA layer at the 262,144-token pool and allocates a 4,718,592-byte ring (engine log at boot, not published), about 740 MiB per GPU over its 6 DSA layers. At 96,100 the GPUs hold 4 more routed experts on each MoE layer than at 100,104 and still keep 226 and 184 MiB more free after warmup than the controls.
- The c1 score and the server's time per token move by more than the two controls differ from each other (0.3 % and 0.3 %). The distinct c1 and c4 rounds put B above both controls, but the controls themselves differ by 4 to 5 %, so those two figures carry less weight; c2 is unchanged. Each arm is one boot (`docs/GOTCHAS.md`, c1 spread).
- Served from 2026-10-09 about 07:15 UTC: `scripts/glm-daily.env` with image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_ring3`, `INDEX_RING=1` and `EXL3_MOE_CPU_SPLIT_BY_DEVICE=96,100`. After the restart the daily kept 1,085 / 557 MiB free and answered the vision probe (operator check, not published).
- `docker inspect` on the served container shows `EXL3_DSA_INDEX_RING=0`, the image default; the launcher passes the effective value, and the engine log shows whether the ring is on (`docs/GOTCHAS.md`).
