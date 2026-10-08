# R924: MXFP8 attention output projection for prefill: no prefill gain, fails the quality gate; not served

Results directory on the box: `/srv/qwen5090/results/2026-10-08-r924-glm53-lowp-165046/`. Driver: `scripts/r924-glm53-lowp.sh` (helpers `scripts/glm_arms.sh`). Raw records: `results/2026-10-08-r924-glm53-lowp/` (`<arm>/c1.jsonl`, `<arm>/dec.jsonl`, `<arm>/vision.jsonl`, `<arm>/prefill-r1/prefill.jsonl` and `prefill-r2/`, `<arm>/server-step.txt`, free VRAM, configs, kernel tests in `K/`, the quality summary `Q/quality.json`, `summary.txt`). The quality gate's reference token ids and engine logs are not published. The overlay (`lowp-tc-r1`, written by an OpenAI Codex agent, a second native extension for sm_120a) is not in this repository.

## What the overlay does

`EXL3_LOWP_MXFP8=1` runs the attention output projection of the prefill path as a block-scaled MXFP8 tensor-core GEMM, with the weights converted at load (11 projections converted, 34 skipped by the VRAM budget). It changes the numerics; decode is unchanged. Off by default.

## Kernel tests

16 cases over the two GPUs (`K/kernels.json`), PASS: maximum relative error at most 3.7e-4 and normalised RMSE at most 2.1e-4 against the reference GEMM at the tested shapes.

## Served arms

One session on 2026-10-08 from 16:50 UTC, image `tabbyapi:cheapswap-r3-agent-r2_mtpfast1_overhead-r2_splitdev2_lowp1`, `scripts/glm-daily.env` otherwise. With the flag on, the served CPU counts (100,104) did not boot (`Insufficient VRAM in split for model and cache`); the ON arm ran at 100,106. Method of `bench/RESULTS.md`; every request ended with `finish_reason: length`; vision 4 of 4 in every arm. Cold prefill: one completion per size with a fresh random word-list prompt, `cached_tokens` 0, the engine's prompt time, two runs per arm.

| arm | flag, CPU counts | cold prefill at about 8,200 prompt tokens, run 1 / run 2 (tok/s) | at about 32,700 (tok/s) | c1 score (tok/s) | distinct c1 / c2 / c4 sum (tok/s) | free VRAM after warmup, GPU0 / GPU1 |
|---|---|---|---|---|---|---|
| OFF1 | 0, 100,104 | 2,018 / 1,996 | 2,031 / 2,029 | 61.6 | 63.3 / 73.3 / 85.0 | 847 / 365 MiB |
| ON | 1, 100,106 | 1,949 / 2,035 | 1,890 / 2,038 | 64.3 | 65.1 / 76.7 / 91.5 | 623 / 363 MiB |
| OFF2 | 0, 100,104 | 2,000 / 1,940 | 2,048 / 2,060 | 65.6 | 63.0 / 75.4 / 87.2 | 847 / 365 MiB |

## Quality gate

Teacher-forced, static placement at 106 routed experts per layer on the CPU on every layer (104 did not fit with the flag on), six kinds (agentic, chat, code, edit-diff, html, prose) of 3,500 tokens each, three boots: off, on, off again. Limits: mean KL 0.001, p99 KL 0.01, top-1 agreement 0.995.

| comparison | mean KL | p99 KL | top-1 agreement |
|---|---|---|---|
| on against off (21,000 tokens) | 0.048 | 0.42 | 0.908 |
| on against off, chat only | 0.131 | 0.63 | 0.823 |
| off against off again (the control) | 0.019 | 0.15 | 0.939 |

FAIL. The control is itself 19 times the mean-KL limit and 40 times the in-process floor R893 measured (mean KL 0.00048, top-1 0.9959): the three arms ran in separate boots, and output differs from boot to boot on this configuration (R916, `docs/GOTCHAS.md`). The flag's mean KL is above the control's on every kind: 1.5 to 1.75 times on five kinds, 7.7 times on chat, 2.6 times overall.

## Reading

- Prefill does not move: with the flag on, 1,890 to 2,038 tok/s against 1,940 to 2,060 without it. The attention output projection is not where prefill time goes on this configuration; this round did not measure where it goes.
- The flag fails the quality gate, and its mean KL is above the boot-to-boot control on every kind.
- ON's decode figures (4-stream sum 91.5 against 85.0 and 87.2) come with 106 instead of 104 CPU experts on GPU1's layers, and the flag does not touch decode; the round has no arm at 106 without the flag to compare them with.
- A quality gate that compares separate boots cannot resolve a change of 0.001 mean KL here. Numerics gates run in one process or with the determinism flags of R916.
