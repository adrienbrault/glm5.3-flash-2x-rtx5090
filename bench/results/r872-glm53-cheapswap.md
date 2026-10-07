# R872: checkpoint-free expert exchange (cheapswap r2) passes its gates and reaches 60.1 tok/s at c1 from an uninformed start

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r872-glm53-cheapswap-073653/`. Driver: `scripts/r872-glm53-cheapswap.sh`. Raw records: `results/2026-10-07-r872-glm53-cheapswap/` (`T/*.log` gate outputs, per arm `c1.jsonl`, `D/depth.jsonl`, `D/prof-tail.txt`, configs, `S/vision.jsonl`, `summary.txt`).

## Gates

Image `tabbyapi:cheapswap-r2` (revision 2 of `docker/cheapswap-r3/`, without the profile initialisation and the score policy).

- CPU native self-test: pass (K2/K8, 1/4/8 rows, fixed arena pointers).
- GPU self-test, 96 forced swaps, rows 1/4/8: pass on cuda:0 with and without the swizzled layout and on cuda:1, maximum absolute error 2.3e-4.
- Worker ring on both cards, rows 1/4/8/128: pass.
- DMA probe (`T/dma.log`): one expert host-to-device 0.220 ms (28.7 GB/s), device-to-host 0.221 ms, one exchange pair 0.46 ms; 57 GB/s across both cards.

## c1

Base configuration, 96 experts per layer on the CPU, MTP off; c1 score as in `bench/RESULTS.md`, five kinds, 2 runs each. The exchange arms start from the identity placement (no counts file).

| arm | placement | c1 score (tok/s) | code | prose | chat | html | edit |
|---|---|---|---|---|---|---|---|
| SB | static, broad counts | 57.4 | 57.7 | 63.1 | 60.4 | 54.3 | 51.6 |
| XS | exchange: exact cadence every 64 tokens, global budget 64, floor 4, hysteresis 2.0 | 56.9 | 56.4 | 59.2 | 56.9 | 60.9 | 51.2 |
| XF | exchange: served cadence every 16 tokens, per-layer budget 64, floor 2, hysteresis 1.2 | 60.1 | 57.2 | 61.0 | 64.9 | 63.2 | 54.2 |
| SB2 | SB again | 56.9 | 57.1 | 62.3 | 60.2 | 54.9 | 49.9 |

XF is 4.7 % above SB with no counts file; the largest gain is on html (+16 %), the kind the static file fits worst.

## Depth on static placement (arm D, `PROFILE=1`)

| prompt tokens (in order) | prefill (tok/s) | decode (tok/s) |
|---|---|---|
| 4,104, first after boot | 1,808 | 42.4 |
| 4,087 | 1,554 | 60.8 |
| 32,733 | 1,886 | 53.8 |
| 131,102 | 1,907 | 43.9 |

The first request after a boot is slow with static placement too, which never adapts. The word-list filler routes poorly under the broad counts: gate/up GEMV about 580 µs per CPU job, 2.4 times the usual picks.
