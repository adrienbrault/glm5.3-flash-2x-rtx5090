# R883: MTP depth 1 and 2 on the exchange-swap configuration: no gain at c1, −16 % summed at c4

Results directory on the box: `/srv/qwen5090/results/2026-10-07-r883-glm53-swap-mtp-221219/`. Driver: `scripts/r883-glm53-swap-mtp.sh`. Raw records: `results/2026-10-07-r883-glm53-swap-mtp/` (`<arm>/c1.jsonl`, `<arm>/measure.jsonl` for the MTP arms, configs, `summary.txt`).

## Arms

All arms use image `tabbyapi:cheapswap-r3-agent-r2` with the served exchange-swap settings (`README.md`), in one session, in this order.

| arm | MTP draft depth | routed experts per layer on the CPU | why that count |
|---|---|---|---|
| C | off | 96 | the served configuration, same-session control |
| M1 | 1 | 104 | depth 1 does not fit the VRAM at 96 with the default chunk and batch (R867) |
| M2 | 2 | 112 | depth 2 needs the VRAM of 16 more GPU experts per layer |

## c1 by content kind

Method of `bench/RESULTS.md`: one chat request per run forced with `min_tokens` to 1,024 tokens (html 2,048), temperature 0, `reasoning_effort: low`, 2 runs; every run ended with `finish_reason: length`.

| kind | C runs (tok/s) | M1 runs (tok/s), draft acceptance | M2 runs (tok/s), draft acceptance |
|---|---|---|---|
| code | 57.2, 60.6 | 52.5, 64.6 (0.87, 0.81) | 57.1, 48.6 (0.68, 0.67) |
| prose | 64.5, 64.5 | 58.2, 60.0 (0.66, 0.63) | 55.7, 56.3 (0.43, 0.46) |
| chat | 65.6, 65.9 | 68.5, 64.9 (0.67, 0.67) | 61.3, 61.4 (0.49, 0.48) |
| html | 64.1, 65.4 | 61.4, 63.0 (0.61, 0.59) | 48.7, 52.1 (0.35, 0.39) |
| edit | 50.9, 52.7 | 49.9, 55.0 (0.96, 0.87) | 49.7, 44.8 (0.81, 0.81) |
| c1 score (mean of the kind medians) | 61.1 | 59.8 | 53.6 |

## Concurrency and prefill (MTP arms, one run each)

`glm53_probe.py --phase measure --runs 1`, the method of R882b. The control's own figures are R882b's on the same image and settings: 59.9, 82.2 and 109.9 tok/s summed at c1, c2 and c4; prefill 1,999 and 2,146 tok/s.

| arm | c1 (tok/s) | c2, sum of the stream rates (tok/s) | c4, sum of the stream rates (tok/s) | cold prefill 8,192 / 32,768 prompt tokens (tok/s) |
|---|---|---|---|---|
| M1 | 61.2 | 73.7 | 92.5 | 1,932 / 2,012 |
| M2 | 45.0 | 62.8 | 71.7 | 1,758 / 1,963 |

## Reading

- Depth 1 ties the control at c1 (59.8 against 61.1, inside the run-to-run spread of code and edit) at 0.81 to 0.96 acceptance on code and edits. A verify step costs close to two plain steps here: the draft row routes to other experts, a share of them on the CPU, and 8 more experts per layer run on the CPU to make room for the draft.
- Depth 1 lowers the sum over four streams from 109.9 to 92.5 tok/s (−16 %): the draft rows add CPU expert work to every batch step.
- Depth 2 is lower on every measure.
- The driver's gate (c1 at least 1.08 times the control, at least 104 tok/s summed at c4) selected no MTP arm; the box serves arm C's configuration after this round.
- Each MTP boot took about 13 minutes against about 5 for the control; the cause is not measured.
