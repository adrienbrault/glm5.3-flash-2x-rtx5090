# R874 and R874c: the CPU expert GEMV streams cold experts at 61 to 62 GB/s, 97 % of the read ceiling; kernel variants are flat

Results directories on the box: `/srv/qwen5090/results/2026-10-07-r874-glm53-cpuworker-104952/` (R874, run as `r874b`) and `/srv/qwen5090/results/2026-10-07-r874c-glm53-cpuworker-161052/` (R874c). Drivers: `scripts/r874-glm53-cpuworker.sh`, `scripts/r874c-glm53-cpuworker.sh`. Raw records: `results/2026-10-07-r874-glm53-cpuworker/` and `results/2026-10-07-r874c-glm53-cpuworker/` (`*.summary.txt` tables, one JSON per trial with phase timings, `summary.txt`). The microbenchmark package (`cpuworker-r1`, written by an OpenAI Codex agent) is not in this repository.

## Method

CPU only, under the GPU lock so no decode runs beside it. A synthetic expert arena with the served CPU kernel; 8, 6 or 4 worker threads on physical cores or on SMT siblings; cold data (evicted from L3), L3-hot data, and a pure-read ceiling on the same memory. Each value is the median of 3 trial means.

## R874: thread count and layout, served kernel

| threads | cold GEMV (µs) | hot GEMV (µs) | cold GB/s | read ceiling GB/s |
|---|---|---|---|---|
| 8 physical | 281.9 | 213.3 | 61.4 | 63.1 |
| 6 physical | 290.9 | 282.1 | 59.5 | 63.2 |
| 4 physical | 426.1 | 419.0 | 40.6 | 63.1 |
| 8 on SMT siblings | 342.7 | 341.7 | 50.5 | 63.2 |

The served kernel reads cold experts at 97 % of the single-CCD read ceiling (about 63 GB/s on the 9800X3D). The 47 GB/s seen per job in R860b comes from the job's other phases (prep, barriers) and from small jobs, not from the GEMV.

## R874c: kernel variants, 8 physical threads

| variant | cold GEMV (µs) | cold GB/s |
|---|---|---|
| baseline | 281.2 | 61.5 |
| CPU affinity | 281.2 | 61.5 |
| spin instead of park | 280.9 | 61.6 |
| software prefetch distance 2 / 4 / 8 | 279.0 / 279.3 / 279.8 | 62.0 / 61.9 / 61.8 |
| 4 kB pages | 280.9 | 61.6 |
| transparent hugepages (THP) | 281.1 | 61.5 |

All variants are within 1 %. With 2 ms between jobs (`detailparked`), a parked worker pool costs 518 µs per job against 304 µs with spinning: waking the pool costs about 210 µs. With 100 µs between jobs (`detail100`, the decode cadence) the two are equal (293 and 293 µs). Waking matters for the first job after an idle gap (the first token after prefill, the first request after a pause), not for steady decode.
