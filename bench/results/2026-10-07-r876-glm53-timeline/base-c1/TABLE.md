# Decode timeline

32 steps × 1 rows; wall mean 25.290 ms/step, median 26.256 ms. Wall/row is aggregate throughput cost at c4.

| step | wall ms | GPU0 busy / idle ms | GPU1 busy / idle ms |
|---:|---:|---:|---:|
| 0 | 27.876 | 5.629 / 22.247 | 5.420 / 22.456 |
| 1 | 19.113 | 5.699 / 13.415 | 5.541 / 13.573 |
| 2 | 27.054 | 5.507 / 21.548 | 5.383 / 21.671 |
| 3 | 20.205 | 5.725 / 14.481 | 5.539 / 14.666 |
| 4 | 25.163 | 5.593 / 19.570 | 5.440 / 19.723 |
| 5 | 18.396 | 5.642 / 12.755 | 5.640 / 12.757 |
| 6 | 21.731 | 5.680 / 16.051 | 5.472 / 16.259 |
| 7 | 25.821 | 5.551 / 20.270 | 5.455 / 20.366 |
| 8 | 25.319 | 5.576 / 19.743 | 5.413 / 19.906 |
| 9 | 21.566 | 5.687 / 15.880 | 5.489 / 16.077 |
| 10 | 27.760 | 5.568 / 22.192 | 5.382 / 22.378 |
| 11 | 27.057 | 5.489 / 21.568 | 5.424 / 21.633 |
| 12 | 24.168 | 5.565 / 18.602 | 5.477 / 18.691 |
| 13 | 27.015 | 5.521 / 21.493 | 5.394 / 21.621 |
| 14 | 26.406 | 5.526 / 20.879 | 5.348 / 21.058 |
| 15 | 26.443 | 5.601 / 20.841 | 5.390 / 21.052 |
| 16 | 26.995 | 5.516 / 21.479 | 5.387 / 21.608 |
| 17 | 23.641 | 5.605 / 18.036 | 5.446 / 18.195 |
| 18 | 26.236 | 5.542 / 20.694 | 5.444 / 20.792 |
| 19 | 26.829 | 5.550 / 21.279 | 5.435 / 21.394 |
| 20 | 25.903 | 5.524 / 20.379 | 5.403 / 20.500 |
| 21 | 26.277 | 5.528 / 20.749 | 5.408 / 20.869 |
| 22 | 27.221 | 5.546 / 21.675 | 5.419 / 21.802 |
| 23 | 26.707 | 5.596 / 21.110 | 5.401 / 21.306 |
| 24 | 25.268 | 5.646 / 19.622 | 5.361 / 19.906 |
| 25 | 26.902 | 5.542 / 21.360 | 5.409 / 21.492 |
| 26 | 24.928 | 5.570 / 19.358 | 5.375 / 19.552 |
| 27 | 26.109 | 5.545 / 20.564 | 5.465 / 20.644 |
| 28 | 25.327 | 5.536 / 19.791 | 5.454 / 19.873 |
| 29 | 26.412 | 5.577 / 20.834 | 5.423 / 20.988 |
| 30 | 26.974 | 5.563 / 21.411 | 5.425 / 21.549 |
| 31 | 26.459 | 5.553 / 20.906 | 5.440 / 21.019 |

| category | GPU0 kernel ms/step | GPU1 kernel ms/step |
|---|---:|---:|
| attention_dsa | 0.362 | 0.311 |
| attention_dsa_indexer | 0.185 | 0.162 |
| attention_kda | 0.517 | 0.475 |
| shared_expert | 0.000 | 0.000 |
| router | 0.152 | 0.158 |
| routed_expert | 0.000 | 0.000 |
| routed_and_shared | 2.154 | 2.202 |
| handoff | 0.128 | 0.131 |
| hc_norm | 0.521 | 0.473 |
| dense_mlp | 0.000 | 0.000 |
| device_transfer | 0.000 | 0.000 |
| sampling_head | 0.000 | 0.003 |
| unknown | 1.557 | 1.519 |

See layers.csv, gaps.csv, cpu-ranges.csv, kernels.csv and gpu-events.csv for the detailed tables.

CPU worker spin per layer: **N/A**. handoff-aggregate.csv contains only whole-run 64-job native reports.

- CPU ranges measure enqueue wall time, not completed GPU time.
- Idle means no visible GPU activity; stream memop waits and missing graph children can be invisible.
- Graph launches are host API events, not substitute device durations.
- Kernel sums can exceed busy union under overlap; GPU totals cannot be added into token wall.
- Per-layer native worker spin is unavailable with the required Python-only overlay.
