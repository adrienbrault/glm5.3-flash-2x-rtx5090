# Decode timeline

32 steps × 1 rows; wall mean 28.646 ms/step, median 28.910 ms. Wall/row is aggregate throughput cost at c4.

| step | wall ms | GPU0 busy / idle ms | GPU1 busy / idle ms |
|---:|---:|---:|---:|
| 0 | 28.861 | 5.531 / 23.331 | 5.424 / 23.437 |
| 1 | 26.536 | 5.633 / 20.902 | 5.365 / 21.171 |
| 2 | 29.317 | 5.552 / 23.764 | 5.345 / 23.972 |
| 3 | 27.667 | 5.694 / 21.973 | 5.356 / 22.311 |
| 4 | 29.103 | 5.550 / 23.553 | 5.332 / 23.772 |
| 5 | 22.976 | 5.653 / 17.323 | 5.635 / 17.341 |
| 6 | 26.842 | 5.638 / 21.204 | 5.401 / 21.441 |
| 7 | 28.931 | 5.540 / 23.390 | 5.365 / 23.565 |
| 8 | 27.782 | 5.558 / 22.224 | 5.433 / 22.349 |
| 9 | 27.142 | 5.651 / 21.491 | 5.477 / 21.665 |
| 10 | 29.095 | 5.531 / 23.564 | 5.367 / 23.727 |
| 11 | 30.284 | 5.432 / 24.852 | 5.399 / 24.885 |
| 12 | 27.587 | 5.561 / 22.027 | 5.468 / 22.119 |
| 13 | 30.105 | 5.467 / 24.638 | 5.359 / 24.746 |
| 14 | 29.373 | 5.496 / 23.877 | 5.339 / 24.034 |
| 15 | 29.613 | 5.579 / 24.034 | 5.353 / 24.259 |
| 16 | 28.889 | 5.501 / 23.388 | 5.404 / 23.485 |
| 17 | 27.474 | 5.593 / 21.881 | 5.450 / 22.025 |
| 18 | 28.793 | 5.544 / 23.249 | 5.430 / 23.363 |
| 19 | 30.398 | 5.560 / 24.838 | 5.341 / 25.056 |
| 20 | 28.433 | 5.523 / 22.910 | 5.481 / 22.952 |
| 21 | 28.495 | 5.517 / 22.978 | 5.408 / 23.087 |
| 22 | 29.278 | 5.552 / 23.726 | 5.402 / 23.876 |
| 23 | 30.052 | 5.535 / 24.517 | 5.345 / 24.707 |
| 24 | 28.678 | 5.622 / 23.056 | 5.351 / 23.327 |
| 25 | 29.904 | 5.542 / 24.362 | 5.338 / 24.566 |
| 26 | 28.652 | 5.562 / 23.090 | 5.378 / 23.274 |
| 27 | 29.695 | 5.483 / 24.212 | 5.399 / 24.296 |
| 28 | 28.611 | 5.521 / 23.090 | 5.426 / 23.185 |
| 29 | 29.070 | 5.583 / 23.487 | 5.451 / 23.619 |
| 30 | 29.706 | 5.523 / 24.183 | 5.366 / 24.339 |
| 31 | 29.329 | 5.518 / 23.811 | 5.398 / 23.931 |

| category | GPU0 kernel ms/step | GPU1 kernel ms/step |
|---|---:|---:|
| attention_dsa | 0.854 | 0.733 |
| attention_dsa_indexer | 0.186 | 0.161 |
| attention_kda | 1.429 | 1.311 |
| shared_expert | 0.904 | 0.931 |
| router | 0.153 | 0.158 |
| routed_expert | 1.255 | 1.273 |
| routed_and_shared | 0.000 | 0.000 |
| handoff | 0.092 | 0.094 |
| hc_norm | 0.523 | 0.476 |
| dense_mlp | 0.157 | 0.000 |
| device_transfer | 0.000 | 0.000 |
| sampling_head | 0.000 | 0.003 |
| unknown | 0.000 | 0.259 |

See layers.csv, gaps.csv, cpu-ranges.csv, kernels.csv and gpu-events.csv for the detailed tables.

CPU worker spin per layer: **N/A**. handoff-aggregate.csv contains only whole-run 64-job native reports.

- CPU ranges measure enqueue wall time, not completed GPU time.
- Idle means no visible GPU activity; stream memop waits and missing graph children can be invisible.
- Graph launches are host API events, not substitute device durations.
- Kernel sums can exceed busy union under overlap; GPU totals cannot be added into token wall.
- Per-layer native worker spin is unavailable with the required Python-only overlay.
