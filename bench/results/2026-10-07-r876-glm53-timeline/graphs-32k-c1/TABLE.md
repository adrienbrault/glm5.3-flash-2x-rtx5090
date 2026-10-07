# Decode timeline

32 steps × 1 rows; wall mean 24.074 ms/step, median 23.999 ms. Wall/row is aggregate throughput cost at c4.

| step | wall ms | GPU0 busy / idle ms | GPU1 busy / idle ms |
|---:|---:|---:|---:|
| 0 | 26.012 | 5.526 / 20.486 | 5.561 / 20.451 |
| 1 | 24.391 | 5.558 / 18.833 | 5.541 / 18.851 |
| 2 | 23.458 | 5.590 / 17.868 | 5.620 / 17.838 |
| 3 | 26.592 | 5.616 / 20.976 | 5.594 / 20.998 |
| 4 | 25.192 | 5.548 / 19.644 | 5.562 / 19.630 |
| 5 | 23.513 | 5.600 / 17.913 | 5.559 / 17.954 |
| 6 | 27.026 | 5.654 / 21.371 | 5.552 / 21.474 |
| 7 | 24.451 | 5.603 / 18.848 | 5.510 / 18.940 |
| 8 | 23.551 | 5.615 / 17.937 | 5.559 / 17.993 |
| 9 | 24.661 | 5.598 / 19.063 | 5.516 / 19.145 |
| 10 | 24.197 | 5.556 / 18.641 | 5.599 / 18.599 |
| 11 | 23.011 | 5.649 / 17.362 | 5.589 / 17.422 |
| 12 | 24.004 | 5.643 / 18.361 | 5.522 / 18.482 |
| 13 | 23.470 | 5.609 / 17.862 | 5.550 / 17.920 |
| 14 | 23.751 | 5.665 / 18.086 | 5.527 / 18.223 |
| 15 | 22.838 | 5.638 / 17.200 | 5.588 / 17.250 |
| 16 | 22.989 | 5.555 / 17.435 | 5.581 / 17.409 |
| 17 | 23.372 | 5.642 / 17.730 | 5.540 / 17.832 |
| 18 | 22.438 | 5.661 / 16.777 | 5.591 / 16.847 |
| 19 | 22.596 | 5.670 / 16.925 | 5.547 / 17.048 |
| 20 | 23.810 | 5.625 / 18.184 | 5.541 / 18.269 |
| 21 | 24.138 | 5.605 / 18.532 | 5.528 / 18.610 |
| 22 | 22.856 | 5.594 / 17.262 | 5.564 / 17.292 |
| 23 | 23.650 | 5.594 / 18.056 | 5.618 / 18.032 |
| 24 | 23.994 | 5.640 / 18.355 | 5.521 / 18.473 |
| 25 | 24.791 | 5.610 / 19.182 | 5.486 / 19.306 |
| 26 | 24.984 | 5.505 / 19.479 | 5.532 / 19.452 |
| 27 | 24.324 | 5.564 / 18.760 | 5.602 / 18.722 |
| 28 | 23.310 | 5.624 / 17.685 | 5.521 / 17.789 |
| 29 | 24.469 | 5.632 / 18.837 | 5.522 / 18.947 |
| 30 | 24.194 | 5.626 / 18.568 | 5.570 / 18.624 |
| 31 | 24.347 | 5.576 / 18.772 | 5.580 / 18.767 |

| category | GPU0 kernel ms/step | GPU1 kernel ms/step |
|---|---:|---:|
| attention_dsa | 0.863 | 0.741 |
| attention_dsa_indexer | 0.198 | 0.171 |
| attention_kda | 1.432 | 1.313 |
| shared_expert | 0.907 | 0.931 |
| router | 0.152 | 0.158 |
| routed_expert | 1.253 | 1.386 |
| routed_and_shared | 0.000 | 0.000 |
| handoff | 0.120 | 0.116 |
| hc_norm | 0.523 | 0.478 |
| dense_mlp | 0.157 | 0.000 |
| device_transfer | 0.000 | 0.000 |
| sampling_head | 0.000 | 0.003 |
| unknown | 0.000 | 0.260 |

See layers.csv, gaps.csv, cpu-ranges.csv, kernels.csv and gpu-events.csv for the detailed tables.

CPU worker spin per layer: **N/A**. handoff-aggregate.csv contains only whole-run 64-job native reports.

- CPU ranges measure enqueue wall time, not completed GPU time.
- Idle means no visible GPU activity; stream memop waits and missing graph children can be invisible.
- Graph launches are host API events, not substitute device durations.
- Kernel sums can exceed busy union under overlap; GPU totals cannot be added into token wall.
- Per-layer native worker spin is unavailable with the required Python-only overlay.
