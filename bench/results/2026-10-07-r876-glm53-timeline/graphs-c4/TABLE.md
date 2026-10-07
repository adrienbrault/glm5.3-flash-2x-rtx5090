# Decode timeline

32 steps × 4 rows; wall mean 60.539 ms/step, median 56.615 ms. Wall/row is aggregate throughput cost at c4.

| step | wall ms | GPU0 busy / idle ms | GPU1 busy / idle ms |
|---:|---:|---:|---:|
| 0 | 56.410 | 7.695 / 48.716 | 7.606 / 48.804 |
| 1 | 50.314 | 8.279 / 42.035 | 8.915 / 41.399 |
| 2 | 56.073 | 7.651 / 48.422 | 7.392 / 48.681 |
| 3 | 68.292 | 8.578 / 59.714 | 9.073 / 59.219 |
| 4 | 52.869 | 7.641 / 45.228 | 7.530 / 45.338 |
| 5 | 56.820 | 8.033 / 48.786 | 8.314 / 48.506 |
| 6 | 55.095 | 8.083 / 47.012 | 8.021 / 47.074 |
| 7 | 73.833 | 8.036 / 65.797 | 7.818 / 66.015 |
| 8 | 56.893 | 7.795 / 49.098 | 7.426 / 49.467 |
| 9 | 58.805 | 7.880 / 50.925 | 7.907 / 50.899 |
| 10 | 58.812 | 7.622 / 51.190 | 7.355 / 51.457 |
| 11 | 76.482 | 8.161 / 68.320 | 7.771 / 68.711 |
| 12 | 55.370 | 7.853 / 47.517 | 7.542 / 47.828 |
| 13 | 58.122 | 7.609 / 50.513 | 7.505 / 50.617 |
| 14 | 57.324 | 8.087 / 49.237 | 7.725 / 49.599 |
| 15 | 76.490 | 8.057 / 68.434 | 7.756 / 68.734 |
| 16 | 53.666 | 7.655 / 46.010 | 7.537 / 46.128 |
| 17 | 47.377 | 7.706 / 39.671 | 7.579 / 39.799 |
| 18 | 52.659 | 7.618 / 45.041 | 7.434 / 45.225 |
| 19 | 81.284 | 8.362 / 72.922 | 8.158 / 73.126 |
| 20 | 54.495 | 7.641 / 46.854 | 7.441 / 47.054 |
| 21 | 53.523 | 7.643 / 45.880 | 7.492 / 46.031 |
| 22 | 60.683 | 7.746 / 52.937 | 7.495 / 53.188 |
| 23 | 75.174 | 8.032 / 67.142 | 7.787 / 67.387 |
| 24 | 58.011 | 8.050 / 49.962 | 8.232 / 49.779 |
| 25 | 56.334 | 7.670 / 48.664 | 7.401 / 48.933 |
| 26 | 52.594 | 7.708 / 44.886 | 7.517 / 45.077 |
| 27 | 81.268 | 8.346 / 72.922 | 8.638 / 72.631 |
| 28 | 55.151 | 8.207 / 46.945 | 8.170 / 46.982 |
| 29 | 54.484 | 7.610 / 46.873 | 7.499 / 46.984 |
| 30 | 56.300 | 7.717 / 48.583 | 7.391 / 48.908 |
| 31 | 76.254 | 7.994 / 68.260 | 7.799 / 68.455 |

| category | GPU0 kernel ms/step | GPU1 kernel ms/step |
|---|---:|---:|
| attention_dsa | 1.513 | 1.291 |
| attention_dsa_indexer | 0.165 | 0.140 |
| attention_kda | 2.097 | 1.928 |
| shared_expert | 1.167 | 1.206 |
| router | 0.270 | 0.279 |
| routed_expert | 1.705 | 1.943 |
| routed_and_shared | 0.000 | 0.000 |
| handoff | 0.159 | 0.162 |
| hc_norm | 0.553 | 0.508 |
| dense_mlp | 0.228 | 0.000 |
| device_transfer | 0.000 | 0.000 |
| sampling_head | 0.000 | 0.011 |
| unknown | 0.000 | 0.280 |

See layers.csv, gaps.csv, cpu-ranges.csv, kernels.csv and gpu-events.csv for the detailed tables.

CPU worker spin per layer: **N/A**. handoff-aggregate.csv contains only whole-run 64-job native reports.

- CPU ranges measure enqueue wall time, not completed GPU time.
- Idle means no visible GPU activity; stream memop waits and missing graph children can be invisible.
- Graph launches are host API events, not substitute device durations.
- Kernel sums can exceed busy union under overlap; GPU totals cannot be added into token wall.
- Per-layer native worker spin is unavailable with the required Python-only overlay.
