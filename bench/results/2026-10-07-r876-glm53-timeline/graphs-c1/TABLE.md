# Decode timeline

32 steps × 1 rows; wall mean 26.713 ms/step, median 26.688 ms. Wall/row is aggregate throughput cost at c4.

| step | wall ms | GPU0 busy / idle ms | GPU1 busy / idle ms |
|---:|---:|---:|---:|
| 0 | 27.253 | 5.588 / 21.665 | 5.418 / 21.835 |
| 1 | 26.404 | 5.578 / 20.826 | 5.374 / 21.030 |
| 2 | 26.911 | 5.585 / 21.325 | 5.427 / 21.483 |
| 3 | 25.225 | 5.716 / 19.509 | 5.405 / 19.820 |
| 4 | 26.457 | 5.578 / 20.879 | 5.426 / 21.031 |
| 5 | 24.915 | 5.638 / 19.277 | 5.479 / 19.436 |
| 6 | 24.904 | 5.661 / 19.243 | 5.380 / 19.524 |
| 7 | 27.180 | 5.545 / 21.635 | 5.387 / 21.793 |
| 8 | 26.472 | 5.546 / 20.925 | 5.442 / 21.030 |
| 9 | 25.285 | 5.633 / 19.652 | 5.490 / 19.795 |
| 10 | 28.150 | 5.568 / 22.582 | 5.393 / 22.757 |
| 11 | 27.775 | 5.470 / 22.305 | 5.449 / 22.326 |
| 12 | 26.671 | 5.500 / 21.172 | 5.447 / 21.224 |
| 13 | 27.540 | 5.527 / 22.013 | 5.383 / 22.157 |
| 14 | 24.687 | 5.602 / 19.085 | 5.466 / 19.221 |
| 15 | 26.125 | 5.618 / 20.506 | 5.435 / 20.690 |
| 16 | 26.629 | 5.515 / 21.114 | 5.422 / 21.207 |
| 17 | 24.587 | 5.638 / 18.949 | 5.467 / 19.120 |
| 18 | 26.698 | 5.567 / 21.131 | 5.437 / 21.261 |
| 19 | 27.836 | 5.574 / 22.262 | 5.407 / 22.429 |
| 20 | 26.943 | 5.542 / 21.401 | 5.423 / 21.520 |
| 21 | 26.678 | 5.593 / 21.085 | 5.452 / 21.225 |
| 22 | 28.228 | 5.536 / 22.692 | 5.394 / 22.834 |
| 23 | 28.182 | 5.603 / 22.580 | 5.367 / 22.815 |
| 24 | 26.048 | 5.677 / 20.372 | 5.416 / 20.632 |
| 25 | 28.538 | 5.584 / 22.954 | 5.373 / 23.166 |
| 26 | 26.040 | 5.593 / 20.447 | 5.470 / 20.569 |
| 27 | 26.918 | 5.611 / 21.308 | 5.412 / 21.506 |
| 28 | 26.363 | 5.594 / 20.769 | 5.421 / 20.941 |
| 29 | 27.150 | 5.607 / 21.543 | 5.422 / 21.729 |
| 30 | 28.401 | 5.567 / 22.834 | 5.368 / 23.033 |
| 31 | 27.633 | 5.566 / 22.067 | 5.391 / 22.242 |

| category | GPU0 kernel ms/step | GPU1 kernel ms/step |
|---|---:|---:|
| attention_dsa | 0.854 | 0.733 |
| attention_dsa_indexer | 0.187 | 0.161 |
| attention_kda | 1.428 | 1.311 |
| shared_expert | 0.903 | 0.931 |
| router | 0.153 | 0.158 |
| routed_expert | 1.259 | 1.277 |
| routed_and_shared | 0.000 | 0.000 |
| handoff | 0.120 | 0.113 |
| hc_norm | 0.522 | 0.475 |
| dense_mlp | 0.156 | 0.000 |
| device_transfer | 0.000 | 0.000 |
| sampling_head | 0.000 | 0.003 |
| unknown | 0.000 | 0.257 |

See layers.csv, gaps.csv, cpu-ranges.csv, kernels.csv and gpu-events.csv for the detailed tables.

CPU worker spin per layer: **N/A**. handoff-aggregate.csv contains only whole-run 64-job native reports.

- CPU ranges measure enqueue wall time, not completed GPU time.
- Idle means no visible GPU activity; stream memop waits and missing graph children can be invisible.
- Graph launches are host API events, not substitute device durations.
- Kernel sums can exceed busy union under overlap; GPU totals cannot be added into token wall.
- Per-layer native worker spin is unavailable with the required Python-only overlay.
