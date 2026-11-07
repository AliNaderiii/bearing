# Task-level report (main_benchmark, metric=acc, reference=phyq_transnet)

## Reference model: clean test accuracy per task and seed

| task | 0 | 1 | 2 |
|---|---|---|---|
| pu:art2real | 61.82 | 57.36 | 60.17 |
| pu:fold0 | 57.56 | 72.06 | 58.44 |
| pu:fold1 | 96.98 | 62.40 | 69.17 |
| pu:fold2 | 34.58 | 38.65 | 32.92 |
| pu:fold3 | 65.83 | 61.67 | 65.83 |
| pu:fold4 | 100.00 | 100.00 | 100.00 |

## Share of runs whose best epoch is <= 2 (val on held-out bearings uninformative)

| model | share (median best epoch) |
|---|---|
| cnn_transformer | 6% (median 7) |
| drsn | 33% (median 10) |
| phyq_transnet | 28% (median 8) |
| qcnn | 22% (median 8) |
| stats_mlp | 17% (median 14) |
| wdcnn | 11% (median 12) |

## clean: seed-averaged per task

| task | phyq_transnet | cnn_transformer | drsn | qcnn | stats_mlp | wdcnn |
|---|---|---|---|---|---|---|
| pu:art2real | 59.78 | 46.85 | 44.85 | 58.62 | 40.11 | 60.14 |
| pu:fold0 | 62.69 | 45.94 | 42.21 | 48.90 | 42.52 | 46.90 |
| pu:fold1 | 76.18 | 36.94 | 33.37 | 35.90 | 42.85 | 35.66 |
| pu:fold2 | 35.38 | 17.19 | 23.51 | 31.77 | 21.84 | 31.63 |
| pu:fold3 | 64.44 | 53.23 | 66.56 | 52.26 | 49.90 | 63.68 |
| pu:fold4 | 100.00 | 82.57 | 59.38 | 86.67 | 89.10 | 85.17 |
| MEAN | 66.41 | 47.12 | 44.98 | 52.35 | 47.72 | 53.86 |

Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)

| vs | mean Δ | median Δ | wins/tasks | p | p_holm |
|---|---|---|---|---|---|
| cnn_transformer | +19.29 | +17.09 | 6/6 | 0.0312 | 0.1562 |
| drsn | +21.43 | +17.71 | 5/6 | 0.0625 | 0.1562 |
| qcnn | +14.06 | +12.76 | 6/6 | 0.0312 | 0.1562 |
| stats_mlp | +18.69 | +17.11 | 6/6 | 0.0312 | 0.1562 |
| wdcnn | +12.55 | +9.29 | 5/6 | 0.0625 | 0.1562 |

## white absolute -6 dB: seed-averaged per task

| task | phyq_transnet | cnn_transformer | drsn | qcnn | stats_mlp | wdcnn |
|---|---|---|---|---|---|---|
| pu:art2real | 54.20 | 38.32 | 37.10 | 51.45 | 38.89 | 47.17 |
| pu:fold0 | 58.44 | 25.96 | 34.96 | 40.31 | 47.77 | 36.54 |
| pu:fold1 | 64.55 | 34.44 | 35.17 | 33.26 | 25.10 | 31.22 |
| pu:fold2 | 47.99 | 26.42 | 36.42 | 33.06 | 15.76 | 33.40 |
| pu:fold3 | 65.69 | 57.33 | 58.89 | 46.15 | 36.63 | 54.90 |
| pu:fold4 | 88.75 | 72.53 | 63.47 | 58.82 | 82.12 | 65.97 |
| MEAN | 63.27 | 42.50 | 44.34 | 43.84 | 41.05 | 44.87 |

Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)

| vs | mean Δ | median Δ | wins/tasks | p | p_holm |
|---|---|---|---|---|---|
| cnn_transformer | +20.77 | +18.89 | 6/6 | 0.0312 | 0.1562 |
| drsn | +18.93 | +20.29 | 6/6 | 0.0312 | 0.1562 |
| qcnn | +19.43 | +18.84 | 6/6 | 0.0312 | 0.1562 |
| stats_mlp | +22.22 | +22.18 | 6/6 | 0.0312 | 0.1562 |
| wdcnn | +18.40 | +18.24 | 6/6 | 0.0312 | 0.1562 |

## white absolute -10 dB: seed-averaged per task

| task | phyq_transnet | cnn_transformer | drsn | qcnn | stats_mlp | wdcnn |
|---|---|---|---|---|---|---|
| pu:art2real | 51.01 | 32.35 | 35.47 | 46.16 | 37.25 | 39.93 |
| pu:fold0 | 40.25 | 28.88 | 40.65 | 40.96 | 30.29 | 41.81 |
| pu:fold1 | 46.81 | 31.88 | 34.48 | 32.78 | 33.47 | 32.67 |
| pu:fold2 | 54.93 | 33.23 | 40.87 | 40.59 | 29.86 | 37.01 |
| pu:fold3 | 66.98 | 54.51 | 55.10 | 45.24 | 43.23 | 51.63 |
| pu:fold4 | 75.45 | 63.82 | 53.09 | 50.14 | 51.22 | 55.62 |
| MEAN | 55.90 | 40.78 | 43.28 | 42.65 | 37.55 | 43.12 |

Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)

| vs | mean Δ | median Δ | wins/tasks | p | p_holm |
|---|---|---|---|---|---|
| cnn_transformer | +15.13 | +13.70 | 6/6 | 0.0312 | 0.1562 |
| drsn | +12.63 | +13.19 | 5/6 | 0.0625 | 0.1875 |
| qcnn | +13.26 | +14.18 | 5/6 | 0.0625 | 0.1875 |
| stats_mlp | +18.35 | +18.75 | 6/6 | 0.0312 | 0.1562 |
| wdcnn | +12.79 | +14.74 | 5/6 | 0.0625 | 0.1875 |

## pink absolute 0 dB: seed-averaged per task

| task | phyq_transnet | cnn_transformer | drsn | qcnn | stats_mlp | wdcnn |
|---|---|---|---|---|---|---|
| pu:art2real | 37.14 | 45.30 | 42.72 | 52.03 | 37.82 | 53.62 |
| pu:fold0 | 42.42 | 33.90 | 17.58 | 34.52 | 25.67 | 38.88 |
| pu:fold1 | 47.50 | 34.31 | 37.22 | 34.58 | 33.33 | 36.63 |
| pu:fold2 | 32.05 | 26.98 | 26.01 | 36.53 | 33.06 | 35.62 |
| pu:fold3 | 39.44 | 57.29 | 48.09 | 56.18 | 60.42 | 60.80 |
| pu:fold4 | 58.99 | 74.93 | 35.56 | 74.10 | 33.44 | 78.72 |
| MEAN | 42.92 | 45.45 | 34.53 | 47.99 | 37.29 | 50.71 |

Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)

| vs | mean Δ | median Δ | wins/tasks | p | p_holm |
|---|---|---|---|---|---|
| cnn_transformer | -2.53 | -1.55 | 3/6 | 0.6875 | 1.0000 |
| drsn | +8.39 | +8.16 | 4/6 | 0.2188 | 1.0000 |
| qcnn | -5.07 | -9.68 | 2/6 | 0.3125 | 1.0000 |
| stats_mlp | +5.64 | +6.74 | 3/6 | 0.6875 | 1.0000 |
| wdcnn | -7.79 | -10.03 | 2/6 | 0.2188 | 1.0000 |

## impulsive absolute 0 dB: seed-averaged per task

| task | phyq_transnet | cnn_transformer | drsn | qcnn | stats_mlp | wdcnn |
|---|---|---|---|---|---|---|
| pu:art2real | 39.39 | 44.82 | 42.06 | 42.00 | 27.68 | 55.53 |
| pu:fold0 | 42.77 | 40.00 | 40.19 | 42.75 | 51.40 | 46.17 |
| pu:fold1 | 32.47 | 33.33 | 33.33 | 31.70 | 35.97 | 32.88 |
| pu:fold2 | 34.90 | 33.30 | 36.56 | 33.06 | 31.84 | 25.59 |
| pu:fold3 | 41.63 | 57.78 | 55.59 | 45.45 | 49.83 | 62.26 |
| pu:fold4 | 63.99 | 77.78 | 53.37 | 62.22 | 43.61 | 77.81 |
| MEAN | 42.53 | 47.83 | 43.52 | 42.86 | 40.05 | 50.04 |

Task-level Wilcoxon (exact, n = tasks; min two-sided p with n=6 is 0.031)

| vs | mean Δ | median Δ | wins/tasks | p | p_holm |
|---|---|---|---|---|---|
| cnn_transformer | -5.31 | -3.15 | 2/6 | 0.3125 | 1.0000 |
| drsn | -0.99 | -1.27 | 2/6 | 0.6875 | 1.0000 |
| qcnn | -0.34 | +0.39 | 4/6 | 1.0000 | 1.0000 |
| stats_mlp | +2.47 | -0.23 | 3/6 | 0.8438 | 1.0000 |
| wdcnn | -7.51 | -8.61 | 1/6 | 0.1562 | 0.7812 |
