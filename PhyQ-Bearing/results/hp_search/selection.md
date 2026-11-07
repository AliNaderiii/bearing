# Hyper-parameter selection on VALIDATION (test never evaluated)

| candidate | cwru:within_0HP:0 | cwru:within_3HP:0 | ims:chrono:0 | mean | |
|---|---|---|---|---|---|
| phyq_transnet__lr1e-3_ep40 | 80.32 | 81.48 | 74.28 | 78.69 | |
| phyq_transnet__lr2e-3_ep40 | 81.80 | 82.11 | 74.08 | 79.33 | |
| phyq_transnet__lr1e-3_ep80 | 79.37 | 83.94 | 73.60 | 78.97 | |
| phyq_transnet__lr2e-3_ep80 | 83.42 | 84.47 | 72.70 | 80.20 | |
| wdcnn__lr1e-3_ep40 | 85.89 | 90.92 | 76.43 | 84.41 | |
| wdcnn__lr2e-3_ep40 | 87.94 | 92.78 | 77.31 | 86.01 | |
| wdcnn__lr1e-3_ep80 | 89.38 | 92.01 | 77.64 | 86.35 | |
| wdcnn__lr2e-3_ep80 | 92.02 | 93.52 | 77.04 | 87.53 | |
| qcnn__lr1e-3_ep40 | 88.65 | 92.62 | 77.72 | 86.33 | |
| qcnn__lr2e-3_ep40 | 90.92 | 92.75 | 78.61 | 87.43 | |
| qcnn__lr1e-3_ep80 | 92.69 | 94.38 | 77.97 | 88.34 | |
| qcnn__lr2e-3_ep80 | 90.12 | 91.55 | 78.07 | 86.58 | |
| drsn__lr1e-3_ep40 | 89.91 | 93.81 | 77.58 | 87.10 | |
| drsn__lr2e-3_ep40 | 91.22 | 94.79 | 79.06 | 88.36 | |
| drsn__lr1e-3_ep80 | 91.78 | 94.73 | 79.14 | 88.55 | |
| drsn__lr2e-3_ep80 | 90.60 | 93.45 | 76.51 | 86.85 | |
| cnn_transformer__lr1e-3_ep40 | 92.50 | 93.97 | 80.59 | 89.02 | |
| cnn_transformer__lr2e-3_ep40 | 92.39 | 94.86 | 81.08 | 89.44 | |
| cnn_transformer__lr1e-3_ep80 | 93.81 | 94.48 | 78.06 | 88.78 | |
| cnn_transformer__lr2e-3_ep80 | 93.50 | 94.11 | 78.22 | 88.61 | |
| stats_mlp__lr1e-3_ep40 | 71.45 | 78.36 | 63.52 | 71.11 | |
| stats_mlp__lr2e-3_ep40 | 73.04 | 80.00 | 63.67 | 72.23 | |
| stats_mlp__lr1e-3_ep80 | 72.97 | 79.19 | 63.23 | 71.80 | |
| stats_mlp__lr2e-3_ep80 | 73.34 | 80.11 | 63.33 | 72.26 | |
| classbd__lr1e-3_ep40 | 43.43 | 40.46 | 41.17 | 41.68 | |
| classbd__lr1e-3_ep80 | 66.94 | 60.31 | 40.95 | 56.06 | |
| classbd__lr2e-3_ep40 | 64.07 | 41.62 | 37.58 | 47.75 | |
| classbd__lr2e-3_ep80 | 51.81 | 44.00 | 40.23 | 45.35 | |
| classbd__lr1e-3_ep40_nophys | 96.73 | 97.29 | 84.03 | 92.69 | |
| classbd__lr1e-3_ep80_nophys | 96.22 | 98.75 | 81.47 | 92.14 | |
| classbd__lr2e-3_ep40_nophys | 96.43 | 98.74 | 81.96 | 92.38 | |
| classbd__lr2e-3_ep80_nophys | 97.11 | 96.52 | 83.14 | 92.26 | |

## Winners

* phyq_transnet: **lr2e-3_ep80**
* wdcnn: **lr2e-3_ep80**
* qcnn: **lr1e-3_ep80**
* drsn: **lr1e-3_ep80**
* cnn_transformer: **lr2e-3_ep40**
* stats_mlp: **lr2e-3_ep80**
* classbd: **lr1e-3_ep40_nophys**