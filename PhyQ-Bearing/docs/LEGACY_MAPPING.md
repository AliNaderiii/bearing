# Legacy scripts → new modules

| legacy (`code/QCNN_for_bearing_diagnosis/`) | new | change |
|---|---|---|
| `utils/Preprocess.py` (CWRU) | `phyq/data/datasets/cwru.py` | exact `X{id}_DE_time` keys (99.mat used X098 keys); BA channel removed (zero-filled for normal files → 100 % shortcut); one-recording chronological split instead of window shuffling |
| `utils/Preprocess_PU.py` | `phyq/data/datasets/pu.py` | struct parsing of `Y['vibration_1']`; bearing-disjoint folds / artificial→real instead of random file split |
| `utils/Preprocess_IMS_Pro.py` | `phyq/data/datasets/ims.py` | data-driven onset (reported) instead of hand-picked file ranges; chronological blocks per stream |
| `utils/Preprocess_NREL.py` | `phyq/data/datasets/nrel.py` | same experiment-wise split; instance normalisation now **after** noise (inside the model) |
| `utils/Preprocess_SDUST.py` | `phyq/data/datasets/sdust.py` | same label tokens; chronological split per file instead of per-class random 60/20/20 window shuffle (overlapping windows leaked) |
| offline features on clean signals, saved in NPZ | `phyq/features/stats.py` | computed on-the-fly from the noisy input inside the model |
| `test_snr_{snr}` arrays precomputed with one noise draw | `phyq/data/noise.py`, `phyq/engine/evaluate.py` | deterministic paired noise per batch; relative **and** absolute SNR; white/pink/impulsive |
| `model.py` (PhyQ-TransNet) | `phyq/models/phyqnet.py` + `layers.py` | every component switchable from YAML (ablation/search); width multiplier |
| `train_hybridmodel.py` (edit + comment/uncomment per dataset) | `scripts/run.py`, `scripts/run_suite.py` | one CLI, YAML suites, resumable, per-run folder with config + metrics + probabilities |
| manual tables | `scripts/make_tables.py` | Markdown + LaTeX + figures + significance tests generated from results |

Workflow change: never edit code to switch dataset/model/protocol – add or override a YAML key.
