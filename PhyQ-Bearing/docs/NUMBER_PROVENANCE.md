# Number provenance (advisor comment 3)

Every number in the MSSP manuscript is produced by a script from files that are **in this repository**.
No number is copied from the earlier PDF of the paper.

## 1. Per-run raw results (released)

Each run writes `results/<experiment>/<dataset>/<variant>/<model>/seed<s>/`:

| file | content |
|---|---|
| `config.yaml` | full resolved configuration (data, split, model, hyper-parameters, noise) |
| `metrics.json` | validation/test accuracy, macro-F1 per noise type and SNR, best epoch, wall time, latency |
| `history.json` | training curve |
| `test_probs.npz` | test labels `y`, recording index `rec_idx`, softmax for `clean` and every `{type}__{mode}__{snr}` |
| `train.log` | log |

`results/_data_report/<dataset>.json` (written by `scripts/sanity_check_data.py`) lists, for every split,
the recordings / bearings / experiments in train, validation and test and the window counts per class.

## 2. Manuscript table / figure -> script -> source runs

| Manuscript item | Script | Source |
|---|---|---|
| Table main (white, absolute SNR) | `scripts/paper_assets.py` -> `tab_main_white_abs.tex` | `results/main_tuned` (CWRU, IMS, NREL, SDUST), `results/main_benchmark` (PU fold0-4, art2real) |
| Noise types (white/pink/impulsive) | `scripts/paper_assets.py` -> `tab_noise_types.tex` | same |
| Relative SNR | `scripts/paper_assets.py` -> `tab_relative.tex` | same |
| Complexity (params, MACs, GPU latency) | `scripts/paper_assets.py`, `scripts/profile_models.py` | models + `metrics.json` |
| End-to-end latency bs = 1/8/64, CPU 1 thread, peak memory | `scripts/profile_e2e.py` | `results/_paper_assets/profile_e2e.json` |
| Statistics (McNemar, grouped bootstrap, Holm) | `scripts/mcnemar_recheck.py`, `scripts/bootstrap_recheck.py`, `scripts/task_level_report.py` | `test_probs.npz` |
| Calibration (ECE, Brier, NLL, non-finite share, reliability) | `scripts/calibration_report.py` | `test_probs.npz` |
| Remedies E1-E3 | `scripts/robust_report.py`, `scripts/robust_table.py` | `results/robust_*` |
| Ablation | `scripts/val_summary.py`, `scripts/make_tables.py` | `results/ablation`, `results/review_cwru` |
| Units per partition | `scripts/units_table.py` | `results/_data_report` |
| Simulation, prominence, H1-H3 | `scripts/sim_mechanism.py`, `scripts/sim_figures.py`, `scripts/sim_saturation.py` | `results/sim_mechanism` |
| Detector laws (incl. Hilbert^2, TKEO, clipping) | `scripts/sim_detector_laws.py` | `results/sim_mechanism/detector_laws.json` |
| Interpretability (filter responses, envelope spectra at BPFO/BPFI/BSF) | `scripts/interpret.py` | `results/interp` |

An independent audit recomputed all 1082 stored runs from `test_probs.npz` and found 0 mismatches with
`metrics.json`.

## 3. Numbers of the earlier PDF

The earlier PDF contained values that are mutually inconsistent, e.g. SDUST at -10 dB (89.15 vs 99.72),
WDCNN (87.21 vs 82.15) and DRSN (45.60 vs 88.30). The original code saved no per-run result files, so
these values cannot be traced to a run, a split or a noise definition. They are **not used** anywhere in
the revised manuscript. All of them are superseded by the tables above, which are produced under one
protocol: leakage-controlled splits, train-fitted normalisation, the same deterministic test noise for
every model, per-model hyper-parameters chosen on validation data, and three seeds (five bearing-disjoint
folds x two seeds on PU).

## 4. What is not used

- No test-time augmentation: `eval.tta_shifts = 0` everywhere. The phrase "Bayesian TTA" of the earlier
  draft does not describe any computation in this code and was removed.
- No ensembling, no temperature scaling, no test-set model selection.
