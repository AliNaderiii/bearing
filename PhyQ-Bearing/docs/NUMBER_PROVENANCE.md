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
| Ablation around the final model on CWRU (Table tab:ablation, Wilcoxon over 12 runs) | `scripts/review_report.py` (section `ablation`) | `results/review_cwru` -> `results/_paper_assets/review_report.json` |
| CWRU cross-load matrix (Section sec:crossload; clean/impulsive accuracy, per pair and source load, Wilcoxon over 36 runs) | `scripts/review_report.py` (section `crossload`) | `results/cross_load_tuned` -> `results/_paper_assets/review_report.json` |
| Inside the trained network (Section sec:interp; excess prominence, gate alpha, RMS through depth, non-finite windows) | `scripts/interpret.py` (probes), `scripts/review_report.py` (section `interp`) | `results/interp/_analysis/*.npz` -> `results/_paper_assets/review_report.json` |
| Figures 10-11 (stem spectra, excess prominence vs off-frequency controls; gate, RMS, t-SNE with silhouette) | `scripts/pro_figures.py --only interp` | `results/interp/_analysis/*.npz` |

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

## v10.9 additions
| Number(s) in manuscript | Source | Script |
|---|---|---|
| Table A.5 (tab:std), mean ± s.d. over seeds | tables/summary.csv | `scripts/std_table.py` |
| Table A.6 (tab:faulttypes), BPFO/BPFI/BSF prominence; text in sec:interp (7.1/9.2, 6.4/9.6, −2.1/2.5, 9.1/3.8, BSF 0.1 dB) | results/interp/_analysis/*.npz | `scripts/interp_fault_types.py --analysis results/interp/_analysis` |
| Table A.7 (tab:uncert) and calibration paragraph (epistemic share ≤13 %, up to 37/45 %, PU 11–36 %, AUROC 0.82–0.99 / 0.48–0.76) | test_probs.npz of main_tuned + main_benchmark/pu (phyq_nomed, phyq_nomed_noquad, classbd, wdcnn) | `scripts/uncertainty_decomp.py --results results` |
| PU non-monotonicity paragraph (fold 2: 11.7/27.0 % clean; Wilcoxon p=0.19 PhyQ, 0.008 WDCNN, 0.027 ClassBD; 6/10 higher) | results/main_benchmark/pu/fold*/<model>/seed*/metrics.json (white absolute −6 vs clean, 10 fold-seed pairs; WDCNN 15) | inline, see docs/V10_9_NOTES_FA.md |
| Training bearings per fold (9–11 train / 3–5 test) | metrics.json → data.train_groups / test_groups | — |
| Linear twin 0.79 M vs 1.40 M; wide twin 1.400 M at width_mult=1.65 | `tests/test_v10_9.py` | — |
