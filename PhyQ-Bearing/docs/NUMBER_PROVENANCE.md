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

## v10.10 additions (width-matched linear twin, PU speed shift)
Results: branch `results-v10-9` (merged into main via PR), `results/main_tuned/*/phyq_nomed_noquad_wide`,
`results/main_benchmark/pu/*/phyq_nomed_noquad_wide`, `results/cross_load_tuned/cwru/*/phyq_nomed_noquad_wide`,
`results/pu_speed_shift/pu/sp_N15to{N09,N15}_f*/<model>`. One script reproduces every number below:
`python scripts/v10_10_report.py --results results --out docs` (also writes `docs/tab_speedshift.tex`).

| Number(s) in manuscript | Source |
|---|---|
| Row "Linear twin, width-matched" in Table 7 (tab:main), tab:noisetypes, tab:relative, tab:std | metrics.json (task mean of seed means; tab:std: mean/s.d. over seeds) |
| Width-matched twin 1.40 M params, 321.0 M MACs (CWRU) | metrics.json → profile (same profiler as tab:complexity) |
| Ablation paragraph: SDUST +7.8/+12.0/+13.2 pts, recording-level Wilcoxon p<1e-27 (381 recordings) | SDUST test_probs.npz, section 3 of the report |
| Pooled 34 paired runs (incl. NREL): I-6 −10.8 (28 lower, p=3e-5); clean/white/pink −1.0…+1.9 (p>0.09); CWRU within ±1.3 (p≥0.3) | section 2 of the report |
| NREL width-matched twin: 100.0 clean, 99.9 / 93.4 at −6 / −10 dB white, 95.5 pink, 62.0 impulsive (52.2/48.0/85.6 per seed) | results/main_tuned/nrel/expwise/phyq_nomed_noquad_wide |
| Cross-load: wide 90.1 % (worst 74.8), +0.6 vs small twin (p=0.64); PhyQ − wide +2.3 (20/9, p=0.032); by source +3.1/+2.8/+4.6/−1.3; W-6 +1.6 (p=0.26); impulsive 31.0 % | cross-load section of the report; column "Lin. wide" of tab:crossload |
| tab:speedshift and speed-shift paragraph (ClassBD 63.8→36.7, p=0.002; 8/10 single-class runs at 900 rpm; PhyQ −3.4, Lin +6.2, WDCNN +5.0, p≥0.38; white −6…−16 pts) | section 4 of the report |
| Abstract/highlight/conclusion: "8–13 points" (SDUST, equal capacity), 92.4 vs 90.1 % (p=0.032) | as above |

Not evaluated: width-matched twin in the PU speed shift.
NREL note: the first NREL attempts of the width-matched twin ended with exit code 3221225477 (Windows access
violation). `python -X faulthandler` located the crash in pandas/pyarrow while `nrel.py` read the CSV headers
(not in CUDA); v10.10b reads the headers with the csv module, after which all three seeds ran normally on the GPU.

## v10.10d (text/tables only, no new runs)

| Number in the paper | Source | Script |
|---|---|---|
| tab:nll (NLL clean / white -6 / pink -6, 5 models × 6 tasks); "13 of 18", PU clean 1.47 vs 1.33, NREL pink 0.39 vs 0.22 | `metrics.json` → `test.clean.nll`, `test.noise.<fam>.absolute.-6.nll` (main_tuned; PU from main_benchmark) | `scripts/nll_table.py` (`docs/tab_nll.tex`) |
| IMS PhyQ clean 78.3 ± 3.3 → 0 dB 79.2 ± 1.1; per seed +0.5 / −2.2 / +4.3 | `main_tuned/ims/chrono/phyq_nomed/seed0-2` | `scripts/std_table.py` |
| Width-matched twin cost: 1.40 M / 321.0 M MACs / 8.46 ms (CWRU), all tasks in `docs/tab_complexity.tex` | `metrics.json` → `profile` (params, macs, latency_ms_bs1; GTX 1650, bs 1), mean over runs | `scripts/paper_assets.py` (MODELS now includes `phyq_nomed_noquad_wide`) |

## v10.10f (figures only, no new runs)

- Figs. 5, 6, 7 and S1 regenerated with `scripts/pro_figures.py` (width-matched twin `phyq_nomed_noquad_wide` added to MODELS).
  Without the twin, the script reproduces the previous PNGs pixel-identically from the results on main (6cf03ba).
- Fig. 7: twins are plotted as the first two rows. The Holm family is unchanged (the 10 comparisons whose p_Holm values are quoted in the text).
  The width-matched twin is tested with the same paired Wilcoxon test, uncorrected: n.s. in CWRU and PU at −6 and −10 dB.
- IMS window-level 95% grouped-bootstrap CIs (PhyQ − twin, white noise, `_window_ci`):
  - small twin: −6 dB [−3.56, −1.76], −10 dB [−4.85, −3.05];
  - width-matched twin: −6 dB [−3.43, −0.72], −10 dB [−2.06, −0.04].
- SDUST, PhyQ − width-matched twin: −6 dB [10.36, 13.63], −10 dB [11.71, 14.81]. Consistent with the recording-level values in the text.
