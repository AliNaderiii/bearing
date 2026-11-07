# Evaluation protocols

All protocols are defined in `configs/datasets/*.yaml` (`split_variants`) and implemented in
`phyq/data/splits.py`. Every run executes `assert_no_leakage` (time-overlap, unit-overlap,
empty-partition and unseen-label checks) before training.

Common settings (`configs/base.yaml`): window 2048 samples; training stride 512 (overlap only
inside the training segment); validation/test stride 2048 (non-overlapping); 60 epochs max,
AdamW + OneCycle, label smoothing 0.05, random cyclic shift, white-noise augmentation with
SNR ~ U(-10, 10) dB applied **identically to all models**; early stopping on validation
macro-F1 averaged over {clean, 0 dB, -6 dB}; 5 seeds.

## Noise

| setting | definition |
|---|---|
| relative SNR | noise power = window power / 10^(SNR/10) (literature standard; keeps amplitude information) |
| absolute SNR | noise power = mean clean power of the **training set** / 10^(SNR/10) (fixed sensor noise floor; removes the amplitude shortcut) |
| types | white Gaussian, pink (1/f), impulsive (Bernoulli-Gaussian, p = 1 %) |

Test noise is generated from `crc32(seed | snr | type | batch)` → bit-identical across models.

## CWRU (12 kHz drive end, 10 classes)

* files: 0–3 HP, Normal + IR/B/OR(@6) × 0.007/0.014/0.021 in. (`phyq/data/datasets/cwru.py`).
* exact variable keys `X{id:03d}_DE_time`; FE optional; **BA forbidden**.
* `within_xHP`: each recording split chronologically 60/20/20 with 2048-sample gaps.
  Windows per class are capped (train 140, test 12, evenly spaced) because normal files are 2–4× longer.
* `cross_aTob`: train/val = source load (chronological), test = **test segment** of the target load.
* Limitation: each fault class is ONE physical seeded fault; CWRU therefore measures
  within-unit temporal generalisation (within-load) or load shift (cross-load), never
  generalisation to new bearings. This is stated in the paper.

## PU (Paderborn, 3 classes: Healthy / OR / IR, real damage)

* vibration_1 (64 kHz) resampled to 16 kHz with an anti-aliasing polyphase filter.
* 4 operating conditions × 20 files per bearing.
* `fold0..4`: stratified 5-fold over **bearings** (Healthy K001–K005, OR KA04/15/16/22/30,
  IR KI04/14/16/18/21); validation = the next fold's bearings; test = the fold's bearings.
  Results are averaged over folds and seeds.
* `art2real`: Lessmeier et al. protocol – train on artificial damage (+ healthy), validate on
  held-out artificial/healthy bearings, test on real damage bearings.
* v9 – literature-comparable protocols (`configs/experiments/pu_protocols.yaml`, review in
  `docs/PU_REFERENCES_REVIEW_FA.md`). Protocol `recording`: whole recordings are assigned
  chronologically per (bearing, condition) — files 1–12 train, 13–16 val, 17–20 test — so no
  window or recording is shared, but bearings ARE shared ("seen bearings, unseen measurements").
  * `seen_all` (4 conditions), `seen_A..D` (one condition each), `paper_dong7` (7 classes = 7 bearings,
    N15_M01_F10; bearing identification, Dong et al. MSSP 2025).
  * `cc_X2Y`: cross-condition, same bearings; train/val = files 1–16 of source condition X,
    test = files 17–20 of target condition Y (A=N15_M07_F10, B=N09_M07_F10, C=N15_M01_F10,
    D=N15_M07_F04). cc_X2Y has the same train/val split as seen_X, so it is evaluated with the
    seen_X model (`extra_tests`; identity of train/val windows and normalisation is verified,
    otherwise refused) – 4 trainings yield all 12 pairs.
  * Always report these under their own names and next to `fold0..4` (unseen bearings).

## IMS (run-to-failure, 4 classes: Normal / Inner / Roller / Outer)

* streams: test 1 B3 (inner), test 1 B4 (roller), test 2 B1 (outer), test 3 B3 (outer).
* fault onset detected automatically (baseline mean + 3σ of RMS or kurtosis, 5 consecutive files);
  Normal = early-life files of the same channels before onset − margin; faulty = files after onset.
  Onsets are written to `results/_data_report/ims.json` and must be reported in the paper.
* `chrono`: per stream, contiguous blocks of files 60/20/20 with a 5-file gap.
* Limitation: classes are tied to 1–2 physical bearings; IMS measures temporal generalisation.

## NREL (gearbox, binary Healthy / Damaged)

* experiment-wise split (identical to the legacy experiment lists):
  train D3 D4 D5 D7 D8 H1–H6 H8 · val D1 D2 D6 H7 · test D9 D10 H9 H10.
* With only 4 test experiments, report the per-experiment accuracy and a grouped bootstrap CI.

## SDUST (10 classes, 3 channels)

* labels from path tokens (nc / if0.2 … rf0.6), condition from `condition_regex`.
* `chrono`: chronological 60/20/20 per file. Cross-condition variants can be added in
  `configs/datasets/sdust.yaml` once the condition names of the local copy are known
  (`python scripts/sanity_check_data.py sdust` prints them).

## Model selection (validation only)

`scripts/search.py` trains the candidate variants in `configs/experiments/search.yaml` on
development tasks (CWRU cross-load, PU folds 0–1) and ranks them **by validation score only**
(`val_only: true` – the test set is never evaluated). The winner is written to
`results/search/selected.txt`; copy its overrides into `configs/models/phyqnet.yaml` *before*
running the main suites. Report the full ranking table in the supplementary material.

## Reporting

* mean ± std over seeds (PU: over folds × seeds); accuracy and macro-F1; ECE.
* paired Wilcoxon over (task, seed) with Holm correction; exact McNemar on the paired noisy test
  set; grouped bootstrap CI resampling whole recordings.
* complexity: parameters, MACs (thop), batch-1 latency on the stated GPU.
* TTA / seed ensembles, if used, are reported in a separate table.
