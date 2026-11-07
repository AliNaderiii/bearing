# PhyQ-Bearing

Leakage-safe, config-driven benchmark of **PhyQ-TransNet** (the author's model, `configs/models/phyq_paper.yaml`,
ported 1:1 from the original `QDCA_Hybrid_Pro`; 1,404,715 parameters). `phyqnet` is an optional re-designed variant
(not used by run_all)
(physics-informed quadratic network) for noise-robust rolling-bearing fault diagnosis
on **CWRU, Paderborn (PU), IMS, NREL and SDUST**.

One codebase, one command per experiment suite, no commenting/uncommenting of code.

```
configs/
  base.yaml                 global defaults (training, validation-only selection, evaluation grid)
  paths.example.yaml        -> copy to paths.yaml and point to your local datasets
  datasets/*.yaml           dataset adapters + named split variants (protocols)
  models/*.yaml             PhyQNet (+ candidate variants) and 5 re-implemented baselines
  experiments/*.yaml        suites: main_benchmark, cross_load, ablation, noise_types, search, smoke
phyq/
  data/        recording abstraction, signal cache, splits (+ leakage assertions), windows, noise
  data/datasets/  cwru.py pu.py ims.py nrel.py sdust.py synthetic.py
  features/    on-the-fly statistical features (computed from the noisy input)
  models/      layers (quadratic conv, Meta-ACON, SE, soft-shrinkage, envelope spectrum, MED), PhyQNet, baselines
  engine/      trainer (validation-only early stopping), deterministic paired evaluation, metrics
  analysis/    aggregation, LaTeX/Markdown tables, noise curves, Wilcoxon/McNemar/bootstrap
scripts/       run.py, run_suite.py, search.py, make_tables.py, sanity_check_data.py, profile_models.py,
               prepare_cache.py, run_all.ps1, run_all.sh
tests/         42 unit tests incl. mock files in the on-disk format of every dataset
docs/          PROTOCOLS.md, LEGACY_MAPPING.md, CHANGES_vs_paper.md, RUN_GUIDE_FA.md
```

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.yaml                       # edit the paths
python -m pytest -q tests                                              # ~20 s
python scripts/check_env.py                                            # packages, GPU, data paths
python scripts/find_cwru.py "<data folder>" --write                    # locate CWRU files (read-only)
python scripts/run_suite.py configs/experiments/smoke.yaml             # synthetic end-to-end (~1 min)
python scripts/sanity_check_data.py cwru pu ims nrel sdust             # verify your local data
```

Full pipeline (resumable – re-running skips finished runs):

```powershell
.\scripts\run_all.ps1          # Windows
./scripts/run_all.sh           # Linux
```

Single run / overrides (any config key can be overridden as `key=value`):

```bash
python scripts/run.py --dataset cwru --variant cross_0to3 --model phyq_paper --seed 0 train.epochs=40
python scripts/run_suite.py configs/experiments/main_benchmark.yaml --datasets pu --models phyq_transnet wdcnn --seeds 0 1
python scripts/make_tables.py main_benchmark --reference phyq_transnet
```

Results: `results/<experiment>/<dataset>/<variant>/<model>/seed<k>/{metrics.json, test_probs.npz, history.json,
config.yaml, train.log}`; reports in `results/<experiment>/_report/`.

## Guarantees enforced by code (and unit-tested)

* windows are cut **inside** time segments / physical units of a single partition
  (`assert_no_leakage` runs before every experiment);
* PU and NREL splits are **unit-disjoint** (bearing-wise / experiment-wise);
* CWRU uses exact `X{id}_DE_time` keys; the BA channel is refused (it is missing in normal files);
* scaling constants and the absolute-SNR noise floor are estimated on **training** data only;
* statistical features are computed on-the-fly from the **noisy** input (no clean-feature leakage);
* test noise is deterministic and identical for all models (paired tests); validation uses a different noise seed;
* early stopping and architecture choice use **validation only** (`scripts/search.py` never evaluates the test set).

See `docs/PROTOCOLS.md` for the exact protocols and `docs/LEGACY_MAPPING.md` for the
differences w.r.t. the previous scripts.

## Data safety
Dataset folders are **read-only**: loaders only call `loadmat` / `whosmat` / `read_csv`; all writes go to
`paths.cache` and `results/`. `prepare()` aborts with `Safety stop` if either lies inside a dataset folder,
and `tests/test_user_layout.py` asserts that names/sizes/mtimes of all data files are unchanged after a run.
Unreadable files are skipped and listed in `summary.json -> data.skipped_recordings` (abort if >20%).

## Speed test / time estimate
```
python scripts/run_suite.py configs/experiments/timing.yaml   # 2 epochs, every model x dataset
python scripts/estimate_time.py [--seeds 0]                   # hours per run_all suite
python scripts/inspect_file.py <file> --fs 25600              # what the loader reads from one file
```

## PU: literature-comparable protocols (v9)
```
python scripts/run_suite.py configs/experiments/pu_quick.yaml      # PhyQ-TransNet, 6 runs
python scripts/run_suite.py configs/experiments/pu_protocols.yaml  # 6 models, 126 runs
```
`seen_*` / `paper_dong7` = seen bearings, unseen recordings; `cc_X2Y` = cross-condition, same
bearings; `fold0..4` (main benchmark) = unseen bearings. Review of the 10 PU reference papers:
`docs/PU_REFERENCES_REVIEW_FA.md`.
