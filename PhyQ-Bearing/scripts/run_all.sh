#!/usr/bin/env bash
# Full pipeline (Linux/macOS). Resumable.
set -e
cd "$(dirname "$0")/.."
python scripts/check_env.py
python scripts/sanity_check_data.py cwru pu ims nrel sdust
python -m pytest -q tests
for e in main_benchmark cross_load ablation; do
  python scripts/run_suite.py configs/experiments/$e.yaml
done
for e in main_benchmark cross_load ablation; do python scripts/make_tables.py $e; done
python scripts/profile_models.py
