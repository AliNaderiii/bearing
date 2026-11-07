"""Run ONE experiment.

    python scripts/run.py --dataset cwru --variant within_3HP --model phyq_paper --seed 0 train.epochs=30
"""
import argparse
import os

import _bootstrap  # noqa: F401
from phyq.runner import run_experiment
from phyq.suite import RunSpec, build_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="phyq_paper")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="single")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args()
    cfg = build_run_config(a.dataset, a.model, a.variant, a.overrides)
    spec = RunSpec(a.exp, a.dataset, a.variant or "default", a.model, a.seed, cfg)
    out = run_experiment(cfg, spec.out_dir, a.seed, force=a.force)
    if out is None:
        print(f"Already done: {spec.out_dir} (use --force)")
    else:
        print(f"Saved to {spec.out_dir}")


if __name__ == "__main__":
    main()
