"""Estimate GPU hours of every run_all suite from the speed test (configs/experiments/timing.yaml).

    python scripts/run_suite.py configs/experiments/timing.yaml
    python scripts/estimate_time.py                # optional: --epochs 40 --seeds 0
Per run: time = epochs * (train seconds per epoch) + evaluation overhead, measured per (dataset, model).
Max epochs are read from each run's config; early stopping usually ends runs earlier, so a range
(typical .. worst case) is printed.
"""
import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from phyq.suite import expand

SUITES = ["main_benchmark", "cross_load", "ablation", "pu_quick", "pu_quick_w2048", "pu_protocols"]
BASELINES = {"wdcnn", "qcnn", "drsn", "cnn_transformer", "stats_mlp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, help="same as run_all -Seeds (search keeps its own seeds)")
    ap.add_argument("--typical", type=float, default=0.75,
                    help="typical fraction of the max epochs actually run (early stopping)")
    a = ap.parse_args()
    speed = {}
    for p in glob.glob(os.path.join(ROOT, "results", "timing", "*", "*", "*", "seed*", "metrics.json")):
        m = json.load(open(p))
        tr = m["train"]
        ep = max(1, int(tr.get("epochs_run", 1)))
        per_ep = tr["train_time_s"] / ep
        over = max(0.0, m.get("wall_time_s", tr["train_time_s"]) - tr["train_time_s"])
        speed[(m["dataset"], m["model_tag"])] = (per_ep, over)
    if not speed:
        print("No timing results. Run first:  python scripts/run_suite.py configs/experiments/timing.yaml")
        return 1
    print("measured (seconds per epoch | eval overhead per run):")
    for (ds, tag), (pe, ov) in sorted(speed.items()):
        print(f"   {ds:6s} {tag:16s} {pe:8.1f} s/epoch | {ov:7.1f} s")
    total_typ = total_max = 0.0
    print("\nestimated time per suite (runs already finished are excluded), typical .. worst case:")
    for name in SUITES:
        path = os.path.join(ROOT, "configs", "experiments", f"{name}.yaml")
        seeds = None if name == "search" else a.seeds
        t_typ = t_max = 0.0
        n = n_todo = 0
        missing = set()
        for s in expand(path, [], None, None, seeds):
            n += 1
            if os.path.exists(os.path.join(s.out_dir, "metrics.json")):
                continue
            n_todo += 1
            tag = s.model_tag if s.model_tag in BASELINES else "phyq_transnet"   # ablation tags = PhyQ-TransNet variants
            if (s.dataset, tag) not in speed:
                missing.add((s.dataset, tag))
                continue
            pe, ov = speed[(s.dataset, tag)]
            e_max = int(s.cfg["train"].get("epochs", 40))
            e_typ = a.typical * e_max
            t_typ += e_typ * pe + ov
            t_max += e_max * pe + ov
        total_typ += t_typ
        total_max += t_max
        warn = f"  (no timing for {sorted(missing)})" if missing else ""
        print(f"   {name:15s} {n_todo:4d}/{n:<4d} runs   {t_typ / 3600:6.1f} .. {t_max / 3600:6.1f} h{warn}")
    print(f"   {'TOTAL':15s} {'':14s} {total_typ / 3600:6.1f} .. {total_max / 3600:6.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
