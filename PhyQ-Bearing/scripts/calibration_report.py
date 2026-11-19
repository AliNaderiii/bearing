"""Calibration of the test predictions (reviewer request): ECE (15 equal-width bins, top-label),
multi-class Brier score and negative log-likelihood, on clean data and under -6 dB absolute-SNR
white, pink and impulsive noise. Computed from the stored test_probs.npz of every run (no retraining).

    python scripts/calibration_report.py --exp main_tuned main_benchmark --out results/_paper_assets/calibration
    python scripts/calibration_report.py --fig        # + reliability diagrams (pooled over runs), PDF/PNG

Outputs: <out>_runs.csv (one row per run and condition), <out>.json (mean over runs),
<out>_reliability.json (per-bin window counts, mean confidence and accuracy, pooled over runs).
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

CONDS = {"clean": "clean", "W-6": "white__absolute__-6", "P-6": "pink__absolute__-6",
         "I-6": "impulsive__absolute__-6"}
MODELS = ["phyq_nomed", "phyq_nomed_noquad", "classbd", "wdcnn", "qcnn", "drsn", "cnn_transformer"]


def calib(p: np.ndarray, y: np.ndarray, bins: int = 15) -> dict:
    """Windows whose output is non-finite (float32 overflow of the network) are counted as errors in
    'acc' (as in the paper: argmax of NaN = class 0 in the original metric) but excluded from ECE/Brier/NLL,
    whose values refer to the finite windows; their share is reported as 'nonfinite' (%)."""
    p = np.asarray(p, np.float64)
    fin = np.isfinite(p).all(1)
    acc_paper = 100 * float((p.argmax(1) == y).mean())
    p, y = p[fin], y[fin]
    if not len(y):
        return {"acc": acc_paper, "nonfinite": 100.0, "ece": float("nan"), "brier": float("nan"),
                "nll": float("nan"), "conf": float("nan")}
    p = p / p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    onehot = np.eye(p.shape[1])[y]
    return {"acc": acc_paper, "nonfinite": 100 * (1 - fin.mean()), "ece": 100 * ece, "brier": float(((p - onehot) ** 2).sum(1).mean()),
            "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()),
            "conf": 100 * conf.mean()}


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 15):
    """Per-bin (count, sum of confidence, sum of correct) of the finite windows."""
    p = np.asarray(p, np.float64)
    fin = np.isfinite(p).all(1)
    p, y = p[fin], y[fin]
    out = np.zeros((3, bins))
    if not len(y):
        return out
    p = p / p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    np.add.at(out[0], idx, 1)
    np.add.at(out[1], idx, conf)
    np.add.at(out[2], idx, (pred == y).astype(np.float64))
    return out


def task_of(dataset: str, variant: str) -> str:
    if dataset == "pu":
        return "PU-a2r" if variant == "art2real" else ("PU5" if variant.startswith("fold") else None)
    return dataset.upper()


def run_npz(path: str, rel=None, key=None) -> dict:
    z = np.load(path)
    y = z["y"]
    if rel is not None:
        for c, k in CONDS.items():
            if k in z.files:
                rel[key + (c,)] = rel.get(key + (c,), 0) + reliability_bins(z[k], y)
    return {c: calib(z[k], y) for c, k in CONDS.items() if k in z.files}


def plot_reliability(rel: dict, out: str, tasks=("CWRU", "PU5"), conds=("clean", "I-6"),
                     models=("phyq_nomed", "phyq_nomed_noquad", "classbd")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"phyq_nomed": "PhyQ-TransNet", "phyq_nomed_noquad": "Linear twin", "classbd": "ClassBD",
             "wdcnn": "WDCNN", "drsn": "DRSN-CW", "qcnn": "QCNN", "cnn_transformer": "CNN-Transformer"}
    cols = ["#1b6ca8", "#7f7f7f", "#d1495b", "#2a9d8f", "#e9c46a", "#6a4c93", "#264653"]
    fig, axes = plt.subplots(len(conds), len(tasks), figsize=(3.2 * len(tasks), 3.0 * len(conds)), squeeze=False)
    for i, c in enumerate(conds):
        for j, t in enumerate(tasks):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
            for k, m in enumerate(models):
                b = rel.get((t, m, c))
                if b is None or b[0].sum() == 0:
                    continue
                ok = b[0] >= 20
                ax.plot(b[1][ok] / b[0][ok], b[2][ok] / b[0][ok], "o-", ms=3, lw=1.1, color=cols[k % len(cols)],
                        label=names.get(m, m))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{t}, {c}", fontsize=9)
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}_reliability.{ext}", dpi=200)
    plt.close(fig)


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["task"], r["model"])][r["cond"]].append(r)
    out = []
    for (task, model), d in sorted(agg.items()):
        for cond, rs in d.items():
            out.append({"task": task, "model": model, "cond": cond, "n": len(rs),
                        **{m: float(np.nanmean([x[m] for x in rs])) for m in ("acc", "nonfinite", "ece", "brier", "nll", "conf")}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--exp", nargs="*", default=["main_tuned", "main_benchmark"])
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--out", default="results/_paper_assets/calibration")
    ap.add_argument("--fig", action="store_true", help="also draw reliability diagrams")
    args = ap.parse_args()
    rows, rel = [], {}
    for exp in args.exp:
        for f in sorted(glob.glob(os.path.join(args.results, exp, "*", "*", "*", "seed*", "test_probs.npz"))):
            parts = os.path.normpath(f).split(os.sep)
            ds, var, model, seed = parts[-5], parts[-4], parts[-3], parts[-2]
            if model not in args.models or (exp == "main_benchmark" and ds != "pu") or \
                    (exp == "main_tuned" and ds == "pu"):
                continue
            task = task_of(ds, var)
            if task is None:
                continue
            for cond, m in run_npz(f, rel, (task, model)).items():
                rows.append({"task": task, "model": model, "variant": var, "seed": seed, "cond": cond, **m})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = aggregate(rows)
    with open(args.out + ".json", "w") as fh:
        json.dump(agg, fh, indent=1)
    with open(args.out + "_reliability.json", "w") as fh:
        json.dump([{"task": k[0], "model": k[1], "cond": k[2], "count": v[0].tolist(), "sum_conf": v[1].tolist(),
                    "sum_correct": v[2].tolist()} for k, v in sorted(rel.items())], fh)
    if args.fig:
        plot_reliability(rel, args.out)
    print(f"{len(rows)} run-conditions -> {args.out}_runs.csv / .json / _reliability.json")


if __name__ == "__main__":
    main()
