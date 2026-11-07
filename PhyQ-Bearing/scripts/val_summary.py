"""Validation-only summary of an experiment (no test numbers): mean best_val per model tag and task.

best_val is the model-selection score (val macro-F1 averaged over clean, 0 dB and -6 dB white noise,
fixed val noise seed). Use it to take design decisions WITHOUT looking at the test set, e.g.

    python scripts/val_summary.py ablation
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--reference", default=None, help="tag to compare against (default: full / phyq_transnet)")
    a = ap.parse_args()
    res = os.path.join(ROOT, "results", a.experiment)
    sc = defaultdict(lambda: defaultdict(list))        # tag -> task -> [best_val per seed]
    for p in glob.glob(os.path.join(res, "*", "*", "*", "seed*", "metrics.json")):
        m = json.load(open(p, encoding="utf-8"))
        if m.get("trained_in") or "train" not in m or "best_val" not in m["train"]:
            continue                                     # extra-test copies carry no own training
        sc[m["model_tag"]][f"{m['dataset']}:{m['variant']}"].append(float(m["train"]["best_val"]))
    if not sc:
        print("no results")
        return
    tasks = sorted({t for d in sc.values() for t in d})
    ref = a.reference or ("full" if "full" in sc else "phyq_transnet")
    lines = [f"# Validation summary: {a.experiment} (best_val, mean over seeds; test set NOT used)", "",
             "| model | " + " | ".join(tasks) + " | mean | Δ vs " + ref + " |",
             "|---|" + "---|" * (len(tasks) + 2)]
    means = {}
    for tag in sorted(sc):
        row = [np.mean(sc[tag][t]) if sc[tag].get(t) else np.nan for t in tasks]
        means[tag] = float(np.nanmean(row))
    for tag in sorted(sc, key=lambda t: -means[t]):
        row = [np.mean(sc[tag][t]) if sc[tag].get(t) else np.nan for t in tasks]
        n = [len(sc[tag].get(t, [])) for t in tasks]
        cells = [("-" if np.isnan(v) else f"{v:.2f} (n={k})") for v, k in zip(row, n)]
        d = means[tag] - means[ref] if ref in means else np.nan
        lines.append(f"| {tag} | " + " | ".join(cells) + f" | **{means[tag]:.2f}** | {d:+.2f} |")
    txt = "\n".join(lines)
    os.makedirs(os.path.join(res, "_report"), exist_ok=True)
    open(os.path.join(res, "_report", "val_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
