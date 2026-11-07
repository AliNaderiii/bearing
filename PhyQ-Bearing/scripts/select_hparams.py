"""Pick each model's hyper-parameters from the VALIDATION-only grid (configs/experiments/hp_search.yaml)
and write configs/experiments/main_tuned.yaml (same runs as main_benchmark minus PU, per-model winners).

    python scripts/run_suite.py configs/experiments/hp_search.yaml
    python scripts/select_hparams.py
    python scripts/run_suite.py configs/experiments/main_tuned.yaml --seeds 0
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
from phyq.utils.config import load_yaml


def main():
    exp = load_yaml(os.path.join(ROOT, "configs", "experiments", "hp_search.yaml"))
    res = os.path.join(ROOT, "results", exp["name"])
    scores = defaultdict(dict)          # tag -> {task: best_val}
    for p in glob.glob(os.path.join(res, "*", "*", "*", "seed*", "metrics.json")):
        m = json.load(open(p, encoding="utf-8"))
        scores[m["model_tag"]][f"{m['dataset']}:{m['variant']}:{m['seed']}"] = float(m["train"]["best_val"])
    tasks = sorted({t for d in scores.values() for t in d})
    cand = {m["tag"]: m for m in exp["models"]}
    by_base = defaultdict(list)
    for tag in cand:
        by_base[tag.split("__")[0]].append(tag)
    lines = ["# Hyper-parameter selection on VALIDATION (test never evaluated)", "",
             "| candidate | " + " | ".join(tasks) + " | mean | |", "|---" * (len(tasks) + 3) + "|"]
    winners, missing = {}, []
    for base, tags in by_base.items():
        best, best_tag = -1.0, None
        for tag in tags:
            row = [scores[tag].get(t) for t in tasks]
            if any(v is None for v in row):
                missing.append(tag)
                lines.append(f"| {tag} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in row) + " | - | |")
                continue
            mu = float(np.mean(row))
            if mu > best:
                best, best_tag = mu, tag
            lines.append(f"| {tag} | " + " | ".join(f"{v:.2f}" for v in row) + f" | {mu:.2f} | |")
        if best_tag:
            winners[base] = best_tag
    lines += ["", "## Winners", ""] + [f"* {b}: **{t.split('__')[1]}**" for b, t in winners.items()]
    if missing:
        lines += ["", f"WARNING: {len(missing)} candidates incomplete -> finish hp_search first: {missing}"]
    txt = "\n".join(lines)
    os.makedirs(res, exist_ok=True)
    open(os.path.join(res, "selection.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    if missing or len(winners) < len(by_base):
        print("\nmain_tuned.yaml NOT written (incomplete search).")
        return
    out = ["# Written by scripts/select_hparams.py -- per-model hyper-parameters chosen on VALIDATION",
           "# (results/hp_search/selection.md). Same data/protocols/noise tests as main_benchmark.",
           "name: main_tuned", "runs:",
           "  - {dataset: cwru, variants: [within_0HP, within_1HP, within_2HP, within_3HP], seeds: [0, 1, 2, 3, 4]}",
           "  - {dataset: ims, variants: [chrono], seeds: [0, 1, 2]}",
           "  - {dataset: nrel, variants: [expwise], seeds: [0, 1, 2]}",
           "  - {dataset: sdust, variants: [chrono], seeds: [0, 1, 2]}", "models:"]
    for base, tag in winners.items():
        m = cand[tag]
        ovs = ", ".join(m.get("overrides", []))
        out.append(f"  - {{tag: {base}, config: {m['config']}, overrides: [{ovs}]}}   # {tag}")
    out += ["seeds: [0, 1, 2]", "overrides: [\"eval.noise_types=[white, pink, impulsive]\"]", ""]
    path = os.path.join(ROOT, "configs", "experiments", "main_tuned.yaml")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print(f"\n-> {path} written")


if __name__ == "__main__":
    main()
