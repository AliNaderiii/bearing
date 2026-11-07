"""Validation-only model selection.

1. runs configs/experiments/search.yaml (val_only: the test set is never evaluated);
2. ranks candidates by mean validation score across development tasks and seeds;
3. writes results/search/ranking.md and the winning tag to results/search/selected.txt.

    python scripts/search.py                         # run + rank
    python scripts/search.py --rank-only
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT


def write_selected(exp_path, tag):
    """Write configs/models/phyqnet_selected.yaml = winner's base config + its overrides."""
    import yaml
    from phyq.utils.config import load_yaml
    exp = load_yaml(exp_path)
    m = [x for x in exp["models"] if (x if isinstance(x, str) else x["tag"]) == tag][0]
    if isinstance(m, str):
        m = {"tag": m, "config": m}
    cfg = {"extends": f"{m['config']}.yaml"}
    for ov in m.get("overrides", []) or []:
        k, v = ov.split("=", 1)
        d = cfg
        parts = k.strip().split(".")
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = yaml.safe_load(v.strip())
    path = os.path.join(ROOT, "configs", "models", "phyqnet_selected.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Written by scripts/search.py: winner on VALIDATION = '{tag}'\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"-> {path} updated (all main suites now use it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=os.path.join(ROOT, "configs", "experiments", "search.yaml"))
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    if not a.rank_only:
        rc = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "run_suite.py"), a.experiment]
                             + a.overrides)
        if rc != 0:
            print("search stopped: fix the message above, then run  python scripts\\search.py  again "
                  "(finished candidates are kept)")
            sys.exit(rc)
    from phyq.utils.config import load_yaml
    exp_name = load_yaml(a.experiment).get("name", "search")
    res_dir = os.path.join(ROOT, "results", exp_name)
    tags = [m if isinstance(m, str) else m["tag"] for m in load_yaml(a.experiment)["models"]]
    rows = {}
    for p in glob.glob(os.path.join(res_dir, "*", "*", "*", "seed*", "metrics.json")):
        m = json.load(open(p))
        task = f"{m['dataset']}/{m['variant']}"
        if m["model_tag"] in tags:
            rows.setdefault(m["model_tag"], {}).setdefault(task, []).append(m["train"]["best_val"])
    if not rows:
        print("No search results found")
        return
    tasks = sorted({t for r in rows.values() for t in r})
    table = []
    for tag, r in rows.items():
        per = [np.mean(r[t]) if t in r else np.nan for t in tasks]
        table.append((tag, np.nanmean(per), per, sum(len(v) for v in r.values())))
    incomplete = [t for t in table if np.isnan(t[2]).any()]
    if incomplete:
        print(f"WARNING: incomplete candidates (not all tasks finished): {[t[0] for t in incomplete]}")
    table.sort(key=lambda x: (np.isnan(x[2]).any(), -x[1]))
    lines = ["| candidate | mean val score | " + " | ".join(tasks) + " | runs |",
             "|---|---|" + "---|" * len(tasks) + "---|"]
    for tag, mu, per, n in table:
        lines.append(f"| {tag} | {mu:.2f} | " + " | ".join(f"{v:.2f}" for v in per) + f" | {n} |")
    out = res_dir
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "ranking.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(out, "selected.txt"), "w").write(table[0][0] + "\n")
    write_selected(a.experiment, table[0][0])
    print("\n".join(lines))
    print(f"\nSelected on validation: {table[0][0]}")


if __name__ == "__main__":
    main()
