"""v10.10d: NLL table (tab:nll) from metrics.json (test.clean.nll, test.noise.<fam>.absolute.-6.nll).
Usage: python scripts/nll_table.py --results results --out results/_paper_assets/nll"""
import argparse, glob, json, os
import numpy as np
GROUPS = {"CWRU": ["main_tuned/cwru/within_%dHP" % i for i in range(4)], "IMS": ["main_tuned/ims/chrono"],
          "NREL": ["main_tuned/nrel/expwise"], "SDUST": ["main_tuned/sdust/chrono"],
          "PU (5 folds)": ["main_benchmark/pu/fold%d" % i for i in range(5)], "PU art2real": ["main_benchmark/pu/art2real"]}
MODELS = [("phyq_nomed", "PhyQ-TransNet"), ("phyq_nomed_noquad", "Linear twin"), ("classbd", "ClassBD"),
          ("wdcnn", "WDCNN"), ("drsn", "DRSN-CW")]
def files(root, g, mod):
    return [f for v in GROUPS[g] for f in sorted(glob.glob(f"{root}/{v}/{mod}/seed*/metrics.json"))]
def nll(m, k):
    if k == "clean":
        return m["test"]["clean"].get("nll", np.nan)
    try:
        return m["test"]["noise"][k]["absolute"]["-6"].get("nll", np.nan)
    except KeyError:
        return np.nan
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/_paper_assets/nll"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for g in GROUPS:
        cells = []
        for mod, _ in MODELS:
            ms = [json.load(open(f)) for f in files(a.results, g, mod)]
            v = []
            for k in ["clean", "white", "pink"]:
                x = np.array([nll(m, k) for m in ms], float)
                v.append("%.2f" % x.mean() if len(x) and np.all(np.isfinite(x)) else "--")
            cells.append(" / ".join(v))
        rows.append(g + " & " + " & ".join(cells) + r" \\")
    open(os.path.join(a.out, "tab_nll_rows.tex"), "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
if __name__ == "__main__":
    main()
