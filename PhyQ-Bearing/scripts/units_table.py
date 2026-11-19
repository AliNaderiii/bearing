"""Table of physical units per partition (advisor comment 2/8): reads results/_data_report/<ds>.json
(written by scripts/sanity_check_data.py) and writes results/_paper_assets/tab_units.tex.
  python scripts/units_table.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DR = os.path.join(ROOT, "results", "_data_report")
OUT = os.path.join(ROOT, "results", "_paper_assets", "tab_units.tex")


def load(ds, var):
    return json.load(open(os.path.join(DR, f"{ds}.json")))["variants"][var]


def pc(v):
    return "/".join(str(x) for x in v)


def main():
    rows = []
    try:
        v = load("cwru", "within_0HP")
        rows.append(("CWRU (each load)", f"{len(v['train_groups'])} recordings (1 per class), drive end",
                     "same recordings, disjoint later segments", pc(v["test_per_class"])))
    except (OSError, KeyError):
        pass
    for ds, var, tr, te in (("ims", "chrono", "4 failing bearings; normal = early life of the same bearings",
                             "same 4 bearings, later time block"),
                            ("sdust", "chrono", "10 recordings (1 per class)", "same recordings, disjoint later segments")):
        try:
            v = load(ds, var)
            rows.append((ds.upper(), tr, te, pc(v["test_per_class"])))
        except (OSError, KeyError):
            pass
    try:
        v = load("nrel", "expwise")
        rows.append(("NREL", ", ".join(v["train_groups"]), ", ".join(v["test_groups"]), pc(v["test_per_class"])))
    except (OSError, KeyError):
        pass
    for k in ("fold0", "fold1", "fold2", "fold3", "fold4", "art2real"):
        try:
            v = load("pu", k)
            rows.append((f"PU {k}", ", ".join(v["train_groups"]), ", ".join(v["test_groups"]), pc(v["test_per_class"])))
        except (OSError, KeyError):
            pass
    lines = ["\\begin{table*}[t]", "\\caption{Physical units per partition and test windows per class.}\\label{tab:units}",
             "\\centering\\scriptsize", "\\begin{tabular}{p{1.8cm}p{6.0cm}p{4.4cm}p{2.2cm}}", "\\toprule",
             "Task & Training units & Test units & Test windows per class \\\\", "\\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
