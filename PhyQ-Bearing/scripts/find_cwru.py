"""Locate the CWRU 12k drive-end .mat files on your disk (READ-ONLY: only .mat headers are read).

    python scripts/find_cwru.py "D:/Bearing Project/CWRU/CWRU Case 08/QCNN for bearing diagnosis/data"
    python scripts/find_cwru.py "<folder>" --write      # also sets paths.cwru in configs/paths.yaml

Files are identified by the variable name inside them (X097_DE_time, X105_DE_time, ...),
so file names do not matter. Folders of other datasets are skipped for speed.
"""
import argparse
import os
import re
import sys

import _bootstrap  # noqa: F401
from _bootstrap import ROOT
import scipy.io as sio

from phyq.data.datasets.cwru import CLASSES, FILE_IDS

SKIP = re.compile(r"paderborn|sdust|ims|wind turbine|nrel|processed|results|cache|__pycache__|\.git", re.I)
ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="*")
ap.add_argument("--write", action="store_true", help="write the found folders into configs/paths.yaml")
a = ap.parse_args()
paths_file = os.path.join(ROOT, "configs", "paths.yaml")
roots = a.roots
if not roots:
    import yaml
    v = yaml.safe_load(open(paths_file, encoding="utf-8"))["paths"]["cwru"]
    roots = v if isinstance(v, list) else [v]

wanted = {f for ids in FILE_IDS.values() for f in ids}
found = {}
n_mat = 0
for r in roots:
    for dp, dirs, files in os.walk(r):
        dirs[:] = [d for d in dirs if not SKIP.search(d)]
        for fn in files:
            if not fn.lower().endswith(".mat"):
                continue
            n_mat += 1
            p = os.path.join(dp, fn)
            try:
                names = [w[0] for w in sio.whosmat(p)]
            except Exception:
                continue
            stem = re.findall(r"\d+", os.path.splitext(fn)[0])
            for nm in names:
                m = re.fullmatch(r"X(\d{3})_DE_time", nm)
                if m and int(m.group(1)) in wanted:
                    fid = int(m.group(1))
                    if fid not in found or (stem and int(stem[0]) == fid):
                        found[fid] = p
print(f"scanned {n_mat} .mat files\n")
print("load  " + " ".join(f"{c:>6s}" for c in CLASSES))
complete = []
for ld, ids in FILE_IDS.items():
    row = " ".join(f"{('OK' if f in found else '--'):>6s}" for f in ids)
    print(f"{ld}HP   {row}")
    if all(f in found for f in ids):
        complete.append(ld)
missing = sorted(wanted - set(found))
print(f"\ncomplete loads: {complete or 'none'}")
if missing:
    print(f"missing file ids: {missing}  (download: https://engineering.case.edu/bearingdatacenter)")
folders = sorted({os.path.dirname(p).replace("\\", "/") for p in found.values()})
print("folders containing CWRU files:")
for f in folders:
    print("   ", f)
if missing and complete:
    print(f"\nTip: run only complete loads with:  --set data.loads={complete}")
if a.write and folders:
    txt = open(paths_file, encoding="utf-8").read()
    flow = "[" + ", ".join(f'"{f}"' for f in folders) + "]"
    txt, n = re.subn(r"(?m)^(\s*cwru:).*$", lambda m: f"{m.group(1)} {flow}", txt)
    if n:
        open(paths_file, "w", encoding="utf-8").write(txt)
        print(f"\nconfigs/paths.yaml updated: cwru: {flow}")
sys.exit(0 if not missing else 1)
