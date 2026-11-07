"""Tiny mock files that mimic the on-disk formats of the five public datasets."""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio

RNG = np.random.RandomState(0)


def _sig(n, label=0):
    t = np.arange(n)
    return (np.sin(2 * np.pi * (0.01 + 0.005 * label) * t) + 0.3 * RNG.randn(n)).astype(np.float64)


def make_cwru(root, loads=(3,), n=40000):
    from phyq.data.datasets.cwru import FILE_IDS
    os.makedirs(root, exist_ok=True)
    for ld in loads:
        for lbl, fid in enumerate(FILE_IDS[ld]):
            d = {f"X{fid:03d}_DE_time": _sig(n, lbl)[:, None], f"X{fid:03d}_FE_time": _sig(n, lbl)[:, None]}
            if lbl > 0:
                d[f"X{fid:03d}_BA_time"] = _sig(n, lbl)[:, None]
            if fid == 99:  # real 99.mat also contains X098 keys -> must be ignored
                d["X098_DE_time"] = np.zeros((n, 1))
            sio.savemat(os.path.join(root, f"{fid}.mat"), d)


def make_pu(root, bearings=("K001", "K002", "KA04", "KA15", "KI04", "KI14"), conds=("N15_M07_F10",), files=2,
            n=64000 * 1):
    for b in bearings:
        os.makedirs(os.path.join(root, b), exist_ok=True)
        lbl = 0 if b.startswith("K0") else (1 if b.startswith("KA") else 2)
        for c in conds:
            for k in range(1, files + 1):
                stem = f"{c}_{b}_{k}"
                Y = np.zeros((1, 3), dtype=[("Name", "O"), ("Type", "O"), ("Data", "O")])
                for j, nm in enumerate(["phase_current_1", "vibration_1", "phase_current_2"]):
                    Y[0, j]["Name"] = nm
                    Y[0, j]["Type"] = "Mech"
                    Y[0, j]["Data"] = _sig(n, lbl)[None, :]
                sio.savemat(os.path.join(root, b, stem + ".mat"), {stem: {"Y": Y, "Info": "mock"}})


def make_ims(root, n_files=(60, 40, 50), n=20480, folders=("1st_test", "2nd_test", "4th_test/txt")):
    import datetime as dt
    specs = list(zip(folders, (8, 4, 4)))
    for (folder, ncol), nf in zip(specs, n_files):
        d = os.path.join(root, folder)
        os.makedirs(d, exist_ok=True)
        t0 = dt.datetime(2004, 2, 12, 10, 32, 39)
        for i in range(nf):
            amp = 1.0 + (4.0 if i > 0.8 * nf else 0.0)
            a = 0.1 * RNG.randn(n, ncol) * amp
            name = (t0 + dt.timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S")
            np.savetxt(os.path.join(d, name), a, fmt="%.3f", delimiter="\t")


def make_nrel(root, n=20000, chans=3):
    import pandas as pd
    os.makedirs(root, exist_ok=True)
    cols = {}
    for e in [f"H{i}" for i in range(1, 11)] + [f"D{i}" for i in range(1, 11)]:
        for c in range(3, 3 + chans):
            cols[f"{e}_AN{c}"] = _sig(n, 1 if e[0] == "D" else 0)
    pd.DataFrame(cols).to_csv(os.path.join(root, "nrel_all.csv"), index=False)


def make_sdust(root, n=30000):
    labels = ["nc", "if0.2", "if0.4", "if0.6", "of0.2", "of0.4", "of0.6", "rf0.2", "rf0.4", "rf0.6"]
    for cond in ("1000rpm", "2000rpm"):
        for li, l in enumerate(labels):
            d = os.path.join(root, cond)
            os.makedirs(d, exist_ok=True)
            sio.savemat(os.path.join(d, f"{l}_{cond}.mat"), {"data": np.stack([_sig(n, li) for _ in range(3)], 1)})
