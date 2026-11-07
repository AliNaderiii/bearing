"""Show what is inside ONE data file (read-only) - e.g. to confirm the SDUST channels are real vibration.

    python scripts/inspect_file.py "D:/.../SDUST_bearing/NC/some_file.mat" --fs 25600
Prints every numeric variable, then the channels the loader keeps (constant, time-index and duplicate
columns are dropped) with simple statistics and the dominant frequency of each channel.
"""
import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401
from phyq.data.datasets.sdust import _clean_channels, _numeric_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fs", type=float, default=25600.0)
    a = ap.parse_args()
    ext = os.path.splitext(a.path)[1].lower()
    raw = []
    if ext == ".mat":
        import scipy.io as sio
        try:
            d = sio.loadmat(a.path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:          # MATLAB v7.3 (HDF5)
            import h5py
            d = {k: np.array(v) for k, v in h5py.File(a.path, "r").items() if isinstance(v, h5py.Dataset)}
        for k, v in d.items():
            if not k.startswith("__"):
                print(f"variable {k!r}: type {type(v).__name__}, shape {getattr(v, 'shape', None)}, "
                      f"dtype {getattr(v, 'dtype', None)}")
        _numeric_arrays(d, raw)
    else:
        import pandas as pd
        df = pd.read_csv(a.path, sep=None, engine="python", header=None, nrows=200000)
        print(df.head(3).to_string())
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0)
        raw = [df.iloc[:, j].to_numpy(np.float32) for j in range(df.shape[1])]
    kept = _clean_channels(raw)
    print(f"\nnumeric arrays found: {len(raw)} | channels kept by the loader: {len(kept)}")
    for i, x in enumerate(kept):
        x = x.astype(np.float64)
        xc = x - x.mean()
        kurt = float(np.mean(xc ** 4) / (np.mean(xc ** 2) ** 2 + 1e-12))
        seg = xc[: min(len(xc), 1 << 16)]
        f = np.fft.rfftfreq(len(seg), 1 / a.fs)
        peak = f[1 + int(np.argmax(np.abs(np.fft.rfft(seg))[1:]))]
        uniq = len(np.unique(x[:20000]))
        print(f"  ch{i}: n={len(x):,} ({len(x) / a.fs:.1f} s @ fs) mean={x.mean():.4g} std={x.std():.4g} "
              f"min={x.min():.4g} max={x.max():.4g} kurtosis={kurt:.2f} peak={peak:.1f} Hz "
              f"unique(first 20k)={uniq}")
    print("\nA vibration channel: mean ~0, many unique values. A tacho/speed/torque channel: few unique values,"
          " large non-zero mean or a square-wave look -> tell me its index.")


if __name__ == "__main__":
    main()
