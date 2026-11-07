"""Recording abstraction + on-disk signal cache.

Every dataset adapter returns a list of :class:`Recording`. A recording is one
continuous multichannel measurement of ONE physical unit (bearing / gearbox) in
ONE operating condition. Everything downstream (splitting, windowing, noise,
features) is dataset-agnostic and works on recordings, which makes leakage
checks explicit and testable:

* ``group``      -> physical unit identity (e.g. PU bearing code ``KA04``).
                    Group-wise splits guarantee that a unit never appears in two splits.
* ``condition``  -> operating condition (load / speed / torque).
* ``time_order`` -> chronological position (IMS run-to-failure snapshots).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly


@dataclass
class Recording:
    rid: str                              # globally unique id, also the cache file name
    dataset: str
    label: int
    label_name: str
    group: str                            # physical unit identity
    condition: str                        # operating condition
    fs: float                             # native sampling rate [Hz]
    loader: Callable[[], np.ndarray]      # returns float32 array (C, T) in native fs
    channels: List[str] = field(default_factory=list)
    time_order: float = 0.0               # for run-to-failure ordering
    meta: Dict = field(default_factory=dict)


def resample_signal(x: np.ndarray, fs_in: float, fs_out: Optional[float]) -> np.ndarray:
    """Anti-aliased polyphase resampling along the last axis."""
    if not fs_out or abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=-1).astype(np.float32)


class SignalCache:
    """Materialises each recording once as ``<cache_dir>/<dataset>/<fs>/<rid>.npy``.

    Arrays are later opened with ``mmap_mode='r'`` (or preloaded) so that large
    datasets (PU, IMS) can be windowed without re-parsing the raw files.
    """

    def __init__(self, cache_dir: str, target_fs: Optional[float] = None):
        self.cache_dir = cache_dir
        self.target_fs = target_fs
        self._mm: Dict[str, np.ndarray] = {}
        self._shape: Dict[str, tuple] = {}

    def path(self, rec: Recording) -> str:
        tag = f"fs{int(self.target_fs)}" if self.target_fs else "native"
        safe = rec.rid.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(self.cache_dir, rec.dataset, tag, f"{safe}.npy")

    def fs_of(self, rec: Recording) -> float:
        return float(self.target_fs or rec.fs)

    def materialize(self, rec: Recording, force: bool = False) -> str:
        p = self.path(rec)
        if os.path.exists(p) and not force:
            return p
        os.makedirs(os.path.dirname(p), exist_ok=True)
        x = rec.loader()
        if x.ndim == 1:
            x = x[None, :]
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = resample_signal(x, rec.fs, self.target_fs)
        tmp = p + ".tmp.npy"
        np.save(tmp, np.ascontiguousarray(x))
        os.replace(tmp, p)
        return p

    def get(self, rec: Recording) -> np.ndarray:
        if rec.rid not in self._mm:
            self._mm[rec.rid] = np.load(self.materialize(rec), mmap_mode="r")
        return self._mm[rec.rid]

    def shape(self, rec: Recording) -> tuple:
        """Array shape read from the .npy header only (no memory map is kept open).

        Keeping one memmap per recording exhausted the per-process file limit on Linux
        (ulimit -n 1024) for PU's 1360 recordings ("Too many open files")."""
        shp = self._shape.get(rec.rid)
        if shp is None:
            with open(self.materialize(rec), "rb") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shp = np.lib.format.read_array_header_1_0(f)[0]
                else:
                    shp = np.lib.format.read_array_header_2_0(f)[0]
            self._shape[rec.rid] = shp = tuple(int(v) for v in shp)
        return shp

    def length(self, rec: Recording) -> int:
        return int(self.shape(rec)[-1])
