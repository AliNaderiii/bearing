"""Torch Dataset over (recording, start) window indices of cached arrays."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Returns clean, globally-scaled windows ``(C, L)`` and labels.

    * Arrays are opened lazily per process -> safe for Windows ``spawn`` workers.
    * ``scale`` is a per-channel constant estimated on TRAIN clean data only.
    * Noise is NOT added here; it is applied batch-wise (see ``phyq.data.noise``).
    """

    def __init__(self, paths: List[str], index: np.ndarray, labels: np.ndarray, window: int,
                 channels: Optional[List[int]] = None, scale: Optional[np.ndarray] = None,
                 preload: bool = False, shared_cache: Optional[dict] = None):
        self.paths = paths
        self.index = np.asarray(index, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.window = int(window)
        self.channels = channels
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32).reshape(-1, 1)
        self.preload = preload
        # train/val/test of the same recordings share ONE in-memory copy (was 3 copies -> MemoryError on SDUST)
        self._arrs: dict = shared_cache if shared_cache is not None else {}

    MAX_OPEN_MEMMAPS = 256

    def __getstate__(self):
        d = self.__dict__.copy()
        if not self.preload:
            d["_arrs"] = {}
        return d

    def _arr(self, ri: int) -> np.ndarray:
        a = self._arrs.get(ri)
        if a is None:
            a = np.load(self.paths[ri], mmap_mode=None if self.preload else "r")
            if self.channels is not None:
                a = a[self.channels]
                if self.preload:
                    a = np.ascontiguousarray(a)
            if not self.preload and len(self._arrs) >= self.MAX_OPEN_MEMMAPS:
                self._arrs.pop(next(iter(self._arrs)))   # bounded: each memmap holds an open file
            self._arrs[ri] = a
        return a

    def __len__(self) -> int:
        return len(self.index)

    def get_window(self, i: int) -> np.ndarray:
        ri, st = self.index[i]
        x = np.asarray(self._arr(int(ri))[..., st:st + self.window], dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        if self.scale is not None:
            x = x * self.scale
        elif not x.flags.writeable:          # memory-mapped view -> private copy
            x = x.copy()
        return x

    def __getitem__(self, i: int):
        return torch.from_numpy(self.get_window(i)), int(self.labels[i])

    def materialize(self, max_items: Optional[int] = None, seed: int = 0) -> np.ndarray:
        n = len(self)
        idx = np.arange(n)
        if max_items and n > max_items:
            idx = np.random.RandomState(seed).choice(n, max_items, replace=False)
        return np.stack([self.get_window(int(i)) for i in idx], 0)
