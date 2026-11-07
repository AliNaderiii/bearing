"""train/val/test share one array cache; preload=auto falls back to memory-mapping for big data."""
import numpy as np

import phyq.data.pipeline as pl
from phyq.data.torch_dataset import WindowDataset


def test_shared_cache_and_auto(tmp_path, monkeypatch):
    p = tmp_path / "r0.npy"
    np.save(p, np.random.randn(2, 10000).astype(np.float32))
    idx = {"train": np.array([[0, 0], [0, 100]]), "val": np.array([[0, 5000]]), "test": np.zeros((0, 2), int)}
    shared = {}
    tr = WindowDataset([str(p)], idx["train"], np.array([0, 0]), 256, None, None, True, shared)
    va = WindowDataset([str(p)], idx["val"], np.array([0]), 256, None, None, True, shared)
    tr[0], va[0]
    assert len(shared) == 1 and tr._arrs is va._arrs           # one copy, not one per split
    assert pl._decide_preload("auto", [str(p)], idx) is True   # 80 kB << RAM
    monkeypatch.setattr(pl, "_total_ram_bytes", lambda: 100_000)
    assert pl._decide_preload("auto", [str(p)], idx) is False  # too big -> memmap
    assert pl._decide_preload(True, [str(p)], idx) is True
    mm = WindowDataset([str(p)], idx["val"], np.array([0]), 256, None, None, False, {})
    assert mm[0][0].shape == (2, 256)
