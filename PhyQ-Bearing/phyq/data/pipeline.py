"""End-to-end data preparation: recordings -> cache -> split -> leakage check -> windows."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .datasets import build_recordings
from .recording import Recording, SignalCache
from .splits import assert_no_leakage, build_split
from .torch_dataset import WindowDataset
from .windows import build_all_windows, window_labels


@dataclass
class PreparedData:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    recordings: List[Recording]
    class_names: List[str]
    n_channels: int
    fs: float
    ref_power: torch.Tensor          # (C,) mean clean power of TRAIN windows after scaling
    summary: Dict


def _class_names(recs: List[Recording]) -> List[str]:
    m = {}
    for r in recs:
        m.setdefault(r.label, r.label_name)
    return [m[k] for k in sorted(m)]


def _inside(child: str, parent: str) -> bool:
    try:
        c, p = os.path.abspath(child), os.path.abspath(parent)
        return os.path.commonpath([c, p]) == p
    except ValueError:          # different drives on Windows
        return False


def assert_read_only_layout(dcfg: dict, results_dir: str = "results") -> None:
    """Refuse to run if the cache or results folder would be written INSIDE a dataset folder."""
    roots = dcfg.get("root")
    roots = roots if isinstance(roots, (list, tuple)) else [roots]
    for r in roots:
        if not r or str(r).lower() == "none":
            continue
        for w in (dcfg.get("cache_dir", "cache"), results_dir):
            if _inside(w, str(r)):
                raise RuntimeError(f"Safety stop: '{w}' is inside dataset folder '{r}'. Datasets are "
                                   f"read-only; put cache/results outside the data folders.")


def _total_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return 8 << 30


def _decide_preload(setting, paths, idx_w, logger=None) -> bool:
    """preload: true | false | auto. auto = keep arrays in RAM only if they use < 30% of physical RAM;
    otherwise memory-map them from the cache on disk (same results, a bit slower on the first epoch)."""
    if isinstance(setting, bool):
        return setting
    if str(setting).lower() in ("true", "false"):
        return str(setting).lower() == "true"
    used = sorted({int(r) for part in idx_w.values() if len(part)
                   for r in np.unique(np.asarray(part).reshape(-1, 2)[:, 0])}) if idx_w else range(len(paths))
    need = sum(os.path.getsize(paths[r]) for r in used if os.path.exists(paths[r]))
    ram = _total_ram_bytes()
    ok = need < 0.30 * ram
    if logger:
        logger.info(f"[memory] cached signals {need / 2**30:.1f} GB, RAM {ram / 2**30:.1f} GB -> "
                    f"{'load into RAM' if ok else 'memory-map from disk'}")
    return ok


def prepare(cfg: dict, seed: int = 0, logger=None) -> PreparedData:
    dcfg = cfg["data"]
    assert_read_only_layout(dcfg, cfg.get("results_dir", "results"))
    recs = build_recordings(dcfg)
    if not recs:
        raise RuntimeError(f"No recordings found for dataset '{dcfg['name']}' under {dcfg.get('root')}")
    cache = SignalCache(dcfg.get("cache_dir", "cache"), dcfg.get("target_fs"))
    ok, skipped = [], []
    for i, r in enumerate(recs, 1):
        try:
            cache.materialize(r)
            ok.append(r)
        except OSError as e:
            if getattr(e, "errno", None) == 28:   # disk full -> never hide it as a "skipped file"
                raise RuntimeError(f"Disk full while writing the cache ({dcfg.get('cache_dir')})") from e
            skipped.append({"rid": r.rid, "path": str(r.meta.get("path")), "error": repr(e)[:300]})
            if logger:
                logger.warning(f"[skip] {r.meta.get('path', r.rid)}: {e}")
        except Exception as e:  # unreadable/corrupt file -> skipped and REPORTED in summary
            skipped.append({"rid": r.rid, "path": str(r.meta.get("path")), "error": repr(e)[:300]})
            if logger:
                logger.warning(f"[skip] {r.meta.get('path', r.rid)}: {e}")
        if logger and i % 200 == 0:
            logger.info(f"[cache] {i}/{len(recs)} recordings")
    if skipped and (len(skipped) > 0.2 * len(recs) or not ok):
        raise RuntimeError(f"{len(skipped)}/{len(recs)} recordings could not be read, e.g. {skipped[:3]}")
    recs = ok
    n_chs = sorted({int(cache.shape(r)[0]) for r in recs})
    if len(n_chs) > 1 and dcfg.get("channels_idx") is None:
        dcfg["channels_idx"] = list(range(n_chs[0]))
        if logger:
            logger.warning(f"[channels] recordings have {n_chs} channels -> using the first {n_chs[0]}")
    L = int(dcfg["window"])
    split = build_split(recs, cache, dcfg["split"], L)
    assert_no_leakage(recs, split, dcfg["split"]["protocol"])

    idx_w = build_all_windows(split, dcfg, seed)
    paths = [cache.path(r) for r in recs]
    ch = dcfg.get("channels_idx")
    preload = _decide_preload(dcfg.get("preload", "auto"), paths, idx_w, logger)
    shared: dict = {}

    def mk(part, scale=None):
        return WindowDataset(paths, idx_w[part], window_labels(idx_w[part], recs), L, ch, scale, preload, shared)

    tr_raw = mk("train")
    sample = tr_raw.materialize(max_items=2000, seed=seed)          # TRAIN only
    std = sample.std(axis=(0, 2)) + 1e-8
    scale = (1.0 / std).astype(np.float32)
    train, val, test = mk("train", scale), mk("val", scale), mk("test", scale)
    ref_power = torch.tensor(((sample * scale[None, :, None]) ** 2).mean(axis=(0, 2)), dtype=torch.float32)

    names = _class_names(recs)
    n_ch = sample.shape[1]

    def counts(ds):
        return np.bincount(ds.labels, minlength=len(names)).tolist()

    summary = {
        "dataset": dcfg["name"], "protocol": dcfg["split"]["protocol"], "n_recordings": len(recs),
        "fs": cache.fs_of(recs[0]), "window": L, "channels": n_ch, "classes": names,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_per_class": counts(train), "val_per_class": counts(val), "test_per_class": counts(test),
        "test_groups": sorted({recs[s.rec_idx].group for s in split["test"]}),
        "train_groups": sorted({recs[s.rec_idx].group for s in split["train"]}),
        "skipped_recordings": skipped,
    }
    if logger:
        logger.info(f"[data] {summary['dataset']} | {summary['protocol']} | train/val/test = "
                    f"{len(train)}/{len(val)}/{len(test)} | C={n_ch} | fs={summary['fs']}")
    return PreparedData(train, val, test, recs, names, n_ch, cache.fs_of(recs[0]), ref_power, summary)
