"""Window index generation inside split segments."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .recording import Recording
from .splits import Segment


def make_windows(segments: List[Segment], window: int, stride: int,
                 max_per_segment: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Return int64 array (N, 2) of (rec_idx, start). Windows never leave their segment.

    If ``max_per_segment`` is set, windows are sub-sampled *evenly in time*
    (deterministic, not random) so that long recordings do not dominate.
    """
    rows = []
    for s in segments:
        n = s.stop - s.start
        if n < window:
            continue
        starts = np.arange(s.start, s.stop - window + 1, stride, dtype=np.int64)
        if max_per_segment and len(starts) > max_per_segment:
            pick = np.linspace(0, len(starts) - 1, int(max_per_segment)).round().astype(int)
            starts = starts[pick]
        rows.append(np.stack([np.full_like(starts, s.rec_idx), starts], 1))
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(rows, 0)


def window_labels(index: np.ndarray, recs: List[Recording]) -> np.ndarray:
    lab = np.array([r.label for r in recs], dtype=np.int64)
    return lab[index[:, 0]] if len(index) else np.zeros(0, dtype=np.int64)


def build_all_windows(split: Dict[str, List[Segment]], cfg: dict, seed: int = 0) -> Dict[str, np.ndarray]:
    L = int(cfg["window"])
    tr_stride = int(cfg.get("train_stride", L // 2))
    ev_stride = int(cfg.get("eval_stride", L))  # non-overlapping by default
    mx_tr = cfg.get("max_windows_per_segment_train")
    mx_ev = cfg.get("max_windows_per_segment_eval")
    return {
        "train": make_windows(split["train"], L, tr_stride, mx_tr, seed),
        "val": make_windows(split["val"], L, ev_stride, mx_ev, seed),
        "test": make_windows(split["test"], L, ev_stride, mx_ev, seed),
    }
