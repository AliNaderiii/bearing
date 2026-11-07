"""Statistical tests for paired model comparison."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats


def mcnemar(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> Dict:
    """Exact (binomial) McNemar test on paired predictions of the same test windows."""
    a = pa == y
    b = pb == y
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p": p}


def grouped_bootstrap_ci(correct: np.ndarray, groups: np.ndarray, labels: np.ndarray = None, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> Dict:
    """Accuracy CI that respects the correlation of windows from the same recording.

    Recordings are resampled *within each class* (class composition stays fixed). If a class
    has a single recording (CWRU, IMS) its windows are resampled in contiguous blocks instead.
    """
    rng = np.random.RandomState(seed)
    labels = np.zeros_like(groups) if labels is None else labels
    strata = []
    for c in np.unique(labels):
        m = labels == c
        gs = np.unique(groups[m])
        strata.append([correct[m & (groups == g)] for g in gs])
    accs = []
    for _ in range(n_boot):
        parts = []
        for st in strata:
            if len(st) > 1:
                parts += [st[i] for i in rng.choice(len(st), len(st), replace=True)]
            else:
                w = st[0]
                blk = max(1, len(w) // 4)
                starts = rng.randint(0, max(1, len(w) - blk + 1), size=int(np.ceil(len(w) / blk)))
                parts.append(np.concatenate([w[s:s + blk] for s in starts])[: len(w)])
        accs.append(np.concatenate(parts).mean())
    lo, hi = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"acc": float(correct.mean() * 100), "lo": float(lo * 100), "hi": float(hi * 100)}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) == 0 or np.allclose(d, 0):
        return {"stat": 0.0, "p": 1.0, "n": len(d), "mean_diff": float(d.mean()) if len(d) else 0.0}
    r = stats.wilcoxon(a, b, zero_method="zsplit")
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n": int(len(d)), "mean_diff": float(d.mean())}


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj
