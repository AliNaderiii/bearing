"""Classical signal-processing baseline: Fast Kurtogram -> envelope spectrum -> optimised classifier.

Pipeline (per window and channel)
---------------------------------
1. Fast Kurtogram (Antoni, MSSP 21 (2007) 108-124): spectral kurtosis of the complex envelope on the
   1/3-binary tree of frequency bands (levels 0, 1, 1.6, 2, 2.6, ..., ``nlevel``), computed with ideal
   FFT-domain band-pass filters.  The band with maximum kurtosis is selected automatically
   (no label information, no knowledge of the fault frequencies).
2. Squared-envelope spectrum (SES) of the selected band, normalised by its mean (scale invariant),
   integrated in ``n_bins`` linear bins over [0, ``f_max``] Hz and log-compressed.
3. Optional auxiliary features: centre frequency / bandwidth / kurtosis of the selected band and the
   same 12 statistical features used by the proposed model (``include_stats``).
4. Classifier chosen ON THE VALIDATION SET (same selection score as the deep models) among
   a small grid of RandomForest / HistGradientBoosting / RBF-SVM / logistic regression.

The training features include ``n_aug`` noisy copies of each training window drawn from the SAME
noise-augmentation distribution as the deep models (``train.noise_aug``), so the comparison is fair.
The fitted pipeline is wrapped in an ``nn.Module`` -> the standard evaluation code (bit-identical
noisy test sets, paired statistics) is used unchanged.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..features.stats import stat_features


# ------------------------------------------------------------------ fast kurtogram
def kurtogram_bands(fs: float, nlevel: int) -> List[Tuple[float, float, float]]:
    """(level, f_low, f_high) of the 1/3-binary tree up to ``nlevel``."""
    nyq = fs / 2
    bands = [(0.0, 0.0, nyq)]
    for k in range(1, nlevel + 1):
        bw = nyq / 2 ** k
        bands += [(float(k), i * bw, (i + 1) * bw) for i in range(2 ** k)]
        bw3 = nyq / (3 * 2 ** (k - 1))                       # level k + 0.6 (= log2(3)-1 offset)
        if k < nlevel:
            bands += [(k + 0.6, i * bw3, (i + 1) * bw3) for i in range(3 * 2 ** (k - 1))]
    return bands


def _band_masks(L: int, fs: float, bands) -> np.ndarray:
    f = np.fft.fftfreq(L, 1 / fs)
    M = np.zeros((len(bands), L), dtype=np.float32)
    for j, (_, lo, hi) in enumerate(bands):
        sel = (f >= lo) & (f < hi) if hi < fs / 2 else (f >= lo)
        M[j, sel] = 2.0                                       # analytic (positive frequencies only)
    M[:, 0] = np.where(np.array([b[1] for b in bands]) == 0, 1.0, 0.0)
    return M


def ses_features(x: np.ndarray, fs: float, nlevel: int = 5, f_max: float = 1000.0, n_bins: int = 100,
                 chunk: int = 512) -> np.ndarray:
    """x: (N, C, L) float -> features (N, C * (n_bins + 3))."""
    N, C, L = x.shape
    bands = kurtogram_bands(fs, nlevel)
    M = _band_masks(L, fs, bands)                             # (J, L)
    fl = np.array([b[1] for b in bands]); fh = np.array([b[2] for b in bands])
    f_r = np.fft.rfftfreq(L, 1 / fs)
    edges = np.linspace(0, f_max, n_bins + 1)
    bin_id = np.digitize(f_r, edges) - 1
    valid = (bin_id >= 0) & (bin_id < n_bins) & (f_r > 0)
    out = np.zeros((N, C, n_bins + 3), dtype=np.float32)
    for s in range(0, N, chunk):
        xc = x[s:s + chunk].astype(np.float32)
        X = np.fft.fft(xc - xc.mean(-1, keepdims=True), axis=-1)            # (n, C, L)
        for c in range(C):
            Xc = X[:, c]                                                     # (n, L)
            best_k = np.full(len(Xc), -np.inf); best_env = np.zeros((len(Xc), L), np.float32)
            best_j = np.zeros(len(Xc), int)
            for j in range(len(bands)):
                env2 = np.abs(np.fft.ifft(Xc * M[j], axis=-1)) ** 2          # squared envelope
                m2 = env2.mean(-1)
                kurt = (env2 ** 2).mean(-1) / (m2 ** 2 + 1e-20) - 2.0       # SK of complex envelope
                upd = kurt > best_k
                best_k[upd] = kurt[upd]; best_env[upd] = env2[upd]; best_j[upd] = j
            ses = np.abs(np.fft.rfft(best_env - best_env.mean(-1, keepdims=True), axis=-1))
            ses = ses / (ses[:, valid].mean(-1, keepdims=True) + 1e-20)
            feat = np.zeros((len(Xc), n_bins), np.float32)
            for b in range(n_bins):
                sel = valid & (bin_id == b)
                if sel.any():
                    feat[:, b] = ses[:, sel].max(-1)
            out[s:s + chunk, c, :n_bins] = np.log1p(feat)
            out[s:s + chunk, c, n_bins] = (fl[best_j] + fh[best_j]) / fs      # normalised centre freq
            out[s:s + chunk, c, n_bins + 1] = (fh[best_j] - fl[best_j]) / fs  # normalised bandwidth
            out[s:s + chunk, c, n_bins + 2] = np.log1p(np.maximum(best_k, 0))
    return out.reshape(N, -1)


# ------------------------------------------------------------------ wrapper module
class ClassicalModel(nn.Module):
    """Wraps feature extractor + fitted sklearn classifier; forward returns log-probabilities."""

    def __init__(self, clf, scaler, n_classes: int, fs: float, fcfg: Dict, include_stats: bool):
        super().__init__()
        self.clf, self.scaler, self.K, self.fs, self.fcfg = clf, scaler, n_classes, fs, fcfg
        self.include_stats = include_stats
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)   # keeps .to()/.parameters() happy

    def features(self, x: torch.Tensor) -> np.ndarray:
        xn = x.detach().float().cpu()
        f = ses_features(xn.numpy(), self.fs, **self.fcfg)
        if self.include_stats:
            f = np.concatenate([f, stat_features(xn).numpy()], 1)
        return self.scaler.transform(np.nan_to_num(f, posinf=0, neginf=0))

    @torch.no_grad()
    def forward(self, x):
        p = self.clf.predict_proba(self.features(x))
        P = np.full((len(p), self.K), 1e-6, dtype=np.float32)
        P[:, self.clf.classes_.astype(int)] = np.maximum(p, 1e-6)
        return torch.from_numpy(np.log(P / P.sum(1, keepdims=True))).to(x.device)


# ------------------------------------------------------------------ fitting
def _candidates(n_train: int, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    c = []
    for leaf in (1, 3):
        c.append((f"rf_leaf{leaf}", RandomForestClassifier(500, min_samples_leaf=leaf, n_jobs=-1, random_state=seed)))
    for lr in (0.05, 0.1):
        c.append((f"hgb_lr{lr}", HistGradientBoostingClassifier(learning_rate=lr, max_iter=400, early_stopping=True,
                                                                 random_state=seed)))
    for C in (0.1, 1.0, 10.0):
        c.append((f"logreg_C{C}", LogisticRegression(C=C, max_iter=3000)))
    if n_train <= 20000:
        for C in (1.0, 10.0, 100.0):
            c.append((f"svm_C{C}", SVC(C=C, gamma="scale", probability=True, random_state=seed)))
    return c


def fit_classical(cfg: Dict, data, device, seed: int, logger, K: int):
    from sklearn.preprocessing import StandardScaler

    from ..data.noise import TrainNoiseAugment
    from ..engine.evaluate import selection_score

    mcfg = cfg["model"]
    fcfg = {"nlevel": int(mcfg.get("nlevel", 5)), "f_max": float(mcfg.get("f_max", 1000.0)),
            "n_bins": int(mcfg.get("n_bins", 100))}
    include_stats = bool(mcfg.get("include_stats", True))
    n_aug = int(mcfg.get("n_aug", 2))
    max_train = int(mcfg.get("max_train_windows", 30000))
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    ds = data.train
    idx = np.arange(len(ds))
    if len(idx) > max_train:
        idx = np.sort(rng.choice(len(idx), max_train, replace=False))
    Xc = torch.from_numpy(np.stack([ds.get_window(int(i)) for i in idx], 0)).float()
    yc = np.asarray(ds.labels)[idx]
    aug = TrainNoiseAugment(cfg["train"].get("noise_aug", {}) or {}, data.ref_power)
    Xs, ys = [Xc], [yc]
    for _ in range(n_aug):
        Xs.append(aug(Xc) if aug.enabled else Xc); ys.append(yc)
    X = torch.cat(Xs); y = np.concatenate(ys)
    logger.info(f"[classical] features for {len(X)} training windows (n_aug={n_aug}, fs={data.fs}) ...")
    F = ses_features(X.numpy(), data.fs, **fcfg)
    if include_stats:
        F = np.concatenate([F, stat_features(X).numpy()], 1)
    F = np.nan_to_num(F, posinf=0, neginf=0)
    scaler = StandardScaler().fit(F)
    Fs = scaler.transform(F)
    logger.info(f"[classical] {Fs.shape[1]} features, {time.time() - t0:.0f}s")
    scfg = cfg.get("selection", {}) or {}
    hist, best = [], (-1.0, None, None)
    for name, clf in _candidates(len(Fs), seed):
        t1 = time.time()
        clf.fit(Fs, y)
        model = ClassicalModel(clf, scaler, K, data.fs, fcfg, include_stats)
        v = selection_score(model, data.val, torch.device("cpu"), K, scfg, data.ref_power)
        hist.append({"epoch": len(hist) + 1, "candidate": name, "val": v, "fit_s": time.time() - t1})
        logger.info(f"[classical] {name}: val {v:.4f} ({time.time() - t1:.0f}s)")
        if v > best[0]:
            best = (v, name, model)
    tr = {"history": hist, "best_val": best[0], "best_epoch": [h["candidate"] for h in hist].index(best[1]) + 1,
          "best_candidate": best[1], "train_time_s": time.time() - t0, "epochs_run": len(hist)}
    prof = {"params": 0, "macs": None, "n_features": int(Fs.shape[1]), "classifier": best[1]}
    return best[2], prof, tr
