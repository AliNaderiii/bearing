"""Phenomenological rolling-element-bearing vibration simulator with KNOWN ground truth.

Used only for the mechanism / interpretability study (MSSP Section "Why quadratic convolution
demodulates"), never for the benchmark numbers.

Model (McFadden & Smith, JSV 96 (1984) 69-82; Randall & Antoni, MSSP 25 (2011) 485-520):

    x(t) = sum_k A_k * h(t - t_k)                       fault impulses (localized defect)
         + r(t)                                         resonance-filtered random background
         + sum_h b_h cos(2*pi*h*f_r*t + phi_h)          shaft harmonics
         + n(t)                                         additive measurement noise

* t_k = k * T_f * (1 + eps_k), eps_k ~ N(0, slip^2)  (1-2 % random slip -> pseudo-cyclostationary)
* h(t) = exp(-zeta*w_n*t) sin(w_d*t) u(t)            structural resonance excited by each impact
* A_k : constant (outer race, stationary load zone); modulated by the shaft rotation
        (inner race) or by the cage rotation (rolling element).
* Every class (including "normal") contains the SAME resonance-filtered random background and
  shaft harmonics, and the fault impulses of all fault classes share the same amplitude
  distribution.  Hence the classes can only be told apart by the *repetition rate and modulation
  of the impulses*, i.e. by information that lives in the ENVELOPE spectrum
  (at BPFO, BPFI +/- f_r, 2*BSF +/- FTF), not in the raw spectrum.

Geometry defaults: SKF 6205-2RS (CWRU drive end): 9 balls, d = 7.94 mm, D = 39.04 mm, alpha = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("normal", "outer", "inner", "ball")


@dataclass
class BearingGeometry:
    n_balls: int = 9
    d_ball: float = 7.94e-3
    d_pitch: float = 39.04e-3
    contact_deg: float = 0.0

    def orders(self) -> Dict[str, float]:
        """Characteristic fault frequencies as multiples (orders) of the shaft frequency."""
        r = self.d_ball / self.d_pitch * np.cos(np.deg2rad(self.contact_deg))
        n = self.n_balls
        return {
            "BPFO": n / 2 * (1 - r),
            "BPFI": n / 2 * (1 + r),
            "FTF": 0.5 * (1 - r),
            "BSF": self.d_pitch / (2 * self.d_ball) * (1 - r ** 2),
        }


@dataclass
class SimConfig:
    fs: float = 12_000.0
    length: int = 2048
    rpm_range: Tuple[float, float] = (1700.0, 1800.0)
    fn_range: Tuple[float, float] = (2_500.0, 4_000.0)   # resonance frequency (Hz)
    zeta_range: Tuple[float, float] = (0.04, 0.10)        # damping ratio
    slip: float = 0.01                                     # random slip (std of period fraction)
    impulse_amp_range: Tuple[float, float] = (0.6, 1.4)
    inner_mod_depth: Tuple[float, float] = (0.5, 1.0)
    ball_mod_depth: Tuple[float, float] = (0.5, 1.0)
    background_rel: Tuple[float, float] = (0.3, 0.6)      # rms of resonant random background / impulse rms
    shaft_rel: Tuple[float, float] = (0.2, 0.8)           # amplitude of shaft harmonics / impulse rms
    geometry: BearingGeometry = field(default_factory=BearingGeometry)


def _resonance_kernel(fs: float, fn: float, zeta: float, dur: float = 0.004) -> np.ndarray:
    t = np.arange(int(dur * fs)) / fs
    wn = 2 * np.pi * fn
    wd = wn * np.sqrt(1 - zeta ** 2)
    return np.exp(-zeta * wn * t) * np.sin(wd * t)


def simulate_one(cls: str, rng: np.random.Generator, cfg: SimConfig = SimConfig(),
                 rpm: Optional[float] = None, fn: Optional[float] = None,
                 return_info: bool = False):
    """Return one clean window (float32, length cfg.length) of class ``cls``."""
    if cls not in CLASSES:
        raise ValueError(cls)
    fs, L = cfg.fs, cfg.length
    rpm = float(rpm if rpm is not None else rng.uniform(*cfg.rpm_range))
    fr = rpm / 60.0
    fn = float(fn if fn is not None else rng.uniform(*cfg.fn_range))
    zeta = float(rng.uniform(*cfg.zeta_range))
    kern = _resonance_kernel(fs, fn, zeta)
    pad = len(kern)
    n_tot = L + 2 * pad
    t = np.arange(n_tot) / fs
    o = cfg.geometry.orders()

    # ---- fault impulse train
    imp = np.zeros(n_tot)
    f_fault = {"outer": o["BPFO"], "inner": o["BPFI"], "ball": 2 * o["BSF"]}.get(cls)
    if f_fault is not None:
        T = 1.0 / (f_fault * fr)
        tk = rng.uniform(0, T)
        phase_mod = rng.uniform(0, 2 * np.pi)
        a0 = rng.uniform(*cfg.impulse_amp_range)
        while tk < t[-1]:
            if cls == "inner":
                m = rng.uniform(*cfg.inner_mod_depth)
                amp = a0 * (1 - m / 2 + m / 2 * np.cos(2 * np.pi * fr * tk + phase_mod))
            elif cls == "ball":
                m = rng.uniform(*cfg.ball_mod_depth)
                amp = a0 * (1 - m / 2 + m / 2 * np.cos(2 * np.pi * o["FTF"] * fr * tk + phase_mod))
            else:
                amp = a0 * (1 + 0.05 * rng.standard_normal())
            idx = int(round(tk * fs))
            if idx < n_tot:
                imp[idx] += amp
            tk += T * (1 + cfg.slip * rng.standard_normal())
    fault = np.convolve(imp, kern)[:n_tot]
    ref_rms = 0.15   # typical rms of the impulse response train at a0=1 (keeps all classes on one scale)

    # ---- resonance-filtered random background (present in ALL classes, incl. normal)
    bg = np.convolve(rng.standard_normal(n_tot), kern)[:n_tot]
    bg *= rng.uniform(*cfg.background_rel) * ref_rms / (bg.std() + 1e-12)

    # ---- shaft harmonics (1x, 2x, 3x)
    shaft = np.zeros(n_tot)
    for h in (1, 2, 3):
        shaft += rng.uniform(*cfg.shaft_rel) * ref_rms / h * np.cos(2 * np.pi * h * fr * t + rng.uniform(0, 2 * np.pi))

    x = (fault + bg + shaft)[pad:pad + L].astype(np.float32)
    if return_info:
        info = {"rpm": rpm, "fr": fr, "fn": fn, "zeta": zeta,
                "f_fault": None if f_fault is None else f_fault * fr, "orders": o}
        return x, info
    return x


def make_dataset(n_per_class: int, seed: int, cfg: SimConfig = SimConfig(),
                 classes: Sequence[str] = CLASSES, **kw) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(classes):
        for _ in range(n_per_class):
            X.append(simulate_one(c, rng, cfg, **kw))
            y.append(ci)
    X = np.stack(X)[:, None, :]
    y = np.asarray(y, dtype=np.int64)
    p = rng.permutation(len(y))
    return X[p], y[p]


# ------------------------------------------------------------------ noise models
def add_white(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Relative SNR w.r.t. each window's own power."""
    p = np.mean(x ** 2, axis=-1, keepdims=True)
    n = rng.standard_normal(x.shape) * np.sqrt(p / 10 ** (snr_db / 10))
    return (x + n).astype(np.float32)


def add_impulsive(x: np.ndarray, rng: np.random.Generator, rate: float = 2e-3,
                  amp_rel: Tuple[float, float] = (5.0, 15.0), fs: float = 12_000.0) -> np.ndarray:
    """Non-fault random impacts (Bernoulli process, broadband exponential decay, random polarity):
    a stand-in for e.g. gear-mesh shocks, loose parts or electrical spikes."""
    out = x.copy()
    std = x.std(axis=-1, keepdims=True)
    B, C, L = x.shape
    decay = np.exp(-np.arange(int(0.0015 * fs)) / (0.0003 * fs))
    for b in range(B):
        for c in range(C):
            spikes = np.zeros(L)
            hits = rng.random(L) < rate
            spikes[hits] = rng.uniform(*amp_rel, size=hits.sum()) * rng.choice([-1, 1], size=hits.sum())
            spikes = np.convolve(spikes, decay * rng.standard_normal(len(decay)))[:L]
            out[b, c] += (spikes * std[b, c]).astype(np.float32)
    return out
