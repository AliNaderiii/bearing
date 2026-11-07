"""Differentiable, on-the-fly statistical features (computed from the *noisy* input).

Computing features inside the model (instead of offline on clean data) removes
the train/test mismatch of the legacy pipeline and guarantees that the stats
branch sees exactly the same corrupted signal as the CNN branch.

Per channel (12):
  amplitude-dependent : log RMS, log std, log peak, log peak-to-peak
  dimensionless       : skewness, log kurtosis, crest, shape, impulse, margin factor
  spectral            : normalised spectral centroid, normalised spectral entropy
With ``amplitude=False`` only the 8 scale-invariant features are returned.
"""
from __future__ import annotations

import math

import torch

AMP_NAMES = ["log_rms", "log_std", "log_peak", "log_p2p"]
INV_NAMES = ["skew", "log_kurt", "crest", "shape", "impulse", "margin", "spec_centroid", "spec_entropy"]


def n_features(amplitude: bool = True) -> int:
    return len(INV_NAMES) + (len(AMP_NAMES) if amplitude else 0)


def stat_features(x: torch.Tensor, amplitude: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """x: (B, C, L) -> (B, C * F). Always computed in float32."""
    x = x.float()
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = xc.pow(2).mean(-1)
    std = (var + eps).sqrt()
    rms = (x.pow(2).mean(-1) + eps).sqrt()
    absx = x.abs()
    peak = absx.amax(-1)
    p2p = x.amax(-1) - x.amin(-1)
    mabs = absx.mean(-1) + eps
    skew = xc.pow(3).mean(-1) / std.pow(3)
    kurt = xc.pow(4).mean(-1) / (var + eps).pow(2)
    crest = peak / rms
    shape = rms / mabs
    impulse = peak / mabs
    margin = peak / ((absx + eps).sqrt().mean(-1).pow(2))
    P = torch.fft.rfft(xc, dim=-1).abs().pow(2)
    P = P / (P.sum(-1, keepdim=True) + eps)
    f = torch.linspace(0, 1, P.shape[-1], device=x.device)
    centroid = (P * f).sum(-1)
    entropy = -(P * (P + eps).log()).sum(-1) / math.log(P.shape[-1])
    inv = [skew, torch.log(kurt + eps), torch.log(crest), torch.log(shape), torch.log(impulse),
           torch.log(margin), centroid, entropy]
    feats = ([torch.log(rms), torch.log(std), torch.log(peak + eps), torch.log(p2p + eps)] if amplitude else []) + inv
    F = torch.stack(feats, -1)  # (B, C, F)
    return torch.nan_to_num(F.flatten(1), nan=0.0, posinf=0.0, neginf=0.0)
