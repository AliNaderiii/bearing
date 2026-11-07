"""Building blocks: quadratic convolutions, MetaACON, attention, soft shrinkage, envelope spectrum, MED."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- convolutions
class QuadConv1d(nn.Module):
    """Quadratic neuron (Fan et al., IEEE TMI 2018; Liao et al., MSSP 2023):

        y = (W_r * x + b_r) ⊙ (W_g * x + b_g) + W_b * (x ⊙ x) + c

    Initialised so that the layer starts as an ordinary linear convolution
    (W_g = 0, b_g = 1, W_b = 0), the standard trick for stable training.
    ``gated=True`` replaces the product branch by a sigmoid gate (bounded variant).
    """

    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, bias=True, gated=False):
        super().__init__()
        self.gated = gated
        self.r = nn.Conv1d(cin, cout, k, stride, padding, groups=groups, bias=bias)
        self.g = nn.Conv1d(cin, cout, k, stride, padding, groups=groups, bias=True)
        self.b = nn.Conv1d(cin, cout, k, stride, padding, groups=groups, bias=False)
        nn.init.zeros_(self.g.weight)
        nn.init.constant_(self.g.bias, 0.0 if gated else 1.0)
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        if self.gated:
            return self.r(x) * (2 * torch.sigmoid(self.g(x))) + self.b(x * x)
        return self.r(x) * self.g(x) + self.b(x * x)


def make_conv(kind: str, cin, cout, k, stride=1, padding=None, groups=1, bias=False):
    if padding is None:
        padding = k // 2
    if kind == "quad":
        return QuadConv1d(cin, cout, k, stride, padding, groups, bias)
    if kind == "quad_gated":
        return QuadConv1d(cin, cout, k, stride, padding, groups, bias, gated=True)
    if kind == "linear":
        return nn.Conv1d(cin, cout, k, stride, padding, groups=groups, bias=bias)
    raise ValueError(f"Unknown conv kind '{kind}'")


# --------------------------------------------------------------------------- activations
class MetaACON(nn.Module):
    """Meta-ACON-C (Ma et al., CVPR 2021) for 1-D feature maps."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 4)
        self.p1 = nn.Parameter(torch.randn(1, c, 1) * 0.1 + 1.0)
        self.p2 = nn.Parameter(torch.zeros(1, c, 1))
        self.fc1 = nn.Conv1d(c, h, 1)
        self.bn = nn.BatchNorm1d(h)
        self.fc2 = nn.Conv1d(h, c, 1)

    def forward(self, x):
        beta = torch.sigmoid(self.fc2(self.bn(self.fc1(x.mean(-1, keepdim=True)))))
        dp = (self.p1 - self.p2) * x
        return dp * torch.sigmoid(beta * dp) + self.p2 * x


def make_act(kind: str, c: int) -> nn.Module:
    if kind == "metaacon":
        return MetaACON(c)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    if kind == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation '{kind}'")


# --------------------------------------------------------------------------- attention / shrinkage
class SE(nn.Module):
    def __init__(self, c, r=8, temporal=False):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c))
        self.temporal = temporal
        if temporal:
            self.tconv = nn.Conv1d(2, 1, 7, padding=3)

    def forward(self, x):
        w = torch.sigmoid(self.fc(x.mean(-1))).unsqueeze(-1)
        x = x * w
        if self.temporal:
            s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
            x = x * torch.sigmoid(self.tconv(s))
        return x


class SoftShrink(nn.Module):
    """Channel-wise learned soft thresholding (DRSN-CW, Zhao et al., IEEE TII 2020)."""

    def __init__(self, c, r=4):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Linear(h, c))

    def forward(self, x):
        a = x.abs().mean(-1)
        tau = (a * torch.sigmoid(self.fc(a))).unsqueeze(-1)
        return torch.sign(x) * F.relu(x.abs() - tau)


# --------------------------------------------------------------------------- residual block
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride, conv="quad", act="metaacon", attention="se", shrink=False, k=7):
        super().__init__()
        self.c1 = make_conv(conv, cin, cout, k, stride)
        self.b1 = nn.BatchNorm1d(cout)
        self.a1 = make_act(act, cout)
        self.c2 = make_conv(conv, cout, cout, k, 1)
        self.b2 = nn.BatchNorm1d(cout)
        self.shrink = SoftShrink(cout) if shrink else nn.Identity()
        self.att = SE(cout, temporal=(attention == "se_temporal")) if attention in ("se", "se_temporal") else nn.Identity()
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        self.a2 = make_act(act, cout)

    def forward(self, x):
        y = self.a1(self.b1(self.c1(x)))
        y = self.att(self.shrink(self.b2(self.c2(y))))
        return self.a2(y + self.skip(x))


# --------------------------------------------------------------------------- positional encodings
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=1024, kind="sin"):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pe = nn.Parameter(torch.zeros(1, max_len, d))
            nn.init.trunc_normal_(self.pe, std=0.02)
        elif kind == "sin":
            pos = torch.arange(max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(1, max_len, d)
            pe[0, :, 0::2] = torch.sin(pos * div)
            pe[0, :, 1::2] = torch.cos(pos * div)[:, : d // 2]
            self.register_buffer("pe", pe)

    def forward(self, x):
        if self.kind == "none":
            return x
        return x + self.pe[:, : x.shape[1]].to(x.dtype)


# --------------------------------------------------------------------------- envelope spectrum
class EnvelopeSpectrum(nn.Module):
    """Physics-informed branch: learnable band-pass bank -> squared envelope -> envelope spectrum.

    Bearing faults produce periodic impacts that modulate structural resonances;
    the squared-envelope spectrum of a well-chosen band exposes the fault
    characteristic frequencies (BPFO/BPFI/BSF). Filters are initialised as
    Gabor band-passes on a log-frequency grid and fine-tuned end-to-end.
    """

    def __init__(self, cin, n_filters=8, k=63, bins=128, out_dim=64, decim=4):
        super().__init__()
        self.bank = nn.Conv1d(cin, n_filters * cin, k, padding=k // 2, groups=cin, bias=False)
        t = torch.arange(k) - k // 2
        fcs = torch.logspace(math.log10(0.03), math.log10(0.45), n_filters)  # normalised freq (fs=1)
        w = []
        for fc in fcs:
            bw = fc / 2
            g = torch.exp(-0.5 * (t * bw * 2.5) ** 2) * torch.cos(2 * math.pi * fc * t)
            w.append(g / g.norm())
        self.bank.weight.data.copy_(torch.stack(w).unsqueeze(1).repeat(cin, 1, 1))
        self.bins, self.decim = bins, decim
        nb = n_filters * cin
        self.head = nn.Sequential(
            nn.Conv1d(nb, out_dim, 5, padding=2), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
            nn.Conv1d(out_dim, out_dim, 5, padding=2), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        y = self.bank(x).float().pow(2)                       # FFT in fp32 (AMP-safe, any length)
        y = F.avg_pool1d(y, self.decim * 2, self.decim, padding=self.decim // 2) if self.decim > 1 else y
        y = y - y.mean(-1, keepdim=True)
        S = torch.fft.rfft(y, dim=-1).abs()[..., 1: self.bins + 1]
        S = torch.log1p(S / (S.mean(-1, keepdim=True) + 1e-6))
        return self.head(S).flatten(1)


# --------------------------------------------------------------------------- MED pre-filter
def fit_med(x, length=32, iters=30):
    """Wiggins minimum-entropy deconvolution (kurtosis maximisation). x: (N, L) numpy."""
    import numpy as np
    from scipy.linalg import solve, toeplitz
    x = np.asarray(x, dtype=np.float64)
    R = np.zeros((length, length))
    for s in x:
        ac = np.correlate(s, s, "full")[len(s) - 1: len(s) - 1 + length]
        R += toeplitz(ac)
    R += 1e-6 * np.trace(R) / length * np.eye(length)
    f = np.zeros(length)
    f[length // 2] = 1.0
    for _ in range(iters):
        b = np.zeros(length)
        num = 0.0
        den = 0.0
        for s in x:
            y = np.convolve(s, f, "full")[: len(s)]
            y3 = y ** 3
            for l in range(length):
                b[l] += np.dot(y3[l:], s[: len(s) - l])
            num += np.sum(y ** 2)
            den += np.sum(y ** 4)
        f = solve(R, b) * (num / max(den, 1e-12))
        f = f / (np.linalg.norm(f) + 1e-12)
    return f.astype(np.float32)


class FixedFIR(nn.Module):
    def __init__(self, cin, length=32):
        super().__init__()
        w = torch.zeros(cin, 1, length)
        w[:, 0, length // 2] = 1.0
        self.register_buffer("w", w)
        self.length = length

    def set_filters(self, filters):
        f = torch.as_tensor(filters, dtype=torch.float32)
        if f.ndim == 1:
            f = f.unsqueeze(0).repeat(self.w.shape[0], 1)
        self.w.copy_(f.flip(-1).unsqueeze(1))   # conv1d is correlation -> flip

    def forward(self, x):
        L = self.length
        return F.conv1d(F.pad(x, (L - 1, 0)), self.w.to(x.dtype), groups=x.shape[1])
