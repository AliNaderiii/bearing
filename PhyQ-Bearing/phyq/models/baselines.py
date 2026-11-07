"""Re-implemented baselines (all trained with the SAME pipeline, data and budget).

* WDCNN      : Zhang et al., Sensors 2017 (wide first-layer kernel CNN)
* QCNN       : WDCNN with quadratic neurons (Liao et al., MSSP 2023)
* DRSN-CW    : Zhao et al., IEEE TII 2020 (deep residual shrinkage network)
* CNNTransformer : convolutional tokeniser + Transformer encoder
* StatsMLP   : 12 hand-crafted statistics per channel + MLP (shortcut detector)
"""
from __future__ import annotations

import torch.nn as nn

from ..features.stats import n_features, stat_features
from .layers import PositionalEncoding, ResBlock, make_conv


def _inorm(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)


class WDCNN(nn.Module):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048, conv="linear"):
        super().__init__()
        cfg = cfg or {}
        conv = cfg.get("conv", conv)

        def blk(ci, co, k, s, p):
            return nn.Sequential(make_conv(conv, ci, co, k, s, p), nn.BatchNorm1d(co), nn.ReLU(inplace=True),
                                 nn.MaxPool1d(2, 2))
        self.f = nn.Sequential(blk(in_ch, 16, 64, 16, 24), blk(16, 32, 3, 1, 1), blk(32, 64, 3, 1, 1),
                               blk(64, 64, 3, 1, 1), blk(64, 64, 3, 1, 0), nn.AdaptiveAvgPool1d(1))
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(64, 100), nn.BatchNorm1d(100), nn.ReLU(inplace=True),
                               nn.Linear(100, n_classes))
        self.inorm = bool(cfg.get("instance_norm", True))

    def forward(self, x):
        return self.c(self.f(_inorm(x) if self.inorm else x))


class QCNN(WDCNN):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048):
        cfg = dict(cfg or {})
        cfg.setdefault("conv", "quad")
        super().__init__(in_ch, n_classes, cfg, length)


class DRSN(nn.Module):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048):
        super().__init__()
        cfg = cfg or {}
        c = int(cfg.get("base_channels", 16))
        self.stem = nn.Sequential(nn.Conv1d(in_ch, c, 15, 2, 7, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        layers, cin = [], c
        for m in (1, 2, 4):
            for j in range(int(cfg.get("blocks_per_stage", 2))):
                layers.append(ResBlock(cin, c * m, 2 if j == 0 else 1, "linear", "relu", "none", True, 3))
                cin = c * m
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(_inorm(x))))


class CNNTransformer(nn.Module):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048):
        super().__init__()
        cfg = cfg or {}
        d = int(cfg.get("d_model", 64))
        self.tok = nn.Sequential(
            nn.Conv1d(in_ch, 32, 64, 8, 28, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, d, 7, 2, 3, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True),
            nn.Conv1d(d, d, 5, 2, 2, bias=False), nn.BatchNorm1d(d), nn.ReLU(inplace=True))
        self.pos = PositionalEncoding(d, 1024, "learned")
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, int(cfg.get("layers", 2)), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        t = self.pos(self.tok(_inorm(x)).transpose(1, 2))
        return self.head(self.tf(t).mean(1))


class StatsMLP(nn.Module):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048):
        super().__init__()
        cfg = cfg or {}
        self.amp = bool(cfg.get("amplitude", True))
        nf = n_features(self.amp) * in_ch
        h = int(cfg.get("hidden", 128))
        self.bn = nn.BatchNorm1d(nf, affine=False)
        self.mlp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, n_classes))

    def forward(self, x):
        return self.mlp(self.bn(stat_features(x, self.amp)))


# ----------------------------------------------------------------------------------------------
# ClassBD (Liao et al., MSSP 222 (2025) 111750) -- clean re-implementation, NOT a copy of the
# official (unlicensed) code https://github.com/asdvfghg/ClassBD (commit on 'master', 2025-03).
#   time filter : QuadConv(in,16,k63,'same')-BN-ReLU-QuadConv(16,1,k63)-BN-ReLU-Sigmoid
#   freq filter : |FFT| -> Linear(L,L) -> |IFFT|
#   classifier  : BDWDCNN backbone (Conv(1,16,64,s8,p28) + 5 conv3 blocks, FC 100, dropout 0.5)
# Physics-informed loss: -kurtosis(time-filter output) and the l2/l4 sparsity of the envelope
# spectrum, with the same fixed relative weights as the official training script.  NOTE: in the official code both terms are wrapped
# in nn.Parameter(...) which DETACHES them, so effectively only cross-entropy trains the network.
#   cfg physics_loss: false  -> reproduces the official effective behaviour (CE only)
#   cfg physics_loss: true   -> the method as described in the paper (differentiable terms)
# The trainer adds `model.aux_loss` (if not None) to the CE loss in training mode.
# ----------------------------------------------------------------------------------------------
import math as _math

import torch as _torch
import torch.nn.functional as _F


class _QuadConvLiao(nn.Module):
    """Quadratic convolution with the initialisation of Liao et al. (w_g = w_b = 0, b_g = 1)."""

    def __init__(self, ci, co, k):
        super().__init__()
        self.r = nn.Conv1d(ci, co, k, padding="same")
        self.g = nn.Conv1d(ci, co, k, padding="same")
        self.b = nn.Conv1d(ci, co, k, padding="same")
        nn.init.normal_(self.r.weight, 0.0, _math.sqrt(0.25 / (ci * k)) * 8)
        nn.init.zeros_(self.g.weight); nn.init.ones_(self.g.bias)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return self.r(x) * self.g(x) + self.b(x * x)


def _kurtosis(y, trim=32):
    y = y.flatten(1)[:, trim:-trim]
    y = y - y.mean(-1, keepdim=True)
    m2 = y.pow(2).mean(-1)
    return (y.pow(4).mean(-1) / (m2 * m2 + 1e-12)).mean()


def _l2_l4_envelope(y):
    y = y.flatten(1)
    env = _torch.abs(_torch.fft.ifft(_torch.fft.fft(y) * _hilbert_mask(y.shape[-1], y.device)))
    es = _torch.abs(_torch.fft.rfft(env - env.mean(-1, keepdim=True)))
    return ((es.norm(2, dim=-1) / (es.norm(4, dim=-1) + 1e-12)) ** 2).mean()


def _hilbert_mask(n, device):
    h = _torch.zeros(n, device=device)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1; h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return h


class ClassBD(nn.Module):
    def __init__(self, in_ch, n_classes, cfg=None, length=2048):
        super().__init__()
        cfg = cfg or {}
        k = int(cfg.get("bd_kernel", 63))
        self.inorm = bool(cfg.get("instance_norm", True))
        self.physics = bool(cfg.get("physics_loss", False))
        self.time_filter = nn.Sequential(
            _QuadConvLiao(in_ch, 16, k), nn.BatchNorm1d(16), nn.ReLU(),
            _QuadConvLiao(16, 1, k), nn.BatchNorm1d(1), nn.ReLU(), nn.Sigmoid())
        self.freq_filter = nn.Linear(length, length)
        f = nn.Sequential()
        f.add_module("c1", nn.Conv1d(1, 16, 64, 8, 28)); f.add_module("bn1", nn.BatchNorm1d(16))
        f.add_module("r1", nn.ReLU()); f.add_module("p1", nn.MaxPool1d(2, 2))
        for i, (ci, co, p) in enumerate([(16, 32, 1), (32, 64, 1), (64, 64, 1), (64, 64, 1), (64, 64, 0)], 2):
            f.add_module(f"c{i}", nn.Conv1d(ci, co, 3, 1, p)); f.add_module(f"bn{i}", nn.BatchNorm1d(co))
            f.add_module(f"r{i}", nn.ReLU()); f.add_module(f"p{i}", nn.MaxPool1d(2, 2))
        self.cnn = f
        with _torch.no_grad():
            n_flat = self.cnn(_torch.zeros(1, 1, length)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 100), nn.ReLU(), nn.Dropout(0.5),
                                  nn.Linear(100, n_classes))
        # Official train_local.py re-creates its 'uncertainty weights' at every step (fixed log-scale -0.5),
        # i.e. CE, -kurtosis and l2/l4 get EQUAL fixed weights. A learnable UW would diverge here because
        # -kurtosis is negative and unbounded. -> fixed relative weight (1.0 = official ratio).
        self.phys_w = float(cfg.get("physics_weight", 1.0))
        self.aux_loss = None

    def forward(self, x):
        if self.inorm:
            x = _inorm(x)
        a1 = self.time_filter(x)
        with _torch.autocast(device_type=x.device.type, enabled=False):   # FFTs in float32
            a1 = a1.float()
            spec = _torch.abs(_torch.fft.fft(a1 - a1.mean(-1, keepdim=True)))
            es = self.freq_filter(spec)
            a2 = _torch.abs(_torch.fft.ifft(es))
            if self.physics and self.training:
                self.aux_loss = self.phys_w * (_l2_l4_envelope(a2) - _kurtosis(a1))
            else:
                self.aux_loss = None
        return self.head(self.cnn(a2))
