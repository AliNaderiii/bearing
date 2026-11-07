"""PhyQ-TransNet = the author's own model, ported 1:1 from the original repository
(``Model/model.py :: QDCA_Hybrid_Pro`` + ``utils/Preprocess.py :: LearnedFilter / GPUFeatureExtractor``).

The architecture, layer sizes, kernel sizes, initialisations and the gated fusion are IDENTICAL to the
original code (for in_ch=3, 10 classes, 33 features: 1,404,715 trainable parameters, as in the paper).
Only the *plumbing* changed, so that the model runs inside the leakage-free pipeline:

* the 11 statistical descriptors per channel (original GPUFeatureExtractor) are computed INSIDE the
  model from the (noisy) input, and standardised with training statistics (BatchNorm without affine
  parameters = a StandardScaler fitted on training batches). Previously they were computed offline;
* the Neural-MED pre-filter (original LearnedFilter, kurtosis + reconstruction loss) is fitted on
  TRAINING windows only, then frozen and applied to every input - also to the noisy test signal
  (the original applied it before noise was added, so it never saw the test noise);
* input normalisation: per-window Instance Normalisation (Eq. 8 of the paper). ``input_norm: none``
  keeps only the pipeline's global train-fitted scaling (closest to the original min-max code);
* the positional embedding has L/16 tokens: exactly 128 for L=2048 (original), and it also works for
  longer windows (PU: 8192 -> 512 tokens).

Ablation switches (each removes ONE component of the full model): quad, attention, shrink,
transformer, stats, act, prefilter.enabled, input_norm.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ============================================================ original components (verbatim logic)
class MetaAconC(nn.Module):
    """Adaptive Context Activation (original implementation)."""

    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv1d(width, max(r, width // r), 1)
        self.fc2 = nn.Conv1d(max(r, width // r), width, 1)
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = x.mean(dim=2, keepdim=True)
        beta = self.sigmoid(self.fc2(self.fc1(context)))
        return (self.p1 * x - self.p2 * x) * self.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


def _act(kind: str, c: int) -> nn.Module:
    return MetaAconC(c) if kind == "metaacon" else nn.ReLU(inplace=False)


class CoordinateAttention(nn.Module):
    """Original 'CoordinateAttention' block (global pooling -> 1x1 conv -> BN -> MetaAconC -> sigmoid gate)."""

    def __init__(self, inp, oup, reduction=32, act="metaacon"):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, 1)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = _act(act, mip)
        self.conv_w = nn.Conv1d(mip, oup, 1)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(self.pool(x))))
        return x * self.conv_w(y).sigmoid()


class QuadraticConv1d(nn.Module):
    """y = (Wr*x + br)(Wg*x + bg) + Wb*x^2 + c   (original implementation and initialisation)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_b = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if bias:
            self.c = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("c", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
        if self.conv_r.bias is not None:
            init.zeros_(self.conv_r.bias)
        init.xavier_normal_(self.conv_g.weight)
        if self.conv_g.bias is not None:
            init.ones_(self.conv_g.bias)
        init.kaiming_uniform_(self.conv_b.weight, a=math.sqrt(5))
        if self.c is not None:
            init.zeros_(self.c)

    def forward(self, x):
        out = self.conv_r(x) * self.conv_g(x) + self.conv_b(x * x)
        return out + self.c.view(1, -1, 1) if self.c is not None else out


def _conv(quad: bool, cin, cout, k, s, p):
    """Quadratic conv (full model) or an ordinary linear conv of the same shape (w/o Quadratic ablation)."""
    return QuadraticConv1d(cin, cout, k, s, p) if quad else nn.Conv1d(cin, cout, k, s, p, bias=True)


class SoftThresholding(nn.Module):
    """Original learnable soft thresholding (DRSN-inspired)."""

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // 4), nn.ReLU(inplace=True),
                                nn.Linear(channels // 4, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return torch.sign(x) * F.relu(torch.abs(x) - y)


class QResBlock_Pro(nn.Module):
    def __init__(self, channels, quad=True, attention=True, shrink=True, act="metaacon"):
        super().__init__()
        self.conv1 = _conv(quad, channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = _act(act, channels)
        self.conv2 = _conv(quad, channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.attn = CoordinateAttention(channels, channels, act=act) if attention else nn.Identity()
        self.shrinkage = SoftThresholding(channels) if shrink else nn.Identity()
        self.act2 = _act(act, channels)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shrinkage(self.attn(out))
        return self.act2(out + x)


# ============================================================ original pre-processing modules
class LearnedFilter(nn.Module):
    """Neural MED pre-filter (original utils/Preprocess.py::LearnedFilter)."""

    def __init__(self, length=2048):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 7, padding=3, bias=False)
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)
        nn.init.dirac_(self.conv.weight)
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, 1.0)

    def forward(self, x):            # x: (B, L)
        xt = self.act(self.conv(x.unsqueeze(1))).squeeze(1)
        X = torch.fft.rfft(xt)
        g = torch.clamp(torch.sigmoid(self.gate(torch.abs(X))), min=0.01, max=0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1])


def stat_features_11(x: torch.Tensor, nperseg: int = 256) -> torch.Tensor:
    """Original GPUFeatureExtractor (11 descriptors per channel). x: (B, C, L) -> (B, 11*C)."""
    B, C, L = x.shape
    x = x.float().reshape(B * C, L)
    window = torch.hann_window(nperseg, device=x.device)
    hop = max(1, nperseg // 4)
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-12)
    peak = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
    crest = peak / (rms + 1e-9)
    mean_t = torch.mean(x, dim=1, keepdim=True)
    std_t = torch.std(x, dim=1, keepdim=True) + 1e-9
    skew_t = torch.mean((x - mean_t) ** 3, dim=1, keepdim=True) / (std_t ** 3 + 1e-12)
    kurt = torch.mean((x - mean_t) ** 4, dim=1, keepdim=True) / (std_t ** 4 + 1e-12)
    shape_factor = rms / (torch.mean(torch.abs(x), dim=1, keepdim=True) + 1e-9)
    frames = F.pad(x, (0, nperseg)).unfold(1, nperseg, hop) * window
    mag = torch.abs(torch.fft.rfft(frames, dim=2))
    spec_energy = torch.mean(torch.sum(mag ** 2, dim=2), dim=1, keepdim=True) / (L + 1e-12)
    h = torch.zeros(nperseg, device=x.device)
    h[0] = 1
    h[1:nperseg // 2] = 2
    env = torch.abs(torch.fft.ifft(torch.fft.fft(frames, dim=2) * h, dim=2))
    env_mean = torch.mean(torch.mean(env, dim=2), dim=1, keepdim=True)
    env_std = torch.mean(torch.std(env, dim=2), dim=1, keepdim=True)
    f = torch.cat([mean_t, std_t, rms, skew_t, kurt, peak, crest, shape_factor, spec_energy, env_mean, env_std], 1)
    f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f.reshape(B, C * 11)


N_STATS_PER_CHANNEL = 11


# ============================================================ the model
class PhyQTransNet(nn.Module):
    """QDCA_Hybrid_Pro (original) + leakage-free plumbing. See module docstring."""

    def __init__(self, in_ch: int, n_classes: int, cfg: Optional[Dict] = None, length: int = 2048):
        super().__init__()
        cfg = cfg or {}
        quad = bool(cfg.get("quad", True))
        att = bool(cfg.get("attention", True))
        shrink = bool(cfg.get("shrink", True))
        act = str(cfg.get("act", "metaacon"))
        self.use_tf = bool(cfg.get("transformer", True))
        self.use_stats = bool(cfg.get("stats", True))
        self.input_norm = str(cfg.get("input_norm", "instance"))
        pcfg = cfg.get("prefilter", {}) or {}
        self.pcfg = pcfg
        self.length = int(length)
        if length % 16:
            raise ValueError("PhyQTransNet needs a window length divisible by 16")

        # ---- frozen Neural-MED pre-filter (one per channel, fitted on TRAIN windows only)
        self.prefilter = None
        if pcfg.get("enabled", True):
            self.prefilter = nn.ModuleList([LearnedFilter(length) for _ in range(in_ch)])
            for p in self.prefilter.parameters():
                p.requires_grad_(False)
            self.register_buffer("prefilter_fitted", torch.zeros((), dtype=torch.bool))

        # ---- A. CNN backbone (original)
        self.stem = nn.Sequential(_conv(quad, in_ch, 32, 15, 2, 7), nn.BatchNorm1d(32), _act(act, 32),
                                  nn.MaxPool1d(2))
        self.branch_1 = nn.Sequential(_conv(quad, 32, 64, 31, 1, 15), nn.BatchNorm1d(64), _act(act, 64), nn.MaxPool1d(2))
        self.branch_2 = nn.Sequential(_conv(quad, 32, 64, 15, 1, 7), nn.BatchNorm1d(64), _act(act, 64), nn.MaxPool1d(2))
        self.branch_3 = nn.Sequential(_conv(quad, 32, 64, 7, 1, 3), nn.BatchNorm1d(64), _act(act, 64), nn.MaxPool1d(2))
        self.fusion_conv = nn.Conv1d(64, 128, 1)
        self.deep_stage = nn.Sequential(QResBlock_Pro(128, quad, att, shrink, act), nn.MaxPool1d(2),
                                        QResBlock_Pro(128, quad, att, shrink, act))

        # ---- B. Transformer (original: learned positional embedding, 2 layers, 4 heads, FF 512, dropout 0.2)
        if self.use_tf:
            self.pos_embedding = nn.Parameter(torch.randn(1, length // 16, 128))
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.2,
                                               activation="gelu", batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

        # ---- C. statistical (physics-informed) stream (original aux_net) + D. gated fusion
        if self.use_stats:
            feat_dim = N_STATS_PER_CHANNEL * in_ch
            self.stats_scaler = nn.BatchNorm1d(feat_dim, affine=False)   # train-statistics standardisation
            self.aux_net = nn.Sequential(nn.Linear(feat_dim, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.3),
                                         nn.Linear(64, 128), nn.SiLU())
            self.fusion_gate = nn.Sequential(nn.Linear(128 * 2, 1), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.3), nn.Linear(128, n_classes))
        self.streams = ["deep", "stats"] if self.use_stats else ["deep"]
        self.last_alpha = None
        self._init_weights()

    def _init_weights(self):        # original
        if self.use_tf:
            init.trunc_normal_(self.pos_embedding, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and not self._in_prefilter(m):
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _in_prefilter(self, mod) -> bool:
        return self.prefilter is not None and any(mod is q for q in self.prefilter.modules())

    # ------------------------------------------------------------ Neural MED fitting (train only)
    def fit_prefilter(self, train_windows: np.ndarray, seed: int = 0, device=None, logger=None):
        """Original train_filter(): loss = -a*kurtosis + b*MSE(out, in); Adam, few epochs, frozen afterwards.
        ``train_windows``: (N, C, L) clean TRAINING windows (val/test are never used)."""
        if self.prefilter is None:
            return
        device = device or torch.device("cpu")
        epochs = int(self.pcfg.get("epochs", 3))
        bs = int(self.pcfg.get("batch_size", 64))
        lr = float(self.pcfg.get("lr", 1e-3))
        ka = float(self.pcfg.get("kurt_alpha", 0.5))
        ra = float(self.pcfg.get("recon_alpha", 0.5))
        g = torch.Generator().manual_seed(int(seed))
        for c, lf in enumerate(self.prefilter):
            lf.to(device).train()
            for p in lf.parameters():
                p.requires_grad_(True)
            opt = torch.optim.Adam(lf.parameters(), lr=lr)
            data = torch.from_numpy(np.ascontiguousarray(train_windows[:, c, :])).float()
            for ep in range(epochs):
                perm = torch.randperm(len(data), generator=g)
                tot, n = 0.0, 0
                for i in range(0, len(data), bs):
                    xb = data[perm[i:i + bs]].to(device)
                    out = lf(xb)
                    d = out - out.mean(dim=1, keepdim=True)
                    kurt = torch.mean(d ** 4, dim=1) / (torch.mean(d ** 2, dim=1) ** 2 + 1e-6)
                    loss = -ka * kurt.mean() + ra * torch.mean((out - xb) ** 2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lf.parameters(), max_norm=5.0)
                    opt.step()
                    tot += float(loss.detach()) * len(xb)
                    n += len(xb)
                if logger:
                    logger.info(f"  [neural-MED] ch{c} ep {ep + 1}/{epochs} loss {tot / max(n, 1):.4f}")
            for p in lf.parameters():
                p.requires_grad_(False)
            lf.eval()
        self.prefilter_fitted.fill_(True)

    def _apply_prefilter(self, x):
        if self.prefilter is None or not bool(self.prefilter_fitted):
            return x                                      # not fitted yet (e.g. profiling) -> identity
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            return torch.stack([lf(xf[:, c]) for c, lf in enumerate(self.prefilter)], 1).to(x.dtype)

    # ------------------------------------------------------------ forward
    def forward(self, x):
        x = self._apply_prefilter(x)
        xs = x
        if self.input_norm == "instance":                 # Eq. (8)
            xs = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        h = self.stem(xs)
        fused = (self.branch_1(h) + self.branch_2(h) + self.branch_3(h)) / 3.0
        cnn_out = self.deep_stage(self.fusion_conv(fused))
        if self.use_tf:
            t = cnn_out.permute(0, 2, 1)
            t = t + self.pos_embedding[:, :t.size(1), :].to(t.dtype)
            deep_rep = self.transformer(t).mean(dim=1)
        else:
            deep_rep = cnn_out.mean(dim=2)
        if self.use_stats:
            with torch.autocast(device_type=x.device.type, enabled=False):
                feats = self.stats_scaler(stat_features_11(x.float()))
                aux_rep = self.aux_net(feats)
                deep32 = deep_rep.float()
                alpha = self.fusion_gate(torch.cat([deep32, aux_rep], dim=1))
                final = (1 - alpha) * deep32 + alpha * aux_rep
            self.last_alpha = alpha.detach()               # weight of the statistical stream
            return self.classifier(final)
        return self.classifier(deep_rep)
