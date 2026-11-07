"""PhyQNet: configurable physics-informed quadratic network.

Streams (each optional, fused by a learned softmax gate):
  1. CNN stream   : [MED] -> instance standardisation -> stem -> multi-scale block
                    -> quadratic residual stages -> [Transformer] -> pooled vector
  2. Stats stream : 12 statistical features per channel computed on the raw (noisy)
                    input -> BatchNorm (learned scaler, train statistics only) -> MLP
  3. Envelope     : learnable band-pass bank -> squared-envelope spectrum -> CNN
Every component can be switched off from the config, which is what the ablation
study and the validation-based architecture search use.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..features.stats import n_features, stat_features
from .layers import EnvelopeSpectrum, FixedFIR, PositionalEncoding, ResBlock, make_act, make_conv


class MultiScale(nn.Module):
    def __init__(self, cin, cout, kernels, conv, act):
        super().__init__()
        self.br = nn.ModuleList([nn.Sequential(make_conv(conv, cin, cout, k, 1), nn.BatchNorm1d(cout))
                                 for k in kernels])
        self.fuse = nn.Sequential(nn.Conv1d(cout * len(kernels), cout, 1, 2, bias=False),
                                  nn.BatchNorm1d(cout), make_act(act, cout))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.br], 1))


class PhyQNet(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, cfg: Dict, length: int = 2048):
        super().__init__()
        w = float(cfg.get("width", 1.0))
        c0 = max(8, int(round(cfg.get("base_channels", 32) * w)))
        conv = cfg.get("conv", "quad")
        scope = cfg.get("quad_scope", "all")
        act = cfg.get("act", "metaacon")
        att = cfg.get("attention", "se")
        shrink = bool(cfg.get("shrink", False))
        stage_conv = conv if scope == "all" else "linear"
        stem_conv = conv if scope in ("all", "stem") else "linear"
        d = int(cfg.get("fusion_dim", 128))
        self.instance_norm = bool(cfg.get("instance_norm", True))
        self.streams: List[str] = []

        mcfg = cfg.get("med", {}) or {}
        self.med = FixedFIR(in_ch, int(mcfg.get("length", 32))) if mcfg.get("enabled", False) else None

        # ---------------- CNN stream
        ccfg = cfg.get("cnn", {}) or {}
        self.use_cnn = bool(ccfg.get("enabled", True))
        if self.use_cnn:
            ks = int(cfg.get("stem_kernel", 63))
            self.stem = nn.Sequential(make_conv(stem_conv, in_ch, c0, ks, int(cfg.get("stem_stride", 4))),
                                      nn.BatchNorm1d(c0), make_act(act, c0))
            self.ms = MultiScale(c0, c0, cfg.get("multiscale"), stem_conv, act) \
                if cfg.get("multiscale") else nn.Identity()
            chans = [c0 * m for m in cfg.get("stage_mult", [1, 2, 4])]
            blocks = cfg.get("blocks", [1, 1, 1])
            layers, cin = [], c0
            for c, nb in zip(chans, blocks):
                for j in range(nb):
                    layers.append(ResBlock(cin, c, 2 if j == 0 else 1, stage_conv, act, att, shrink,
                                           int(cfg.get("block_kernel", 7))))
                    cin = c
            self.stages = nn.Sequential(*layers)
            tcfg = cfg.get("transformer", {}) or {}
            self.use_tf = bool(tcfg.get("enabled", True))
            if self.use_tf:
                dm = int(tcfg.get("d_model", cin))
                self.tf_in = nn.Linear(cin, dm) if dm != cin else nn.Identity()
                self.pos = PositionalEncoding(dm, 1024, tcfg.get("pos", "learned"))
                layer = nn.TransformerEncoderLayer(dm, int(tcfg.get("heads", 4)), int(dm * tcfg.get("ff_mult", 2)),
                                                   float(tcfg.get("dropout", 0.1)), batch_first=True,
                                                   norm_first=True, activation="gelu")
                self.tf = nn.TransformerEncoder(layer, int(tcfg.get("layers", 2)), enable_nested_tensor=False)
                cin = dm
            self.cnn_proj = nn.Sequential(nn.Linear(cin, d), nn.LayerNorm(d))
            self.streams.append("cnn")

        # ---------------- Stats stream
        scfg = cfg.get("stats", {}) or {}
        self.use_stats = bool(scfg.get("enabled", True))
        if self.use_stats:
            self.stats_amp = bool(scfg.get("amplitude", True))
            nf = n_features(self.stats_amp) * in_ch
            self.stats_bn = nn.BatchNorm1d(nf, affine=False)
            self.stats_mlp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, d),
                                           nn.LayerNorm(d))
            self.streams.append("stats")

        # ---------------- Envelope stream
        ecfg = cfg.get("envelope", {}) or {}
        self.use_env = bool(ecfg.get("enabled", False))
        if self.use_env:
            self.env = EnvelopeSpectrum(in_ch, int(ecfg.get("n_filters", 8)), int(ecfg.get("kernel", 63)),
                                        int(ecfg.get("bins", 128)), int(ecfg.get("dim", 64)),
                                        int(ecfg.get("decim", 4)))
            self.env_proj = nn.Sequential(nn.Linear(int(ecfg.get("dim", 64)), d), nn.LayerNorm(d))
            self.streams.append("env")

        if not self.streams:
            raise ValueError("PhyQNet needs at least one stream")
        self.fusion = cfg.get("fusion", "gate")
        S = len(self.streams)
        if self.fusion == "gate" and S > 1:
            self.gate = nn.Sequential(nn.Linear(d * S, d), nn.GELU(), nn.Linear(d, S))
            head_in = d
        elif self.fusion == "concat":
            head_in = d * S
        else:
            head_in = d
        self.head = nn.Sequential(nn.Dropout(float(cfg.get("dropout", 0.2))), nn.Linear(head_in, n_classes))
        self.last_alpha = None

    def set_med(self, filters):
        if self.med is not None:
            self.med.set_filters(filters)

    def _cnn(self, x):
        z = self.stages(self.ms(self.stem(x)))
        if self.use_tf:
            t = self.pos(self.tf_in(z.transpose(1, 2)))
            z = self.tf(t).mean(1)
        else:
            z = z.mean(-1)
        return self.cnn_proj(z)

    def forward(self, x):
        if self.med is not None:
            x = self.med(x)
        feats = []                                   # order: cnn, stats, env (== self.streams)
        xs = x
        if self.instance_norm:
            xs = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
        if self.use_cnn:
            feats.append(self._cnn(xs))
        if self.use_stats:
            feats.append(self.stats_mlp(self.stats_bn(stat_features(x, self.stats_amp))))
        if self.use_env:
            feats.append(self.env_proj(self.env(xs)))
        if len(feats) == 1:
            z = feats[0]
        elif self.fusion == "gate":
            feats = [f.float() for f in feats]
            a = torch.softmax(self.gate(torch.cat(feats, 1)).float(), -1)
            self.last_alpha = a.detach()
            z = sum(a[:, i: i + 1] * f for i, f in enumerate(feats))
        elif self.fusion == "concat":
            z = torch.cat([f.float() for f in feats], 1)
        else:
            z = sum(f.float() for f in feats) / len(feats)
        return self.head(z)
