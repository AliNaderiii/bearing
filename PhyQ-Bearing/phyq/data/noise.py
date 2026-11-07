"""Additive noise models operating on torch batches ``(B, C, L)``.

SNR definitions
---------------
relative : P_noise = P_signal(window, channel) / 10^(SNR/10)
           (common in the literature; noise scales with each window -> the
           amplitude of the noisy window still encodes the clean amplitude).
absolute : P_noise = P_ref(channel) / 10^(SNR/10) with P_ref = mean clean power
           of the TRAINING set (fixed sensor noise floor). More realistic and
           removes the amplitude shortcut.

Noise types: ``white`` (Gaussian), ``pink`` (1/f power), ``impulsive``
(Bernoulli-Gaussian impulses on a small Gaussian floor), all normalised to the
requested power.

Evaluation noise is **deterministic**: generated on CPU from a generator seeded
by ``(base_seed, snr, type, batch_id)``. With a fixed evaluation order and batch
size, every model sees the identical noisy test set -> paired tests.
"""
from __future__ import annotations

import zlib
from typing import Optional, Sequence

import torch


def _unit_noise(shape, kind: str, gen: Optional[torch.Generator], device, impulse_p: float = 0.01) -> torch.Tensor:
    B, C, L = shape
    if kind == "white":
        n = torch.randn(shape, generator=gen, device=device)
    elif kind == "pink":
        w = torch.randn(shape, generator=gen, device=device)
        W = torch.fft.rfft(w, dim=-1)
        f = torch.arange(W.shape[-1], device=device, dtype=torch.float32)
        f[0] = 1.0
        n = torch.fft.irfft(W / torch.sqrt(f), n=L, dim=-1)
    elif kind == "impulsive":
        floor = 0.3 * torch.randn(shape, generator=gen, device=device)
        mask = (torch.rand(shape, generator=gen, device=device) < impulse_p).float()
        imp = torch.randn(shape, generator=gen, device=device) * 10.0
        n = floor + mask * imp
    else:
        raise ValueError(f"Unknown noise type '{kind}'")
    p = n.pow(2).mean(-1, keepdim=True).clamp_min(1e-12)   # unit power per (b, c)
    return n / p.sqrt()


def add_noise(x: torch.Tensor, snr_db, kind: str = "white", mode: str = "relative",
              ref_power: Optional[torch.Tensor] = None, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """x: (B, C, L). snr_db: scalar or (B,) tensor. ref_power: (C,) for absolute mode."""
    B, C, L = x.shape
    x = x.float()
    if not torch.is_tensor(snr_db):
        snr_db = torch.full((B,), float(snr_db))
    snr_db = snr_db.to(x.device, torch.float32).view(B, 1, 1)
    if mode == "relative":
        p_sig = x.pow(2).mean(-1, keepdim=True)
    elif mode == "absolute":
        if ref_power is None:
            raise ValueError("absolute SNR requires ref_power (train clean power per channel)")
        p_sig = ref_power.to(x.device, torch.float32).view(1, C, 1).expand(B, C, 1)
    else:
        raise ValueError(f"Unknown SNR mode '{mode}'")
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    gdev = gen.device if gen is not None else x.device
    n = _unit_noise((B, C, L), kind, gen, gdev).to(x.device)
    return x + n * p_noise.sqrt()


def eval_generator(base_seed: int, snr: float, kind: str, batch_id: int) -> torch.Generator:
    key = f"{base_seed}|{snr:.3f}|{kind}|{batch_id}".encode()
    g = torch.Generator(device="cpu")
    g.manual_seed(zlib.crc32(key))
    return g


class TrainNoiseAugment:
    """On-the-fly training augmentation: random SNR per sample, random type, mode relative/absolute/mixed."""

    def __init__(self, cfg: dict, ref_power: Optional[torch.Tensor]):
        self.enabled = bool(cfg.get("enabled", False))
        self.p = float(cfg.get("prob", 0.8))
        self.snr_range = tuple(cfg.get("snr_range", (-10.0, 10.0)))
        self.types: Sequence[str] = cfg.get("types", ["white"])
        self.mode = cfg.get("mode", "relative")
        self.ref_power = ref_power

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        B = x.shape[0]
        lo, hi = self.snr_range
        snr = torch.empty(B, device=x.device).uniform_(lo, hi)
        apply = (torch.rand(B, device=x.device) < self.p).float().view(B, 1, 1)
        kind = self.types[int(torch.randint(len(self.types), (1,)).item())]
        mode = self.mode
        if mode == "mixed":   # half of the batches relative, half absolute SNR
            mode = "relative" if torch.rand(1).item() < 0.5 else "absolute"
        xn = add_noise(x, snr, kind, mode, self.ref_power)
        return apply * xn + (1 - apply) * x
