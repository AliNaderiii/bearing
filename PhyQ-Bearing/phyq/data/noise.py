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

``impacts`` (v10.4, TRAINING augmentation only - experiment E2): sparse, randomly
decaying impact bursts (Poisson-like arrivals, rate ~1e-3/sample, random sign,
log-normal amplitudes, exponential decay over a few samples, random oscillation)
on a Gaussian floor. It is deliberately a DIFFERENT impulse model from the
``impulsive`` test noise (i.i.d. single-sample spikes, p = 0.01, fixed scale), so
an E2 model is evaluated on an impulse model it never saw during training.

``harmonic`` (v10.5, TEST only): periodic interference of a gear stage - a mesh tone at a random
normalised frequency f_m in [0.02, 0.12] cycles/sample with harmonics 2 f_m and 3 f_m, each amplitude-
modulated at a random shaft rate f_m / Z (Z = 18..40 teeth) -> sidebands at k f_m +- f_m/Z, random phases,
plus a weak Gaussian floor (10 % of the power). Parameters are drawn per (window, channel).

``recorded`` (v10.5, TEST only): real machine vibration recorded on another machine (by default the
healthy experiments of the NREL wind-turbine gearbox, resampled to the target sampling rate; see
``phyq/data/noise_bank.py``). Random crops of the standardised recording are used as noise.

Evaluation noise is **deterministic**: generated on CPU from a generator seeded
by ``(base_seed, snr, type, batch_id)``. With a fixed evaluation order and batch
size, every model sees the identical noisy test set -> paired tests.
"""
from __future__ import annotations

import zlib
from typing import Optional, Sequence

import torch


def _impact_bursts(shape, gen: Optional[torch.Generator], device, rate: float = 1e-3, klen: int = 24) -> torch.Tensor:
    """Train-only impulse model (see module docstring): decaying oscillating bursts + Gaussian floor."""
    B, C, L = shape
    floor = 0.2 * torch.randn(shape, generator=gen, device=device)
    spikes = (torch.rand(shape, generator=gen, device=device) < rate).float()
    amp = torch.exp(0.75 * torch.randn(shape, generator=gen, device=device))          # log-normal amplitudes
    sign = torch.where(torch.rand(shape, generator=gen, device=device) < 0.5, -1.0, 1.0)
    drive = spikes * amp * sign
    # one random burst shape per call: exp decay (tau 2..8 samples) x cosine (random normalised frequency)
    u = torch.rand(2, generator=gen, device=device)
    tau = 2.0 + 6.0 * float(u[0])
    f = 0.05 + 0.35 * float(u[1])
    t = torch.arange(klen, device=device, dtype=torch.float32)
    k = torch.exp(-t / tau) * torch.cos(2 * torch.pi * f * t)
    burst = torch.nn.functional.conv1d(drive.view(B * C, 1, L), k.flip(0).view(1, 1, -1), padding=klen - 1)
    burst = burst[..., :L].view(B, C, L)
    return floor + 8.0 * burst


_RECORDED_BANK: Optional[torch.Tensor] = None


def set_recorded_bank(bank) -> None:
    """Register the 1-D noise recording used by kind='recorded' (None to clear)."""
    global _RECORDED_BANK
    _RECORDED_BANK = None if bank is None else torch.as_tensor(bank, dtype=torch.float32).flatten().contiguous()


def _harmonic(shape, gen: Optional[torch.Generator], device) -> torch.Tensor:
    B, C, L = shape
    t = torch.arange(L, device=device, dtype=torch.float32).view(1, 1, L)
    fm = 0.02 + 0.10 * torch.rand(B, C, 1, generator=gen, device=device)
    z = torch.randint(18, 41, (B, C, 1), generator=gen, device=device).float()
    fs = fm / z
    n = torch.zeros(shape, device=device)
    for k, a in ((1, 1.0), (2, 0.5), (3, 0.25)):
        ph = 2 * torch.pi * torch.rand(B, C, 1, generator=gen, device=device)
        ph_m = 2 * torch.pi * torch.rand(B, C, 1, generator=gen, device=device)
        depth = 0.3 + 0.5 * torch.rand(B, C, 1, generator=gen, device=device)
        n = n + a * (1 + depth * torch.cos(2 * torch.pi * fs * t + ph_m)) * torch.cos(2 * torch.pi * k * fm * t + ph)
    n = n / n.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return 0.9487 * n + 0.3162 * torch.randn(shape, generator=gen, device=device)   # 90 % / 10 % power


def _recorded(shape, gen: Optional[torch.Generator], device) -> torch.Tensor:
    B, C, L = shape
    if _RECORDED_BANK is None:
        raise RuntimeError("noise type 'recorded' needs a noise recording: set eval.recorded_noise "
                           "(see phyq/data/noise_bank.py)")
    bank = _RECORDED_BANK
    if bank.numel() < 2 * L:
        raise RuntimeError(f"recorded noise bank too short ({bank.numel()} samples) for windows of {L}")
    start = torch.randint(0, bank.numel() - L, (B * C,), generator=gen, device="cpu")
    idx = start.view(-1, 1) + torch.arange(L).view(1, -1)
    return bank[idx].view(B, C, L).to(device)


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
    elif kind == "impacts":
        n = _impact_bursts(shape, gen, device)
    elif kind == "harmonic":
        n = _harmonic(shape, gen, device)
    elif kind == "recorded":
        n = _recorded(shape, gen, device)
        n = n - n.mean(-1, keepdim=True)
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
