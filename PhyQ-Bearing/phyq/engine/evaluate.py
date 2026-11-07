"""Deterministic (paired) evaluation under clean and noisy conditions."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.noise import add_noise, eval_generator
from .metrics import compute_metrics


@torch.no_grad()
def predict(model, ds, device, batch_size: int = 256, snr: Optional[float] = None, kind: str = "white",
            mode: str = "relative", ref_power: Optional[torch.Tensor] = None, noise_seed: int = 12345,
            tta_shifts: int = 1, amp: bool = False) -> np.ndarray:
    """Return softmax probabilities. Noise is generated on CPU per batch from a fixed seed,
    so every model sees bit-identical noisy inputs (requires the same ``batch_size``)."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for b, (x, _) in enumerate(dl):
        if snr is not None:
            g = eval_generator(noise_seed, float(snr), kind, b)
            x = add_noise(x, float(snr), kind, mode, ref_power, gen=g)
        x = x.to(device, non_blocking=True)
        p = 0
        L = x.shape[-1]
        for s in range(max(1, tta_shifts)):
            xs = torch.roll(x, shifts=int(s * L / tta_shifts), dims=-1) if s else x
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(xs)
            p = p + torch.softmax(logits.float(), -1)
        out.append((p / max(1, tta_shifts)).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 1))


def evaluate_grid(model, ds, device, n_classes: int, ecfg: Dict, ref_power: torch.Tensor,
                  amp: bool = False, tta_shifts: int = 1, keep_probs: bool = True) -> Dict:
    """Evaluate on clean data and on every (noise type, SNR mode, SNR) of the grid."""
    bs = int(ecfg.get("batch_size", 256))
    seed = int(ecfg.get("noise_seed", 12345))
    y = ds.labels
    res = {"clean": None, "noise": {}}
    probs_store = {}
    p = predict(model, ds, device, bs, None, ref_power=ref_power, tta_shifts=tta_shifts, amp=amp)
    res["clean"] = compute_metrics(p, y, n_classes)
    probs_store["clean"] = p
    for kind in ecfg.get("noise_types", ["white"]):
        for mode in ecfg.get("snr_modes", ["relative"]):
            for snr in ecfg.get("snrs", []):
                p = predict(model, ds, device, bs, snr, kind, mode, ref_power, seed, tta_shifts, amp)
                res["noise"].setdefault(kind, {}).setdefault(mode, {})[str(snr)] = compute_metrics(p, y, n_classes)
                probs_store[f"{kind}|{mode}|{snr}"] = p
    if keep_probs:
        res["_probs"] = probs_store
    return res


def selection_score(model, ds, device, n_classes: int, scfg: Dict, ref_power, amp=False) -> float:
    """Validation score used for early stopping / model selection: mean metric over the
    configured conditions (clean + SNRs x SNR definitions). Uses VAL only."""
    scores = []
    bs = int(scfg.get("batch_size", 256))
    modes = scfg.get("snr_modes") or [scfg.get("snr_mode", "relative")]
    conds = []
    for snr in scfg.get("snrs", [None]):
        if snr in (None, "clean"):
            conds.append((None, "relative"))
        else:
            conds += [(float(snr), m) for m in modes]
    for snr, mode in conds:
        p = predict(model, ds, device, bs, snr, scfg.get("noise_type", "white"), mode, ref_power,
                    int(scfg.get("noise_seed", 777)), amp=amp)
        scores.append(compute_metrics(p, ds.labels, n_classes)[scfg.get("metric", "macro_f1")])
    return float(np.mean(scores))
