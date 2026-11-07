"""Training loop with validation-only model selection."""
from __future__ import annotations

import copy
import math
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.noise import TrainNoiseAugment
from .evaluate import selection_score


def _make_loader(ds, tcfg, seed):
    bs = int(tcfg.get("batch_size", 64))
    workers = int(tcfg.get("workers", 0))
    g = torch.Generator()
    g.manual_seed(seed)
    if tcfg.get("balanced_sampler", False):
        counts = np.bincount(ds.labels)
        w = 1.0 / counts[ds.labels]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), True, generator=g)
        return DataLoader(ds, bs, sampler=sampler, drop_last=len(ds) > bs, num_workers=workers)
    return DataLoader(ds, bs, shuffle=True, drop_last=len(ds) > bs, generator=g,
                      num_workers=workers, persistent_workers=workers > 0)


def _random_shift(x):
    L = x.shape[-1]
    s = int(torch.randint(0, L, (1,)).item())
    return torch.roll(x, s, -1)


def _amp_scale(x, lo=0.8, hi=1.25):
    s = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(math.log(lo), math.log(hi)).exp()
    return x * s


class NonFiniteLoss(RuntimeError):
    """Loss became NaN/inf. Under AMP the runner restarts the run in fp32 (float16 overflow)."""

    def __init__(self, msg, amp: bool):
        super().__init__(msg)
        self.amp = amp


def train(model: nn.Module, data, cfg: Dict, device: torch.device, seed: int, logger=None,
          n_classes: Optional[int] = None) -> Dict:
    tcfg = cfg["train"]
    scfg = cfg.get("selection", {})
    n_classes = n_classes or len(data.class_names)
    model.to(device)
    accum = max(1, int(tcfg.get("grad_accum", 1)))   # >1 only after a CUDA out-of-memory retry
    loader = _make_loader(data.train, dict(tcfg, batch_size=max(2, int(tcfg.get("batch_size", 64)) // accum)), seed)
    epochs = int(tcfg.get("epochs", 60))
    steps = max(1, len(loader) // accum) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-3)),
                            weight_decay=float(tcfg.get("weight_decay", 1e-2)))
    sched_kind = tcfg.get("scheduler", "onecycle")
    if sched_kind == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tcfg.get("lr", 1e-3)), total_steps=steps,
                                                    pct_start=float(tcfg.get("warmup_frac", 0.15)))
    elif sched_kind == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    else:
        sched = None
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    aug = TrainNoiseAugment(tcfg.get("noise_aug", {}), data.ref_power.to(device))
    ls = float(tcfg.get("label_smoothing", 0.0))
    mix_a = float(tcfg.get("mixup_alpha", 0.0))
    shift = bool(tcfg.get("random_shift", True))
    amp_aug = tcfg.get("amp_scale")
    clip = float(tcfg.get("grad_clip", 5.0))
    patience = int(tcfg.get("patience", 15))
    min_ep = int(tcfg.get("min_epochs", 10))
    eval_every = int(tcfg.get("eval_every", 1))
    select = str(tcfg.get("select", "best_val"))     # best_val (early stopping) | last (fixed epochs, no selection)
    if select not in ("best_val", "last"):
        raise ValueError(f"train.select must be best_val or last, got {select!r}")

    best, best_ep, best_state = -1.0, -1, None
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tn, tc = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if shift:
                x = _random_shift(x)
            if amp_aug:
                x = _amp_scale(x, *amp_aug)
            x = aug(x)
            with torch.autocast(device_type=device.type, enabled=amp):
                if mix_a > 0:
                    lam = float(np.random.beta(mix_a, mix_a))
                    perm = torch.randperm(x.shape[0], device=device)
                    out = model(lam * x + (1 - lam) * x[perm])
                    loss = lam * F.cross_entropy(out.float(), y, label_smoothing=ls) + \
                        (1 - lam) * F.cross_entropy(out.float(), y[perm], label_smoothing=ls)
                else:
                    out = model(x)
                    loss = F.cross_entropy(out.float(), y, label_smoothing=ls)
                aux = getattr(model, "aux_loss", None)   # e.g. ClassBD physics-informed terms
                if aux is not None:
                    loss = loss + aux.float()
            if not torch.isfinite(loss.detach()):
                # never continue silently: a NaN forward pass also corrupts BatchNorm running stats
                raise NonFiniteLoss(f"non-finite training loss at epoch {ep}, batch {it} "
                                    f"({'AMP float16' if amp else 'float32'})", amp)
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
            tl += float(loss.detach()) * len(y)
            tn += len(y)
            tc += int((out.argmax(1) == y).sum())
        rec = {"epoch": ep, "train_loss": tl / max(tn, 1), "train_acc": 100 * tc / max(tn, 1)}
        if ep % eval_every == 0 or ep == epochs:
            v = selection_score(model, data.val, device, n_classes, scfg, data.ref_power, amp)
            rec["val_score"] = v
            if v > best + 1e-6:
                best, best_ep = v, ep
                best_state = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
        hist.append(rec)
        if logger and (ep % int(tcfg.get("log_every", 5)) == 0 or ep == 1 or ep == epochs):
            logger.info(f"  ep {ep:3d} loss {rec['train_loss']:.4f} acc {rec['train_acc']:.1f} "
                        f"val {rec.get('val_score', float('nan')):.2f} (best {best:.2f}@{best_ep})")
        if select == "best_val" and ep >= min_ep and best_ep > 0 and ep - best_ep >= patience:
            if logger:
                logger.info(f"  early stop at epoch {ep} (best {best_ep})")
            break
    if select == "last":
        best_ep = len(hist)                           # the final model is the one evaluated
        best = hist[-1].get("val_score", best)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val": best, "best_epoch": best_ep, "select": select, "epochs_run": len(hist), "train_time_s": time.time() - t0,
            "history": hist}
