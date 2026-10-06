# -*- coding: utf-8 -*-
"""
utils/advanced_components.py
High-End Components for Deep Learning (2025)
Author: Ali Naderi
Includes:
  1. AdamP Optimizer
  2. Label Smoothing Loss (LSR)
  3. Meta-ACON Activation
  4. Coordinate Attention 1D (Flexible)
  5. Early Stopping
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable

# ==========================================
# 1. AdamP Optimizer
# ==========================================
class AdamP(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0, delta: float = 0.1, wd_ratio: float = 0.1,
                 nesterov: bool = False, amsgrad: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov, amsgrad=amsgrad)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def _layer_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(1, -1)

    def _cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, eps: float, view_func: Callable) -> torch.Tensor:
        x = view_func(x)
        y = view_func(y)
        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:
            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)
            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / (view_func(p.data).norm(dim=1).view(expand_size) + eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio
                return perturb, wd
        return perturb, wd

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']
                amsgrad = group['amsgrad']

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq'] if amsgrad else None
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (torch.sqrt(max_exp_avg_sq) + group['eps']) if amsgrad else (torch.sqrt(exp_avg_sq) + group['eps'])
                if amsgrad: torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                p.data.add_(perturb, alpha=-step_size)

        return loss

# ==========================================
# 2. Label Smoothing Loss (LSR)
# ==========================================
class LSR(nn.Module):
    def __init__(self, e: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels: torch.Tensor, classes: int, value: float = 1.0) -> torch.Tensor:
        one_hot = torch.zeros(labels.size(0), classes, device=labels.device)
        labels = labels.view(-1, 1).long()
        one_hot.scatter_(1, labels, value)
        return one_hot

    def _smooth_label(self, target: torch.Tensor, length: int, smooth_factor: float) -> torch.Tensor:
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)
        return one_hot

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x_log = self.log_softmax(x)
        loss = torch.sum(- x_log * smoothed_target, dim=1)
        if self.reduction == 'mean': return torch.mean(loss)
        elif self.reduction == 'sum': return torch.sum(loss)
        return loss

# ==========================================
# 3. Meta-ACON Activation
# ==========================================
class MetaAconC(nn.Module):
    """ Adaptive ACON activation (learnable Swish/ReLU) """
    def __init__(self, width: int, r: int = 16):
        super().__init__()
        hidden = max(r, width // r)
        self.fc1 = nn.Conv1d(width, hidden, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Conv1d(hidden, width, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm1d(width)
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x.mean(dim=2, keepdim=True)
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(context)))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

# ==========================================
# 4. Coordinate Attention 1D
# ==========================================
class CoordAtt(nn.Module):
    """ Coordinate Attention for 1D signals (max/avg pool selectable) """
    def __init__(self, inp: int, oup: int, reduction: int = 32, pool: str = 'max'):
        super().__init__()
        assert pool in ['max', 'avg'], "pool must be 'max' or 'avg'"
        self.pool_type = pool
        self.pool_w = nn.AdaptiveMaxPool1d(1) if pool=='max' else nn.AdaptiveAvgPool1d(1)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = MetaAconC(mip)
        self.conv_w = nn.Conv1d(mip, oup, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.pool_w(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_w = self.conv_w(y).sigmoid()
        return identity * a_w

# ==========================================
# 5. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0,
                 path: str = 'checkpoint.pt', trace_func: Callable = print):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
        self.val_loss_min = val_loss
