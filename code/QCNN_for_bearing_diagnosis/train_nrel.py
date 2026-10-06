# -*- coding: utf-8 -*-
r"""
train_nrel_final_v36.py
================================================================================
NREL SOTA Training Pipeline - ULTRA LOW MEMORY EDITION v36
Author: Ali Naderi - 2026

Fixes:
  - Solved 'CUDA error: out of memory' by reducing physical batch size to 16.
  - Maintains effective batch size of 64 via Gradient Accumulation (Steps=4).
  - Explicit memory cleanup to prevent VRAM spikes.

Usage:
  python train_nrel.py
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import datetime
import logging
import warnings
import importlib.util
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

# Suppress Warnings
warnings.filterwarnings("ignore")

# ------------------- Logger -------------------
LOG = logging.getLogger("NREL_V36")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(ch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTLY_TEMPLATE = "plotly_white"

# =============================================================================
# 1. Utilities
# =============================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

# =============================================================================
# 2. Robustness Layers
# =============================================================================
def add_noise_batch(x, min_snr=-5, max_snr=15, p=0.5):
    if random.random() > p: return x
    B, C, L = x.shape
    sig_power = torch.mean(x ** 2, dim=(1, 2), keepdim=True)
    sig_power = torch.clamp(sig_power, min=1e-9)
    
    snr_db = torch.rand(B, 1, 1).to(x.device) * (max_snr - min_snr) + min_snr
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_lin + 1e-9)
    
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise

def mixup_data(x, f, y, alpha=0.4, device=DEVICE):
    if alpha <= 0: return x, f, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(device)
    return lam*x + (1-lam)*x[idx], lam*f + (1-lam)*f[idx], y, y[idx], lam

def corrupt_features(f, p=0.6, noise_std=0.2):
    if random.random() < p:
        return torch.zeros_like(f)
    return f + torch.randn_like(f) * noise_std

class LearnedFilter(nn.Module):
    def __init__(self, length=2048):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 65, padding=32, bias=False)
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        xt = self.act(self.conv(x)).squeeze(1)
        X = torch.fft.rfft(xt)
        g = torch.sigmoid(self.gate(torch.abs(X))).clamp(0.01, 0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1]).unsqueeze(1)

def load_filters(filter_dir):
    filters = []
    if not os.path.exists(filter_dir): return None
    LOG.info(f"Loading filters from: {filter_dir}")
    for i in range(8):
        fp = os.path.join(filter_dir, f"learned_filter_nrel_ch{i}.pt")
        if os.path.exists(fp):
            try:
                m = LearnedFilter().to(DEVICE)
                m.load_state_dict(torch.load(fp, map_location=DEVICE))
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                filters.append(m)
            except: filters.append(None)
        else:
            filters.append(None)
    return filters

@torch.no_grad()
def apply_filters(x, filters):
    if not filters: return x
    out = []
    for c in range(x.shape[1]): 
        fc = filters[c] if c < len(filters) else None
        xc = x[:, c:c+1, :] 
        if fc is not None:
            out.append(fc(xc))
        else:
            out.append(xc)
    return torch.cat(out, dim=1)

# =============================================================================
# 3. Dataset
# =============================================================================
class NRELDataset(Dataset):
    def __init__(self, npz_path, mode='train', snr=None, load_ram=True):
        self.data = np.load(npz_path, mmap_mode='r', allow_pickle=True)
        
        if mode == 'train': k1, k2 = 'X_train_sig', 'y_train'
        elif mode == 'val': k1, k2 = 'X_val_sig', 'y_val'
        else:
            k2 = 'y_test'
            if snr is not None:
                k1 = f'test_snr_{snr}_sig'
                if k1 not in self.data: k1 = f'test_snr_{float(snr)}_sig'
                if k1 not in self.data: k1 = f'test_snr_{int(snr)}_sig'
            else: k1 = 'X_test_sig'

        if k1 not in self.data:
             if mode=='test' and snr is not None:
                 k1 = 'X_test_sig'
             else:
                 raise KeyError(f"{k1} not found")

        self.sig = self.data[k1]
        self.lbl = self.data[k2]
        kf = k1.replace('_sig', '_feat')
        self.feat = self.data[kf] if kf in self.data else np.zeros((self.sig.shape[0], 1))
        
        self.length = self.sig.shape[0]
        if 'classes' in self.data:
            c = self.data['classes']
            self.classes = {i:str(v) for i,v in enumerate(c.item() if c.ndim==0 else c)}
        else: self.classes = {0:'Healthy', 1:'Damaged'}

        self.in_ram = False
        if load_ram:
            try:
                self.sig = np.array(self.sig, dtype=np.float32)
                self.feat = np.array(self.feat, dtype=np.float32)
                self.lbl = np.array(self.lbl, dtype=np.int64)
                self.in_ram = True
                LOG.info(f"[{mode.upper()}] RAM Loaded: {self.length}")
            except: pass

    def __len__(self): return self.length
    def __getitem__(self, idx):
        return torch.from_numpy(self.sig[idx]), torch.from_numpy(self.feat[idx]), torch.tensor(self.lbl[idx])

def get_model_class(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for n in ['QDCA_Hybrid_Pro', 'QDCA_Hybrid']: 
        if hasattr(mod, n): return getattr(mod, n)
    return [o for n,o in vars(mod).items() if isinstance(o,type) and issubclass(o, nn.Module)][0]

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_score = None; self.early_stop = False
    def __call__(self, val_acc, model, path):
        if self.best_score is None:
            self.best_score = val_acc
            torch.save(model.state_dict(), path)
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = val_acc
            torch.save(model.state_dict(), path)
            self.counter = 0

# =============================================================================
# 4. Reporting
# =============================================================================
def generate_report(run_dir, y_true, y_pred, class_map, history, snr_res, seed):
    labels_idx = sorted([int(k) for k in class_map.keys()])
    labels_names = [str(class_map[k]) for k in labels_idx]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    fig_cm = px.imshow(cm, text_auto=True, x=labels_names, y=labels_names, 
                       color_continuous_scale='Blues', title=f"Confusion Matrix (Seed {seed})")
    fig_cm.write_html(os.path.join(run_dir, "confusion_matrix.html"))
    
    if history:
        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist.add_trace(go.Scatter(y=[h['loss'] for h in history], name="Loss"), secondary_y=False)
        fig_hist.add_trace(go.Scatter(y=[h['val_acc'] for h in history], name="Val Acc"), secondary_y=True)
        fig_hist.write_html(os.path.join(run_dir, "training_history.html"))

    if snr_res:
        snrs = sorted([float(k) for k in snr_res.keys()])
        accs = []
        for k in snrs:
            if str(k) in snr_res: v = snr_res[str(k)]
            elif str(int(k)) in snr_res: v = snr_res[str(int(k))]
            elif str(float(k)) in snr_res: v = snr_res[str(float(k))]
            else: v = 0.0
            accs.append(v)
            
        fig_snr = go.Figure()
        fig_snr.add_trace(go.Scatter(x=snrs, y=accs, mode='lines+markers+text', 
                                     text=[f"{a:.1f}%" for a in accs], name="Acc"))
        fig_snr.update_layout(title="Noise Robustness", xaxis_title="SNR", yaxis_title="Accuracy")
        fig_snr.write_html(os.path.join(run_dir, "snr_robustness.html"))

# =============================================================================
# 5. Training Logic (Ultra-Low Memory)
# =============================================================================
def train_one_epoch(model, loader, opt, crit, filters=None):
    model.train()
    try: n_c = loader.dataset[0][0].shape[0]
    except: n_c = 8
    norm = nn.InstanceNorm1d(n_c, affine=False).to(DEVICE)
    
    loss_sum, tot = 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    
    # MEMORY FIX: Reduced Batch Size + High Accumulation
    # Effective Batch Size = 16 * 4 = 64
    ACCUMULATION_STEPS = 4 
    opt.zero_grad(set_to_none=True)
    
    for i, (x, f, y) in enumerate(pbar):
        # Aggressive cache clearing
        if i % 25 == 0: torch.cuda.empty_cache()
            
        x, f, y = x.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
        f = torch.nan_to_num(f, 0.0)

        # 1. Signal Path
        if random.random() < 0.7:
             x = add_noise_batch(x, min_snr=-8, max_snr=10, p=1.0)
        
        if filters: 
            with torch.no_grad(): x = apply_filters(x, filters)
        
        x = norm(x)
        x = torch.clamp(x, -20, 20)
        f = corrupt_features(f, p=0.6, noise_std=0.2)

        # Full Precision
        xm, fm, ya, yb, lam = mixup_data(x, f, y, alpha=0.4)
        out = model(xm, fm)
        loss = lam*crit(out, ya) + (1-lam)*crit(out, yb)
        
        # Scale loss for accumulation
        loss = loss / ACCUMULATION_STEPS
        
        if torch.isnan(loss): continue
        
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        # Detach to free graph memory
        loss_val = loss.item() * ACCUMULATION_STEPS
        loss_sum += loss_val; tot += 1
        pbar.set_postfix(loss=loss_val)
        
        # Explicit cleanup
        del x, f, y, xm, fm, out, loss
        
    return loss_sum/max(tot, 1)

@torch.no_grad()
def evaluate(model, loader, crit, filters=None, tta=False):
    model.eval()
    try: n_c = loader.dataset[0][0].shape[0]
    except: n_c = 8
    norm = nn.InstanceNorm1d(n_c, affine=False).to(DEVICE)
    loss_sum, corr, tot = 0, 0, 0
    preds, targs, logits = [], [], []
    
    for x, f, y in tqdm(loader, desc="Eval", leave=False):
        x, f, y = x.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
        f = torch.nan_to_num(f, 0.0)
        
        if tta:
            x0 = norm(apply_filters(x, filters) if filters else x)
            p = F.softmax(model(torch.clamp(x0, -20, 20), f), 1)
            
            xn = add_noise_batch(x, 0, 10, 1.0)
            xn = norm(apply_filters(xn, filters) if filters else xn)
            p += F.softmax(model(torch.clamp(xn, -20, 20), f), 1)
            
            out = torch.log(p/2 + 1e-9)
        else:
            x = norm(apply_filters(x, filters) if filters else x)
            out = model(torch.clamp(x, -20, 20), f)
            
        loss = crit(out, y)
        loss_sum += loss.item()*x.size(0)
        corr += (out.argmax(1)==y).sum().item()
        tot += x.size(0)
        preds.append(out.argmax(1).cpu().numpy())
        targs.append(y.cpu().numpy())
        logits.append(out.cpu().numpy())
        
        del x, f, y, out
        
    return loss_sum/max(tot,1), corr/max(tot,1), np.concatenate(preds), np.concatenate(targs), np.concatenate(logits)

# =============================================================================
# Main
# =============================================================================
def main():
    ROOT = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis"
    NPZ_PATH = os.path.join(ROOT, "nrel_processed_robust.npz")
    FILTER_DIR = os.path.join(ROOT, "data", "NREL", "filters")
    OUT_DIR = os.path.join(ROOT, "data", "NREL", "runs_final_v37")
    MODEL_PATH = os.path.join(ROOT, "Model", "model.py")
    
    seeds = [44, 101, 2024]
    ensure_dir(OUT_DIR)
    
    if not os.path.exists(NPZ_PATH):
        LOG.error("NPZ not found")
        return

    temp_ds = NRELDataset(NPZ_PATH, 'train', load_ram=False)
    S, F_dim, C = temp_ds[0][0].shape[0], temp_ds[0][1].shape[0], len(temp_ds.classes)
    cls_map = temp_ds.classes
    del temp_ds
    
    LOG.info(f"NREL V37 | In={S}, Feat={F_dim}, Classes={C}")
    filters = load_filters(FILTER_DIR)
    if not filters: LOG.warning("!!! FILTERS NOT FOUND !!!")
    MCls = get_model_class(MODEL_PATH)
    
    results = []
    saved_models = []
    
    PHYSICAL_BS = 16 
    
    for seed in seeds:
        seed_everything(seed)
        rdir = ensure_dir(os.path.join(OUT_DIR, f"seed_{seed}"))
        LOG.info(f"\n>>> Seed {seed} <<<")
        
        tr_ds = NRELDataset(NPZ_PATH, 'train', load_ram=True)
        va_ds = NRELDataset(NPZ_PATH, 'val', load_ram=True)
        te_ds = NRELDataset(NPZ_PATH, 'test', load_ram=True)
        
        tr_dl = DataLoader(tr_ds, PHYSICAL_BS, True, num_workers=0)
        va_dl = DataLoader(va_ds, PHYSICAL_BS, False)
        te_dl = DataLoader(te_ds, PHYSICAL_BS, False)
        
        m = MCls(in_channels=S, num_classes=C, feat_dim=F_dim).to(DEVICE)
        opt = optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
        sch = optim.lr_scheduler.OneCycleLR(opt, 2e-4, epochs=60, steps_per_epoch=len(tr_dl)//4)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        es = EarlyStopping(patience=10)
        
        best_acc = 0.0 
        best_path = os.path.join(rdir, "best.pth")
        hist = []
        
        for ep in range(1, 61):
            ls = train_one_epoch(m, tr_dl, opt, crit, filters)
            _, acc, _, _, _ = evaluate(m, va_dl, crit, filters)
            sch.step()
            
            LOG.info(f"Ep {ep:02d}: Loss {ls:.4f} | Val {acc*100:.2f}%")
            hist.append({'loss': ls, 'val': acc})
            
            if acc >= best_acc:
                best_acc = acc
                torch.save(m.state_dict(), best_path)
                
            es(acc, m, best_path)
            if es.early_stop: 
                LOG.info("Early stopping.")
                break
        
        del tr_ds, va_ds, tr_dl, va_dl
        torch.cuda.empty_cache()
        
        m.load_state_dict(torch.load(best_path))
        saved_models.append(m)
        _, t_acc, yp, yt, _ = evaluate(m, te_dl, crit, filters, tta=True)
        LOG.info(f"Test (Clean): {t_acc*100:.2f}%")
        
        snr_vals = {}
        for snr in [-10, -8, -6, -4, -2]:
            try:
                ds = NRELDataset(NPZ_PATH, 'test', snr=snr, load_ram=True)
                dl = DataLoader(ds, PHYSICAL_BS, False)
                _, s_acc, _, _, _ = evaluate(m, dl, crit, filters, tta=True)
                snr_vals[str(snr)] = s_acc * 100
                LOG.info(f"SNR {snr}: {s_acc*100:.2f}%")
                del ds, dl
                torch.cuda.empty_cache()
            except: pass
            
        results.append({'seed': seed, 'test': t_acc*100, 'snr': snr_vals})
        generate_report(rdir, yt, yp, cls_map, hist, snr_vals, seed)
        del te_ds, te_dl
        torch.cuda.empty_cache()

    LOG.info("\n>>> Ensemble <<<")
    te_ds = NRELDataset(NPZ_PATH, 'test', load_ram=True)
    te_dl = DataLoader(te_ds, PHYSICAL_BS, False)
    
    ens_logits = 0
    for m in saved_models:
        m.eval()
        _, _, _, _, lg = evaluate(m, te_dl, crit, filters, tta=True)
        ens_logits += lg
    
    pred = ens_logits.argmax(1)
    y = np.array([i[2] for i in te_ds])
    acc = accuracy_score(y, pred) * 100
    LOG.info(f"Ensemble Acc: {acc:.2f}%")
    
    # FIX: Use OUT_DIR explicitly instead of args.out_dir
    generate_report(OUT_DIR, y, pred, cls_map, [], {}, "Ensemble")
    save_json(results, os.path.join(OUT_DIR, "final.json"))
    
    print("\n--- Paper Table ---")
    avg_snrs = {}
    keys = ['-10', '-8', '-6', '-4', '-2']
    for k in keys:
        vals = []
        for r in results:
            if k in r['snr']: vals.append(r['snr'][k])
            elif str(float(k)) in r['snr']: vals.append(r['snr'][str(float(k))])
            elif str(int(k)) in r['snr']: vals.append(r['snr'][str(int(k))])
        avg_snrs[k] = np.mean(vals) if vals else 0.0
    print(f"PhyQ-TransNet & {avg_snrs['-10']:.1f} & {avg_snrs['-8']:.1f} & {avg_snrs['-6']:.1f} & {avg_snrs['-4']:.1f} & {avg_snrs['-2']:.1f} \\\\")

if __name__ == "__main__":
    main()
