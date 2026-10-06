# -*- coding: utf-8 -*-
r"""
train_sdust_final_v20.py
================================================================================
SDUST SOTA Training Pipeline - v20 (Publication Ready)
Author: Ali Naderi - 2026

Architecture:
  - Robust Data Loading (RAM/Disk Hybrid)
  - Online Noise Injection (Training robustness for -10dB targets)
  - Ensemble Voting + TTA (Test Time Augmentation)
  - Automatic Mixed Precision (AMP)
  - Professional Plotly Visualization

Outputs:
  - runs_sdust/seed_X/ (Models, Plots, Logs)
  - runs_sdust/final_ensemble_stats.json (Aggregated Results)
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
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm

# Suppress Warnings
warnings.filterwarnings("ignore")

# ------------------- Configuration -------------------
LOG = logging.getLogger("SDUST_SOTA")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(ch)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTLY_TEMPLATE = "plotly_white"

# =============================================================================
# Utilities
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

# =============================================================================
# Robustness & Augmentation
# =============================================================================
def add_noise_batch(x, min_snr=-5, max_snr=15, p=0.5):
    """Injects Gaussian noise on-the-fly during training"""
    if random.random() > p: return x
    B, C, L = x.shape
    # Signal power per sample
    sig_power = torch.mean(x ** 2, dim=2, keepdim=True)
    # Random SNR per sample
    snr_db = torch.rand(B, 1, 1).to(x.device) * (max_snr - min_snr) + min_snr
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_lin + 1e-8)
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise

def mixup_data(x, f, y, alpha=0.4):
    if alpha <= 0: return x, f, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_f = lam * f + (1 - lam) * f[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, mixed_f, y_a, y_b, lam

# =============================================================================
# Learned Filter Support
# =============================================================================
class LearnedFilter(nn.Module):
    def __init__(self, length=2048):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 7, padding=3, bias=False)
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        xt = self.act(self.conv(x)).squeeze(1)
        X = torch.fft.rfft(xt)
        g = torch.sigmoid(self.gate(torch.abs(X))).clamp(0.01, 0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1])

def load_filter(path: str):
    if not os.path.exists(path): return None
    try:
        model = LearnedFilter()
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE).eval()
        for p in model.parameters(): p.requires_grad = False
        return model
    except: return None

@torch.no_grad()
def apply_filters(x, filters):
    if not filters or all(f is None for f in filters): return x
    out = []
    for c in range(3):
        fc = filters[c]
        xc = x[:, c:c+1, :] 
        if fc is not None:
            xc_filt = fc(xc)
            if xc_filt.dim() == 2: xc_filt = xc_filt.unsqueeze(1)
            out.append(xc_filt)
        else:
            out.append(xc)
    return torch.cat(out, dim=1)

# =============================================================================
# Smart Dataset (Fixed & Optimized)
# =============================================================================
class SDUSTDataset(Dataset):
    def __init__(self, npz_path, mode='train', snr=None, load_ram=True):
        self.npz_path = npz_path
        self.mode = mode
        self.in_memory = False
        self.data_disk = None
        
        # 1. Metadata Load
        with np.load(npz_path, allow_pickle=True) as temp:
            self.classes = temp['class_names'].item() if 'class_names' in temp else {i: str(i) for i in range(10)}
            # Fix: Explicitly define lbl_key here
            if mode == 'train': self.lbl_key = 'y_train'
            elif mode == 'val': self.lbl_key = 'y_val'
            elif mode == 'test': self.lbl_key = 'y_test'
            self.length = temp[self.lbl_key].shape[0]

        # 2. Key Mapping
        if mode == 'train':
            self.sig_key, self.feat_key = 'X_train_sig', 'X_train_feat'
        elif mode == 'val':
            self.sig_key, self.feat_key = 'X_val_sig', 'X_val_feat'
        elif mode == 'test':
            if snr is not None:
                self.sig_key, self.feat_key = f'test_snr_{snr}_sig', f'test_snr_{snr}_feat'
            else:
                self.sig_key, self.feat_key = 'X_test_sig', 'X_test_feat'

        # 3. RAM Loading (Try/Except)
        if load_ram:
            try:
                full = np.load(npz_path, allow_pickle=True)
                self.sig_data = full[self.sig_key].astype(np.float32)
                self.feat_data = full[self.feat_key].astype(np.float32)
                self.lbl_data = full[self.lbl_key].astype(np.int64)
                self.in_memory = True
                LOG.info(f"[{mode.upper()}] Loaded {self.length} samples into RAM.")
            except:
                self.in_memory = False
                LOG.warning(f"[{mode.upper()}] RAM load failed. Using Disk Mode.")

    def _ensure_disk(self):
        if self.data_disk is None:
            self.data_disk = np.load(self.npz_path, mmap_mode='r', allow_pickle=True)

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.in_memory:
            sig = self.sig_data[idx]
            feat = self.feat_data[idx]
            lbl = self.lbl_data[idx]
        else:
            self._ensure_disk()
            sig = np.array(self.data_disk[self.sig_key][idx], copy=True).astype(np.float32)
            feat = np.array(self.data_disk[self.feat_key][idx], copy=True).astype(np.float32)
            lbl = int(self.data_disk[self.lbl_key][idx]) # Fixed line
            
        if feat.ndim > 1: feat = feat.flatten()
        return torch.from_numpy(sig), torch.from_numpy(feat), torch.tensor(lbl, dtype=torch.long)

# =============================================================================
# Training Engine
# =============================================================================
def train_epoch(model, loader, optimizer, scheduler, criterion, filters=None):
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    # Instance Norm for Signal Stability
    norm_layer = nn.InstanceNorm1d(3, affine=False).to(DEVICE)
    
    avg_loss = 0
    correct = 0
    total = 0
    
    # Use TQDM but with leave=False to avoid cluttering console
    pbar = tqdm(loader, desc="Train", leave=False, unit="batch")
    
    for i, (x, f, y) in enumerate(pbar):
        # Memory Safety
        if i % 100 == 0: torch.cuda.empty_cache()
        
        x, f, y = x.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
        
        # 1. Normalize
        x = norm_layer(x)
        
        # 2. Inject Noise (Robustness Boost)
        x = add_noise_batch(x, min_snr=-5, max_snr=15, p=0.6)
        
        # 3. Apply Physics Filters
        if filters:
            with torch.no_grad(): x = apply_filters(x, filters)
            
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # MixUp
            lam = np.random.beta(0.4, 0.4)
            idx = torch.randperm(x.size(0)).to(DEVICE)
            x_m = lam*x + (1-lam)*x[idx]
            f_m = lam*f + (1-lam)*f[idx]
            y_a, y_b = y, y[idx]
            
            out = model(x_m, f_m)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            
        if torch.isnan(loss): continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Strict clipping
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        
        avg_loss += loss.item() * x.size(0)
        total += x.size(0)
        
    return avg_loss / (total + 1e-6)

@torch.no_grad()
def evaluate(model, loader, criterion, filters=None, use_tta=False):
    model.eval()
    norm_layer = nn.InstanceNorm1d(3, affine=False).to(DEVICE)
    avg_loss = 0
    correct = 0
    total = 0
    preds, targets, logits = [], [], []
    
    for x, f, y in loader:
        x, f, y = x.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
        if filters: x = apply_filters(x, filters)
        
        if use_tta:
            # TTA: Original + Shift Right + Shift Left + Noised
            p1 = F.softmax(model(norm_layer(x), f), dim=1)
            p2 = F.softmax(model(norm_layer(torch.roll(x, 5, 2)), f), dim=1)
            p3 = F.softmax(model(norm_layer(torch.roll(x, -5, 2)), f), dim=1)
            p4 = F.softmax(model(norm_layer(add_noise_batch(x, 5, 10, 1.0)), f), dim=1)
            out_prob = (p1 + p2 + p3 + p4) / 4.0
            out = torch.log(out_prob + 1e-12) # Fake logits
        else:
            out = model(norm_layer(x), f)
            
        loss = criterion(out, y)
        if not torch.isnan(loss): avg_loss += loss.item() * x.size(0)
        
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        
        preds.append(out.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
        logits.append(out.cpu().numpy())
        
    return avg_loss / (total + 1e-6), (correct / (total + 1e-6)) * 100, np.concatenate(preds), np.concatenate(targets), np.concatenate(logits)

# =============================================================================
# Helper Functions
# =============================================================================
def load_model_class(model_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dyn_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ignore = ['LearnedFilter', 'MetaAconC', 'CoordinateAttention', 'QuadraticConv1d', 'SoftThresholding', 'SpectralGating', 'QResBlock_Pro']
    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and name not in ignore and name not in ['nn', 'Module']:
            return obj
    raise RuntimeError("Valid model class not found in file.")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
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

def generate_plots(run_dir, y_true, y_pred, class_names, history, snr_res):
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = [str(class_names[i]) for i in range(len(class_names))]
    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale='Blues', title="Confusion Matrix")
    fig_cm.update_layout(template=PLOTLY_TEMPLATE)
    fig_cm.write_html(os.path.join(run_dir, "confusion_matrix.html"))
    
    # 2. Training Curves
    fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hist.add_trace(go.Scatter(y=[h['loss'] for h in history], name="Train Loss"), secondary_y=False)
    fig_hist.add_trace(go.Scatter(y=[h['val_acc'] for h in history], name="Val Acc"), secondary_y=True)
    fig_hist.update_layout(title="Training History", template=PLOTLY_TEMPLATE)
    fig_hist.write_html(os.path.join(run_dir, "training_history.html"))
    
    # 3. SNR Robustness
    if snr_res:
        snrs = sorted([int(k) for k in snr_res.keys()])
        accs = [snr_res[str(k)] for k in snrs]
        fig_snr = go.Figure()
        fig_snr.add_trace(go.Scatter(x=snrs, y=accs, mode='lines+markers+text', text=[f"{a:.1f}%" for a in accs], textposition="top center", name="Accuracy", line=dict(color='firebrick', width=3)))
        fig_snr.update_layout(title="Noise Robustness (-10dB to -2dB)", xaxis_title="SNR (dB)", yaxis_title="Accuracy", template=PLOTLY_TEMPLATE)
        fig_snr.write_html(os.path.join(run_dir, "snr_robustness.html"))

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    # Paths (Auto-configured based on your context)
    def_data = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\SDUST\processed_sdust_final"
    def_model = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\Model\model.py"
    def_out = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\SDUST\runs_sdust"

    parser.add_argument("--data_dir", default=def_data)
    parser.add_argument("--out_dir", default=def_out)
    parser.add_argument("--model_py", default=def_model)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 101, 2024])
    args = parser.parse_args()
    
    npz_path = os.path.join(args.data_dir, "processed_sdust_final.npz")
    ModelClass = load_model_class(args.model_py)
    filters = [load_filter(os.path.join(args.data_dir, f"learned_filter_ch{i}.pt")) for i in range(3)]
    
    # Init DS info
    temp = SDUSTDataset(npz_path, 'train', load_ram=False)
    s_dim, f_dim = temp[0][0].shape[0], temp[0][1].shape[0]
    class_names = temp.classes
    del temp
    
    LOG.info(f"Pipeline Config: {len(class_names)} Classes | In={s_dim}, Feat={f_dim} | Noise Injection: ON")
    
    run_stats, saved_models = [], []
    
    for seed in args.seeds:
        seed_everything(seed)
        run_name = f"seed_{seed}"
        run_dir = ensure_dir(os.path.join(args.out_dir, run_name))
        LOG.info(f"\n>>> Run {seed} Started <<<")
        
        # Datasets (RAM Optimized)
        ds_tr = SDUSTDataset(npz_path, 'train', load_ram=True)
        ds_va = SDUSTDataset(npz_path, 'val', load_ram=True)
        ds_te = SDUSTDataset(npz_path, 'test', load_ram=True)
        
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        model = ModelClass(in_channels=s_dim, num_classes=len(class_names), feat_dim=f_dim).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(dl_tr), epochs=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        es = EarlyStopping(patience=10)
        
        best_path = os.path.join(run_dir, "best_model.pth")
        history = []
        best_acc = 0.0
        
        # Loop
        for ep in range(1, args.epochs+1):
            t0 = time.time()
            tr_loss = train_epoch(model, dl_tr, opt, sched, crit, filters)
            va_loss, va_acc, _, _, _ = evaluate(model, dl_va, crit, filters)
            
            LOG.info(f"Ep {ep:02d} | Loss: {tr_loss:.4f} | Val Acc: {va_acc:.2f}% | Time: {time.time()-t0:.1f}s")
            
            history.append({'loss': tr_loss, 'val_acc': va_acc})
            es(va_acc, model, best_path)
            
            if va_acc > best_acc: best_acc = va_acc
            if es.early_stop:
                LOG.info("Early stopping.")
                break
                
        # Final Eval
        model.load_state_dict(torch.load(best_path))
        saved_models.append(model)
        
        # Test + TTA
        _, test_acc, y_pred, y_true, _ = evaluate(model, dl_te, crit, filters, use_tta=True)
        LOG.info(f"Seed {seed} Test Acc (TTA): {test_acc:.2f}%")
        
        # Robustness Check
        snr_res = {}
        for snr in [-10, -8, -6, -4, -2]:
            try:
                ds_s = SDUSTDataset(npz_path, 'test', snr=snr, load_ram=True)
                dl_s = DataLoader(ds_s, batch_size=args.batch_size, num_workers=0)
                _, acc, _, _, _ = evaluate(model, dl_s, crit, filters, use_tta=True)
                snr_res[str(snr)] = acc
                LOG.info(f"SNR {snr}dB: {acc:.2f}%")
                del ds_s, dl_s
                torch.cuda.empty_cache()
            except: pass
            
        # Generate Reports
        generate_plots(run_dir, y_true, y_pred, class_names, history, snr_res)
        with open(os.path.join(run_dir, "classification_report.txt"), 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=[str(v) for k,v in class_names.items()]))
            
        run_stats.append({"seed": seed, "test_acc": test_acc, "snr": snr_res})
        del ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te
        torch.cuda.empty_cache()

    # Ensemble
    if len(saved_models) > 1:
        LOG.info("\n>>> Ensemble Voting <<<")
        ds_te = SDUSTDataset(npz_path, 'test', load_ram=True)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)
        
        ens_logits = 0
        for m in saved_models:
            m.eval()
            _, _, _, _, logits = evaluate(m, dl_te, crit, filters, use_tta=True)
            ens_logits += logits
            
        ens_pred = np.argmax(ens_logits, axis=1)
        y_true = np.array([y.item() for _,_,y in ds_te])
        ens_acc = accuracy_score(y_true, ens_pred) * 100
        
        LOG.info(f"Ensemble Accuracy: {ens_acc:.2f}%")
        generate_plots(args.out_dir, y_true, ens_pred, class_names, [], {})
        
    save_json(run_stats, os.path.join(args.out_dir, "final_results.json"))

if __name__ == "__main__":
    main()



