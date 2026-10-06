# -*- coding: utf-8 -*-
"""
train_hybridmodel_ultimate_v3.py
Author: Ali Naderi - 2025 (Ultra-Pro Edition + Advanced Viz - Fixed)
Description: 
    State-of-the-Art Training Pipeline for Quadratic Neural Networks (QDCA-Net).
    Integrates:
    1. MixUp & TTA (Test Time Augmentation)
    2. Ensemble Learning (Voting)
    3. Smart Parameter Grouping
    4. **NEW: Interactive Plotly Visualizations & Detailed WandB Logging**
"""

import os
import sys
import argparse
import logging
import json
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, roc_curve, precision_recall_curve, f1_score, recall_score, precision_score)

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# WandB
import wandb

# ======================================================
# 1. SMART IMPORT SYSTEM
# ======================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
if os.path.exists(os.path.join(base_dir, 'utils')):
    sys.path.append(os.path.join(base_dir, 'utils'))
if os.path.exists(os.path.join(base_dir, 'Model')):
    sys.path.append(os.path.join(base_dir, 'Model'))

try:
    try:
        from utils.DatasetLoader import RobustBearingDataset, collate_fn
    except ImportError:
        from utils.DatasetLoader import RobustBearingDataset, collate_fn

    try:
        # Corrected import to match your file name
        from Model.model import QDCA_Hybrid_Pro
        MODEL_VERSION = "V3"
    except ImportError:
        from Model.model import QDCA_Hybrid_Pro
        MODEL_VERSION = "Legacy"

except ImportError as e:
    print(f"\n[CRITICAL ERROR] Import failed: {e}")
    sys.exit(1)

# ======================================================
# 2. UTILITIES & HELPERS
# ======================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("TrainerUltra")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    
    fh = logging.FileHandler(os.path.join(out_dir, "training_log.txt"), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        if self.verbose:
            tqdm.write(f'  |-- New Best Acc: {self.best_score:.2f}%. Saving...')
        torch.save(model.state_dict(), self.path)

# ======================================================
# 3. VISUALIZATION ENGINE
# ======================================================
def generate_advanced_charts(y_true, y_pred, y_scores, class_names, out_dir, suffix=""):
    """
    Generates professional Plotly charts for papers/reports.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=class_names, y=class_names, 
        colorscale='Blues', text=cm, texttemplate="%{text}",
        hoverongaps=False
    ))
    fig_cm.update_layout(
        title=f'Confusion Matrix {suffix}',
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        width=800, height=700
    )
    fig_cm.write_image(os.path.join(out_dir, f"confusion_matrix{suffix}.png"), scale=3)
    try: wandb.log({f"Confusion Matrix {suffix}": wandb.Image(os.path.join(out_dir, f"confusion_matrix{suffix}.png"))})
    except: pass

    # 2. Per-Class Metrics Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=class_names, y=precision, name='Precision', marker_color='skyblue'))
    fig_bar.add_trace(go.Bar(x=class_names, y=recall, name='Recall', marker_color='lightgreen'))
    fig_bar.add_trace(go.Bar(x=class_names, y=f1, name='F1 Score', marker_color='salmon'))
    
    fig_bar.update_layout(
        title=f'Per-Class Performance Metrics {suffix}',
        xaxis_title="Class",
        yaxis_title="Score",
        barmode='group',
        template="plotly_white",
        width=1000, height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_bar.write_image(os.path.join(out_dir, f"metrics_bar{suffix}.png"), scale=3)
    try: wandb.log({f"Metrics Bar {suffix}": wandb.Image(os.path.join(out_dir, f"metrics_bar{suffix}.png"))})
    except: pass

    # 3. ROC Curve (Multiclass)
    if y_scores is not None:
        try:
            fig_roc = go.Figure()
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            
            y_bin = np.eye(len(class_names))[y_true]
            
            for i, name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name}", mode='lines'))

            fig_roc.update_layout(
                title=f"ROC Curves {suffix}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                width=800, height=600
            )
            fig_roc.write_image(os.path.join(out_dir, f"roc_curve{suffix}.png"), scale=3)
            try: wandb.log({f"ROC Curve {suffix}": wandb.Image(os.path.join(out_dir, f"roc_curve{suffix}.png"))})
            except: pass
        except Exception as e:
            print(f"Skipping ROC due to error: {e}")

def plot_signals(dataset, num_samples=3, out_dir="./sample_signals"):
    """Save sample signals to check data quality"""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(num_samples, len(dataset))):
        sig, feat, lbl = dataset[i]
        sig_np = sig.numpy()
        
        fig = go.Figure()
        for ch in range(sig_np.shape[0]):
            fig.add_trace(go.Scatter(y=sig_np[ch, :], mode='lines', name=f'Ch {ch}'))
        
        fig.update_layout(
            title=f"Sample Signal {i} (Class {lbl})",
            xaxis_title="Time", yaxis_title="Amplitude",
            template="plotly_white"
        )
        fig.write_image(os.path.join(out_dir, f"signal_{i}.png"), scale=2)
        try: wandb.log({f"Signal Sample {i}": wandb.Image(os.path.join(out_dir, f"signal_{i}.png"))})
        except: pass

# ======================================================
# 4. TRAINING COMPONENTS
# ======================================================
def mixup_data(x, y, alpha=0.4, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class TTAPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, x, feat):
        self.model.eval()
        with torch.no_grad():
            out1 = F.softmax(self.model(x, feat), dim=1)
            x_shift1 = torch.roll(x, shifts=5, dims=2)
            out2 = F.softmax(self.model(x_shift1, feat), dim=1)
            x_shift2 = torch.roll(x, shifts=-5, dims=2)
            out3 = F.softmax(self.model(x_shift2, feat), dim=1)
            out4 = F.softmax(self.model(x * 0.98, feat), dim=1)
            return (out1 + out2 + out3 + out4) / 4.0

# ======================================================
# 5. TRAINING MANAGER
# ======================================================
class TrainingManager:
    def __init__(self, model, device, lr=1e-3, alpha_lr_scale=1.0):
        self.model = model.to(device)
        self.device = device
        self.best_acc = 0.0
        
        params_r, params_g, params_b, params_other = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if 'conv_g' in name: params_g.append(p)
            elif 'conv_b' in name: params_b.append(p)
            elif 'conv_r' in name: params_r.append(p)
            else: params_other.append(p)
        
        param_groups = [
            {"params": params_other, "lr": lr, "weight_decay": 1e-4},
            {"params": params_r, "lr": lr, "weight_decay": 1e-4},
            {"params": params_g, "lr": lr * alpha_lr_scale, "weight_decay": 1e-4},
            {"params": params_b, "lr": lr * alpha_lr_scale, "weight_decay": 1e-4},
        ]
        
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = None
        print(f"Optimizer initialized (FP32 Mode) - Gating: {len(params_g)}, Quad: {len(params_b)} params.")

    def set_scheduler(self, train_loader, epochs):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=[g['lr'] for g in self.optimizer.param_groups], 
            steps_per_epoch=len(train_loader), 
            epochs=epochs, 
            pct_start=0.3
        )

    def train_epoch(self, loader, criterion, epoch, use_mixup=True):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(loader, desc=f"Ep {epoch}", leave=False)
        
        for x, feat, y in pbar:
            x, feat, y = x.to(self.device), feat.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            if use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, device=self.device)
                out = self.model(mixed_x, feat)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                out = self.model(x, feat)
                loss = criterion(out, y)
            
            if torch.isnan(loss):
                print(f"[Warning] NaN loss in Ep {epoch}. Batch skipped.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            
            total_loss += loss.item() * x.size(0)
            if not use_mixup:
                correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / (total + 1e-5), (100.0 * correct / (total + 1e-5)) if not use_mixup else 0.0

    def validate(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, feat, y in loader:
                x, feat, y = x.to(self.device), feat.to(self.device), y.to(self.device)
                out = self.model(x, feat)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        return total_loss / (total + 1e-5), 100.0 * correct / (total + 1e-5)

# ======================================================
# 6. MAIN EXECUTION FLOW
# ======================================================
def train_single_model(model_idx, seed, args, device, logger, num_classes, feat_dim, class_names):
    seed_everything(seed)
    logger.info(f"\n>>> Training Ensemble Member {model_idx+1} (Seed={seed}) <<<")
    
    ds_train = RobustBearingDataset(args.npz, mode="train", load_into_memory=True, instance_norm=True)
    ds_val = RobustBearingDataset(args.npz, mode="val", snr=-6, load_into_memory=True, instance_norm=True)
    
    # NEW: Visualize samples for the first model
    if model_idx == 0:
        plot_signals(ds_train, num_samples=3, out_dir=os.path.join(args.save_dir, "vis_train"))

    train_loader = DataLoader(ds_train, batch_size=args.bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.bs, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=num_classes, feat_dim=feat_dim).to(device)
    manager = TrainingManager(model, device, lr=args.lr, alpha_lr_scale=args.alpha)
    manager.set_scheduler(train_loader, args.epochs)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    save_path = os.path.join(args.save_dir, f"model_seed{seed}.pth")
    early_stop = EarlyStopping(patience=20, verbose=True, path=save_path)
    
    history = {'train_loss': [], 'val_acc': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        t_loss, _ = manager.train_epoch(train_loader, criterion, epoch, use_mixup=True)
        v_loss, v_acc = manager.validate(val_loader, criterion)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        logger.info(f"M{model_idx+1} Ep {epoch:02d} | Loss: {t_loss:.4f} | Val Acc: {v_acc:.2f}%")
        wandb.log({f"M{model_idx+1}_Train_Loss": t_loss, f"M{model_idx+1}_Val_Acc": v_acc})
        
        early_stop(v_acc, model)
        if early_stop.early_stop:
            logger.info("Early stopping triggered.")
            break
    
    # NEW: Plot History
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=history['train_loss'], name="Train Loss"), secondary_y=False)
    fig.add_trace(go.Scatter(y=history['val_loss'], name="Val Loss"), secondary_y=False)
    fig.add_trace(go.Scatter(y=history['val_acc'], name="Val Acc"), secondary_y=True)
    fig.update_layout(title=f"Training History Model {model_idx+1}")
    fig.write_image(os.path.join(args.save_dir, f"history_model_{model_idx+1}.png"))
    try: wandb.log({f"History M{model_idx+1}": wandb.Image(os.path.join(args.save_dir, f"history_model_{model_idx+1}.png"))})
    except: pass
            
    return save_path

def main():
    parser = argparse.ArgumentParser()
    
    # --- Paths for [0, 1, 2, 3 HP] Dataset ---
    default_npz = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\1HP\processed_data_robust_1hp\processed_data_robust_v3.npz"
    default_meta = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\1HP\processed_data_robust_1hp\meta.json"
    default_save = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_1hp"
    
    parser.add_argument('--npz', type=str, default=default_npz)
    parser.add_argument('--meta', type=str, default=default_meta)
    parser.add_argument('--save-dir', type=str, default=default_save)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", nargs='+', type=int, default=[42, 101, 2024])
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.save_dir)
    
    wandb.init(project="Bearing_Fault_SOTA_0HP", entity="alinaderi119-upwork", mode="disabled", config=args, name="Ultimate_Viz_Run_0HP")
    
    logger.info(f"Pipeline V2 (Viz Edition 0HP) Started | Device: {device}")
    
    # 1. Meta
    with open(args.meta, 'r') as f: meta = json.load(f)
    num_classes = len(meta['class_map'])
    class_names = [meta['idx_to_label'].get(str(i), str(i)) for i in range(num_classes)]
    
    # Dummy load
    try:
        temp_ds = RobustBearingDataset(args.npz, mode="train", load_into_memory=False)
        feat_dim = temp_ds[0][1].shape[0]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Train Ensemble
    saved_models = []
    for idx, seed in enumerate(args.seeds):
        path = train_single_model(idx, seed, args, device, logger, num_classes, feat_dim, class_names)
        saved_models.append(path)
        
    # 3. Ensemble Evaluation + Advanced Visualization
    logger.info("\n>>> Starting Ensemble Evaluation & Visualization <<<")
    models = []
    for p in saved_models:
        m = QDCA_Hybrid_Pro(in_channels=3, num_classes=num_classes, feat_dim=feat_dim).to(device)
        m.load_state_dict(torch.load(p))
        m.eval()
        models.append(m)
        
    full_data = np.load(args.npz, allow_pickle=True)
    snrs = ['Clean'] + meta.get('snr_list', [-4, -2, 0, 2, 4])
    
    results = []
    snr_minus_10_data = {'y_true': [], 'y_pred': [], 'y_scores': []}
    
    print(f"\n{'SNR':<10} | {'Acc Ensemble':<12}")
    print("-" * 30)
    
    for snr in snrs:
        k_sig = 'X_test_sig' if snr == 'Clean' else f'test_snr_{snr}_sig'
        k_feat = 'X_test_feat' if snr == 'Clean' else f'test_snr_{snr}_feat'
        
        if k_sig in full_data:
            x_np = full_data[k_sig].astype(np.float32)
            f_np = full_data[k_feat].astype(np.float32)
            y_np = full_data['y_test'].astype(np.int64)
            
            mean = np.mean(x_np, axis=2, keepdims=True)
            std = np.std(x_np, axis=2, keepdims=True) + 1e-8
            x_np = (x_np - mean) / std
            
            ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(y_np))
            loader = DataLoader(ds, batch_size=args.bs, num_workers=0)
            
            total, correct_e = 0, 0
            
            for x, feat, y in loader:
                x, feat, y = x.to(device), feat.to(device), y.to(device)
                ens_probs = torch.zeros(x.size(0), num_classes).to(device)
                for m in models:
                    tta = TTAPredictor(m, device)
                    ens_probs += tta.predict(x, feat)
                
                pred_e = ens_probs.argmax(1)
                total += x.size(0)
                correct_e += (pred_e == y).sum().item()
                
                if str(snr) == "-10":
                    snr_minus_10_data['y_true'].extend(y.cpu().numpy())
                    snr_minus_10_data['y_pred'].extend(pred_e.cpu().numpy())
                    snr_minus_10_data['y_scores'].extend(ens_probs.cpu().numpy())

            acc_e = 100.0 * correct_e / total
            print(f"{str(snr):<10} | {acc_e:.2f}%")
            results.append({'snr': snr, 'acc': acc_e})
            wandb.log({f"Acc_{snr}": acc_e})
            
    pd.DataFrame(results).to_csv(os.path.join(args.save_dir, "final_results.csv"), index=False)
    
    # 4. Generate Advanced Plots for SNR -10
    if len(snr_minus_10_data['y_true']) > 0:
        logger.info("Generating Pro Charts for SNR -10dB...")
        generate_advanced_charts(
            np.array(snr_minus_10_data['y_true']),
            np.array(snr_minus_10_data['y_pred']),
            np.array(snr_minus_10_data['y_scores']),
            class_names,
            args.save_dir,
            suffix="_SNR-10"
        )
        
    logger.info(f"Complete. Results saved to {args.save_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()


