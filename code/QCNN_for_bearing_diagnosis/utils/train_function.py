# train_function_fixed.py
# -*- coding: utf-8 -*-
"""
Robust training loop for bearing-fault models.
Fixes:
 - safe device handling
 - supports dataset batches of (x,y) or (x,hht,y)
 - safer model import and in_channels handling
 - TensorBoard logging (loss / acc / f1)
 - consistent checkpoint naming and best-model load
 - mixed precision support (autocast/scaler) safe for optional hht
"""
from __future__ import annotations

import os
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter

# -------------------------------
# Dataset Import (از فایل اصلاح‌شده)
# -------------------------------
# from DatasetLoader import BearingFaultDataset  # assumed available
# We'll import dynamically inside train_model to better control errors.

# -------------------------------
# Metrics & Evaluation (supports x,y or x,hht,y)
# -------------------------------
def _unpack_batch(batch):
    """Return (x_tensor, maybe_hht_or_None, y_tensor)"""
    if len(batch) == 2:
        x, y = batch
        return x, None, y
    elif len(batch) == 3:
        x, hht, y = batch
        return x, hht, y
    else:
        # try to be permissive if dataset yields dict
        b = batch
        if isinstance(b, dict):
            x = b.get("x") or b.get("X") or b.get("input")
            y = b.get("y") or b.get("label")
            hht = b.get("hht", None)
            return x, hht, y
        raise ValueError("Unsupported batch format from dataset; expected (x,y) or (x,hht,y) or dict.")

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval"
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, hht, y = _unpack_batch(batch)
            x = x.to(device)
            if hht is not None:
                hht = hht.to(device)
                logits = model(x, hht)  # model must accept optional hht
            else:
                logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())
    if len(all_preds) == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"[{desc}] Acc: {acc:.4f} | F1: {f1:.4f}")
    return acc, f1, preds, labels

# -------------------------------
# Checkpoint Management (consistent names)
# -------------------------------
def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str, epoch: int, val_f1: float, is_best: bool = False) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    fname = f"epoch{epoch:03d}_f1{val_f1:.4f}.pt"
    path = os.path.join(checkpoint_dir, fname)
    torch.save(state, path)
    if is_best:
        best_name = f"epoch{epoch:03d}_f1{val_f1:.4f}_best.pt"
        best_path = os.path.join(checkpoint_dir, best_name)
        torch.save(state, best_path)
        return best_path
    return path

# -------------------------------
# Training Loop
# -------------------------------
def train_model(
    npz_path: str,
    model: nn.Module,
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    mixed_precision: bool = True,
    no_cuda: bool = False,
    patience: int = 7,
    grad_clip: float = 5.0,
    use_tensorboard: bool = True
) -> Dict[str, Any]:
    # ------------- device -------------
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    print(f"[Device] Using: {device}")
    model = model.to(device)

    # ------------- dataset import (dynamic) -------------
    try:
        from DatasetLoader import BearingFaultDataset
    except Exception as e:
        raise ImportError(f"Cannot import BearingFaultDataset from DatasetLoader: {e}")

    train_ds = BearingFaultDataset(npz_path, split="train")
    valid_ds = BearingFaultDataset(npz_path, split="valid")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'))

    print(f"[Data] Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    # ------------- optimizer / scheduler / criterion -------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if (mixed_precision and device.type == 'cuda') else None

    # ------------- logging -------------
    tb_writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb")) if use_tensorboard else None

    best_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_acc": [], "val_f1": []}
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_preds, train_labels = [], []
        start_time = time.time()

        for batch in train_loader:
            x, hht, y = _unpack_batch(batch)
            x = x.to(device)
            if hht is not None:
                hht = hht.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    logits = model(x, hht) if hht is not None else model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x, hht) if hht is not None else model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_preds.append(preds.detach().cpu().numpy())
            train_labels.append(y.detach().cpu().numpy())

        # metrics
        train_preds = np.concatenate(train_preds) if train_preds else np.array([])
        train_labels = np.concatenate(train_labels) if train_labels else np.array([])
        train_acc = accuracy_score(train_labels, train_preds) if train_preds.size else 0.0
        train_f1 = f1_score(train_labels, train_preds, average='macro') if train_preds.size else 0.0
        train_loss = epoch_loss / max(1, len(train_loader.dataset))

        # validation
        val_acc, val_f1, val_preds, val_labels = evaluate_model(model, valid_loader, device, "Valid")

        # scheduler / history / tb
        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("acc/train", train_acc, epoch)
            tb_writer.add_scalar("f1/train", train_f1, epoch)
            tb_writer.add_scalar("acc/val", val_acc, epoch)
            tb_writer.add_scalar("f1/val", val_f1, epoch)
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        elapsed = time.time() - start_time
        print(f"[Epoch {epoch:02d}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} F1: {val_f1:.4f} | Time: {elapsed:.1f}s")

        # checkpoint
        is_best = val_f1 > best_f1
        saved_path = save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'history': history
        }, checkpoint_dir, epoch, val_f1, is_best=is_best)
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            print(f"  -> New best model saved: {saved_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # final: load best if exists
    best_ckpt_pattern = f"epoch{best_epoch:03d}_f1{best_f1:.4f}_best.pt"
    best_ckpt = os.path.join(checkpoint_dir, best_ckpt_pattern)
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_ckpt}")
    else:
        print("No best checkpoint found (maybe use last saved).")

    # final eval and confusion matrix
    _, _, val_preds, val_labels = evaluate_model(model, valid_loader, device, "Final Valid")
    if val_preds.size:
        cm = confusion_matrix(val_labels, val_preds)
    else:
        cm = np.zeros((1, 1), dtype=int)

    # save confusion matrix figure
    try:
        sns.set_theme()
        plt.figure(figsize=(10, 8))
        n_classes = cm.shape[0]
        labels = [f"C{i}" for i in range(n_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix (Best F1: {best_f1:.4f})")
        plt.ylabel("True"); plt.xlabel("Predicted")
        cm_path = os.path.join(checkpoint_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {cm_path}")
    except Exception as e:
        print(f"Failed to save confusion matrix: {e}")

    if tb_writer:
        tb_writer.close()

    return {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'history': history,
        'confusion_matrix': cm,
        'model': model
    }

# -------------------------------
# CLI Interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bearing Fault Diagnosis Model")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to preprocessed .npz file (from Preprocess.py)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--in_channels", type=int, default=12, help="Model input channels (default 12 for preprocess_v6_final)")
    args = parser.parse_args()

    # safe device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Try import model names you use; allow fallback dummy
    model = None
    model_candidates = [
        ("Model.QDCA_BiGRU_Res_Model", "QDCA_BiGRU_Res_Final"),
        ("Model.QDCA_BiGRU_Res_Model", "QDCA_BiGRU_Res"),
        ("Model.QDCA_BiGRU_Res", "QDCA_BiGRU_Res_Final"),
        ("Model.QDCA_BiGRU_Res_HHT", "QDCA_BiGRU_Res_HHT_Fusion"),
        ("Model.QDCA_BiGRU_Res_Model", "QDCA_BiGRU_Res_HHT_Fusion"),
    ]
    for modname, clsname in model_candidates:
        try:
            mod = __import__(modname, fromlist=[clsname])
            cls = getattr(mod, clsname)
            model = cls(in_channels=args.in_channels, num_classes=10)
            print(f"Model loaded: {clsname} from {modname}")
            break
        except Exception:
            model = None
    if model is None:
        print("Model import failed for candidates. Using simple dummy model for smoke test.")
        class DummyModel(nn.Module):
            def __init__(self, in_ch=args.in_channels, num_classes=10):
                super().__init__()
                self.conv = nn.Conv1d(in_ch, 16, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(16, num_classes)
            def forward(self, x, hht=None):
                x = self.conv(x)
                x = self.pool(x).squeeze(-1)
                return self.fc(x)
        model = DummyModel().to(device)

    # run training
    result = train_model(
        npz_path=args.npz_path,
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        no_cuda=args.no_cuda,
        patience=args.patience
    )

    print("\nTraining completed successfully!")


