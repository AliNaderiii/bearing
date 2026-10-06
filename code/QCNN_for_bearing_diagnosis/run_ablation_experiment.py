# -*- coding: utf-8 -*-
"""
run_ablation_experiment.py
Author: Ali Naderi
Description: Trains and Evaluates 5 variants of the model for Ablation Study table.
(Silent Mode: Cleaner logs for non-interactive terminals)
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# --- Imports ---
from utils.DatasetLoader import RobustBearingDataset, collate_fn
from Model.AblationModel import QDCA_Ablation

# --- CONFIG ---
VARIANTS = ['full', 'no_quad', 'no_attn', 'no_trans', 'no_stat']
EPOCHS = 20
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
SAVE_DIR = "./ablation_results"

def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = np.mean(FP / (FP + TN + 1e-12))
    
    return acc*100, f1*100, precision*100, recall*100, FPR*100

def train_and_eval(variant):
    print(f"\n{'='*40}")
    print(f"  Processing Variant: {variant.upper()}")
    print(f"{'='*40}")
    
    # 1. Setup Data
    train_ds = RobustBearingDataset(DATA_PATH, mode='train', load_into_memory=True)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # 2. Setup Model
    model = QDCA_Ablation(in_channels=3, num_classes=10, feat_dim=33, mode=variant).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 3. Train Loop
    model.train()
    for ep in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        # Standard loop without tqdm to avoid log flooding
        for x, f, y in loader:
            x, f, y = x.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, f)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        print(f"Ep {ep+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    # 4. Evaluate on Different SNRs
    results = {}
    snrs = [-10, -8, -6, 0] 
    
    model.eval()
    print("\n  Running Evaluation...")
    for snr in snrs:
        try:
            test_ds = RobustBearingDataset(DATA_PATH, mode='test', snr=snr, load_into_memory=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
            
            y_true, y_pred = [], []
            with torch.no_grad():
                for x, f, y in test_loader:
                    x, f = x.to(DEVICE), f.to(DEVICE)
                    out = model(x, f)
                    pred = out.argmax(dim=1)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
            
            acc, f1, prec, rec, fpr = calculate_metrics(y_true, y_pred)
            results[f"{snr}dB"] = [acc, f1, fpr]
            print(f"   SNR {snr}dB -> Acc: {acc:.2f}% | F1: {f1:.2f}% | FPR: {fpr:.2f}%")
        except Exception as e:
            print(f"   SNR {snr}dB -> Failed: {e}")
            results[f"{snr}dB"] = [0, 0, 0]
        
    return results

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data = []
    
    for var in VARIANTS:
        metrics = train_and_eval(var)
        
        for snr, vals in metrics.items():
            all_data.append({
                "Model Variant": var,
                "SNR": snr,
                "Accuracy (%)": vals[0],
                "F1-Score (%)": vals[1],
                "FPR (%)": vals[2]
            })
            
    df = pd.DataFrame(all_data)
    output_path = os.path.join(SAVE_DIR, "ablation_study_report.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n{'-'*50}")
    print(f"Ablation study complete! Results saved to:\n{os.path.abspath(output_path)}")
    print(f"{'-'*50}")
    
    if not df.empty:
        print("\nQuick Summary (SNR = -10dB):")
        print(df[df['SNR'] == '-10dB'][['Model Variant', 'Accuracy (%)', 'FPR (%)']].to_string(index=False))

if __name__ == "__main__":
    main()
