# -*- coding: utf-8 -*-
"""
cross_load_eval.py
Author: Ali Naderi - 2025
Description: 
    Performs Cross-Load Evaluation (Generalization Test).
    - Scenario: Train on 1HP -> Test on 0HP
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- 1. SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
if os.path.exists(os.path.join(base_dir, 'utils')):
    sys.path.append(os.path.join(base_dir, 'utils'))
if os.path.exists(os.path.join(base_dir, 'Model')):
    sys.path.append(os.path.join(base_dir, 'Model'))

try:
    from Model.model import QDCA_Hybrid_Pro
except ImportError:
    from Model.model import QDCA_Hybrid_Pro

# --- 2. CONFIGURATION (CRITICAL PART - MODIFIED FOR 1HP -> 0HP) ---

# A. مدل آموزش دیده روی 1HP (مغز مدل)
# فرض بر این است که مدل نهایی 1HP در این مسیر ذخیره شده است
MODEL_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_ultra_1hp\model_seed42.pth"

# B. داده‌های تست 0HP (داده‌های جدید برای مدل)
# از دیتاست اصلی 0HP استفاده می‌کنیم
TEST_DATA_NPZ = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust\processed_data_robust_v3.npz"
TEST_DATA_META = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust\meta.json"

# خروجی در پوشه جدید
OUTPUT_DIR = r"./cross_load_results_1hp_to_0hp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=== Cross-Load Evaluation: Train 1HP -> Test 0HP ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Metadata (From 0HP test data to ensure correct reading)
    if not os.path.exists(TEST_DATA_META):
        print(f"Error: Meta file not found at {TEST_DATA_META}")
        return

    with open(TEST_DATA_META, 'r') as f: meta = json.load(f)
    num_classes = len(meta['class_map'])
    print(f"Target Classes in Test Data: {num_classes}")

    # 2. Load Model (Weights from 1HP training)
    print(f"Loading 1HP Model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Note: feat_dim is usually 33. We assume it matches.
    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=num_classes, feat_dim=33).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()

    # 3. Load 0HP Data
    print(f"Loading 0HP Data from: {TEST_DATA_NPZ}")
    full_data = np.load(TEST_DATA_NPZ, allow_pickle=True)
    snrs = ['Clean'] + meta.get('snr_list', [-10, -8, -6, -4, -2, 0, 2, 4, 6])
    
    results = []
    print(f"\n{'SNR':<10} | {'Cross-Load Accuracy':<20}")
    print("-" * 35)

    for snr in snrs:
        # Construct keys
        if snr == 'Clean':
            k_sig, k_feat = 'X_test_sig', 'X_test_feat'
        else:
            k_sig, k_feat = f'test_snr_{snr}_sig', f'test_snr_{snr}_feat'
        
        if k_sig not in full_data:
            continue
            
        # Extract Data
        x_np = full_data[k_sig].astype(np.float32)
        f_np = full_data[k_feat].astype(np.float32)
        y_np = full_data['y_test'].astype(np.int64)
        
        # Instance Norm (Critical for Cross-Load)
        # Normalizes 0HP data so it "looks like" the 1HP data distribution to the model
        mean = np.mean(x_np, axis=2, keepdims=True)
        std = np.std(x_np, axis=2, keepdims=True) + 1e-8
        x_np = (x_np - mean) / std
        
        # DataLoader
        ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(y_np))
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        
        # Inference
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, f, y in loader:
                x, f = x.to(DEVICE), f.to(DEVICE)
                logits = model(x, f)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.numpy())
        
        acc = accuracy_score(all_targets, all_preds) * 100
        print(f"{str(snr):<10} | {acc:.2f}%")
        results.append({'SNR': snr, 'Accuracy': acc})

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "cross_load_1hp_to_0hp.csv"), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
