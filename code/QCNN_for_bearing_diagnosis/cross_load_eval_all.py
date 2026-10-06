# -*- coding: utf-8 -*-
"""
cross_load_eval_all.py
Author: Ali Naderi - 2025 (Scientific Edition)
Description: 
    Comprehensive Cross-Load Generalization Matrix for CWRU Dataset.
    Evaluates model transferability across 4 working loads (0, 1, 2, 3 HP).
    
    Output:
    1. Cross-Load Accuracy Matrix (CSV)
    2. Cross-Load Heatmap (High-Res PNG/HTML)
    3. Generalization Gap Analysis
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# --- 1. SETUP PATHS & IMPORTS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
if os.path.exists(os.path.join(base_dir, 'utils')):
    sys.path.append(os.path.join(base_dir, 'utils'))
if os.path.exists(os.path.join(base_dir, 'Model')):
    sys.path.append(os.path.join(base_dir, 'Model'))

try:
    from Model.model import QDCA_Hybrid_Pro
except ImportError:
    try:
        from Model.model import QDCA_Hybrid_Pro
    except:
        print("[Error] Model class not found.")
        sys.exit(1)

# --- 2. CONFIGURATION MATRIX ---
# تعریف مسیرها برای هر ۴ بار کاری
# فرض بر این است که شما مراحل Preprocess و Train را برای هر ۴ بار انجام داده‌اید
# اگر هنوز مدلی را آموزش نداده‌اید، آن سطر در ماتریس خالی می‌ماند.

ROOT_DIR = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_CONFIG = {
    "0HP": {
        "npz": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust\processed_data_robust_v3.npz"),
        "meta": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust\meta.json"),
        "ckpt": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_0hp\model_seed42.pth") # مدل آموزش دیده روی 0HP
    },
    "1HP": {
        "npz": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\1HP\processed_data_robust_1hp\processed_data_robust_v3.npz"),
        "meta": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\1HP\processed_data_robust_1hp\meta.json"),
        "ckpt": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_1hp\model_seed42.pth")
    },
    "2HP": {
        "npz": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\2HP\processed_data_robust_2hp\processed_data_robust_v3.npz"),
        "meta": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\2HP\processed_data_robust_2hp\meta.json"),
        "ckpt": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_2hp\model_seed42.pth") # (فرض: این مدل باید آموزش دیده باشد)
    },
    "3HP": {
        "npz": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"),
        "meta": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\meta.json"),
        "ckpt": os.path.join(ROOT_DIR, r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_3hp\model_seed42.pth") # (فرض: این مدل باید آموزش دیده باشد)
    }
}

OUTPUT_DIR = r"./cross_load_final_report"

# --- 3. EVALUATION ENGINE ---
class CrossLoadEvaluator:
    def __init__(self, config):
        self.config = config
        self.loads = list(config.keys())
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_model(self, train_load):
        """Loads the model trained on a specific load."""
        cfg = self.config[train_load]
        if not os.path.exists(cfg["ckpt"]):
            print(f"   [Warning] Checkpoint not found for {train_load}. Skipping row.")
            return None, None

        # Load Meta to get classes
        with open(cfg["meta"], 'r') as f: meta = json.load(f)
        num_classes = len(meta['class_map'])
        
        # Load Dummy Data to get feat_dim
        try:
            # Quick hack to get feature dim without loading full dataset class
            data = np.load(cfg["npz"], allow_pickle=True)
            feat_dim = data['X_train_feat'].shape[1]
        except:
            feat_dim = 33

        model = QDCA_Hybrid_Pro(in_channels=3, num_classes=num_classes, feat_dim=feat_dim).to(DEVICE)
        model.load_state_dict(torch.load(cfg["ckpt"], map_location=DEVICE))
        model.eval()
        return model, num_classes

    def get_test_loader(self, test_load, batch_size=64):
        """Loads test data for a specific load."""
        cfg = self.config[test_load]
        if not os.path.exists(cfg["npz"]):
            print(f"   [Warning] Data not found for {test_load}.")
            return None

        data = np.load(cfg["npz"], allow_pickle=True)
        
        # We use 'X_test_sig' (Clean) or we can choose a specific SNR like 'test_snr_-10_sig'
        # For Cross-Load tables in papers, usually CLEAN data is used to isolate load effect.
        # If you want noise robustness + cross load, change key to f'test_snr_-10_sig'
        SIG_KEY = 'X_test_sig' 
        FEAT_KEY = 'X_test_feat'
        
        if SIG_KEY not in data:
            return None
            
        x_np = data[SIG_KEY].astype(np.float32)
        f_np = data[FEAT_KEY].astype(np.float32)
        y_np = data['y_test'].astype(np.int64)

        # Instance Normalization (Crucial for Cross-Domain)
        mean = np.mean(x_np, axis=2, keepdims=True)
        std = np.std(x_np, axis=2, keepdims=True) + 1e-8
        x_np = (x_np - mean) / std

        ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(f_np), torch.from_numpy(y_np))
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def run(self):
        # Matrix to store results
        results_matrix = pd.DataFrame(index=self.loads, columns=self.loads)
        
        print(f"{'='*60}")
        print(f"Starting Cross-Load Evaluation Matrix ({len(self.loads)}x{len(self.loads)})")
        print(f"{'='*60}")

        for train_load in self.loads:
            print(f"\n[Source Domain: {train_load}] Loading Model...")
            model, _ = self.load_model(train_load)
            
            if model is None:
                continue

            for test_load in self.loads:
                # print(f"   -> Testing on Target Domain: {test_load}...", end="")
                loader = self.get_test_loader(test_load)
                
                if loader is None:
                    results_matrix.loc[train_load, test_load] = "N/A"
                    print(" No Data")
                    continue

                all_preds, all_labels = [], []
                with torch.no_grad():
                    for x, f, y in loader:
                        x, f = x.to(DEVICE), f.to(DEVICE)
                        logits = model(x, f)
                        preds = logits.argmax(dim=1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y.numpy())
                
                acc = accuracy_score(all_labels, all_preds) * 100
                results_matrix.loc[train_load, test_load] = acc
                print(f" -> Target {test_load}: {acc:.2f}%")

        return results_matrix

# --- 4. VISUALIZATION ---
def plot_cross_load_heatmap(df):
    """Generates a publication-quality heatmap."""
    # Convert to numeric, handle N/A
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Calculate Averages
    diag_vals = np.diag(df)
    off_diag_mask = ~np.eye(df.shape[0], dtype=bool)
    off_diag_vals = df.values[off_diag_mask]
    
    avg_same = np.mean(diag_vals)
    avg_cross = np.mean(off_diag_vals)
    
    print(f"\n{'-'*40}")
    print(f"Average Same-Load Accuracy:  {avg_same:.2f}%")
    print(f"Average Cross-Load Accuracy: {avg_cross:.2f}%")
    print(f"Generalization Gap:          {avg_same - avg_cross:.2f}%")
    print(f"{'-'*40}")

    # Plot
    z_text = [[f"{val:.1f}%" for val in row] for row in df.values]
    
    fig = ff.create_annotated_heatmap(
        z=df.values,
        x=list(df.columns),
        y=list(df.index),
        annotation_text=z_text,
        colorscale='Viridis',
        showscale=True
    )

    fig.update_layout(
        title=dict(
            text="<b>Cross-Load Generalization Matrix</b>",
            y=0.9, x=0.5, xanchor='center', yanchor='top',
            font=dict(family="Times New Roman", size=20)
        ),
        xaxis=dict(title="<b>Target Load (Test)</b>", side="bottom", tickfont=dict(family="Times New Roman", size=14)),
        yaxis=dict(title="<b>Source Load (Train)</b>", tickfont=dict(family="Times New Roman", size=14)),
        width=700, height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        font=dict(family="Times New Roman", size=14)
    )
    
    # Save
    fig.write_image(os.path.join(OUTPUT_DIR, "cross_load_heatmap.png"), scale=3)
    fig.write_html(os.path.join(OUTPUT_DIR, "cross_load_heatmap.html"))
    print(f"\nHeatmap saved to {OUTPUT_DIR}")

# --- MAIN ---
def main():
    evaluator = CrossLoadEvaluator(LOAD_CONFIG)
    results = evaluator.run()
    
    # Save Raw Data
    results.to_csv(os.path.join(OUTPUT_DIR, "cross_load_matrix.csv"))
    print(f"\nMatrix saved to {os.path.join(OUTPUT_DIR, 'cross_load_matrix.csv')}")
    
    # Plot
    plot_cross_load_heatmap(results)

if __name__ == "__main__":
    main()
