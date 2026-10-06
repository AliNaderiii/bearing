# -*- coding: utf-8 -*-
"""
generate_qttention_maps.py
Author: Ali Naderi - 2025
Description: 
    Extracts and saves feature maps/attention maps from the trained QDCA-Net model.
    Useful for visualizing what the model "sees" (Interpretability).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. SETUP PATHS ---
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
if os.path.exists(os.path.join(base_dir, 'utils')):
    sys.path.append(os.path.join(base_dir, 'utils'))
if os.path.exists(os.path.join(base_dir, 'Model')):
    sys.path.append(os.path.join(base_dir, 'Model'))

try:
    from utils.DatasetLoader import RobustBearingDataset, collate_fn
    # Try importing your specific model
    try:
        from Model.model import QDCA_Hybrid_Pro
    except ImportError:
        from Model.model import QDCA_Hybrid_Pro
except ImportError as e:
    print(f"[Critical Error] Setup failed: {e}")
    sys.exit(1)

# --- Global Hooks Storage ---
activations = {}

def get_activation(name):
    def hook(model, input, output):
        # Detach and move to CPU to save memory
        activations[name] = output.detach().cpu()
    return hook

def main():
    parser = argparse.ArgumentParser(description="Generate Qttention/Feature Maps")
    
    # Paths (Adjust defaults as needed)
    default_npz = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
    default_ckpt = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_3hp\model_seed42.pth"
    
    parser.add_argument('--npz', type=str, default=default_npz, help="Path to data NPZ")
    parser.add_argument('--ckpt', type=str, default=default_ckpt, help="Path to trained model checkpoint")
    parser.add_argument('--out_dir', type=str, default="data/qmaps_3hp", help="Output directory for maps")
    parser.add_argument('--snr', type=int, default=-10, help="SNR level to test (e.g. -10)")
    parser.add_argument('--samples', type=int, default=10, help="Number of samples per class to visualize")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"--- Feature Map Extraction Tool ---\nDevice: {device}\nSNR: {args.snr} dB")

    # 1. Load Data
    print("Loading dataset...")
    try:
        # We use the test set with specific SNR
        test_ds = RobustBearingDataset(
            args.npz, 
            mode='test', 
            snr=args.snr, 
            load_into_memory=True, 
            instance_norm=True
        )
        # Create a small loader just to get feature dims or iterate manually
        loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        
        # Get input dimensions from first sample
        sample_x, sample_f, _ = test_ds[0]
        feat_dim = sample_f.shape[0]
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Model
    print(f"Loading model from {args.ckpt}...")
    try:
        model = QDCA_Hybrid_Pro(in_channels=3, num_classes=10, feat_dim=feat_dim).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Register Hooks
    # We want to visualize the output of the Deep Stage (Quadratic layers) and maybe Attention
    # Adjust layer names based on your model definition
    # Example: model.deep_stage[2] is likely the last QResBlock
    
    print("Registering hooks on target layers...")
    # Hook the output of the deep feature extractor (before Transformer)
    model.deep_stage.register_forward_hook(get_activation('deep_stage_out'))
    
    # Optional: Hook specific internal layers if needed
    # model.stem.register_forward_hook(get_activation('stem_out'))

    # 4. Extract Features
    print(f"Extracting features for {args.samples} samples per class...")
    
    classes_found = {i: 0 for i in range(10)}
    saved_count = 0
    
    # Containers for metadata
    meta_data = []

    with torch.no_grad():
        for i, (x, f, y) in enumerate(tqdm(loader)):
            label = y.item()
            
            # Check if we need more samples for this class
            if classes_found[label] >= args.samples:
                if all(c >= args.samples for c in classes_found.values()):
                    break # Done collecting
                continue
            
            x, f = x.to(device), f.to(device)
            
            # Forward pass (triggers hooks)
            logits = model(x, f)
            pred = logits.argmax(dim=1).item()
            
            # Get captured feature map
            # Shape is likely (1, Channels, Length) e.g., (1, 128, 128)
            feat_map = activations['deep_stage_out'].squeeze(0).numpy() 
            
            # Save Feature Map as CSV
            # We flatten it or save 2D. Let's save as 2D (Channels x Time)
            filename = f"qmap_class{label}_idx{i}_pred{pred}.csv"
            filepath = os.path.join(args.out_dir, filename)
            
            # Save raw feature map (rows=channels, cols=time)
            pd.DataFrame(feat_map).to_csv(filepath, header=False, index=False)
            
            # Record metadata
            meta_data.append({
                "file": filename,
                "true_label": label,
                "pred_label": pred,
                "snr": args.snr,
                "correct": label == pred
            })
            
            classes_found[label] += 1
            saved_count += 1

    # Save index
    pd.DataFrame(meta_data).to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    print(f"\n[Done] Saved {saved_count} feature maps to {args.out_dir}")
    print("You can now plot these CSV files as heatmaps to visualize 'Qttention'.")

if __name__ == "__main__":
    main()