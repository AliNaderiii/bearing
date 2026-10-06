# -*- coding: utf-8 -*-
"""
Fix_Meta_PU.py
Author: AI Assistant
Description: Generates the missing 'meta.json' file for Paderborn Dataset
             by inspecting the existing .npy files.
"""
import os
import json
import numpy as np

# Config - must match your Preprocess settings
DATA_DIR = r"D:\processed_pu_final"
SEQ_LEN = 2048

def generate_meta():
    if not os.path.exists(DATA_DIR):
        print(f"[Error] Directory not found: {DATA_DIR}")
        return

    counts = {}
    
    # Calculate counts from y labels (int64 = 8 bytes per sample)
    for split in ['train', 'val', 'test']:
        y_path = os.path.join(DATA_DIR, f"y_{split}.npy")
        if os.path.exists(y_path):
            file_size = os.path.getsize(y_path)
            # Headerless .npy (memmap) size = N * 8 bytes
            n_samples = file_size // 8
            counts[split] = n_samples
            print(f"Found {split}: {n_samples} samples")
        else:
            print(f"[Warning] {y_path} not found")
            counts[split] = 0

    meta_data = {
        "length": SEQ_LEN,
        "counts": counts,
        "created_at": "Recovered via Fix_Meta_PU"
    }

    out_path = os.path.join(DATA_DIR, "meta.json")
    with open(out_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"\n[SUCCESS] meta.json created at: {out_path}")
    print("You can now run the training script.")

if __name__ == "__main__":
    generate_meta()