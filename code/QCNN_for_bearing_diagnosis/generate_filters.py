# -*- coding: utf-8 -*-
"""
generate_filters.py
Author: Ali Naderi - 2026
Goal: Pre-train Physics-Informed filters on 'Healthy' NREL data to suppress background noise.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# --- Config ---
NPZ_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\nrel_processed_robust.npz"
OUT_DIR = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\NREL\filters"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnedFilter(nn.Module):
    def __init__(self, length=2048):
        super().__init__()
        # Initialize as a Band-Pass like filter
        self.conv = nn.Conv1d(1, 1, 65, padding=32, bias=False) 
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        xt = self.act(self.conv(x)).squeeze(1)
        X = torch.fft.rfft(xt)
        # Gate aims to keep meaningful frequencies
        g = torch.sigmoid(self.gate(torch.abs(X)))
        return torch.fft.irfft(X * g, n=x.shape[-1])

def main():
    if not os.path.exists(NPZ_PATH):
        print(f"Error: {NPZ_PATH} not found.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading NREL data...")
    data = np.load(NPZ_PATH, mmap_mode='r')
    X_train = data['X_train_sig']
    y_train = data['y_train']
    
    # Select only HEALTHY signals (Class 0)
    # Because we want filters that know "normality" and reject anomalies/noise
    healthy_idx = np.where(y_train == 0)[0]
    if len(healthy_idx) > 5000: healthy_idx = healthy_idx[:5000] # Take subset for speed
    
    X_healthy = X_train[healthy_idx] # (N, 8, 2048)
    print(f"Training filters on {len(X_healthy)} healthy samples...")
    
    # Train one filter per channel
    for ch in range(8):
        print(f"\n>>> Training Filter Channel {ch} <<<")
        model = LearnedFilter().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=0.005)
        
        # Data for this channel
        sigs = torch.from_numpy(X_healthy[:, ch, :]).to(DEVICE) # (N, 2048)
        
        # Objective: Reconstruction + Sparsity (Denoising autoencoder logic)
        for epoch in range(10): # Quick training
            optimizer.zero_grad()
            
            # Add noise to input
            noise = torch.randn_like(sigs) * 0.1
            noisy_in = sigs + noise
            
            # Pass through filter
            out = model(noisy_in)
            
            # Loss: Output should match clean Input (Denoising)
            loss = nn.MSELoss()(out.squeeze(1), sigs)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"Ep {epoch}: Loss {loss.item():.5f}")
                
        # Save
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"learned_filter_nrel_ch{ch}.pt"))
        print(f"Saved ch{ch} filter.")

if __name__ == "__main__":
    main()
    