# -*- coding: utf-8 -*-
"""
Model/AblationModel.py
Author: Ali Naderi - 2025
Description: Modified QDCA-Net for Ablation Studies.
Modes:
    - 'full': Original Model
    - 'no_quad': Replaces QuadraticConv with Standard Conv1d
    - 'no_attn': Removes Coordinate Attention
    - 'no_trans': Removes Transformer Encoder
    - 'no_stat': Removes Statistical Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- FIX: Correct Import ---
# Assuming this file is in the 'Model' directory alongside 'QDCA_BiGRU_Res_Model.py'
try:
    from .model import MetaAconC, CoordinateAttention, QuadraticConv1d, SoftThresholding
except ImportError:
    # Fallback if run from root directory
    from Model.model import MetaAconC, CoordinateAttention, QuadraticConv1d, SoftThresholding
    
class QDCA_Ablation(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, feat_dim=33, mode='full'):
        super().__init__()
        self.mode = mode
        
        # --- 1. Convolution Type Selection ---
        if mode == 'no_quad':
            # Use Standard Conv1d instead of Quadratic
            def ConvBlock(cin, cout, k, s, p):
                return nn.Sequential(
                    nn.Conv1d(cin, cout, k, s, p),
                    nn.BatchNorm1d(cout),
                    nn.ReLU()
                )
        else:
            # Use Original Quadratic Conv
            def ConvBlock(cin, cout, k, s, p):
                return nn.Sequential(
                    QuadraticConv1d(cin, cout, k, s, p),
                    nn.BatchNorm1d(cout),
                    MetaAconC(cout)
                )

        # --- A. Backbone ---
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, 15, 2, 7),
            nn.MaxPool1d(2)
        )

        self.branch_1 = nn.Sequential(ConvBlock(32, 64, 31, 1, 15), nn.MaxPool1d(2))
        self.branch_2 = nn.Sequential(ConvBlock(32, 64, 15, 1, 7), nn.MaxPool1d(2))
        self.branch_3 = nn.Sequential(ConvBlock(32, 64, 7, 1, 3), nn.MaxPool1d(2))
        
        self.fusion_conv = nn.Conv1d(64, 128, 1)

        # --- B. Deep Stage (With or Without Attention) ---
        self.deep_stage = nn.ModuleList()
        for _ in range(2):
            layers = [ConvBlock(128, 128, 3, 1, 1)]
            if mode != 'no_attn':
                layers.append(CoordinateAttention(128, 128))
            layers.append(SoftThresholding(128))
            layers.append(nn.MaxPool1d(2) if _ == 0 else nn.Identity())
            self.deep_stage.append(nn.Sequential(*layers))

        # --- C. Transformer (Optional) ---
        if mode != 'no_trans':
            self.pos_embedding = nn.Parameter(torch.randn(1, 128, 128))
            encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- D. Aux Branch (Optional) ---
        if mode != 'no_stat':
            self.aux_net = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Linear(64, 128)
            )
            self.fusion_gate = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, features=None):
        # Backbone
        x = self.stem(x)
        b1 = self.branch_1(x); b2 = self.branch_2(x); b3 = self.branch_3(x)
        fused = (b1 + b2 + b3) / 3.0
        fused = self.fusion_conv(fused)
        
        for block in self.deep_stage:
            fused = block(fused)
        
        # Transformer or Global Pool
        if self.mode != 'no_trans':
            x_t = fused.permute(0, 2, 1) + self.pos_embedding
            trans_out = self.transformer(x_t)
            deep_rep = trans_out.mean(dim=1)
        else:
            deep_rep = fused.mean(dim=2) # Simple Global Pooling
            
        # Statistical Fusion
        if self.mode != 'no_stat' and features is not None:
            aux_rep = self.aux_net(features)
            alpha = self.fusion_gate(torch.cat([deep_rep, aux_rep], dim=1))
            final_rep = (1 - alpha) * deep_rep + alpha * aux_rep
        else:
            final_rep = deep_rep
            
        return self.classifier(final_rep)
