# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import sys

# --- Optional Analysis Libraries ---
try:
    from torchinfo import summary as torchinfo_summary
    SUMMARY_AVAILABLE = True
except ImportError:
    SUMMARY_AVAILABLE = False

try:
    from thop import profile as thop_profile
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False


# ==========================================
# 1. Core Components (Quadratic & Attention)
# ==========================================
class MetaAconC(nn.Module):
    """ Adaptive Context Activation (Non-linear activation) """
    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv1d(width, max(r, width // r), 1)
        self.fc2 = nn.Conv1d(max(r, width // r), width, 1)
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = x.mean(dim=2, keepdims=True)
        beta = self.sigmoid(self.fc2(self.fc1(context)))
        return (self.p1 * x - self.p2 * x) * self.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class CoordinateAttention(nn.Module):
    """ Coordinate Attention for 1D Signals to capture long-range dependencies """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, 1)
        self.bn1 = nn.BatchNorm1d(mip)
        self.act = MetaAconC(mip)
        self.conv_w = nn.Conv1d(mip, oup, 1)

    def forward(self, x):
        identity = x
        x_pool = self.pool(x)
        y = self.act(self.bn1(self.conv1(x_pool)))
        a_w = self.conv_w(y).sigmoid()
        return identity * a_w

class QuadraticConv1d(nn.Module):
    """ 
    Quadratic Convolution: Extract non-linear features directly.
    Equation: (Wr*x + br)(Wg*x + bg) + Wb*x^2 + c 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_b = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if bias: self.c = nn.Parameter(torch.zeros(out_channels))
        else: self.register_parameter('c', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
        if self.conv_r.bias is not None: init.zeros_(self.conv_r.bias)
        init.xavier_normal_(self.conv_g.weight) # Gate uses Xavier
        if self.conv_g.bias is not None: init.ones_(self.conv_g.bias)
        init.kaiming_uniform_(self.conv_b.weight, a=math.sqrt(5))
        if self.c is not None: init.zeros_(self.c)

    def forward(self, x):
        # x shape: (B, C, L)
        # quadratic term x^2 helps capturing energy/power features directly
        return (self.conv_r(x) * self.conv_g(x)) + self.conv_b(x * x) + (self.c.view(1, -1, 1) if self.c is not None else 0)

class SoftThresholding(nn.Module):
    """ Learnable Soft Thresholding for noise reduction (DRSN inspired) """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # Learnable threshold: y
        return torch.mul(torch.sign(x), F.relu(torch.abs(x) - y))


# ==========================================
# 2. Main Block
# ==========================================
class QResBlock_Pro(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = QuadraticConv1d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = MetaAconC(channels)
        
        self.conv2 = QuadraticConv1d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.attn = CoordinateAttention(channels, channels)
        self.shrinkage = SoftThresholding(channels)
        self.act2 = MetaAconC(channels)

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Squeeze-and-Excitation via Coordinate Attention
        out = self.attn(out)
        # Denoising
        out = self.shrinkage(out)
        
        return self.act2(out + residual)


# ==========================================
# 3. QDCA_Hybrid_Pro_v3 Model
# ==========================================
class QDCA_Hybrid_Pro(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, feat_dim=33):
        super().__init__()
        
        # 

        # --- A. CNN Backbone (Feature Extraction) ---
        # 1. Stem (Input 2048 -> 1024)
        self.stem = nn.Sequential(
            QuadraticConv1d(in_channels, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32), MetaAconC(32),
            nn.MaxPool1d(2) # 1024 -> 512
        )

        # 2. Multi-Scale Branching (Length 512 -> 256)
        # Large kernel for low freq, small kernel for high freq
        self.branch_1 = nn.Sequential(
            QuadraticConv1d(32, 64, 31, 1, 15), nn.BatchNorm1d(64), MetaAconC(64), nn.MaxPool1d(2)
        )
        self.branch_2 = nn.Sequential(
            QuadraticConv1d(32, 64, 15, 1, 7), nn.BatchNorm1d(64), MetaAconC(64), nn.MaxPool1d(2)
        )
        self.branch_3 = nn.Sequential(
            QuadraticConv1d(32, 64, 7, 1, 3), nn.BatchNorm1d(64), MetaAconC(64), nn.MaxPool1d(2)
        )
        
        # 3. Deep Denoising Stage (Length 256 -> 128)
        self.fusion_conv = nn.Conv1d(64, 128, 1)
        self.deep_stage = nn.Sequential(
            QResBlock_Pro(128),
            nn.MaxPool1d(2), # 256 -> 128
            QResBlock_Pro(128)
            # Output is (B, 128, 128)
        )
        
        # --- B. Transformer (Global Dependencies) ---
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, 128)) # (1, Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=0.2, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # --- C. Auxiliary Branch (Engineering Features) ---
        self.aux_net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        
        # --- D. Gated Fusion & Classification ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(128 * 2, 1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        init.trunc_normal_(self.pos_embedding, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: init.zeros_(m.bias)

    def forward(self, x, features=None):
        # x: (B, 3, 2048)
        # features: (B, 33) or 88, 14, ...
        
        # 1. CNN Path
        x = self.stem(x) # (B, 32, 512)
        
        b1 = self.branch_1(x)
        b2 = self.branch_2(x)
        b3 = self.branch_3(x)
        
        # Adaptive Multi-scale Fusion (Sum)
        fused = (b1 + b2 + b3) / 3.0 # (B, 64, 256)
        
        fused = self.fusion_conv(fused) # (B, 128, 256)
        cnn_out = self.deep_stage(fused) # (B, 128, 128)
        
        # 2. Transformer Path
        # Permute for Transformer: (B, C, L) -> (B, L, C)
        x_trans = cnn_out.permute(0, 2, 1) 
        x_trans = x_trans + self.pos_embedding[:, :x_trans.size(1), :]
        
        trans_out = self.transformer(x_trans) # (B, 128, 128)
        
        # Global Pooling (Mean)
        deep_rep = trans_out.mean(dim=1) # (B, 128)
        
        # 3. Aux Fusion
        if features is not None:
            aux_rep = self.aux_net(features) # (B, 128)
            
            # Gated Fusion mechanism
            combined = torch.cat([deep_rep, aux_rep], dim=1) # (B, 256) used for gate calc? No, simple gate
            # Calculate alpha based on both
            gate_input = torch.cat([deep_rep, aux_rep], dim=1)
            alpha = self.fusion_gate(gate_input) # (B, 1)
            
            final_rep = (1 - alpha) * deep_rep + alpha * aux_rep
        else:
            final_rep = deep_rep
            
        # 4. Classification
        logits = self.classifier(final_rep)
        return logits

# ----------------------------------
# 4. Analysis Utility (Fixed)
# ----------------------------------
def analyze_model(model, device='cuda'):
    # Convert torch.device to string for printing
    dev_str = str(device).upper()
    
    print("\n" + "="*50)
    print(f"   QDCA_Hybrid_Pro_v3 ANALYSIS ({dev_str})   ")
    print("="*50)
    
    # Dummy Inputs matching V3 Dataset
    # Ensure inputs are created on the correct device
    x_dummy = torch.randn(10, 3, 2048).to(device) # Signal
    f_dummy = torch.randn(10, 33).to(device)      # Eng Features
    
    # 1. Forward Check
    try:
        model.eval()
        with torch.no_grad():
            out = model(x_dummy, f_dummy)
        print(f"Input Signal:   {tuple(x_dummy.shape)}")
        print(f"Input Features: {tuple(f_dummy.shape)}")
        print(f"Output Logits:  {tuple(out.shape)}")
        print("\n[OK] Forward pass successful.")
    except Exception as e:
        print(f"\n[FAIL] Forward pass error: {e}")
        # Print full traceback if needed
        import traceback
        traceback.print_exc()
        return

    # 2. Complexity
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Model Size:       {total_params * 4 / (1024**2):.2f} MB")
    
    if FLOPS_AVAILABLE:
        try:
            class Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x, f): return self.m(x, f)
            
            flops, _ = thop_profile(Wrap(model), inputs=(x_dummy, f_dummy), verbose=False)
            print(f"FLOPs:            {flops / 1e9:.3f} G")
        except Exception as e: 
            print(f"FLOPs calculation skipped: {e}")

    # 3. Layer Summary
    if SUMMARY_AVAILABLE:
        print("\n" + "-"*20 + " Layer Details " + "-"*20)
        try:
            # Reconfigure stdout for unicode compatibility (Windows fix)
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            
            summary_str = torchinfo_summary(model, input_data=[x_dummy, f_dummy], 
                                          col_names=["input_size", "output_size", "num_params", "mult_adds"],
                                          verbose=0)
            print(summary_str)
        except Exception as e:
            print(f"Summary print failed: {e}")
            
    print("="*50 + "\n")

if __name__ == "__main__":
    # Auto-detect device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate Model
    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=10, feat_dim=33).to(dev)
    
    # Run Analysis
    analyze_model(model, dev)