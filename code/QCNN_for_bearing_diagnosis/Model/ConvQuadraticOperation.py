# -*- coding: utf-8 -*-
"""
ConvQuadraticOperation.py
Author: Ali Naderi
Date: 2025
Description: 
    Optimized implementation of Quadratic Convolutional Layers.
    Formula: f(x) = (W_r*x + b_r) * (W_g*x + b_g) + (W_b*x^2 + b_b) + c
    Features:
    - Native PyTorch nn.Module inheritance (Auto device/dtype handling)
    - ReLinear Initialization for training stability
    - Optimized forward pass using standard Conv1d primitives
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init

class ConvQuadraticOperation(nn.Module):
    """
    1D Quadratic Convolution Layer.
    Combines Linear, Gating, and Quadratic branches.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvQuadraticOperation, self).__init__()
        
        # 1. Linear Branch (Red) - Captures standard features
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        
        # 2. Gating Branch (Green) - Controls information flow
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        
        # 3. Quadratic Branch (Blue) - Captures non-linear/energy features from squared input
        self.conv_b = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        # 4. Independent Global Bias
        if bias:
            self.c = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('c', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        ReLinear Initialization:
        Initializes the layer to behave like a standard linear convolution at the start.
        Formula: (Linear * 1) + 0 + 0 = Linear
        """
        # Red (Linear): Standard Kaiming initialization
        init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
        if self.conv_r.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv_r.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.conv_r.bias, -bound, bound)

        # Green (Gate): Weights=0, Bias=1 -> Output is approx 1.0
        init.constant_(self.conv_g.weight, 0)
        if self.conv_g.bias is not None:
            init.constant_(self.conv_g.bias, 1.0) 

        # Blue (Quadratic): Weights=0 -> Output starts at 0
        init.constant_(self.conv_b.weight, 0)
        if self.conv_b.bias is not None:
            init.constant_(self.conv_b.bias, 0)
            
        # Global Bias
        if self.c is not None:
            init.zeros_(self.c)

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        
        out_r = self.conv_r(x)       # Linear term
        out_g = self.conv_g(x)       # Gating term
        out_b = self.conv_b(x * x)   # Quadratic term (input squared)

        # Combine: (Linear * Gate) + Quadratic + Bias
        res = (out_r * out_g) + out_b
        
        if self.c is not None:
            res = res + self.c.view(1, -1, 1)
            
        return res


class ConvTransQuadraticOperation(nn.Module):
    """
    1D Quadratic Transposed Convolution Layer (Deconvolution).
    Useful for Autoencoders or Generative models.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True):
        super(ConvTransQuadraticOperation, self).__init__()
        
        self.conv_tr_r = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias)
        self.conv_tr_g = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias)
        self.conv_tr_b = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias=bias)

        if bias:
            self.c = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('c', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Similar ReLinear strategy for Transposed Conv
        init.kaiming_uniform_(self.conv_tr_r.weight, a=math.sqrt(5))
        if self.conv_tr_r.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv_tr_r.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.conv_tr_r.bias, -bound, bound)

        init.constant_(self.conv_tr_g.weight, 0)
        if self.conv_tr_g.bias is not None:
            init.constant_(self.conv_tr_g.bias, 1.0)

        init.constant_(self.conv_tr_b.weight, 0)
        if self.conv_tr_b.bias is not None:
            init.constant_(self.conv_tr_b.bias, 0)

        if self.c is not None:
            init.zeros_(self.c)

    def forward(self, x):
        out_r = self.conv_tr_r(x)
        out_g = self.conv_tr_g(x)
        out_b = self.conv_tr_b(x * x)
        
        res = (out_r * out_g) + out_b
        
        if self.c is not None:
            res = res + self.c.view(1, -1, 1)
            
        return res

# -------------------------
# Self-Test
# -------------------------
if __name__ == '__main__':
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running tests on: {device}")
    
    # Input: Batch=20, Channels=1, Length=2048
    x = torch.randn(20, 1, 2048).to(device)
    
    # Test 1: Convolution
    try:
        layer = ConvQuadraticOperation(in_channels=1, out_channels=16, kernel_size=64, stride=8, padding=28).to(device)
        out = layer(x)
        print(f"Quadratic Conv Forward Pass Successful.")
        print(f"Input: {x.shape} -> Output: {out.shape}") # Expected: [20, 16, 256]
    except Exception as e:
        print(f"Conv Failed: {e}")

    # Test 2: Transposed Convolution
    try:
        layer_tr = ConvTransQuadraticOperation(in_channels=16, out_channels=1, kernel_size=64, stride=8, padding=28, output_padding=0).to(device)
        recon = layer_tr(out)
        print(f"Quadratic Transpose Forward Pass Successful.")
        print(f"Input: {out.shape} -> Output: {recon.shape}") # Expected: [20, 1, 2048] (approximately)
    except Exception as e:
        print(f"Transpose Failed: {e}")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Setup
t = np.linspace(0, 4*np.pi, 200)
sig = np.sin(t)
x_tensor = torch.tensor(sig, dtype=torch.float32).view(1, 1, -1)

# Standard Conv
conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
with torch.no_grad():
    y_linear = conv(x_tensor).numpy().flatten()

# Quadratic Conv (Simulation of logic)
# y = (w1*x)*(w2*x) + w3*x^2
# Let's simulate a state where weights are active to show non-linearity
w_r = 1.0; w_g = 1.0; w_b = 0.5
y_quad = (sig * w_r) * (sig * w_g) + (sig**2 * w_b)

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, sig, 'k', label='Input (Sine)')
plt.title("Input Signal")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t, y_linear, 'b', label='Standard Conv1d')
plt.title("Linear Layer Output (Preserves Frequency)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t, y_quad, 'r', label='Quadratic Layer')
plt.title("Quadratic Layer Output (Generates Harmonics/Complexity)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quadratic_behavior.png')
