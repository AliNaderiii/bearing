# test_hht.py
import torch
from torchHHT import hht

print("Testing torchHHT import...")

# سیگنال تست: chirp + نویز
fs = 12000
t = torch.linspace(0, 1, fs)
signal = torch.sin(2 * torch.pi * (10 + 50 * t) * t)  # chirp از 10Hz به 60Hz
x = signal + 0.3 * torch.randn_like(signal)

# اعمال HHT
print("Applying HHT...")
imfs, envelopes, frequencies = hht.hilbert_huang(x, fs, num_imf=3)

print(f"IMFs shape: {imfs.shape}")
print(f"Envelopes shape: {envelopes.shape}")
print(f"Frequencies shape: {frequencies.shape}")
print("HHT test passed!")
