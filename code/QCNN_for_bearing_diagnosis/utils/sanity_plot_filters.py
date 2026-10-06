# sanity_plot_filters.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# --- مسیر فایل NPZ و دایرکتوری فیلترها ---
npz_file = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\0HP\processed_data_robust\processed_data_robust_v3.npz"
out_dir = os.path.dirname(npz_file)

assert os.path.exists(npz_file), f"NPZ file not found: {npz_file}"

# --- بارگذاری داده‌ها ---
data = np.load(npz_file, allow_pickle=True)
X_test = data['X_test_sig'].astype(np.float32)  # شکل (N, 3, L)
L = X_test.shape[2]

# --- بارگذاری فیلترها ---
filter_paths = sorted(glob.glob(os.path.join(out_dir, "learned_filter_ch*.pt")))
assert len(filter_paths) == 3, "Expected 3 learned filters for 3 channels"

# --- تعریف کلاس فیلتر محلی ---
class LocalLearnedFilter(nn.Module):
    def __init__(self, length=L):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 7, padding=3, bias=False)
        self.act = nn.PReLU()
        self.n_freq = length // 2 + 1
        self.gate = nn.Linear(self.n_freq, self.n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        xt = self.act(self.conv(x)).squeeze(1)
        X = torch.fft.rfft(xt)
        mag = torch.abs(X)
        g = torch.sigmoid(self.gate(mag))
        g = torch.clamp(g, 0.01, 0.99)
        return torch.fft.irfft(X * g, n=x.shape[-1])

# --- انتخاب چند نمونه تصادفی ---
num_samples = 5
indices = random.sample(range(X_test.shape[0]), num_samples)

# --- آماده‌سازی فیلترها ---
filters = []
for fp in filter_paths:
    lf = LocalLearnedFilter(length=L)
    state = torch.load(fp, map_location='cpu', weights_only=True)
    try:
        lf.load_state_dict(state)
    except:
        # حذف prefix احتمالی
        new_state = {k.split("module.",1)[-1]:v for k,v in state.items()}
        lf.load_state_dict(new_state)
    lf.eval()
    filters.append(lf)

# --- رسم سیگنال و طیف ---
for idx in indices:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Sample index: {idx}", fontsize=16)
    
    for ch in range(3):
        sig = X_test[idx, ch, :]
        with torch.no_grad():
            filtered = filters[ch](torch.from_numpy(sig)).numpy()
        
        # --- طیف ---
        fft_orig = np.abs(np.fft.rfft(sig))
        fft_filt = np.abs(np.fft.rfft(filtered))
        ratio = (fft_filt + 1e-12) / (fft_orig + 1e-12)
        freqs = np.fft.rfftfreq(L, d=1.0)  # assuming normalized sampling
        
        # سیگنال قبل و بعد
        axes[ch,0].plot(sig, color='blue', label='Original')
        axes[ch,0].plot(filtered, color='red', alpha=0.7, label='Filtered')
        axes[ch,0].set_title(f"Channel {ch} - Time Domain")
        axes[ch,0].legend()
        
        # طیف قبل و بعد
        axes[ch,1].plot(freqs, fft_orig, color='blue', label='Original')
        axes[ch,1].plot(freqs, fft_filt, color='red', alpha=0.7, label='Filtered')
        axes[ch,1].set_title(f"Channel {ch} - Frequency Spectrum")
        axes[ch,1].set_xlim([0, L//2])
        axes[ch,1].legend()
        
        # نسبت تقویت
        axes[ch,2].plot(freqs, ratio, color='green')
        axes[ch,2].set_title(f"Channel {ch} - Amplification Ratio")
        axes[ch,2].set_ylim([0, np.max(ratio)*1.1])
        axes[ch,2].set_xlim([0, L//2])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
