# -*- coding: utf-8 -*-
"""
analyze_spectral_qttention.py
Author: Ali Naderi - 2026 (Publication Quality Edition)
Description: 
    Generates high-resolution, IEEE/Elsevier-style spectral analysis plots.
    Includes Dynamic Axis Scaling and Dynamic FS Correction.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # جلوگیری از کرش در ویندوز
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack
import torch
import math
from matplotlib import rcParams

# --- CONFIGURATION ---
DATA_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
MODEL_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_3hp\model_seed42.pth"
SAVE_DIR = r"./paper_results_final_3hp/spectral_analysis"

# SKF 6205-2RS JEM - Drive End (3HP)
BEARING_PARAMS = {
    'n': 9,         
    'd': 0.3126,    
    'D': 1.537,     
    'alpha': 0,     
    'rpm': 1730     
}
FS = 12000 # Sampling Frequency

# --- ELITE PLOT STYLING (IEEE/Elsevier Standard) ---
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 13
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 15
rcParams['axes.titleweight'] = 'bold'
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18
rcParams['axes.linewidth'] = 1.2
rcParams['axes.labelweight'] = 'bold'

sys.path.append(os.path.dirname(__file__))
try:
    from utils.DatasetLoader import RobustBearingDataset
    from Model.model import QDCA_Hybrid_Pro
except ImportError:
    print("[Error] Required custom modules not found.")
    sys.exit(1)

# --- HELPER FUNCTIONS ---
def calculate_fault_freqs(rpm):
    fr = rpm / 60.0
    n, d, D, alpha = BEARING_PARAMS['n'], BEARING_PARAMS['d'], BEARING_PARAMS['D'], BEARING_PARAMS['alpha']
    cos_a = math.cos(math.radians(alpha))
    bpfo = (n / 2) * (1 - (d / D) * cos_a) * fr
    return {'BPFO': bpfo, 'Fr': fr}

def get_envelope_spectrum(signal, fs):
    signal = signal - np.mean(signal)
    analytic_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope = envelope - np.mean(envelope)
    n = len(envelope)
    f = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(envelope)) * 2 / n
    return f, mag

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def plot_fancy_spectrum(ax, f, mag, title, bpfo, harmonics=3, color='#1f77b4'):
    ax.plot(f, mag, color=color, linewidth=1.5, label='Envelope Spectrum', alpha=0.9)
    ax.fill_between(f, mag, 0, color=color, alpha=0.1)
    
    ax.set_title(title, pad=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    max_mag = np.max(mag)
    ax.set_ylim(0, max_mag * 1.2) 
    
    for k in range(1, harmonics + 1):
        freq = k * bpfo
        if freq < ax.get_xlim()[1]:
            ax.axvspan(freq-2, freq+2, color='#d62728', alpha=0.15)
            ax.axvline(x=freq, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.5)
            
            idx = (np.abs(f - freq)).argmin()
            peak_mag = mag[idx]
            
            if peak_mag > max_mag * 0.05: 
                ax.annotate(f'{k}$\\times$BPFO', 
                            xy=(freq, peak_mag), 
                            xytext=(freq + 8, peak_mag + max_mag*0.1),
                            arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=6, alpha=0.8),
                            fontsize=12, fontweight='bold', color='#333333', rotation=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    TARGET_CLASS = 7  
    SNR_LEVEL = -10
    
    dataset = RobustBearingDataset(DATA_PATH, mode='test', snr=SNR_LEVEL, load_into_memory=True, instance_norm=True)
    
    sample_idx = -1
    for i in range(len(dataset)):
        _, _, lbl = dataset[i]
        if lbl == TARGET_CLASS:
            sample_idx = i
            break
            
    if sample_idx == -1: return

    sig, feat, label = dataset[sample_idx]
    sig_tensor = sig.unsqueeze(0).to(device)
    feat_tensor = feat.unsqueeze(0).to(device)
    
    temp_ds = RobustBearingDataset(DATA_PATH, mode='train', load_into_memory=False)
    feat_dim = temp_ds[0][1].shape[0]
    del temp_ds 
    
    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=10, feat_dim=feat_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    model.stem.register_forward_hook(get_activation('stem'))
    with torch.no_grad(): 
        _ = model(sig_tensor, feat_tensor)
        
    raw_signal = sig[0, :].numpy()
    feature_map = activations['stem'].squeeze(0).cpu().numpy()
    feature_signal = np.mean(feature_map, axis=0)
    
    # ---------------------------------------------------------
    # DYNAMIC LENGTH AND FREQUENCY CALCULATION (THE FIX)
    # ---------------------------------------------------------
    len_raw = len(raw_signal)
    len_feat = len(feature_signal)
    downsample_factor = len_raw / len_feat
    FS_feat = FS / downsample_factor  # محاسبه دقیق فرکانس نمونه‌برداری لایه ویژگی
    
    print(f"Raw Signal Length: {len_raw}, Feature Signal Length: {len_feat}")
    print(f"Downsample Factor: {downsample_factor}, New FS: {FS_feat} Hz")
    # ---------------------------------------------------------

    freqs = calculate_fault_freqs(BEARING_PARAMS['rpm'])
    bpfo = freqs['BPFO']
    
    f_raw, mag_raw = get_envelope_spectrum(raw_signal, FS)
    f_feat, mag_feat = get_envelope_spectrum(feature_signal, FS_feat) 
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.45, wspace=0.25)
    
    # (a) Raw Signal (Dynamic X-Lim)
    axes[0,0].plot(raw_signal, color='#7f8c8d', linewidth=1, alpha=0.8)
    axes[0,0].set_title("(a) Raw Input Signal (-10 dB SNR)")
    axes[0,0].set_xlabel("Time (Samples)")
    axes[0,0].set_ylabel("Amplitude")
    axes[0,0].set_xlim(0, len_raw) # پر کردن کل کادر به صورت اتوماتیک
    axes[0,0].grid(True, linestyle='--', alpha=0.5)
    axes[0,0].spines['top'].set_visible(False)
    axes[0,0].spines['right'].set_visible(False)
    
    plot_fancy_spectrum(axes[0,1], f_raw, mag_raw, "(b) Envelope Spectrum of Raw Signal", bpfo, color='#e74c3c')
    axes[0,1].set_xlim(0, 400)
    
    # (c) Feature Signal (Dynamic X-Lim)
    axes[1,0].plot(feature_signal, color='#16a085', linewidth=1.5)
    axes[1,0].set_title("(c) Extracted Feature Map (Quadratic Layer)")
    axes[1,0].set_xlabel("Time (Downsampled Ticks)")
    axes[1,0].set_ylabel("Activation Intensity")
    axes[1,0].set_xlim(0, len_feat) # فیت شدن دقیقِ سیگنال ویژگی در کادر
    axes[1,0].grid(True, linestyle='--', alpha=0.5)
    axes[1,0].spines['top'].set_visible(False)
    axes[1,0].spines['right'].set_visible(False)
    
    plot_fancy_spectrum(axes[1,1], f_feat, mag_feat, "(d) Envelope Spectrum of Extracted Feature", bpfo, color='#2980b9')
    axes[1,1].set_xlim(0, 400)
    
    fig.savefig(os.path.join(SAVE_DIR, "Spectral_Analysis_Full_IEEE.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(SAVE_DIR, "Spectral_Analysis_Full_IEEE.png"), dpi=600, bbox_inches='tight')
    
    print(f"Masterpiece plots saved to {SAVE_DIR}")
    plt.close(fig)

if __name__ == "__main__":
    main()