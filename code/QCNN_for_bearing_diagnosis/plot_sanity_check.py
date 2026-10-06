# -*- coding: utf-8 -*-
"""
plot_sanity_check.py
Author: Ali Naderi
Description: Generates a comparison plot: Raw Noisy Signal vs. Quadratic Feature Map.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# --- تنظیمات مسیرها (مطمئن شوید درست هستند) ---
# استفاده از دیتای 3HP که نتایج عالی داشت
DATA_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp\processed_data_robust_v3.npz"
MODEL_PATH = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\checkpoints_final_vis_3hp\model_seed42.pth"
SAVE_DIR = r"./paper_results_final_3hp/sanity_check"

# --- ایمپورت کلاس‌ها ---
import sys
sys.path.append(os.path.dirname(__file__)) # افزودن پوشه جاری به مسیر
from utils.DatasetLoader import RobustBearingDataset, collate_fn
try:
    from Model.model import QDCA_Hybrid_Pro
except:
    from Model.model import QDCA_Hybrid_Pro

# --- هوک برای گرفتن خروجی لایه ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. لود کردن دیتاست (تست با نویز -10dB)
    print("Loading Dataset (-10dB)...")
    dataset = RobustBearingDataset(DATA_PATH, mode='test', snr=-10, load_into_memory=True, instance_norm=True)
    
    # پیدا کردن یک نمونه خاص (مثلاً Ball Fault - کلاس 0 یا 1)
    target_class = 0  # Ball Fault typically
    sample_idx = -1
    
    for i in range(len(dataset)):
        _, _, lbl = dataset[i]
        if lbl == target_class:
            sample_idx = i
            break
            
    if sample_idx == -1:
        print("Class not found!")
        return

    # گرفتن داده
    sig, feat, label = dataset[sample_idx]
    sig = sig.unsqueeze(0).to(device)     # (1, 3, 2048)
    feat = feat.unsqueeze(0).to(device)   # (1, 33)
    
    # 2. لود کردن مدل
    print("Loading Model...")
    # دریافت ابعاد فیچر به صورت اتوماتیک
    temp_ds = RobustBearingDataset(DATA_PATH, mode='train', load_into_memory=False)
    feat_dim = temp_ds[0][1].shape[0]
    
    model = QDCA_Hybrid_Pro(in_channels=3, num_classes=10, feat_dim=feat_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 3. اتصال هوک به لایه اول (Stem - Quadratic Conv)
    # این لایه همان جایی است که انتظار داریم "جادوی دمدولاسیون" رخ دهد
    model.stem.register_forward_hook(get_activation('stem_output'))
    
    # 4. اجرای مدل
    with torch.no_grad():
        _ = model(sig, feat)
        
    # گرفتن خروجی هوک
    # خروجی استم معمولاً (Batch, Channels, Length/stride) است
    feature_map = activation['stem_output'].squeeze(0).cpu().numpy()
    
    # 5. رسم نمودار حرفه‌ای
    print("Plotting...")
    
    fig = plt.figure(figsize=(12, 8))
    
    # --- ردیف بالا: سیگنال خام نویزی ---
    ax1 = fig.add_subplot(2, 1, 1)
    raw_signal = sig[0, 0, :].cpu().numpy() # کانال اول
    ax1.plot(raw_signal, color='#1f77b4', linewidth=0.8)
    ax1.set_title(f"Raw Input Signal (Ball Fault) @ -10 dB SNR", fontsize=14, fontname="Times New Roman", weight='bold')
    ax1.set_ylabel("Amplitude (Normalized)", fontsize=12, fontname="Times New Roman")
    ax1.set_xlim(0, len(raw_signal))
    ax1.grid(True, alpha=0.3)
    # اضافه کردن متن توضیحی روی نمودار
    ax1.text(0.02, 0.9, "Defect impulses are buried in noise", transform=ax1.transAxes, 
             fontsize=11, color='red', fontname="Times New Roman", style='italic', bbox=dict(facecolor='white', alpha=0.7))

    # --- ردیف پایین: هیت‌مپ ویژگی‌های استخراج شده ---
    ax2 = fig.add_subplot(2, 1, 2)
    # انتخاب مثلاً 32 کانال اول برای نمایش واضح‌تر
    sns.heatmap(feature_map[:32, :], cmap="viridis", cbar=True, ax=ax2, rasterized=True)
    ax2.set_title("Quadratic Feature Response (Layer 1 Output)", fontsize=14, fontname="Times New Roman", weight='bold')
    ax2.set_xlabel("Time Steps (Downsampled)", fontsize=12, fontname="Times New Roman")
    ax2.set_ylabel("Feature Channels", fontsize=12, fontname="Times New Roman")
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Sanity_Check_Demodulation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sanity check plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()