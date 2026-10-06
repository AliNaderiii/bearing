import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات
INPUT_DIR = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\qmaps_3hp"
OUTPUT_DIR = os.path.join(INPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# پیدا کردن فایل‌های CSV
files = glob.glob(os.path.join(INPUT_DIR, "qmap_*.csv"))

print(f"Found {len(files)} maps. Plotting...")

for f in files:
    try:
        # بارگذاری داده
        df = pd.read_csv(f, header=None)
        data = df.values
        
        # استخراج نام کلاس از اسم فایل
        basename = os.path.basename(f)
        # اسم فایل: qmap_classX_idxY_predZ.csv
        parts = basename.split('_')
        true_label = parts[1].replace('class', '')
        
        # رسم Heatmap حرفه‌ای
        plt.figure(figsize=(10, 4))
        sns.heatmap(data, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
        plt.title(f"Quadratic Feature Map (Class {true_label}) @ -10dB SNR", fontsize=12, fontname="Times New Roman")
        plt.xlabel("Time Steps (Temporal Dimension)", fontsize=10, fontname="Times New Roman")
        plt.ylabel("Feature Channels", fontsize=10, fontname="Times New Roman")
        
        # ذخیره
        save_name = basename.replace('.csv', '.png')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error plotting {f}: {e}")

print(f"Done! Images saved in: {OUTPUT_DIR}")