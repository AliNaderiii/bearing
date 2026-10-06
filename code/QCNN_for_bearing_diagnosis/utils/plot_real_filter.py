import os
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# وارد کردن کلاس فیلتر از فایل پیش‌پردازش شما
from Preprocess import LearnedFilter

# ==========================================
# 1. تنظیم مسیرها و بارگذاری داده‌های واقعی
# ==========================================
out_dir = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\3HP\processed_data_robust_3hp"

# بارگذاری سیگنال خام
snapshot_path = os.path.join(out_dir, "raw_snapshot.npz")
rs = np.load(snapshot_path)
x_test_raw = rs['X_test_raw']  

# انتخاب یک نمونه
sample_idx = 0
channel_idx = 0
sig_raw = x_test_raw[sample_idx, channel_idx, :].astype(np.float32)
L = len(sig_raw)

# ==========================================
# 2. بارگذاری فیلتر آموزش‌دیده (Neural MED)
# ==========================================
filter_path = os.path.join(out_dir, f"learned_filter_ch{channel_idx}.pt")
state = torch.load(filter_path, map_location='cpu')

# بازسازی فیلتر
lf = LearnedFilter(length=L)
lf.load_state_dict(state)
lf.eval()

# عبور سیگنال واقعی از فیلتر
with torch.no_grad():
    sig_tensor = torch.from_numpy(sig_raw).unsqueeze(0) 
    sig_filtered = lf(sig_tensor).numpy()[0]            

# ==========================================
# 3. ایجاد محور زمان
# ==========================================
fs = 12000  # فرکانس نمونه‌برداری CWRU
t = np.arange(L) / fs

# ==========================================
# 4. رسم نمودار حرفه‌ای با Plotly
# ==========================================
# ایجاد Subplots با محور X مشترک
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.12,
    subplot_titles=(
        "<b>(a) Raw Vibration Signal (Before Pre-Filtering)</b>", 
        "<b>(b) Output of the Unsupervised Learnable Filter (Neural MED)</b>"
    )
)

# رنگ‌های استاندارد و شیک (Paper-Ready Palette)
color_raw = "#25648B"      # آبی نفتی تیره (Deep Navy Blue)
color_filtered = "#97830D" # قرمز زرشکی (Crimson Red)

# افزودن سیگنال خام (نمودار بالا)
fig.add_trace(
    go.Scatter(
        x=t, y=sig_raw, 
        mode='lines', 
        line=dict(color=color_raw, width=1.2), 
        name='Raw Signal'
    ),
    row=1, col=1
)

# افزودن سیگنال فیلتر شده (نمودار پایین)
fig.add_trace(
    go.Scatter(
        x=t, y=sig_filtered, 
        mode='lines', 
        line=dict(color=color_filtered, width=1.5), 
        name='Filtered Signal'
    ),
    row=2, col=1
)

# ==========================================
# 5. استایل‌دهی پیشرفته برای مقالات Q1
# ==========================================
fig.update_layout(
    font=dict(family="Times New Roman", size=15, color="black"),
    plot_bgcolor="white",  # پس‌زمینه کاملا سفید
    paper_bgcolor="white",
    width=1000,            # عرض تصویر
    height=700,            # ارتفاع تصویر
    showlegend=False,
    margin=dict(l=70, r=40, t=70, b=70) # حاشیه‌ها
)

# استایل محورهای X و Y (اضافه کردن خطوط گرید ملایم و کادر مشکی دور نمودار)
axis_style = dict(
    showline=True, 
    linewidth=1.5, 
    linecolor='black', 
    mirror=True, 
    showgrid=True, 
    gridcolor='#E5E7E9', # خاکستری بسیار روشن برای گریدها
    gridwidth=1,
    zeroline=True,
    zerolinecolor='#E5E7E9',
    zerolinewidth=1
)

# اعمال استایل به تمام محورها
fig.update_xaxes(**axis_style)
fig.update_yaxes(**axis_style)

# افزودن نام محورها
fig.update_yaxes(title_text="Amplitude (g)", row=1, col=1)
fig.update_yaxes(title_text="Amplitude (g)", row=2, col=1)
fig.update_xaxes(title_text="Time (Seconds)", row=2, col=1)

# تغییر سایز عنوان‌های ساب‌پلات‌ها
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=16, family="Times New Roman", color="black")

# ==========================================
# 6. نمایش و ذخیره با کیفیت بالا
# ==========================================
# ذخیره به صورت PDF (وکتور خالص برای LaTeX)
fig.write_image("real_learned_filter_effect_plotly.pdf", engine="kaleido")

# ذخیره به صورت PNG (با رزولوشن بالا - scale=3 یعنی 3 برابر کیفیت عادی)
fig.write_image("real_learned_filter_effect_plotly.png", scale=3, engine="kaleido")

print("Plotly Image generated successfully!")
print("Saved as 'real_learned_filter_effect_plotly.pdf' and '.png'")

# نمایش تصویر در مرورگر یا ادیتور
fig.show()