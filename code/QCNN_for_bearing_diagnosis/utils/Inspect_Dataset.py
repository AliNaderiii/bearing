import os
import pandas as pd
import glob

# --- مسیر اصلی پوشه دیتاست NREL خود را اینجا وارد کنید ---
# (پوشه‌ای که شامل زیرپوشه‌های Healthy و Damaged و ... است)
ROOT_DIR = r"D:\Bearing Project\CWRU\CWRU Case 08\QCNN for bearing diagnosis\data\Wind Turbine Gearbox Condition Monitoring Vibration\Data Information"

def inspect_files():
    # پیدا کردن تمام فایل‌های CSV در تمام زیرپوشه‌ها
    csv_files = glob.glob(os.path.join(ROOT_DIR, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        print("No CSV files found! Check the ROOT_DIR path.")
        return

    print(f"Found {len(csv_files)} CSV files.\n")
    print("-" * 60)
    print(f"{'Filename':<40} | {'All Columns'}")
    print("-" * 60)

    # بررسی هر فایل
    for file_path in csv_files:
        try:
            # خواندن فقط هدر فایل
            df = pd.read_csv(file_path, nrows=1)
            filename = os.path.basename(file_path)
            
            # لیست ستون‌ها
            cols = df.columns.tolist()
            
            # نمایش نام فایل و ۵ ستون اول (یا همه ستون‌های لرزش)
            # معمولا ستون‌های لرزش با AN شروع می‌شوند
            vib_cols = [c for c in cols if "AN" in c or "vib" in c.lower()]
            
            if vib_cols:
                display_cols = vib_cols[:] # نمایش ۵ تای اول
            else:
                display_cols = cols[:]
                
            print(f"{filename:<40} | {display_cols}")
            
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    print("-" * 60)

if __name__ == "__main__":
    inspect_files()
