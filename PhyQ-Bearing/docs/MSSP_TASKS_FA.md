# کارهای اجرایی برای ارسال به MSSP (نسخه‌ی v10.2)

همه‌ی دستورها را در پوشه‌ی پروژه (`D:\PhyQ-Bearing`) و با venv فعال اجرا کنید:

```powershell
cd D:\PhyQ-Bearing
.\.venv\Scripts\Activate.ps1
```

هیچ دستوری به داده‌ها یا مسیرهای اصلی شما چیزی نمی‌نویسد. همه‌ی خروجی‌ها داخل `results\` پروژه ذخیره می‌شوند.
اجراهایی که قبلاً تمام شده‌اند (یعنی `metrics.json` دارند) دوباره اجرا **نمی‌شوند**.

---

## ۰-الف. به‌روزرسانی به v10.2 (بدون از دست دادن نتایج)

این بار پوشه‌ی قدیم را **جابه‌جا نکنید**، چون `results\` و `cache\` و `main_tuned.yaml` شما در آن است.
zip را در یک پوشه‌ی موقت باز کنید و فقط کد را روی پروژه کپی کنید:

```powershell
Expand-Archive -Path D:\PhyQ-Bearing_v10_2.zip -DestinationPath D:\PhyQ_tmp
robocopy D:\PhyQ_tmp\PhyQ-Bearing D:\PhyQ-Bearing /E /XF paths.yaml main_tuned.yaml /XD results cache .venv
Remove-Item -Recurse D:\PhyQ_tmp
cd D:\PhyQ-Bearing
.\.venv\Scripts\Activate.ps1
python -c "import phyq; print(phyq.__version__)"     # باید 10.2 باشد
python -m pytest tests -q                             # باید 89 passed باشد
```

* `robocopy` فایل‌ها را کپی و جایگزین می‌کند، ولی **هیچ فایلی را پاک نمی‌کند**. پوشه‌های `results`، `cache` و `.venv` و فایل‌های `paths.yaml` و `main_tuned.yaml` اصلاً لمس نمی‌شوند.
* کتابخانه‌ی جدیدی لازم نیست. اگر `scikit-learn` نصب نیست، این دستور را بزنید: `pip install scikit-learn`. این کتابخانه برای baseline کلاسیک لازم است.

## ۰. چه چیزی در v10.2 اضافه شده است

| فایل | کار |
|---|---|
| `phyq/sim/bearing_sim.py` | شبیه‌ساز پدیدارشناختی یاتاقان (McFadden–Smith / Randall–Antoni)، با فرکانس‌های خرابی معلوم |
| `scripts/sim_mechanism.py` | مطالعه‌ی مکانیزم: `analytic` (بدون آموزش)، `train`، `analyze`، `summary` |
| `scripts/sim_figures.py` | شکل‌های ۲ و ۳ مقاله |
| `phyq/models/baselines.py` → `ClassBD` | پیاده‌سازی مجدد ClassBD (Liao و همکاران، MSSP 2025). کد رسمی لایسنس ندارد، پس کپی نشده است |
| `phyq/classical/kurtogram.py` | baseline کلاسیک: Fast Kurtogram + طیف پوش مربعی + طبقه‌بند انتخاب‌شده روی validation |
| `configs/models/classbd.yaml`، `kurtogram_ses.yaml` | پیکربندی مدل‌های جدید |
| `configs/experiments/hp_search.yaml` | ۸ ردیف ClassBD اضافه شده است (همان شبکه‌ی lr×epochs، به‌علاوه‌ی physics loss روشن/خاموش) |
| `configs/experiments/classical_tuned.yaml` | baseline کلاسیک روی CWRU/IMS/NREL/SDUST |
| `configs/experiments/mssp_baselines_pu.yaml` | ClassBD و baseline کلاسیک روی PU |
| `scripts/paper_assets.py` | جدول‌های LaTeX و شکل‌های ۵ و ۶ را مستقیم از فایل‌های نتیجه می‌سازد (بدون کپی دستی عدد) |
| `phyq/engine/trainer.py` | اگر مدل `aux_loss` داشته باشد، آن را به loss اضافه می‌کند (فقط ClassBD). مدل‌های دیگر تغییری نمی‌کنند |

**نکته‌ی مهم درباره‌ی ClassBD:** در کد رسمی، دو جمله‌ی physics loss (kurtosis و l2/l4) داخل `nn.Parameter(...)` قرار گرفته‌اند. این کار آن‌ها را از گراف محاسباتی جدا می‌کند، پس در عمل فقط cross-entropy آموزش می‌دهد. به همین دلیل هر دو حالت را به‌عنوان ابرپارامتر روی validation امتحان می‌کنیم. این نکته در مقاله هم ذکر شده است.

---

## ۱. ClassBD: جست‌وجوی ابرپارامتر (فقط validation)

```powershell
python scripts\run_suite.py configs\experiments\hp_search.yaml
python scripts\select_hparams.py
```

* فقط ۸ × ۳ = ۲۴ اجرای جدید ClassBD انجام می‌شود. بقیه از قبل موجودند و رد می‌شوند.
* `select_hparams.py` فایل `configs\experiments\main_tuned.yaml` را با برنده‌ی هر مدل (از جمله ClassBD) **دوباره می‌نویسد**. برنده‌های قبلی همان قبلی‌ها می‌مانند، چون نتایجشان تغییری نکرده است.
* در `results\hp_search\selection.md` ببینید کدام ردیف ClassBD برنده شده است و physics_loss در آن true است یا false.

## ۲. ClassBD روی تسک‌های اصلی (۳ seed)

```powershell
python scripts\run_suite.py configs\experiments\main_tuned.yaml --models classbd --seeds 0 1 2
```

## ۳. baseline کلاسیک Fast Kurtogram + SES (۳ seed)

```powershell
python scripts\run_suite.py configs\experiments\classical_tuned.yaml
```

* این مدل روی CPU اجرا می‌شود و GPU لازم ندارد. زمان اصلی صرف استخراج ویژگی می‌شود (روی SDUST هر اجرا حدود ۱ ساعت).
* اگر خیلی کند بود، `model.nlevel=4` را قبل از `--seeds` اضافه کنید. این تغییر باید برای همه‌ی دیتاست‌ها یکسان باشد.

## ۴. PU: ClassBD و baseline کلاسیک

اول در `configs\experiments\mssp_baselines_pu.yaml` مقدار physics_loss را برابر برنده‌ی مرحله‌ی ۱ بگذارید (پیش‌فرض true است):

```powershell
python scripts\run_suite.py configs\experiments\mssp_baselines_pu.yaml model.physics_loss=false --models classbd
python scripts\run_suite.py configs\experiments\mssp_baselines_pu.yaml --models kurtogram_ses
```

* اگر برنده physics_loss=true بود، `model.physics_loss=false` را از دستور اول حذف کنید.
* روی PU پنجره ۸۱۹۲ است، پس لایه‌ی خطی ClassBD بزرگ می‌شود (حدود ۶۷M پارامتر و ۱ GB حافظه). این برای GTX 1650 جا می‌شود. اگر OOM شد، برنامه خودش grad_accum را بالا می‌برد.

## ۵. مطالعه‌ی مکانیزم روی شبیه‌ساز (GPU، ۳ seed)

نسخه‌ی کوچک (۱ seed، CPU) را خودم در sandbox اجرا کردم. نسخه‌ی کامل برای مقاله این است:

```powershell
python scripts\sim_mechanism.py analytic
python scripts\sim_mechanism.py train --seeds 0 1 2 --n-train 1000 --epochs 30 --batch 64 --lr 0.002 --models quad noquad classbd wdcnn --force
python scripts\sim_mechanism.py analyze --seeds 0 1 2
python scripts\sim_mechanism.py summary
python scripts\sim_figures.py
```

* **نکته:** نتیجه‌های sandbox با تنظیم دیگری گرفته شده‌اند (۴۰۰ پنجره در هر کلاس، ۱۲ epoch، batch 16). `--force` آن‌ها را بازنویسی می‌کند تا همه‌ی seedها با یک تنظیم باشند.
* خروجی‌ها در `results\sim_mechanism\` ذخیره می‌شوند: `accuracy_summary.md`، `prominence.json` و `figures\`.

## ۶. ساخت جدول‌ها و شکل‌های مقاله

```powershell
python scripts\make_tables.py main_tuned --reference phyq_nomed
python scripts\make_tables.py main_benchmark --reference phyq_nomed
python scripts\paper_assets.py
```

بعد این‌ها را برای من بفرستید (zip):
* `results\_paper_assets\` (همه‌ی فایل‌ها)
* `results\main_tuned\_report\significance.md` و `results\main_benchmark\_report\significance.md`
* `results\hp_search\selection.md`
* `results\sim_mechanism\accuracy_summary.md`، `prominence.json` و `figures\`

با این فایل‌ها همه‌ی `[TBD]`های جدول‌ها را در `paper_mssp\manuscript.tex` پر می‌کنم.

## ترتیب پیشنهادی و زمان تقریبی (GTX 1650)

| مرحله | زمان تقریبی |
|---|---|
| ۱ (۲۴ اجرای ClassBD، val) | ۳ تا ۵ ساعت |
| ۲ (۲۱ اجرای ClassBD) | ۴ تا ۶ ساعت |
| ۳ (کلاسیک، CPU، می‌تواند هم‌زمان با ۲ اجرا شود) | ۶ تا ۱۰ ساعت |
| ۴ (PU) | ۴ تا ۶ ساعت |
| ۵ (شبیه‌ساز) | ۲ تا ۳ ساعت |
