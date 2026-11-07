# راهنمای قدم‌به‌قدم اجرا روی سیستم شما (Windows 11 + GTX 1650)

> همهٔ دستورها را در **PowerShell** و داخل پوشهٔ پروژه اجرا کنید.
> هر جا نوشته‌ام «باید ببینید»، اگر چیز دیگری دیدید **همان‌جا متوقف شوید** و خروجی را برایم بفرستید.

---

## مرحلهٔ ۰ — پیش‌نیازها (یک‌بار)

1. **Python 3.11 (64-bit)** را از python.org نصب کنید. هنگام نصب تیک گزینهٔ **Add python.exe to PATH** را بزنید.
2. **درایور کارت گرافیک NVIDIA** را به‌روز کنید (از GeForce Experience یا سایت NVIDIA).
3. دست‌کم **۳۰ گیگابایت** فضای خالی داشته باشید (برای cache و نتایج).
4. در تنظیمات Power ویندوز، **Sleep را روی Never بگذارید** تا اجرای طولانی وسط کار قطع نشود.

> **v10.2 (MSSP):** برای به‌روزرسانی و کارهای اجرایی جدید (ClassBD، baseline کلاسیک، شبیه‌ساز، ساخت جدول‌های مقاله) فایل `docs/MSSP_TASKS_FA.md` را ببینید. روش به‌روزرسانی v10.2 با روش پایین **فرق دارد**، چون نتایج قبلی باید حفظ شوند.

## مرحلهٔ ۱ — قرار دادن پروژه

فایل zip در `D:\PhyQ-Bearing.zip` است. در PowerShell بزنید:

```powershell
Expand-Archive -Path D:\PhyQ-Bearing.zip -DestinationPath D:\
cd D:\PhyQ-Bearing
```

با این کار یک پوشهٔ **جدید** به نام `D:\PhyQ-Bearing` ساخته می‌شود. به پوشه‌های داده‌تان دست زده نمی‌شود.

### اگر نسخهٔ قدیمی را قبلاً باز کرده‌اید (به‌روزرسانی)

اول نسخهٔ قدیم را کنار بگذارید، بعد نسخهٔ جدید را باز کنید و `.venv` را برگردانید تا نصب‌ها تکرار نشود:

```powershell
cd D:\
Rename-Item D:\PhyQ-Bearing PhyQ-Bearing_old
Expand-Archive -Path D:\PhyQ-Bearing.zip -DestinationPath D:\
Move-Item D:\PhyQ-Bearing_old\.venv D:\PhyQ-Bearing\.venv
cd D:\PhyQ-Bearing
Get-Content configs\paths.yaml      # باید مسیرهای واقعی شما را ببینید، نه D:/Datasets
```

پوشهٔ `PhyQ-Bearing_old` را بعداً می‌توانید پاک کنید. نتایج آن فقط خطا بوده است.

## مرحلهٔ ۲ — ساخت محیط مجازی پایتون

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

* اگر خطای *running scripts is disabled* گرفتید، یک‌بار دستور زیر را بزنید، `Y` را وارد کنید و دوباره Activate کنید:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

* بعد از فعال شدن، ابتدای خط باید `(.venv)` دیده شود.
* **هر بار که PowerShell جدیدی باز می‌کنید**، اول `cd D:\PhyQ-Bearing` و بعد `.\.venv\Scripts\Activate.ps1` را بزنید.

## مرحلهٔ ۳ — نصب کتابخانه‌ها

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## مرحلهٔ ۴ — مسیر داده‌ها (از قبل با مسیرهای خودتان پر شده)

فایل `configs\paths.yaml` **از قبل با مسیرهایی که فرستادید پر شده** و لازم نیست چیزی را تغییر دهید. یک نسخهٔ پشتیبان از همین فایل هم در `configs\paths.user.yaml` هست.

```yaml
paths:
  cache: "D:/PhyQ-Bearing/cache"          # همهٔ فایل‌های موقت اینجا نوشته می‌شوند
  cwru: [".../data/0HP", ".../data/1HP", ".../data/2HP", ".../data/3HP"]
  pu: ".../data/Paderborn"
  ims: "D:/Bearing Project/4. Bearings/IMS"
  nrel: ".../data/Wind Turbine Gearbox Condition Monitoring Vibration/Data Information"
  sdust: ".../data/SDUST/SDUST_bearing"
```

(`...` یعنی `D:/Bearing Project/CWRU/CWRU Case 08/QCNN for bearing diagnosis`)

**مسیر CWRU را برایم نفرستاده بودید.** طبق کد قبلی‌تان، فایل‌های آن در پوشه‌های `data\0HP` تا `data\3HP` هستند. برای اطمینان، این دستور را بزنید. فقط سرآیند فایل‌ها خوانده می‌شود:

```powershell
python scripts\find_cwru.py "D:/Bearing Project/CWRU/CWRU Case 08/QCNN for bearing diagnosis/data" --write
```

جدولی چاپ می‌شود که نشان می‌دهد کدام ۴۰ فایل (۱۰ کلاس × ۴ بار) پیدا شده‌اند. با `--write` فقط مسیر CWRU داخل `configs\paths.yaml` به‌روز می‌شود. پوشه‌های Paderborn، SDUST و Wind Turbine برای سرعت بررسی نمی‌شوند. اگر فایلی کم بود، آدرس دانلود هم چاپ می‌شود.

### تضمین «فقط خواندنی»

* کد روی فایل‌های داده **فقط عمل خواندن** انجام می‌دهد: `loadmat`، `read_csv`، `whosmat`.
* همهٔ نوشتن‌ها فقط در `D:\PhyQ-Bearing\cache` و `D:\PhyQ-Bearing\results` انجام می‌شود.
* اگر به اشتباه cache یا results داخل یکی از پوشه‌های داده تنظیم شود، اجرا با پیام `Safety stop` متوقف می‌شود و چیزی نوشته نمی‌شود.
* تست‌های خودکار ثابت می‌کنند که نام، اندازه و زمان تغییر همهٔ فایل‌های داده بعد از اجرا عیناً یکسان می‌ماند.

### نکته‌های مخصوص دیتاست‌های شما

* برای اینکه ببینید داخل یک فایل چیست و loader کدام کانال‌ها را نگه می‌دارد (فقط خواندنی):
  `python scripts\inspect_file.py "<مسیر یک فایل>" --fs 25600`

| دیتاست | رفتار کد |
|---|---|
| CWRU | فایل‌ها بر اساس **کلید داخل فایل** (`X097_DE_time` و …) شناسایی می‌شوند، نه نام فایل. پوشه‌های `processed_data_robust*` اثری ندارند. |
| PU | پوشه‌های اضافه مثل `KI17`، `KB23/24/27` و `Dataset Information` در کلاس‌های پیش‌فرض استفاده نمی‌شوند. فایل خراب **رد و گزارش** می‌شود. اگر بیش از ۲۰٪ فایل‌ها خراب باشند، اجرا متوقف می‌شود. |
| IMS | پوشهٔ زمان‌دار حتی اگر تودرتو باشد (مثل `1st_test\1st_test` یا `4th_test\txt`) خودکار پیدا می‌شود. `4th_test` همان آزمون سوم است. |
| NREL | دو فایل حجیم `Healthy_data.csv` و `Damaged_data.csv` **فقط یک بار** و تکه‌تکه خوانده می‌شوند و ستون‌ها در cache ذخیره می‌شوند. بار اول کند است، بعد سریع. |
| SDUST | برچسب از نام پوشه (`NC`، `IF0.2`، …، `RF0.6`) گرفته می‌شود. **کانال تکراری ساخته نمی‌شود.** کد قبلی اگر فایل ۱ یا ۲ کانال داشت، همان کانال را کپی می‌کرد تا ۳ کانال شود. حالا فقط کانال‌های واقعی استفاده می‌شوند. بردار زمان و ستون‌های تکراری هم حذف می‌شوند. تعداد کانال‌ها در `sanity_check` چاپ می‌شود (`channels C=`). |

## مرحلهٔ ۵ — بررسی محیط

```powershell
python scripts\check_env.py
```

باید ببینید:
* خطی شبیه `[OK] CUDA GPU: NVIDIA GeForce GTX 1650, 4.0 GB`
* جلوی همهٔ دیتاست‌ها `OK`
* و در انتها `ALL GOOD`

اگر نوشت `CUDA NOT available`، مرحلهٔ ۳ را تکرار کنید (احتمالاً نسخهٔ CPU نصب شده است).

## مرحلهٔ ۶ — تست کد (حدود ۳۰ ثانیه)

```powershell
python -m pytest -q tests
```

باید ببینید: `42 passed`

## مرحلهٔ ۷ — اجرای آزمایشی روی دادهٔ مصنوعی (حدود ۱ دقیقه)

```powershell
python scripts\run_suite.py configs\experiments\smoke.yaml
```

باید ببینید: در انتها `Finished in ... min` و هیچ خطی شامل `FAILED` نباشد. دقت‌های این مرحله مهم نیستند.

## مرحلهٔ ۸ — بررسی دیتاست‌های واقعی

```powershell
python scripts\sanity_check_data.py cwru pu ims nrel sdust
```

* بار اول همهٔ فایل‌ها خوانده و در cache ذخیره می‌شوند. برای PU و IMS ممکن است **ده‌ها دقیقه** طول بکشد. اجراهای بعدی سریع‌اند.
* برای هر دیتاست تعداد رکوردها در هر کلاس، تقسیم train/val/test و عبارت `leakage checks OK` چاپ می‌شود.
* اگر برای دیتاستی `FAILED` دیدید، **کل خروجی همان دیتاست را برایم بفرستید**. معمولاً فقط یک خط در `configs\datasets\<name>.yaml` باید عوض شود، مثلاً `fs`، `n_channels` یا `condition_regex` برای SDUST.
* می‌توانید دیتاست‌ها را جدا هم بررسی کنید: `python scripts\sanity_check_data.py sdust`
* اگر فایلی خراب باشد، خط `[SKIPPED unreadable file]` همراه با نام فایل چاپ می‌شود.
* اگر خطای `Disk full` دیدید، روی درایو D جا خالی کنید. cache برای PU و IMS ممکن است چند گیگابایت شود.

## مرحلهٔ ۹ — تست سرعت روی GPU شما و تخمین زمان کل (حدود ۳۰ تا ۴۰ دقیقه)

**در v8 لازم است دوباره اجرا شود.** مدل پیشنهادی حالا مدل خود شما (PhyQ-TransNet) است، CWRU دو کانال دارد (DE و FE) و پنجرهٔ PU بلندتر شده است. برای همین زمان‌های قبلی دیگر معتبر نیستند.

```powershell
Remove-Item -Recurse -Force results\timing -ErrorAction SilentlyContinue
python scripts\run_suite.py configs\experiments\timing.yaml
python scripts\estimate_time.py
```

* دستور اول هر ۶ مدل را روی هر ۵ دیتاست فقط با ۲ epoch اجرا می‌کند. نتایجش در `results\timing` ذخیره می‌شود و **در جدول‌های مقاله استفاده نمی‌شود**.
* دستور دوم زمان هر suite را به ساعت تخمین می‌زند. بازهٔ چاپ‌شده از حالت «معمول» (با توقف زودهنگام) تا «حداکثر» است.
* اگر در لاگ `CUDA out of memory -> restarting this run with grad_accum=2` دیدید، **مشکلی نیست**. همان اجرا با micro-batch کوچک‌تر و همان batch مؤثر از اول شروع می‌شود.

## مرحلهٔ ۱۰ — مدل پیشنهادی: PhyQ-TransNet خودتان (بدون جست‌وجوی معماری)

* مدل پیشنهادی در همهٔ جدول‌ها (`phyq_transnet`) دقیقاً `QDCA_Hybrid_Pro` شماست.
  * تعداد پارامترها 1,404,715 است، همان 1.40M مقاله.
  * نام لایه‌ها، اندازه‌ها و مقداردهی اولیه همان کد اصلی است.
  * اگر وزن‌های مدل اصلی در آن بارگذاری شوند، خروجی‌اش با مدل اصلی **یکسان** است (این در تست‌ها چک می‌شود).
* فقط این چند مورد در اطرافِ مدل اصلاح شده‌اند، چون باعث نشت داده یا مقایسهٔ ناعادلانه می‌شدند:
  * ۱۱ ویژگی آماری هر کانال داخل مدل و از **سیگنال نویزی** محاسبه می‌شوند.
  * Neural MED فقط روی پنجره‌های **train** آموزش می‌بیند، بعد منجمد می‌شود و روی ورودی نویزی تست هم اعمال می‌شود.
  * نرمال‌سازی ورودی Instance Norm است، طبق معادلهٔ (8) مقاله.
* `scripts\search.py` دیگر در `run_all` نیست. نتایج `results\search` و `results\search_v2` (اگر دارید) را نگه دارید. در مقاله استفاده نمی‌شوند.
* فهرست اصلاحات لازم در **متن** مقاله در `docs\PAPER_FIXES_FA.md` آمده است.

## مرحلهٔ ۱۱ — اجرای کامل

| مرحله | محتوا |
|---|---|
| main benchmark | ۶ مدل × ۵ دیتاست؛ CWRU با ۵ seed، PU با ۵ fold × ۲ seed، بقیه با ۳ seed؛ نویز سفید، صورتی و ضربه‌ای |
| cross-load | CWRU، ۱۲ جفت بار × ۶ مدل × ۳ seed |
| ablation | همان ۵ ردیف جدول 9 مقاله (Full، بدون Quadratic، بدون Attention، بدون Transformer، بدون Statistical Stream) به‌علاوهٔ ۵ جزء دیگر (Soft-Thresholding، Meta-ACON، Neural MED، Instance Norm، augmentation نویز)؛ CWRU 3HP و دو تسک cross-load × ۳ seed (۹۰ اجرا) |

زمان دقیق را `estimate_time.py` در مرحلهٔ ۹ می‌دهد.

```powershell
.\scripts\run_all.ps1
```

* `run_all` اول `check_env` و `sanity_check_data` را اجرا می‌کند. اگر مسیر دیتاستی اشتباه باشد، در چند ثانیه با پیغام روشن **متوقف می‌شود**.
* **هر وقت خواستید متوقف کنید** (Ctrl+C) و بعد همان دستور را دوباره بزنید. اجراهای تمام‌شده تکرار نمی‌شوند. می‌توانید کار را در چند شب انجام دهید.
* اگر برق رفت یا پنجره بسته شد، **همان دستور را دوباره بزنید**؛ کار از جایی که مانده بود ادامه پیدا می‌کند.
* اجرای بخشی از کار، مثلاً فقط CWRU برای دیدن سریع اولین نتایج:

```powershell
python scripts\run_suite.py configs\experiments\main_benchmark.yaml --datasets cwru
python scripts\make_tables.py main_benchmark
python scripts\run_suite.py configs\experiments\main_benchmark.yaml --dry      # فقط فهرست اجراهای باقی‌مانده
```

### مرحلهٔ ۱۱-ب — PU با پروتکل‌های مقالات مرجع (جدید در v9)

جزئیات و دلایل در `docs\PU_REFERENCES_REVIEW_FA.md` آمده است.

```powershell
python scripts\run_suite.py configs\experiments\pu_quick.yaml       # ۳ آموزش (۹ نتیجه)، فقط مدل شما، ۱ seed
python scripts\make_tables.py pu_quick
python scripts\run_suite.py configs\experiments\pu_protocols.yaml   # کامل: ۶ مدل × ۶ پروتکل × ۳ seed = ۱۰۸ آموزش
python scripts\make_tables.py pu_protocols
```

| نام | معنی | عددی که باید با آن مقایسه شود |
|---|---|---|
| `seen_A`…`seen_D`, `seen_all` | یاتاقان‌های دیده‌شده، **فایل‌های تازه** (1–12 آموزش، 13–16 اعتبارسنجی، 17–20 آزمون) | اعداد ~۹۹٪ مقالات (که در واقع تقسیم تصادفی پنجره‌ای و نشت‌دار دارند) |
| `paper_dong7` | ۷ کلاس = ۷ یاتاقان (Dong 2025) | 99.90 گزارش‌شده در MSSP 2025 |
| `cc_A2B` … `cc_D2C` | شرط کاری تازه با همان یاتاقان‌ها. **آموزش جداگانه ندارد**: همان مدلِ `seen_X` روی آزمون شرط Y سنجیده می‌شود (تقسیم آموزش/اعتبارسنجی یکسان است و کد این یکسانی را بررسی می‌کند) | جدول ۴ Wang 2025 (AEI) |
| `fold0..4` (main benchmark) | یاتاقان‌های **کاملاً تازه** | Almeida 2026 (~77٪ LOBO)، Vieira 2026 |

## مرحلهٔ ۱۲ — ارسال نتایج برای من

```powershell
Compress-Archive -Path results -DestinationPath results.zip -Force
```

فایل `results.zip` را بفرستید (پوشهٔ cache را **نفرستید**). جدول‌ها و نمودارها در `results\<exp>\_report\` هستند. از روی همین‌ها بخش Results، جدول‌های LaTeX و متن مقاله را می‌نویسیم.

---

## رفع مشکلات رایج

* **`CUDA error: out of memory` یا `illegal memory access` (v9.2):**
  * هر اجرا در `run_suite` در یک پروسه‌ی جداگانه انجام می‌شود، پس خطای CUDA در یک اجرا دیگر بقیه را خراب نمی‌کند.
  * پیش از شروع آموزش، مصرف حافظه‌ی GPU اندازه‌گیری می‌شود و micro-batch و grad_accum طوری انتخاب می‌شوند که batch مؤثر تغییر نکند.
  * اگر خطای CUDA رخ دهد، همان اجرا با حالت امن (cuda-safe 1 و سپس 2) تکرار می‌شود. محاسبات در این حالت معادل‌اند و فقط کرنل‌ها عوض می‌شوند.
  * بعد از چنین خطایی بهتر است ویندوز را ری‌استارت کنید و برنامه‌هایی را که از GPU استفاده می‌کنند (مرورگر و غیره) ببندید.
  * برای عیب‌یابی: `python scripts\gpu_diag.py`

* **`loss nan` در لاگ آموزش (v9.1):** در float16 (AMP)، کانولوشن‌های درجه‌دو مقادیر را تا حد سرریز بالا می‌برند (حتی در شروع آموزش حدود 4×10⁴، در حالی که سقف float16 برابر 65504 است). از v9.1 مدل PhyQ-TransNet همیشه در float32 اجرا می‌شود. اگر مدل دیگری هم loss نامعتبر بدهد، آن اجرا خودکار از ابتدا بدون AMP تکرار می‌شود. نتایجی که با `loss nan` ثبت شده‌اند را پاک کنید و دوباره اجرا کنید.

| مشکل | راه‌حل |
|---|---|
| `Unable to allocate ... MiB` (کمبود RAM) | در `configs\datasets\<name>.yaml` زیر `data:` خط `preload: false` را اضافه کنید. حالت `auto` معمولاً خودش این را تشخیص می‌دهد. |
| `device cpu` در لاگ یا `NO CUDA GPU VISIBLE` | `.venv` فعال نیست. در **هر پنجرهٔ جدید** PowerShell اول بزنید: `cd D:\PhyQ-Bearing` و بعد `.\.venv\Scripts\Activate.ps1`. باید `(.venv)` در ابتدای خط دیده شود. |
| `CUDA out of memory` | از v8 این مورد خودکار است: همان اجرا با gradient accumulation (همان batch مؤثر) دوباره شروع می‌شود. فقط اگر با `grad_accum=8` هم خطا داد، `batch_size` همان دیتاست را در `configs\datasets\<name>.yaml` کم کنید. |
| کمبود RAM هنگام PU یا IMS | در `configs\datasets\pu.yaml` (یا `ims.yaml`) زیر `data:` خط `preload: false` را اضافه کنید |
| `CWRU: no variable X<id>_DE_time found` | `python scripts\find_cwru.py "<پوشهٔ data>" --write` را بزنید، یا فقط بارهای کامل را اجرا کنید: `data.loads=[3]` |
| `Safety stop` | cache یا results داخل پوشهٔ داده تنظیم شده است. مسیر `cache` را در `paths.yaml` خارج از پوشه‌های داده بگذارید. |
| `PRE-FLIGHT FAILED ... is the EXAMPLE path` | هنوز نسخهٔ قدیم یا `paths.yaml` نمونه را دارید. بخش «به‌روزرسانی» در مرحلهٔ ۱ را انجام دهید. **هیچ اجرایی انجام نشده و به هیچ فایلی دست نخورده است.** |
| `PRE-FLIGHT FAILED ... folder not found` | آن مسیر در `configs\paths.yaml` وجود ندارد. آن را در File Explorer باز کنید و مسیر را از نوار آدرس کپی کنید (به‌جای `\` از `/` استفاده کنید). |
| `DATA problem for <dataset>/*: all its remaining runs are skipped` | آن دیتاست پیدا شد ولی فایل‌های لازم داخلش نبود. بقیهٔ اجراهای همان دیتاست رد شدند تا وقت هدر نرود. اسکریپت `run_all` همان‌جا می‌ایستد. پیغام خطای بالای این خط را برایم بفرستید. |
| `FileNotFoundError ... not found under` | مسیر داده در `configs\paths.yaml` اشتباه است، یا ساختار پوشه با جدول مرحلهٔ ۴ نمی‌خواند |
| اسکریپت `.ps1` اجرا نمی‌شود | دستور `Set-ExecutionPolicy` در مرحلهٔ ۲ |
| وسط اجرا تنظیمات را عوض کردید | پوشهٔ `results\<exp>` همان آزمایش را پاک کنید، وگرنه نتایج قدیم و جدید قاطی می‌شوند |
| در ابتدای خط `(.venv)` نیست | `.\.venv\Scripts\Activate.ps1` را بزنید |

## قانون‌های طلایی برای مقاله

1. **هیچ تنظیمی را بر اساس نتایج تست عوض نکنید.** معماری PhyQ-TransNet ثابت است (طراحی خودتان) و epoch بهینه فقط با validation انتخاب می‌شود.
2. اعداد مقاله فقط از `make_tables.py` می‌آیند؛ هیچ عدد دستی یا عددی از کد قدیمی وارد مقاله نشود.
3. اگر PhyQ-TransNet جایی بازنده بود، همان را گزارش می‌کنیم. داوران Q1 گزارش صادقانه و تحلیل‌شده را به ادعای «بهترین همه‌جا» ترجیح می‌دهند.
