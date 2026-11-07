# بازبینی فایل‌های LaTeX مقاله (paper.tex + references.bib)

- **منبع بررسی‌شده:** شاخه‌ی `add-latex-20261022` (commit `879d0b3`) در مخزن `AliNaderiii/bearing`
- **تاریخ:** 2026-10-25
- **قاعده‌ی کار:** هیچ فایلی در مسیرهای شما (`D:\Bearing Project\...\Bearing Paper - Ali Naderi\`) تغییر نکرده و نباید تغییر کند. همه‌ی نسخه‌های اصلاح‌شده در پوشه‌ی جدید و جداگانه قرار گرفته‌اند.
- **شماره‌ی خط‌ها:** همه‌ی شماره‌خط‌ها مربوط به **فایل اصلی شما** (`paper.tex` در همان commit) است، مگر آنکه جایی صریحاً نوشته شده باشد «نسخه‌ی اصلاح‌شده».

---

## §0. خلاصه و فایل‌های تحویلی

| فایل (در workspace) | توضیح |
|---|---|
| `paper_latex_revised/paper.tex` | نسخه‌ی اصلاح‌شده: کلیدهای ارجاع اصلاح شده‌اند و خطای کامپایل برطرف شده است. محتوای علمی و نتایج **دست نخورده‌اند** و فقط با `% REVIEW[...]` علامت خورده‌اند. |
| `paper_latex_revised/references.bib` | ۵۴ مدخل که **همه** با Crossref یا doi.org (و Rosa با arxiv.org) تطبیق داده شده‌اند. بالای هر مدخل وضعیتش (VERIFIED / REPLACED / NEW / FLAGGED) نوشته شده است. |
| `paper_latex_revised/paper_tex_changes.diff` | تفاوت خط‌به‌خط فایل اصلی و نسخه‌ی اصلاح‌شده (۷۵ خط تغییر کرده). |
| `paper_latex_revised/paper_preview_NO_IMAGES.pdf` | PDF آزمایشی کامپایل‌شده. پوشه‌ی `images/` در مخزن نبود، به همین دلیل جای هر شکل یک کادر «[image]» گذاشته شده است. هدف این فایل فقط کنترل ارجاع‌ها و فهرست منابع است. |
| `PhyQ-Bearing/docs/LATEX_REVIEW_FA.md` | همین گزارش. |

**نتیجه‌ی کلی:**
1. **فایل اصلی کامپایل نمی‌شود.** خطای قطعی `Option clash for package xcolor` در خط ۲۴ رخ می‌دهد (توضیح در §1).
2. **از ۴۷ ارجاعِ استفاده‌شده در متن، ۲۰ ارجاع در هیچ پایگاهی یافت نشد.** این‌ها عملاً مراجع ساختگی‌اند و احتمالاً با ابزار تولید متن ساخته شده‌اند. چند ارجاع دیگر هم واقعی‌اند ولی مشخصاتشان (ژورنال، نویسنده، صفحه) غلط است.
   - این مهم‌ترین ریسک مقاله است. یک داور MSSP یا Measurement فقط با کلیک روی DOI متوجه آن می‌شود و نتیجه‌اش معمولاً **reject فوری و گاهی گزارش اخلاقی** است.
   - همه‌ی این موارد در نسخه‌ی اصلاح‌شده با مراجع **واقعی و متناسب با جمله** جایگزین شده‌اند (§3).
3. نسخه‌ی اصلاح‌شده با موتور Tectonic (XeTeX + BibTeX) **بدون هیچ ارجاع تعریف‌نشده‌ای** کامپایل شد:
   - ۵۳ مرجع در فهرست منابع چاپ شد؛
   - `Citation undefined = 0` و `Reference undefined = 0`.
4. مشکلات محتوایی (اعداد نشت‌دار، جدول‌های کپی‌شده، PU file-wise و غیره) در این مرحله **بازنویسی نشده‌اند.** فقط با `% REVIEW[Cn]` علامت خورده‌اند (§5)، چون بازنویسی بخش نتایج مرحله‌ی بعدی کار است.

---

## §1. نتیجه‌ی کامپایل

| مورد | فایل اصلی | نسخه‌ی اصلاح‌شده |
|---|---|---|
| خطای مهلک | `paper.tex:24 LaTeX Error: Option clash for package xcolor` | ندارد ✅ |
| BibTeX | `paper.bbl` و `references.log` شما **خالی** بودند، یعنی BibTeX هیچ‌وقت درست اجرا نشده بود. `paper.log` قدیمی ۷۸ هشدار natbib دارد. | ۵۳ مرجع؛ فقط ۲ هشدار بی‌اهمیت `empty pages` برای دو مقاله‌ی ICLR |
| ارجاع تعریف‌نشده | ۳ مورد در log قدیمی (log کهنه است؛ در tex فعلی همه‌ی labelها تعریف شده‌اند) | ۰ |
| Overfull hbox | — | ۲ مورد: جدول IMS (`tab:ims_results`، حدود 23pt) و جدول PU (`tab:pu_results`، حدود 17pt). اگر این جدول‌ها پس از بازنویسی نتایج باقی ماندند، با `\resizebox{\columnwidth}{!}{...}` یا `\small` درست می‌شوند. |
| هشدار فونت `TU/txr` | — | فقط مخصوص XeTeX است. در pdfLaTeX (Overleaf یا MiKTeX) دیده نمی‌شود. |

**علت خطای xcolor:** بسته‌های `tikz` (خط ۱۹) و `colortbl` (خط ۲۳) خودشان `xcolor` را **بدون** گزینه بارگذاری می‌کنند. وقتی خط ۲۴ دوباره `\usepackage[table]{xcolor}` را صدا می‌زند، LaTeX خطای option clash می‌دهد.
- MiKTeX در حالت تعاملی با زدن Enter از این خطا رد می‌شود. احتمالاً به همین دلیل تا امروز PDF ساخته می‌شد، ولی سامانه‌های ارسال Elsevier (Editorial Manager) روی خطا متوقف می‌شوند.
- **اصلاح:** `\usepackage[table]{xcolor}` قبل از `tikz` آمده و `colortbl` حذف شده، چون گزینه‌ی `table` خودش colortbl را بارگذاری می‌کند.

---

## §2. اصلاحات پیش‌درآمد (Preamble)

| خط اصلی | مشکل | اصلاح در نسخه‌ی جدید |
|---|---|---|
| 2–3 | `MANUSCRIPT SKELETON` و `Target Journal: ???` | به توضیح «Revised manuscript» تغییر کرد. |
| 8 | توضیحی درباره‌ی گزینه‌ی `demo` که اصلاً استفاده نشده است | حذف شد. |
| 19، 23–24 | خطای option clash بسته‌ی xcolor | `\usepackage[table]{xcolor}` پیش از tikz منتقل شد و colortbl حذف شد. |
| 21–22 | `\usepackage{multirow}` دو بار تکرار شده | یکی حذف شد. |
| 29 | `\journal{International Journal of Machine Tools and Manufacture}`. این ژورنال برای موضوع مقاله نامناسب است و با هدف‌گذاری ما هم نمی‌خواند. | `\journal{Measurement}` با توضیح «REVIEW: set to the actual target journal». اگر EAAI انتخاب شد، فقط همین خط را عوض کنید. |
| 11 | بسته‌ی `lineno` بارگذاری شده ولی `\linenumbers` وجود ندارد | `\linenumbers` بعد از `\begin{document}` اضافه شد (اکثر ژورنال‌های Elsevier هنگام ارسال شماره‌خط می‌خواهند). |
| 280 | توضیح فارسی `% تغییر به textwidth` در کد. فایل منبع برای ژورنال ارسال می‌شود. | به `% width changed to textwidth` ترجمه شد. |

---

## §3. جدول حکم مراجع (کلید قدیم → وضعیت → کلید جدید)

راهنمای وضعیت‌ها:
- ✅ **درست**: مرجع واقعی است و فقط مشخصاتش از روی DOI تکمیل یا اصلاح شد.
- 🔧 **مشخصات غلط**: مقاله واقعی است ولی ژورنال، نویسنده یا صفحه اشتباه بود.
- ❌ **ساختگی**: در Crossref هیچ مقاله‌ای با این مشخصات وجود ندارد.

### 3-الف. مراجع مقدمه و پیشینه

| خط | کلید قدیم | وضعیت | کلید جدید | DOI / توضیح |
|---|---|---|---|---|
| 59 | P7_Hosseini | ✅ | Hosseini2024_RSER | 10.1016/j.rser.2024.115227 |
| 59 | P1_Lu, P5_Yu (جمله‌ی «weakest link») | 🔧 هر دو واقعی‌اند ولی **ادعای جمله را پشتیبانی نمی‌کنند**: Lu درباره‌ی وزن‌دهی ویژگی physics-informed است و Yu درباره‌ی پیش‌بینی روند تخریب. | Guo2020_NREL_RSER، deAzevedo2016_RSER | 10.1016/j.rser.2020.109888 ؛ 10.1016/j.rser.2015.11.032. مقاله‌ی Yu به جمله‌ی اول (تنش مکانیکی گیربکس) و Lu به پاراگراف physics-informed منتقل شد. |
| 61 | P3_Dibaj | ✅ | Dibaj2023_RenewEnergy | 10.1016/j.renene.2022.12.049 |
| 61 | P38_Chen | ❌ | Feng2013_TFA_MSSP | 10.1016/j.ymssp.2013.01.017 (مرور روش‌های زمان-فرکانس) |
| 61، 103 | P11_Li | ✅ | Li2022_WaveletKernelNet | 10.1109/tsmc.2020.3048950 |
| 61، 103 | P31_He | ❌ | He2023_EWSNet | 10.1016/j.jmsy.2023.08.014 (همان مقاله‌ی واقعی P41 در پوشه‌ی مراجع شما) |
| 61 | P27_Zhang (در فهرست «تجزیه‌ی تطبیقی») | ✅، ولی در جای نادرست | Zhang2023_FastNBD ← به فهرست BD منتقل شد | 10.1016/j.ymssp.2022.109918. این مقاله BD است، نه تجزیه. |
| 61 | P28_He | 🔧 نویسنده‌ی واقعی Miao است | Miao2023_FMD | 10.1109/tie.2022.3156156 |
| 61 | P26_Chen | ❌ | Lei2013_EMD_MSSP + Dragomiretskiy2014_VMD | 10.1016/j.ymssp.2012.09.015 ؛ 10.1109/tsp.2013.2288675 |
| 61 | P34_Huang | ❌ | Wiggins1978_MED | 10.1016/0016-7142(78)90005-4. MED پایه‌ی بلوک Neural MED شماست و ذکرش لازم است. |
| 61 | P22_He | ❌ | McDonald2012_MCKD | 10.1016/j.ymssp.2012.06.010 |
| 61 | P24_Cheng، P25_Zhang، P33_Wang | ❌ | ادغام در فهرست BD (MED، MCKD، FastNBD) | — |
| 61 | P30_Li | ❌ | Zhao2019_DLMHM_MSSP | 10.1016/j.ymssp.2018.05.050 |
| 61 | P32_Liao | ❌ | Zhang2020_Review_Access | 10.1109/ACCESS.2020.2972859 |
| 63، 96، 461، 907 | P21_Zhang | ✅ | Zhang2017_WDCNN | 10.3390/s17020425 |
| 63، 96، 117، 461، 908 | P20_Zhao | 🔧 فهرست نویسندگان اصلاح شد | Zhao2020_DRSN | 10.1109/tii.2019.2943898 |
| 70، 108 | P36_Fan | 🔧 | Fan2025_TPAMI | 10.1109/TPAMI.2025.3588894 |
| 70، 108 | P41_Xu | ❌ | Fan2020_UniversalQuadratic | 10.1016/j.neunet.2020.01.007. در خط 108 عبارت «Xu et al.» هم به «Fan et al.» تغییر کرد. |
| 72، 113 | P9_Yu | 🔧 این مقاله **Bi-LSTM** است | Yu2020_BiLSTM | 10.1016/j.neucom.2020.03.041 |
| 72، 113 | P17_Yin | ❌ | Zhang2022_DCABiGRU | 10.1016/j.measurement.2021.110242 |
| 74، 117 | P12_Yang | ❌ | Hou2021_CA | 10.1109/cvpr46437.2021.01350 (مقاله‌ی اصلی Coordinate Attention که مدل شما از آن استفاده می‌کند) |
| 74 | P40_Wang | ❌ | Hu2018_SE | 10.1109/cvpr.2018.00745 |
| 96، 909 | P34_Huang، P16_Zhu (multi-scale) | ❌ | Chen2021_MCCNNLSTM (+ Ref_Ji_MDQCNN در خط 96) | 10.1007/s10845-020-01600-2. نام آن دقیقاً با ردیف «MC-CNN-LSTM» جدول ۷ می‌خواند. |
| 105 | P32_Liao، P33_Wang (unfolding) | ❌ | Lu2023_MSSP، Liao2023_MNNBD | 10.1016/j.ymssp.2023.110171 ؛ 10.1088/1361-6501/accbdb |
| 110، 910 | P37_Liao، P13_Liao | ❌ (نسخه‌ی ساختگیِ Qttention) | Ref_Liao_QCNN | 10.1109/TIM.2023.3259031 (مقاله‌ی واقعی Qttention) |
| 911 | P10_Ji | ❌ (ژورنال IJMTM و نویسندگان غلط) | ادغام در Ref_Ji_MDQCNN | 10.1007/s11071-025-10918-6 |
| 115 | Ref_Naderi_BrainTumor_2025 | ✅ واقعی ولی **بی‌ربط** (self-citation در حوزه‌ی پزشکی) | حفظ شد، با `% REVIEW[C10]` | 10.1155/cplx/1644859. پیشنهاد: جمله حذف شود (§5). |

### 3-ب. مراجع مقایسه‌ای ۲۰۲۵–۲۰۲۶ (همه واقعی‌اند؛ مشخصات از DOI تکمیل شد)

| کلید | DOI | اصلاح |
|---|---|---|
| Ref_Guo_DiffUCD | 10.1016/j.measurement.2024.115951 | — |
| Ref_Zhang_GAN | 10.1016/j.measurement.2025.118583 | جای نام و نام خانوادگی نویسندگان در Crossref جابه‌جا بود و اصلاح شد (Zhang, Shuntang …). |
| Ref_Li_ESC_DGCN | 10.1016/j.mechmachtheory.2025.106267 | — |
| Ref_Xu_MSSP_2026 | 10.1016/j.ymssp.2026.113917 | — |
| Ref_Dong_MSSP_2025 | 10.1016/j.ymssp.2025.113371 | — |
| Ref_Wang_AEI_2025 | 10.1016/j.aei.2025.103405 | — |
| Ref_Liu_AEI_2025 | 10.1016/j.aei.2025.103491 | — |
| Ref_Wu_TIM_2026 | 10.1109/tim.2026.3654735 | DOI اضافه شد. |
| Asyari_NREL_2025 | 10.1109/access.2025.3553122 | صفحات (54116–54129) و فهرست نویسندگان اصلاح شد. |
| Heo2021_AdamP، zhang2017mixup | ICLR 2021 / ICLR 2018 | عنوان AdamP اصلاح شد. |

### 3-ج. مراجع داده و پروتکل

| خط | کلید | وضعیت | توضیح |
|---|---|---|---|
| 626 | paderborn_reference | 🔧 | عنوان اصلاح شد («…Motor Current Signals of Electric Motors…») و DOI 10.36001/phme.2016.v3i1.1577 اضافه شد. |
| 726، 762 | nrel_reference | 🔧 | Sheng, NREL/TP-5000-54530 (2012)، DOI 10.2172/1048981 |
| 662 | paderborn_cnn_eren | ❌ مقاله‌ی Rev. Sci. Instrum. 90:025004 وجود ندارد | محتوای مدخل با مقاله‌ی واقعی Eren و همکاران، J. Signal Process. Syst. 91 (2019)، DOI 10.1007/s11265-018-1378-3 جایگزین شد. **ولی آن مقاله ۸۶٫۴۰٪ روی PU گزارش نکرده است**، پس ردیف جدول همچنان نادرست است (C4). |
| 663 | paderborn_vit | ❌ (Measurement 190:110698 متعلق به مقاله‌ی دیگری است) | **عمداً** به صورت مدخل «UNVERIFIED REFERENCE…» نگه داشته شد تا در PDF دیده شود و فراموش نشود. این ردیف باید حذف شود. |

### 3-د. مراجع جدیدی که اضافه شد

| کلید | DOI | کجا ارجاع داده شد |
|---|---|---|
| Liao2025_ClassBD ⚠️ **ضروری** | 10.1016/j.ymssp.2024.111750 | خط 110: یک جمله‌ی تمایز اضافه شد (پایین‌تر). ClassBD هم از فیلتر درجه‌دو و هم از BD استفاده می‌کند و **نزدیک‌ترین کار پیشین** به مدل شماست. ذکر نکردنش برای داوران MSSP یک خطای جدی است. |
| Hendriks2022_MSSP ⚠️ ضروری | 10.1016/j.ymssp.2021.108732 | خط 259 (جمله‌ی نشت داده) |
| Rosa2024_arXiv ⚠️ ضروری | arXiv:2407.14625 (Rosa, Braga, Silva) | خط 259 |
| Vieira2026_MSSP ⚠️ ضروری | 10.1016/j.ymssp.2026.114640 | خط 259 |
| Smith2015_CWRU | 10.1016/j.ymssp.2015.04.021 | خط 223 (معرفی CWRU) |
| SDUST_dataset + Chen2026_MSSP_OpenSet | github.com/JRWang-SDUST ؛ 10.1016/j.ymssp.2025.113667 | خط 226 (معرفی SDUST) |
| Qiu2006_IMS | 10.1016/j.jsv.2005.03.007 | خط 553 (معرفی IMS) |
| Guo2020_NREL_RSER، deAzevedo2016_RSER | (بالا) | خط 59 |
| Lu2023_MSSP، Liao2023_MNNBD، Lei2013، VMD، MED، MCKD | (بالا) | مقدمه و پیشینه |
| Lee2023_Access | 10.1109/ACCESS.2023.3281910 | در bib هست ولی هنوز **ارجاع داده نشده** و چاپ نمی‌شود. پیشنهاد: در بخش تنظیم ابرپارامترها کنار AdamP بیاید. |

### 3-هـ. حذف‌شده‌ها

این مدخل‌ها در هیچ‌جای متن ارجاع نشده بودند و غالباً قابل‌تأیید هم نبودند:
P18_Zhao، P4_Dibaj، P35_Miao، P23_Jia، P15_Yan، Ref_Xu_MCCGAN، Chen_ZeroShot_2025، Ogaili_FME_2024، miao_caae، paderborn_eemd.

مدخل‌های ساختگی بالا هم با کلید جدید جایگزین شده‌اند.

---

## §4. تغییرات متنی که در `paper.tex` اعمال شد

همه‌ی این تغییرات فقط در نسخه‌ی جدید اعمال شده‌اند.

| خط اصلی | قبل | بعد |
|---|---|---|
| 63 | `GCNs incur an exponential $\mathcal{O}(N^2)$` | `a quadratic $\mathcal{O}(N^2)$`. پیچیدگی N² نمایی نیست و این خطای ریاضی آشکاری بود. |
| 108 | `Xu et al. \cite{P41_Xu}` | `Fan et al. \cite{Fan2020_UniversalQuadratic}` |
| 110 | جمله‌ی Qttention | با `\cite{Ref_Liao_QCNN}`، به‌اضافه‌ی جمله‌ی تمایز با ClassBD (متن زیر) |
| 113 | `RNNs like Bi-GRUs \cite{P9_Yu, P17_Yin}` | `RNNs such as Bi-LSTMs and Bi-GRUs \cite{Yu2020_BiLSTM, Zhang2022_DCABiGRU}` |
| 223، 226، 553 | معرفی CWRU، SDUST و IMS بدون مرجع | ارجاع به مقاله‌ی اصلی هر دیتاست اضافه شد. |
| 259 | پاراگراف «File-Level Partitioning» | یک جمله درباره‌ی نشت داده با سه مرجع ضروری اضافه شد (متن زیر). |

**متن اضافه‌شده در خط 110** (علامت `% REVIEW-ADDED` دارد؛ لطفاً خودتان تأیید کنید):
> The same group subsequently proposed ClassBD \cite{Liao2025_ClassBD}, a classifier-guided neural blind-deconvolution front-end built on quadratic filters and trained jointly with the downstream classifier. ClassBD is a plug-in denoising module placed before a conventional classifier, whereas the present work embeds quadratic operators inside the feature backbone, couples them with a Transformer encoder for global context, and adds a parallel statistical stream.

**متن اضافه‌شده در خط 259:**
> It should be noted that overlapping windows and file-wise splits are known to cause data leakage and over-optimistic accuracies on CWRU and PU \cite{Hendriks2022_MSSP, Rosa2024_arXiv, Vieira2026_MSSP}; the protocols used here are therefore stated explicitly for each dataset.

**نکته‌ی مهم درباره‌ی پاراگراف خط 105** («…these methods fundamentally rely on *linear* convolution operations»): این ادعا برای ClassBD **صادق نیست**، چون ClassBD از نورون‌های درجه‌دو استفاده می‌کند. به همین دلیل ClassBD در خط 105 ارجاع داده نشد و تمایزش جداگانه در خط 110 آمد.

---

## §5. مشکلات محتوایی که علامت خورده‌اند ولی بازنویسی نشده‌اند

این موارد در `docs/PAPER_FIXES_FA.md` و `review/Review_Report_FA.md` به‌تفصیل آمده‌اند. اینجا فقط نگاشتشان به خط‌های tex و نشانگری که در نسخه‌ی جدید گذاشته شده آمده است.

| نشانگر | خط اصلی | مشکل | اقدام لازم |
|---|---|---|---|
| C1 | 43–46 (Abstract) | همه‌ی اعداد کلیدی (99.72٪ CWRU و 99.34٪ PU و غیره) از پروتکل‌های نشت‌دار آمده‌اند. | با نتایج re-run جایگزین شود: مدل نهایی phyq_nomed، ۳ seed، پروتکل بدون نشت. |
| C2 | 522–531 (`tab:sdust_results`) | زیرمجموعه و پروتکل SDUST مشخص نیست. SDUST حداقل ۵ زیرمجموعه دارد. | نتیجه‌ی ما (clean / 0 / −6 / −10 dB): 99.10 / 91.89 / 78.65 / 66.28. صریحاً بنویسید که با اعداد مقالات دیگر قابل مقایسه نیست. |
| C3 | 552–581 (`tab:ims_results`) | جابه‌جایی تصادفی فایل‌ها + SNR نسبی | با نتایج re-run جایگزین شود: 78.29 / 79.16 / 73.70 / 64.49. CNN-Transformer روی IMS بهتر است و باید صادقانه گزارش شود. |
| C4 | 640–698 (`tab:paderborn_comparison`، `tab:pu_results`) | تقسیم file-wise نشت‌دار است. ۸ عدد جدول مقایسه (86.40 تا 98.89) **روی PU و با این پروتکل گزارش نشده‌اند**؛ مثلاً MDQCNN اصلاً از PU استفاده نکرده و 98.81٪ آن روی Torino در −10 dB است. ViT هم وجود ندارد. | جدول ۴ کاملاً حذف و با جدول پروتکل‌های PU جایگزین شود. متن پیشنهادی پایین‌تر آمده است. |
| C5 | 895–916 (`tab:comparative_results`) | ردیف‌های baseline از مقالات دیگر کپی شده‌اند و تحت پروتکل یکسان آموزش ندیده‌اند. caption هم «3 HP→3 HP» است. | با baselineهای بازآموزی‌شده‌ی ما جایگزین شود: WDCNN، DRSN، QCNN، CNN-Transformer و Stats-MLP، هر کدام با ۳ seed. |
| C6 | 918–935 (`tab:load_stability`) | «cross-load» در واقع within-load است. | برچسب اصلاح شود، یا آزمایش cross-load واقعی اجرا شود. |
| C7 | 940–975 (`tab:ablation`) | ablation باید روی validation و با ۳ seed باشد. **در مدل نهایی MED خاموش است.** | اعداد ablation ما روی validation، اختلاف نسبت به مدل کامل (Δ): no_prefilter +11.53؛ no_quad +0.81؛ no_stats −3.55؛ no_instance_norm −7.00؛ no_noise_aug −44. |
| C8 | 1230–1240 (`tab:complexity`) | پیچیدگی باید مربوط به مدل نهایی باشد. | 1.40 M پارامتر / 343.6 M MACs |
| C9 | 265–284 (بخش Neural MED و شکل `fig:filter_effect`) | مدل نهایی (nomed) **بدون** MED است. | به یکی از دو صورت: (الف) بخش MED به «optional pre-filter, evaluated in ablation» تبدیل شود، با ذکر اینکه روی validation زیان‌آور بود (+11.53)؛ یا (ب) کامل حذف شود. |
| C10 | 115 | self-citation پزشکی بی‌ربط | پیشنهاد: کل جمله حذف شود. |
| — | 113 | ادعای «graph-based … quadratically» برای ESC-DGCN | قابل دفاع است. |

**متن انگلیسی پیشنهادی برای جایگزینی پاراگراف PU** (خط 640). این متن هنوز در tex اعمال نشده و در مرحله‌ی بعدی، همراه با بازنویسی کامل Results، اعمال می‌شود:
> On the Paderborn University (PU) dataset, reported accuracies depend strongly on the partitioning protocol: random or file-wise splits allow windows of the same physical bearing to appear in both training and test sets and routinely yield >95% accuracy, whereas bearing-wise evaluation is substantially harder \cite{Vieira2026_MSSP}. We therefore report two leakage-free protocols over 13 bearings: a bearing-wise 5-fold split and an artificial-to-real transfer (train on artificially damaged bearings, test on naturally damaged ones). Under the bearing-wise 5-fold protocol PhyQ-TransNet attains 57.7% (clean) and 62.6% at −6 dB; under artificial-to-real transfer it attains 55.8% (clean) and 52.8% at −6 dB, outperforming all re-trained baselines at −6 dB by 14.7–17.0 percentage points (Holm-corrected $p \le 0.004$). These figures are not comparable to file-wise accuracies reported elsewhere.

---

## §6. استفاده از نسخه‌ی اصلاح‌شده (بدون دست زدن به پوشه‌ی فعلی)

1. پوشه‌ی `paper_latex_revised/` را از workspace بگیرید و در مخزن، روی یک شاخه‌ی جدید (مثلاً `paper-revised-20261025`) در مسیر `paper_revised/` بگذارید. یا روی سیستم، در یک پوشه‌ی **جدید و جدا** قرار دهید؛ مثلاً `D:\PhyQ-Bearing\paper_revised\`.
2. پوشه‌ی `images\` فعلاً فقط روی سیستم شماست. برای کامپایل کامل، یک **کپی** از آن در پوشه‌ی جدید بگذارید. کپی کردن فقط از مبدأ می‌خواند و چیزی را تغییر نمی‌دهد:
   ```powershell
   Copy-Item -Recurse "D:\Bearing Project\CWRU\CWRU Case 08\Bearing Paper - Ali Naderi\images" "D:\PhyQ-Bearing\paper_revised\images"
   ```
3. کامپایل با MiKTeX یا Overleaf، به ترتیب: `pdflatex → bibtex → pdflatex → pdflatex`. نسخه‌ی جدید نباید هیچ خطایی بدهد.
4. برای دیدن همه‌ی نقاطی که باید تصمیم بگیرید، در فایل جدید عبارت `REVIEW` را جست‌وجو کنید. فهرست منابع هم باید بدون مدخل «UNVERIFIED» باشد، یعنی ردیف ViT جدول ۴ حذف شده باشد.
