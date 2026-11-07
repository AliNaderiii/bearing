# بررسی مراجع جدید (SDUST + P1–P41)

> منبع فایل‌ها: branch `add-references-20261022` در مخزن `AliNaderiii/bearing`، پوشه‌های `references/sdust/` (۷ فایل) و `references/general/` (P1–P41).
> صحت دانلود: SHA256 هر ۴۸ فایل با `references/INDEX.csv` مطابقت دارد (۰ عدم‌تطابق).
> تاریخ بررسی: 2026-10-22. مبنای مقایسه: نتایج v10.1 پروژه (مدل نهایی `phyq_nomed`) و متن مقاله‌ی فعلی (`paper.txt`).

---

## ۰. خلاصه‌ی یافته‌های مهم (به ترتیب اهمیت)

1. **فهرست مراجع مقاله به‌طور سیستماتیک خراب است.** برای بیشتر مراجعی که PDF آن‌ها را فرستادید، نام نویسنده، عنوان، ژورنال یا سال در فهرست مراجع مقاله **با خود مقاله نمی‌خواند** (جدول §۳). نمونه: مرجع [26] به‌صورت «Y. Fan, J. Xu, X. Wu, Quadratic NNs for solving PDEs, Neurocomputing 2019» آمده، در حالی که مقاله‌ی واقعی (P23) «F.-L. Fan et al., One Neuron Saved Is One Neuron Earned, **IEEE TPAMI 2025**» است. داور یا ویراستاری که چند مرجع را چک کند، این را نشانه‌ی مراجع ساختگی می‌بیند و **احتمال desk-reject بالاست**. فهرست مراجع باید کامل و از روی DOI بازنویسی شود.
2. **دو عدد جدول ۴ مقاله (PU) از منبعشان نمی‌آیند:**
   - **DiffUCD [20] = 96.58٪:** این عدد در مقاله‌ی Guo et al. **دقت GAN-Train** (معیار کیفیت داده‌ی تولیدشده روی PU) است، نه دقت تشخیص در پروتکل قابل‌مقایسه.
   - **MDQCNN [28] = 98.81٪:** MDQCNN **اصلاً روی PU آزمایش نشده**. 98.81٪ دقت آن روی دیتاست Politecnico di Torino در −10 dB است.
   - **نتیجه:** جدول ۴ باید کامل حذف شود و جای آن را نتایج baselineهای بازپیاده‌سازی‌شده با پروتکل یکسان بگیرد (که داریم).
3. **«SDUST» یک benchmark واحد نیست.** در ۸ مقاله‌ای که SDUST را استفاده کرده‌اند، دست‌کم ۵ زیرمجموعه یا پیکربندی متفاوت دیده می‌شود. این‌ها در تعداد کلاس (۳، ۴ یا ۱۰)، نوع بلبرینگ (6205 یا N205EU)، سرعت و بار، و تعداد کانال با هم فرق دارند (§۲). پس **مقایسه‌ی مستقیم عدد ما با عدد این مقالات معتبر نیست.**
4. **تقریباً همه‌ی مقالات SDUST کار سخت‌تری از ما انجام داده‌اند:** عبور از یک شرایط کاری به شرایط دیگر، domain generalization، open-set یا Sim2Real. پروتکل ما chronological درون هر فایل است، یعنی همان بلبرینگ و همان شرایط در train و test. پس عدد 99.1٪ ما در «شرایط آسان‌تر» به دست آمده است. صادقانه‌ترین لنگر بیرونی، **Chen et al. (MSSP 2026)** است: مدل source-only در انتقال بار 2000rpm_20N→40N به **KNO ≈ 82.8٪** رسیده است.
5. **نزدیک‌ترین کار پیشین به مدل شما ClassBD است (P2/P27، Liao et al., MSSP 2025، ESI Highly Cited).** ClassBD هم یک **فیلتر blind deconvolution عصبی با کانولوشن quadratic** (زمان) به‌اضافه‌ی فیلتر فرکانسی دارد که با loss کورتوسیس و cross-entropy **همراه طبقه‌بند** آموزش می‌بیند. این تقریباً همان ایده‌ی Neural MED + quad در مدل شماست. **اگر این مقاله ارجاع داده نشود و تفاوت‌ها توضیح داده نشوند، داور نوآوری را زیر سؤال می‌برد.** ablation ما هم نشان داد پیش‌فیلتر MED در پروتکل ما کمک نمی‌کند (حذفش +11.5 در validation). این را باید صریحاً در برابر ادعای ClassBD بحث کرد.
6. **مراجع تکراری و نامرتبط:**
   - ۴۸ فایل در واقع **۴۲ اثر یکتا** هستند: ۳۵ اثر general و ۷ مقاله‌ی SDUST (§۱).
   - P29 مقاله‌ی خود شما در طبقه‌بندی تومور مغزی است. ارجاع به آن ([39]) در مقاله‌ی بلبرینگ خودارجاعی نامرتبط دیده می‌شود. پیشنهاد: حذف.
   - P15 (مقاله‌ی دانشجویی کارشناسی) و IJHSA (ژورنال «Housing Science» برای عیب‌یابی بلبرینگ آسانسور) کیفیت لازم برای ارجاع به‌عنوان SOTA را ندارند.
7. **مراجعی که باید حتماً اضافه یا حفظ شوند:**
   - Rosa et al. 2024 (P28): نشت داده در CWRU. همراه با Vieira et al. MSSP 2026 که قبلاً پیدا کردیم.
   - Lee & Kim 2023 (P33): اثر تنظیم optimizer و ابرپارامترها. این توجیه جستجوی ابرپارامتر هم‌بودجه برای baselineهای ماست.
   - Fan et al. TPAMI 2025 (P23): منبع درست ادعای کارایی پارامتری quadratic.
   - Qttention (P22) و MDQCNN (P10): هر دو درست ارجاع شده‌اند.

---

## ۱. فهرست فایل‌ها و تکراری‌ها

| گروه | فایل‌ها | توضیح |
|---|---|---|
| متن کاملاً یکسان | P3 = P4 | Dibaj et al., Renewable Energy 2023 |
| متن کاملاً یکسان | P6 = P13 | TFPred, Control Eng. Practice 2024 |
| متن کاملاً یکسان | P16 = P21 | MCNN-LSTM, J. Intell. Manuf. |
| نسخه‌ی چاپی + preprint | P2 (MSSP) ≈ P27 (arXiv) | ClassBD |
| نسخه‌ی چاپی + ResearchGate | P12 ≈ P40 | DCA-BiGRU, Measurement 2022 |
| نسخه‌ی چاپی + preprint | P31 ≈ P39 | FaultNet, IEEE Access 2021 |
| نامرتبط با موضوع | P29 | مقاله‌ی خودتان (Brain tumor, Complexity 2025) |

**جمع:** ۴۸ فایل = ۷ SDUST + ۴۱ general. در general، ۶ فایل تکراری‌اند (P4، P13، P21، P27، P39، P40)، پس **۳۵ اثر general یکتا + ۷ SDUST = ۴۲ اثر یکتا**.

---

## ۲. بررسی دقیق مقالات SDUST (مهم‌ترین بخش برای ادعای «بهترین روی SDUST»)

| # | مقاله | زیرمجموعه‌ی SDUST | وظیفه / پروتکل | نتیجه‌ی گزارش‌شده | قابل‌مقایسه با ما؟ |
|---|---|---|---|---|---|
| S1 | Li et al., *Mech. Mach. Theory* 217 (2025) 106267, ESC-DGCN — **مرجع [22]** | بلبرینگ 6205، **۴ کلاس** (H/IR/OR/B)، بار 40 Nm، سرعت تصادفی 1000–2000 rpm، 25.6 kHz، **۲۵۰ نمونه در کلاس**، چندحسگری | طبقه‌بندی زیر سرعت متغیر؛ نحوه‌ی تقسیم گزارش نشده؛ بدون نویز | 99.63٪ (CNN پایه 92.09٪) | ❌ کلاس‌ها، شرایط و پروتکل فرق دارند |
| S2 | Guo et al., *Measurement* 242 (2025) 115951, DiffUCD — **مرجع [20]** | بلبرینگ **N205EU**، **۱۰ کلاس**، ۴ سرعت 1000/1500/2000/2500 rpm، ۱۰۰ نمونه × 2048 در کلاس | **Domain generalization**: آموزش روی ۳ سرعت و آزمون روی سرعت چهارم (T1000…T2500) | میانگین **95.86٪** (WGAN-GP 92.24٪). در PU: GAN-Train 96.58٪ ← همان عددی که مقاله‌ی شما به‌اشتباه به‌عنوان دقت PU آورده | ❌ وظیفه سخت‌تر است (سرعت دیده‌نشده) |
| S3 | Zhang et al., *Measurement* 257 (2026) 118583, CSGAN-GP-FP — **مرجع [21]** | بلبرینگ 6205، ۱۲ برچسب. **هر اندازه‌ی عیب فقط در یک سرعت/بار ثبت شده** (0.2↔1000rpm/20N، 0.4↔1500/40، 0.6↔2000/60) | SDUST فقط برای کیفیت داده‌ی تولیدی (فاصله‌ی اقلیدسی/کسینوسی) استفاده شده؛ دقت طبقه‌بندی روی XJTU-SY | ED=0.332؛ دقت 93.21/95.47/98.36٪ (XJTU-SY) | ❌ هشدار: اندازه‌ی عیب با شرایط کاری کاملاً **confounded** است؛ مدل می‌تواند به‌جای عیب، سرعت را یاد بگیرد |
| S4 | Chen et al., *MSSP* 242 (2026) 113667, MFCL (open-set UDA) | **همان SDUST رسمی** (github.com/JRWang-SDUST/SDUST-Dataset): 6205 DE، **۱۰ کلاس** (NC, IF/OF/RF × 0.2/0.4/0.6)، 2000rpm با 20N و 40N | **انتقال بار** 20N↔40N، open-set (کلاس‌های ناشناخته در هدف)، چند اجرا با ±std | MFCL: OS 92.88٪، HOS 91.83٪. **Source-only (S1): OS 74.53٪، KNO 82.81٪**؛ DANN KNO 98.31٪ | ⚠️ نزدیک‌ترین زیرمجموعه به داده‌ی ما؛ **بهترین لنگر صادقانه** برای ارزیابی cross-load |
| S5 | Wang et al., *Meas. Sci. Technol.* 2026 (accepted), DCLATN + digital twin | SKF6205، ۴ وضعیت، کانال X/Y، سرعت‌های 1500→2000/2500/3000 rpm | **UDA** (داده‌ی بدون‌برچسب هدف در آموزش) + داده‌ی شبیه‌سازی digital twin | میانگین 98.93٪ (Res18-ADA 95.51٪) | ❌ وظیفه و زیرمجموعه فرق دارند. ناسازگاری داخلی: متن پارامتر را 0.26M/0.341 MFLOPs می‌گوید و جدول ۱۰ آن را 0.19M/0.215 |
| S6 | Xu et al., *IEEE TIM* 74 (2025) 3560520, Sim2Real CoGAN | **۳ وضعیت** (N/OR/IR)، 500/1000/1500 rpm، 20/40 N، فقط عیب 0.4 mm، ۲۰۰ نمونه × 1280 | **Single-domain generalization** (یک شرایط → چند شرایط دیده‌نشده) + داده‌ی شبیه‌سازی 4-DOF | جدول IX (عدد به‌صورت تصویر است و قابل استخراج نبود). روش پیشنهادی بهترین است. PU: CNN 75.86٪ ← روش 98.80٪ | ❌ |
| S7 | Li et al., *Int. J. Housing Sci. Appl.* 46(4) 2025 (بلبرینگ آسانسور) | توصیف کلی SDUST (6205، 25.6 kHz)، ویژگی‌های دستی | BPNN + بهینه‌سازی فراابتکاری؛ تقسیم داده گزارش نشده | 96.9٪ (SVM 92.59٪) | ❌ ژورنال نامرتبط و کیفیت پایین؛ **ارجاع ندهید** |
| (P10) | Ji et al., *Nonlinear Dyn.* 113 (2025), MDQCNN — **مرجع [28]** | SDUST **۱۰ کلاس**، بار 0N، **دو موقعیت A1/A2 × سه کانال X/Y/Z**، 25.6 kHz | نویز سفید با SNR −10…−2 dB (تعریف SNR نسبی)؛ نحوه‌ی تقسیم گزارش نشده | فقط نمودار (شکل 17). MDQCNN و DCA-BiGRU بالاترین | ⚠️ تنها مقاله‌ی SDUST با ارزیابی نویز؛ عددی برای مقایسه ارائه نمی‌دهد |

### نتیجه‌گیری برای مقاله‌ی شما

- **ادعای قابل دفاع:**
  > «تحت پروتکل chronological درون‌رکوردی، بدون نشت پنجره‌ها، phyq_nomed در همه‌ی سطوح نویز سفید از baselineهای بازپیاده‌سازی‌شده بهتر است (3 seed، McNemar).»
- **ادعای غیرقابل دفاع:**
  > «از روش‌های 2025–2026 روی SDUST بهتر است.»
  
  آن روش‌ها روی زیرمجموعه‌ها و وظایف دیگری گزارش شده‌اند.
- **کارهای لازم در متن مقاله:**
  1. **زیرمجموعه‌ی دقیق SDUST را مشخص کنید:** بلبرینگ، سرعت/بار، کانال‌ها و تعداد کلاس. در اجراهای شما **C=6 کلاس** شناسایی شد، ولی مقاله «K = 10» نوشته است. باید روشن شود کدام ۶ کلاس در پوشه‌ی داده‌ی شما هست.
  2. **لینک مخزن رسمی را بیاورید** (JRWang-SDUST) و بنویسید که پروتکل ما درون‌رکوردی است.
  3. **توصیف سکوی آزمون در §4.6.3 را بازبینی کنید.** متن فعلی می‌گوید «Zhenjiang Tianyu… planetary gearbox». مقالات S1، S4 و P10 سکو را «موتور + کوپلینگ + gearbox + ترمز پودر مغناطیسی» توصیف می‌کنند. توصیف باید با منبع رسمی داده بخواند.
  4. **ارجاع [21] را درست به کار ببرید.** این مقاله را نباید به‌عنوان مقایسه‌ی دقت SDUST آورد؛ دقت آن روی XJTU-SY است.
- **آزمایش اختیاری پیشنهادی (لنگر بیرونی واقعی):**
  - **SDUST cross-load** با همان تعریف Chen et al.: آموزش روی 2000rpm_20N و آزمون روی 2000rpm_40N، closed-set، همه‌ی مدل‌ها.
  - این آزمایش فقط در صورتی ممکن است که این دو شرایط در داده‌ی شما وجود داشته باشد.
  - **معیار مقایسه:** source-only CNN در آن مقاله KNO ≈ 82.8٪ گرفته است. این تنها مقایسه‌ی بیرونی تقریباً منصفانه برای SDUST خواهد بود.
  - **هزینه:** برای ۷ مدل × ۳ seed احتمالاً ۱ تا ۲ روز روی GPU شما.

---

## ۳. ممیزی فهرست مراجع مقاله در برابر PDFها

راهنمای وضعیت:
- ✅ درست
- ⚠️ اثر درست است ولی مشخصات کتاب‌شناختی غلط است
- ❌ ارجاع به ادعایی که مقاله پشتیبانی نمی‌کند، یا مرجع ساختگی

| مرجع مقاله (آن‌طور که نوشته شده) | اثر واقعی (فایل) | وضعیت و اصلاح |
|---|---|---|
| [1] S. Hosseini, «DL for fault diagnosis: A review», IEEE Access 2020 | P7: **S.A. Hosseini, A. Ahmedi, F. Blaabjerg, S. Peyghami, «Exploring wind farm reliability…», RSER 2024**, doi:10.1016/j.rser.2024.115227 | ⚠️ عنوان، ژورنال و سال غلط است. (عنوان نوشته‌شده شبیه P30، یعنی Zhang et al. IEEE Access 2020، است.) |
| [2] C. Lu, «Physics-informed feature weighting», IEEE TIE 2021 | P1: **H. Lu, V.P. Nemani, … A.T. Zimmerman, MSSP 191 (2023) 110171** | ⚠️ نام، ژورنال و سال غلط است. ❌ در متن برای «بلبرینگ ضعیف‌ترین حلقه‌ی گیربکس است» ارجاع شده که P1 پشتیبانی نمی‌کند. برای این ادعا از P11 (Guo et al., NREL, RSER 2020)، P35 (de Azevedo, RSER 2016) یا P7 استفاده کنید. |
| [4] A. Dibaj, «Application of DL in fault diagnosis», Measurement 2020 | P3/P4: **A. Dibaj, Z. Gao, A.R. Nejad, «Fault detection of offshore wind turbine drivetrains…», Renewable Energy 203 (2023) 161–176** | ⚠️ عنوان، ژورنال و سال غلط است. |
| [16] X. Li, «PINNs for machinery fault diagnosis», JMS 2023 | P41: **C. He, H. Shi, J. Si, J. Li, «Physics-informed interpretable wavelet weight initialization… (EWSNet)», J. Manuf. Syst. 2023**, doi:10.1016/j.jmsy.2023.08.014 | ⚠️ نویسندگان و عنوان غلط است. |
| [17] X. Liao, «Attention-based DL for fault diagnosis», IEEE TII 2022 | به احتمال زیاد P2: **J.-X. Liao, C. He, J. Li, J. Sun, S. Zhang, X. Zhang, «Classifier-guided neural blind deconvolution…», MSSP 222 (2025) 111750** (متن: «unfolding physical constraints into neural layers [15,17]») | ⚠️ همه‌ی مشخصات غلط است. این مرجع کلیدی است (§۴). |
| [26] Y. Fan, J. Xu, X. Wu, «Quadratic NNs for solving PDEs», Neurocomputing 2019 | P23: **F.-L. Fan, H.-C. Dong, Z. Wu, L. Ruan, T. Zeng, Y. Cui, J.-X. Liao, «One Neuron Saved Is One Neuron Earned…», IEEE TPAMI 47(11) (2025) 9702–9717** | ⚠️ اثر کاملاً متفاوتی نوشته شده است. نکته: خود این مقاله می‌گوید **«بهره‌ی کارایی به وظیفه بستگی دارد»**. ادعای «linearity bottleneck» در مقاله‌ی شما باید ملایم‌تر شود. |
| [27] G. Xu, «Polynomial NNs for fault diagnosis», Pattern Recognition 2019 | — (PDF ندارد) | ❌ احتمالاً ساختگی است. حذف کنید یا منبع واقعی بیاورید. |
| [28] Ji et al., MDQCNN, Nonlinear Dyn. 113 (2025) | P10 | ✅ مشخصات درست است. ❌ ولی عدد آن در جدول ۴ (PU) غلط است (§۰ بند ۲). |
| [29] J. Yu, «Residual life… Bi-LSTM & attention», IEEE TIE 2021 | P9: **Y. Yu, C. Hu, X. Si, J. Zheng, J. Zhang, «Averaged Bi-LSTM networks for RUL prognostics…», Neurocomputing 402 (2020) 134–147** | ⚠️ عنوان، ژورنال و سال غلط است. ❌ متن آن را «Bi-GRU» توصیف می‌کند در حالی که مقاله Bi-LSTM است. |
| [30] J. Yin, «Gearbox… GRU temporal attention», Shock & Vib. 2022 | احتمالاً P20: **N. Ghanbari, Y. Riyazi, F.A. Shirazi, A. Kalhor, «Noise-robust gearbox fault detection», JTAVA 10(1) (2024) 54–66** | ⚠️ نامطمئن است. با نسخه‌ی word/bib خودتان تطبیق دهید. |
| [31] R. Yang, «coordinate attention and AdamP», IEEE Access **9 (2023)** | — | ❌ جلد ۹ IEEE Access مربوط به سال 2021 است، پس مرجع ناسازگار و احتمالاً ساختگی است. برای Coordinate Attention باید **Hou, Zhou, Feng, CVPR 2021** ارجاع شود. |
| [33] J. Zhu, N. Chen, C. Shen, Measurement 150 (2020) 107062 | در جدول ۷ با برچسب «MC-CNN-LSTM [33]» آمده. اثر واقعی MC-CNN-LSTM، P16 است: **X. Chen, B. Zhang, D. Gao, J. Intell. Manuf. (2021)**, doi:10.1007/s10845-020-01600-2 | ⚠️ مرجع با مدل جدول نمی‌خواند. |
| [36] Y. Liao, R. Huang, J. Li, «GANs for bearing…», Sensors 20(1) 2020 | متن و جدول ۷ آن را «QCNN / Qttention» معرفی می‌کنند ← P22 = **همان [42]** | ❌ ارجاع به اثر اشتباه. به [42] تغییر دهید. |
| [37] Y. Liao, «Blind deconvolution for fault detection», MSSP 2018 | احتمالاً P14: **J.-X. Liao, H.-C. Dong, L. Luo, J. Sun, S. Zhang, «Multi-task neural network blind deconvolution…», Meas. Sci. Technol. 34 (2023) 075017** | ⚠️ همه‌ی مشخصات غلط است. |
| [39] A. Naderi et al., Brain tumor, Complexity 2025 | P29 | ✅ مشخصات درست است، ولی **از نظر موضوع نامرتبط است**. حذف را توصیه می‌کنم. |
| [42] Liao et al., Qttention, IEEE TIM 72 (2023) | P22 | ✅ درست است. |
| [20]، [21]، [22] (مقالات SDUST) | S2، S3، S1 | ✅ مشخصات درست است. ❌ ولی [20] در جدول ۴ با عدد GAN-Train آمده و [21] برای دقت SDUST مناسب نیست. |
| [47] X. Ji, «MDQCNN…», IJMTM 188 (2025) | — | ❌ ساختگی است (قبلاً هم گزارش شده بود). تکرار نادرست [28] است. |

> **توصیه‌ی عملی:** فهرست مراجع را با **Zotero/Mendeley از روی DOI** بازسازی کنید و هیچ مرجعی را دستی یا با کمک ابزار تولید متن ننویسید. DOI همه‌ی PDFها در جدول §۵ آمده است.

---

## ۴. مقالاتی که روی روش‌شناسی و ادعاهای مقاله اثر مستقیم دارند

### 4-الف. ClassBD (P2/P27): نزدیک‌ترین کار پیشین (باید حتماً بحث شود)
- **ساختار:** فیلتر BD عصبی با **کانولوشن quadratic در حوزه‌ی زمان** + فیلتر خطی فرکانسی، بعد طبقه‌بند WDCNN. آموزش end-to-end با loss = کورتوسیس + نرم l2/l4 + cross-entropy.
- **شباهت با مدل شما:** Neural MED (پیش‌فیلتر deconvolution با هدف کورتوسیس) + لایه‌های quadratic + طبقه‌بند. **همان خانواده‌ی ایده است.**
- **پروتکل PU آن‌ها:** ۱۴ کلاس که هر کدام **عمدتاً یک بلبرینگ** است (K01–K06 سالم، بقیه تک‌بلبرینگ). test = بخش بیستم همان رکوردها؛ train/val = ۱۹ بخش اول به‌صورت تصادفی.
  - یعنی **بلبرینگ test در آموزش دیده شده** (پروتکل same-bearing). کلاس عملاً هویت بلبرینگ است.
  - پس دقت بالای 94٪ آن‌ها با PU ما، که بلبرینگ‌های test جدا از آموزش‌اند (حدود 58–62٪)، **قابل مقایسه نیست**.
- **نکته‌ی مفید:** در جدول آن‌ها DRSN روی PU در −4 dB فقط حدود 52٪ و روی JNU در −10 dB حدود 34٪ گرفته است. ما در پروتکل خودمان با تنظیم هم‌بودجه، DRSN را baseline قوی یافتیم. این **شاهد خوبی برای ضعیف تنظیم‌شدن baselineها در ادبیات** است و جستجوی ابرپارامتر هم‌بودجه‌ی ما را توجیه می‌کند.
- **پیشنهاد متن برای مقاله (Related Work / Discussion):**
  > «Our neural MED prefilter belongs to the same family as ClassBD [Liao et al., MSSP 2025], which jointly trains a quadratic time-domain BD filter with a classifier. In our leakage-free protocol, however, the prefilter did not improve validation performance (+11.5 points when removed on CWRU validation), so the final model disables it. We attribute this to … [overfitting of the kurtosis objective to training-condition impulses / interaction with instance norm]. Unlike ClassBD's same-bearing PU protocol, we evaluate on held-out bearings.»

### 4-ب. منابع درست برای ادعاهای quadratic
- **P23 (Fan et al., TPAMI 2025):** ادعای کارایی پارامتری را با نظریه و آزمایش ثابت می‌کند، ولی **صریحاً می‌گوید بهره به وظیفه بستگی دارد**. با یافته‌ی ما هم‌خوان است: quad آموزش را پایدار می‌کند و روی validation کمک می‌کند، ولی MACها را حدود ۲ برابر و آسیب‌پذیری در برابر نویز impulsive را بیشتر می‌کند.
- **P22 (Qttention, TIM 2023):** منبع baseline `qcnn` ما. درست ارجاع شده ([42]).
- **P10 (MDQCNN):** ادعایش روی CWRU در −10 dB برابر 97.20٪ است و روی Torino 98.81٪. SNR نسبی و تقسیم داده گزارش نشده است. عدد جدول ۷ مقاله‌ی شما برای MDQCNN (96.25 در −10 dB) با هیچ‌کدام نمی‌خواند.
- **P14 (MNNBD, MST 2023):** پیش‌درآمد ClassBD است. اگر MED را در مقاله توصیف می‌کنید، ارجاع به آن هم لازم است.

### 4-ج. پروتکل و نشت داده (برای بخش Experimental Protocol)
- **P28 (Rosa, Braga, Silva, arXiv:2407.14625, 2024):** تقسیم رایج CWRU نشت دارد. به Hendriks et al. 2022 و Abburi et al. 2023 ارجاع می‌دهد. حتی تقسیم پیشنهادی آن‌ها هم کاملاً بدون نشت نیست. فرمول‌بندی multi-label و AUROC پیشنهاد می‌کند (macro AUROC ≈ 0.91).
  - **این مقاله را به همراه Vieira et al. MSSP 2026 (همان گروه) ارجاع دهید.**
  - توصیه: Hendriks 2022 و Abburi 2023 را هم اضافه کنید.
- **P12/P40 (DCA-BiGRU):** پنجره 1024 با گام 400 (همپوشانی حدود ۶۱٪). نحوه‌ی تقسیم گزارش نشده. نمونه‌ی رایج پروتکل‌های نشت‌دار است. MDQCNN هم از آن به‌عنوان رقیب استفاده کرده است.
- **P19 (Shenfield & Howarth, Sensors 2020):** ۵۰٪ هر شرایط به test رفته. پنجره‌های آموزشی با گام 64 همپوشان‌اند و test غیرهمپوشان است. نمونه‌ی بهتری از تفکیک است.
- **P41 (EWSNet):** تقسیم ۷:۳ و ارزیابی با نویز **گاوسی، لاپلاس و صورتی**. منبع خوبی برای توجیه ارزیابی نویز غیرگاوسی (impulsive) ماست.

### 4-د. عادلانه بودن مقایسه
- **P33 (Lee & Kim, IEEE Access 2023):** نشان می‌دهد optimizer و ابرپارامترها نتیجه‌ی مدل‌های CNN را روی CWRU و MFPT به‌طور قابل‌توجهی تغییر می‌دهند. ارزیابی نویز آن‌ها میانگین SNR در بازه‌ی −4 تا 4 dB است.
  - **این بهترین ارجاع برای توجیه جستجوی ابرپارامتر هم‌بودجه است** (`hp_search.yaml` و `select_hparams.py`).
  - مقاله‌ی شما از AdamP استفاده کرده است. این مقاله هم در کنار [40] (AdamP، ICLR 2021) مفید است.

---

## ۵. جدول کامل ۳۵ اثر یکتای general (+ ارجاع به SDUST): مشخصات و توصیه

| فایل | اثر | DOI / شناسه | ارتباط | توصیه |
|---|---|---|---|---|
| P1 | Lu et al., Physics-informed feature weighting, MSSP 191 (2023) | 10.1016/j.ymssp.2023.110171 | physics-informed، تغییر توزیع سرعت | ارجاع (اصلاح [2]) |
| P2 / P27 | Liao et al., ClassBD, MSSP 222 (2025) 111750 | 10.1016/j.ymssp.2024.111750 | **نزدیک‌ترین کار پیشین** | **حتماً** + بحث (§4-الف) |
| P3 / P4 | Dibaj, Gao, Nejad, Renewable Energy 203 (2023) | 10.1016/j.renene.2022.12.049 | مقدمه (توربین بادی) | ارجاع (اصلاح [4]) |
| P5 | Yu et al., Non-temporal NN for WT gearbox degradation, Renew. Energy 243 (2025) | 10.1016/j.renene.2025.122438 | پیش‌بینی روند تخریب | اختیاری (مقدمه) |
| P6 / P13 | Chen et al., TFPred, Control Eng. Pract. 146 (2024) | 10.1016/j.conengprac.2024.105900 | یادگیری با برچسب کم | اختیاری |
| P7 | Hosseini et al., Wind farm reliability, RSER (2024) | 10.1016/j.rser.2024.115227 | مقدمه | ارجاع (اصلاح [1]) |
| P8 | Guo et al., TFNet for WT, JESTCH (2025) | 10.1016/j.jestch.2025.102056 | CWRU و XJTU، سرعت متغیر | اختیاری |
| P9 | Yu et al., Averaged Bi-LSTM RUL, Neurocomputing 402 (2020) | 10.1016/j.neucom.2020.03.041 | RNN | ارجاع (اصلاح [29]) |
| P10 | Ji et al., MDQCNN, Nonlinear Dyn. 113 (2025) | 10.1007/s11071-025-10918-6 | quadratic، نویز، SDUST | ✅ ارجاع؛ عدد جدول ۴ حذف |
| P11 | Guo et al. (NREL), Bearing axial cracking reliability, RSER 127 (2020) | 10.1016/j.rser.2020.109888 | **پشتوانه‌ی «بلبرینگ ضعیف‌ترین حلقه»** | ارجاع (جایگزین [2] در آن جمله) |
| P12 / P40 | Zhang et al., DCA-BiGRU small samples, Measurement 187 (2022) | 10.1016/j.measurement.2021.110242 | پنجره‌ی همپوشان | اختیاری |
| P14 | Liao et al., MNNBD, Meas. Sci. Technol. 34 (2023) 075017 | 10.1088/1361-6501/accbdb | BD عصبی | ارجاع (اصلاح [37]) |
| P15 | Abhijit et al. (SRMIST), RUL CNN-LSTM | — | دانشجویی، RUL | **ارجاع ندهید** |
| P16 / P21 | Chen, Zhang, Gao, MCNN-LSTM, J. Intell. Manuf. (2021) | 10.1007/s10845-020-01600-2 | baseline جدول ۷ | ارجاع (اصلاح [33]) |
| P17 | Pandiyan et al., LSTM encoder–decoder self-lubricating bearings, Friction 11 (2023) | 10.1007/s40544-021-0584-3 | پیش‌بینی خرابی | اختیاری / حذف |
| P18 | Chai et al., PSO-SVM + CDET, JEAS 72 (2025) | 10.1186/s44147-025-00683-z | ML کلاسیک | اختیاری |
| P19 | Shenfield & Howarth, RNN-WDCNN, Sensors 20 (2020) 5112 | 10.3390/s20185112 | نویز، WDCNN | ارجاع (§4-ج) |
| P20 | Ghanbari et al., Noise-robust gearbox, JTAVA 10(1) (2024) | 10.22064/tava.2024.2029883.1242 | گیربکس، نویز | اختیاری |
| P22 | Liao et al., Qttention, IEEE TIM 72 (2023) | — | baseline `qcnn` | ✅ [42] |
| P23 | Fan et al., One Neuron Saved…, IEEE TPAMI 47(11) (2025) | 10.1109/TPAMI.2025.3588894 | نظریه‌ی quadratic | ارجاع (اصلاح [26]) |
| P24 | Liao et al., BearingPGA-Net (arXiv:2307.16363) | — | مدل سبک، FPGA | برای بحث پیچیدگی؛ نسخه‌ی نهایی چاپی را چک کنید |
| P25 | Mirzaeibonehkhater et al., Transformer TDA (arXiv:2309.11006) | — | Transformer روی CWRU | اختیاری (preprint) |
| P26 | Sun et al., GCOS-1DCNN-MLP-Cross-Attention, Machines 13 (2025) 819 | 10.3390/machines13090819 | cross-attention، نویز | اختیاری |
| P28 | Rosa, Braga, Silva, CWRU multi-label benchmark (arXiv:2407.14625) | — | **نشت CWRU** | **حتماً** |
| P29 | Naderi et al., Brain tumor, Complexity (2025) | 10.1155/cplx/1644859 | نامرتبط | حذف [39] |
| P30 | Zhang et al., DL bearing diagnostics review, IEEE Access 8 (2020) | 10.1109/ACCESS.2020.2972859 | مرور | ارجاع (مقدمه) |
| P31 / P39 | Magar et al., FaultNet, IEEE Access 9 (2021) | 10.1109/ACCESS.2021.3056944 | CWRU و PU | اختیاری |
| P32 | Chen et al., Deep transfer learning review, IEEE TIM (2023) | 10.1109/TIM.2023.3244237 | مرور TL | ارجاع (domain shift) |
| P33 | Lee & Kim, Optimizers & HP tuning, IEEE Access 11 (2023) | 10.1109/ACCESS.2023.3281910 | **عادلانه بودن مقایسه** | **حتماً** |
| P34 | Wang et al., Hybrid physics-based & data-driven models, JMS 63 (2022) | 10.1016/j.jmsy.2022.04.004 | physics-informed (مقدمه) | ارجاع |
| P35 | de Azevedo et al., WT bearing CM review, RSER (2016) | 10.1016/j.rser.2015.11.032 | مقدمه | ارجاع |
| P36 | Zhang et al., Few-shot siamese WDCNN, IEEE Access 7 (2019) | 10.1109/ACCESS.2019.2934233 | WDCNN، داده‌ی محدود | اختیاری |
| P37 | Sun et al., Rolling bearing defect detection review, Appl. Sci. 15 (2025) 1001 | 10.3390/app15021001 | مرور | اختیاری |
| P38 | Neupane & Seok, CWRU DL review, IEEE Access 8 (2020) | 10.1109/ACCESS.2020.2990528 | مرور CWRU | ارجاع |
| P41 | He et al., EWSNet, J. Manuf. Syst. (2023) | 10.1016/j.jmsy.2023.08.014 | نویز گاوسی، لاپلاس و صورتی | ارجاع (اصلاح [16]) |
| S1–S7 | (جدول §۲) | | SDUST | S1، S2 و S4 ارجاع؛ S3 فقط در جای درست؛ S7 نه |

---

## ۶. فهرست کارهای اصلاحی مقاله (حاصل این بررسی)

1. **بازسازی کامل فهرست مراجع از روی DOI** (جدول §۳ و §۵). موارد ساختگی ([27]، [31]، [47]) حذف یا جایگزین شوند.
2. **حذف جدول ۴ (PU)** و جایگزینی آن با جدول نتایج بازپیاده‌سازی و پروتکل یکسان (که داریم).
3. **جدول ۷** (baselineهای کپی‌شده) حذف شود و نتایج `main_tuned` جای آن را بگیرد. QCNN به [42] و MC-CNN-LSTM به P16 ارجاع شود.
4. **بند ClassBD در Related Work** اضافه شود، همراه با بحث حذف MED (§4-الف).
5. **بخش Protocol:** ارجاع به Rosa 2024، Vieira 2026 و Lee & Kim 2023 اضافه شود، و نیز توضیح «SDUST benchmark واحد نیست» همراه با مشخصات دقیق زیرمجموعه‌ی ما.
6. **ادعای «linearity bottleneck»** با استناد درست به Fan et al. 2025 ملایم شود، با این جمله که بهره به وظیفه بستگی دارد.
7. **حذف خودارجاعی نامرتبط [39]** و مراجع کم‌کیفیت (P15، S7).
8. **(اختیاری) آزمایش SDUST cross-load** (20N→40N) برای داشتن لنگر بیرونی (§۲).
