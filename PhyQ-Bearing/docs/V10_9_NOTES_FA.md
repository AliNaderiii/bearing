# v10.9 — اجرای کامل اقدامات بازبینی دوم (Advisor_Review_v10_8)

## ۱. چه چیزهایی از خروجی‌های موجود محاسبه و وارد مقاله شد (بدون اجرای جدید)
| مورد | اسکریپت | نتیجه |
|---|---|---|
| جدول ± s.d. (Table A.5, `tab:std`) | `scripts/std_table.py` | همه مدل‌ها × ۶ وظیفه × ۵ شرط |
| per-fault (BPFO/BPFI/BSF) (Table A.6) | `scripts/interp_fault_types.py` | quad ≈ envelope تحلیلگر روی BPFI؛ BSF برای همه ~0 dB |
| تجزیه epistemic/aleatoric (Table A.7) | `scripts/uncertainty_decomp.py` | سهم epistemic داخل توزیع ≤۱۳٪؛ نویز ندیده تا ۳۷–۴۵٪؛ PU: ۱۱–۳۶٪ ولی AUROC خطا ۰٫۴۸–۰٫۷۶ |
| غیریکنواختی PU (noisy > clean) | metrics.json فولدها | علت: fold2 زیر شانس (11.7/27.0٪)؛ Wilcoxon PhyQ p=0.19، WDCNN 0.008، ClassBD 0.027 |
| تعداد یاتاقان آموزشی در هر فولد | metrics.json (`data.train_groups`) | 9–11 آموزش / 3–5 تست |

متن‌های جدید مقاله: Remark «Strided implementation» (aliasing جمله 2ωc)، خانواده bounded-influence (Antoni & Borghesani 2019، Smith 2019)، ReLinear (Fan 2025) برای رشد 2^k، QCNNهای جدید (Tang 2024، Luo 2025)، تمایز با Borghesani 2023، Hu 2026 (غیرقابل‌مقایسه)، معنی نام «PhyQ»، اختلاف پارامتر 0.79/1.40M در cross-load و Limitations، highlight اصلاح شد.

## ۲. کد جدید (تست‌ها: 103 passed)
- `phyq/models/phyq_paper.py`: گزینه `model.width_mult` (پیش‌فرض 1.0 = مدل اصلی، بیت‌به‌بیت). با `quad=false, width_mult=1.65` → 1.400M پارامتر (= PhyQ).
- `phyq/data/splits.py`: `group_kfold` با `source/target` → bearing-disjoint + تغییر سرعت.
- `configs/datasets/pu.yaml`: واریانت‌های `sp_N15toN09_f0..4` و کنترل `sp_N15toN15_f0..4`.

## ۳. اجرای GPU (به ترتیب اولویت)
| فایل | اجرا | زمان تقریبی (GTX 1650) |
|---|---|---|
| `capacity_twin.yaml` | 36 (cross-load) | ~1–1.5 h |
| `capacity_twin_main.yaml` | 21 (SDUST اول) | ~4–6 h |
| `pu_speed_shift.yaml` | 80 (+20 با wide) | ~6–10 h |
| `capacity_twin_pu.yaml` | 13 | ~2–3 h |

دستورها (هر کدام یک خط):
```
python scripts\run_suite.py configs\experiments\capacity_twin.yaml
python scripts\run_suite.py configs\experiments\capacity_twin_main.yaml
python scripts\run_suite.py configs\experiments\pu_speed_shift.yaml --models phyq_nomed phyq_nomed_noquad classbd wdcnn
python scripts\run_suite.py configs\experiments\capacity_twin_pu.yaml
```
fp32 (پیش‌فرض phyq_paper). لپ‌تاپ نباید sleep شود. بعد از هر مرحله: `git add -f results/...` و push.
