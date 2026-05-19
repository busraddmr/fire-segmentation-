# BIM320 — Yangın Görüntülerinde Alev Segmentasyonu

> **Üniversite:** İstanbul Sabahattin Zaim Üniversitesi  
> **Ders:** BIM 320 Görüntü İşleme | **Danışman:** Dr. Hasibe Büşra AYTEKİN  
> **Öğrenciler:** Büşra Demir · Elif Bilge Güleç · Ümmü Habibe Yüce  
> **Teslim:** 19.05.2026 Salı 23:59 | **Sunum:** 20–21 Mayıs 2026

Klasik (deterministik) görüntü işleme yöntemleriyle yangın alevi segmentasyonu. ML/DL kullanılmamıştır.

---

## Sonuçlar

| Metrik | Ortalama | Medyan |
|---|---|---|
| Precision | %85.5 | %88.9 |
| Recall | %87.8 | %90.8 |
| Dice (F1) | %85.5 | %88.0 |
| IoU | %75.6 | %78.5 |

- Maks. IoU: **%96.5** — IoU ≥ %80: **551 / 1278** görsel

---

## Veri Seti

1278 görsel (640×640 .jpg) + binary maske (.png). Üç Roboflow kaynağından kalite filtrelenerek derlendi.

---

## Pipeline (14 Adım)

1. **Ön İşleme:** GaussianBlur → BilateralFilter → CLAHE
2. **Renk Maskeleri (5 kanal):** HSV, YCrCb, RGB, LAB, Parlaklık → Oylama ≥ 3/5
3. **Morfoloji:** Morph Open → Morph Close → Delik Doldurma → Dilatasyon → Kontur Filtresi (≥ 200 px²)

---

## Çalıştırma

```bash
pip install opencv-python numpy matplotlib scipy

# Ana pipeline + metrikler
python src/segmentation.py

# Rapor görselleri (isteğe bağlı)
python src/analysis_report.py
```

Çıktılar `outputs/` klasörüne kaydedilir.

---

## Klasör Yapısı

```
├── dataset/
│   ├── images/    (1278 × .jpg)
│   └── masks/     (1278 × .png)
├── src/
│   ├── segmentation.py
│   └── analysis_report.py
└── outputs/
```
