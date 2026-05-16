# BIM320 — Yangın Görüntülerinde Alev Segmentasyonu

> **Üniversite:** İstanbul Sabahattin Zaim Üniversitesi — Mühendislik ve Doğa Bilimleri Fakültesi
> **Ders:** BIM 320 Görüntü İşleme | **Danışman:** Dr. Hasibe Büşra AYTEKİN
> **Öğrenciler:** Büşra Demir · Elif Bilge Güleç · Ümmü Habibe Yüce
> **Son Teslim:** 19.05.2026 Salı 23:59 | **Sunum:** 20–21 Mayıs 2026

Klasik (deterministik) görüntü işleme yöntemleriyle yangın alevi segmentasyonu.
Makine öğrenmesi, derin öğrenme veya önceden eğitilmiş model **kullanılmamıştır.**

---

## Sonuçlar

| Metrik | Değer |
|---|---|
| Ortalama IoU | **%75.6** |
| Medyan IoU | %78.5 |
| Maksimum IoU | %96.5 |
| Ortalama Dice | **%85.5** |
| IoU ≥ %80 olan görsel | 551 / 1278 |
| Toplam test görseli | **1278** |

---

## Proje Gereksinimleri

### Uygulanan
- Klasik / deterministik CV pipeline (OpenCV + NumPy + SciPy)
- Ground Truth maskesiyle nicel değerlendirme (IoU + Dice — iki ayrı metrik)
- İki farklı parametre karşılaştırması (VOTE_THR = 2 vs 3)
- Çalışır kaynak kod + açıklayıcı README

### Kullanılmayan (Yasak) Yöntemler
- CNN, SVM, Random Forest, herhangi bir ML/DL modeli
- LLM veya yapay zeka API tabanlı görsel analiz
- Web tabanlı hazır araçlar

---

## Veri Seti

### Kaynak ve Kompozisyon

Dataset üç farklı Roboflow kaynağından kalite filtrelenerek oluşturulmuştur:

| Kaynak | Ham Görsel | Elenen | Final'e Giren |
|---|---|---|---|
| Fire Detection Veri Seti | 441 | 120 (boş / düşük kalite) | **321** |
| Fire Segmentation Dataset | 798 | 45 (hatalı maske) | **753** |
| Fire and Smoke Dataset | 204 | — | **204** |
| **Toplam** | **1.443** | **165** | **1.278** |

> **Filtre kararları:**
> - Alev içermeyen görseller, boş maskeler ve çok uzakta/çok küçük piksel kaplayan alev içeren düşük kaliteli görseller elenmiştir.

### Klasör Yapısı

```
dataset/
├── images/    (1278 adet .jpg — 640×640 piksel)
└── masks/     (1278 adet .png — binary, 0/255)
```

### Veri Seti Özellikleri
- Tüm görseller **640×640** piksel
- Maskeler binary (0 = arka plan, 255 = alev)
- Farklı aydınlatma koşulları, gece/gündüz, duman içeren senaryolar

---

## Algoritma (Pipeline)

Pipeline tamamen deterministik olup 14 adımdan oluşmaktadır:

```
Girdi Görüntüsü
      │
      ▼
┌─────────────────────────────┐
│       ÖN İŞLEME             │
│  1. GaussianBlur (5×5)      │
│  2. BilateralFilter (d=9)   │
│  3. CLAHE (LAB uzayı)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   5 KANAL RENK MASKESİ      │
│  4. HSV Maskesi             │
│  5. YCrCb Maskesi           │
│  6. RGB Oran Maskesi        │
│  7. LAB Maskesi             │
│  8. Parlaklık Maskesi       │
│         ↓                   │
│  OYLAMA: ≥ 3/5 kanal        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   MORFOLOJİ & TEMİZLEME     │
│  9.  Morph Open  (1×, 9×9)  │
│  10. Morph Close (3×, 9×9)  │
│  11. Delik Doldurma         │
│  12. Dilatasyon (5×5, 1×)   │
│  13. Kontur Filtresi ≥200px²│
└─────────────┬───────────────┘
              │
              ▼
       Final Tahmin Maskesi
```

### Ön İşleme Parametreleri

| Adım | Yöntem | Parametre |
|---|---|---|
| Gürültü azaltma | GaussianBlur | kernel=5×5, σ=1.0 |
| Kenar koruyucu düzeltme | BilateralFilter | d=9, σColor=80, σSpace=80 |
| Kontrast artırma | CLAHE (LAB L kanalı) | clipLimit=2.5, tileGrid=8×8 |

### 5 Kanal Renk Maskesi (Oylama ≥ 3/5)

| # | Renk Uzayı | Eşik Koşulu |
|---|---|---|
| 1 | HSV | H∈[0,35]∪[145,180], S≥100, V≥120 |
| 2 | YCrCb | Cr>150, Cb<140, Y>50 |
| 3 | RGB | R>150, R>G≥B, R−B>40 |
| 4 | LAB | L>80, A>135, B>130 |
| 5 | Parlaklık | R>200, G>150, B<180, R−B>30 |

> Her piksel için 5 kanalın en az **3'ünde** pozitif olan pikseller alev adayı sayılır.
> Bu "majority voting" yaklaşımı tek kanala göre çok daha gürbüz sonuç verir.

### Morfolojik İşleme

| Adım | İşlem | Parametre |
|---|---|---|
| Gürültü temizleme | Morph Open | 1×, 9×9 elips kernel |
| Boşluk kapatma | Morph Close | 3×, 9×9 elips kernel |
| İç boşluk doldurma | binary_fill_holes | — |
| Alan genişletme | Dilatasyon | 1×, 5×5 elips kernel |
| Küçük bölge eleme | Kontur filtresi | min alan ≥ 200 px² |

---

## Parametre Analizi

Proje gereksinimlerine uygun olarak iki farklı oylama eşiği karşılaştırılmıştır:

| VOTE_THR | Açıklama | Avantaj | Dezavantaj |
|---|---|---|---|
| **2 / 5** | Daha gevşek eşik | Daha fazla alan kapsanır (yüksek Recall) | Daha fazla yanlış pozitif |
| **3 / 5** | Seçilen optimal değer | Denge: precision-recall | — |

> Karşılaştırma görseli: `outputs/parameter_analysis.png`

---

## Çalıştırma

### Gereksinimler

```bash
pip install opencv-python numpy matplotlib scipy
```

### Ana Pipeline (Metrik + Çıktı Görselleri)

```bash
python src/segmentation.py
```

Üç çıktı görseli `outputs/` klasörüne kaydedilir:

| Dosya | İçerik |
|---|---|
| `segmentation_results.png` | 6 örnek: Orijinal / GT / Tahmin / Hata haritası (IoU & Dice) |
| `pipeline_visualization.png` | 14 adımlık pipeline görselleştirmesi |
| `parameter_analysis.png` | VOTE_THR=2 vs VOTE_THR=3 karşılaştırması |

### Rapor Görselleri (İsteğe Bağlı)

```bash
python src/analysis_report.py
```

`outputs/report_visuals/` klasörüne kaydedilir:

| Dosya | İçerik |
|---|---|
| `best_worst_summary.png` | En iyi 3 + en kötü 3 görsel (4'lü panel) |
| `best_01~03_pipeline.png` | En iyi 3 görsel için 14 adım pipeline |
| `worst_01~03_pipeline.png` | En kötü 3 görsel için 14 adım pipeline |
| `kernel_size_analysis.png` | K_SIZE = 3,5,7,9,11,13 karşılaştırma grafiği |
| `iou_distribution.png` | IoU dağılım histogramı |

---

## Klasör Yapısı

```
fire-segmentation/
├── dataset/
│   ├── images/    (1278 × .jpg — 640×640)
│   └── masks/     (1278 × .png — binary)
├── src/
│   ├── segmentation.py      ← Ana pipeline + metrik hesaplama
│   └── analysis_report.py   ← Rapor görselleri üretici
├── outputs/
│   ├── segmentation_results.png
│   ├── pipeline_visualization.png
│   └── parameter_analysis.png
└── README.md
```

---

## Nicel Değerlendirme Metrikleri

### IoU (Jaccard Index)

```
IoU = |Tahmin ∩ GT| / |Tahmin ∪ GT|
```

### Dice Katsayısı (F1 Skoru)

```
Dice = 2 × |Tahmin ∩ GT| / (|Tahmin| + |GT|)
```

### Hata Haritası Renk Kodu

| Renk | Anlam |
|---|---|
| Yeşil | TP — Doğru Pozitif (alevin doğru tespit edildiği piksel) |
| Turuncu | FP — Yanlış Pozitif (aslında alev olmayan ama alev denen piksel) |
| Mavi | FN — Yanlış Negatif (alev olan ama kaçırılan piksel) |

---

## Teknik Notlar

- **Deterministik:** Aynı girdi her zaman aynı çıktıyı verir, rastgelelik yoktur.
- **Bağımlılık yok:** Eğitim, model dosyası veya GPU gerektirmez.
- **Hız:** Tek görsel için ortalama ~50–100 ms (CPU, 640×640).
- **Genellenebilirlik:** Parametre seti sabit; yeni bir yangın fotoğrafına doğrudan uygulanabilir.
