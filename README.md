# BIM320 — Yangın Görüntülerinde Alev Segmentasyonu

**İstanbul Sabahattin Zaim Üniversitesi — Görüntü İşleme Dersi**  
Büşra Demir · Elif Bilge Güleç · Ümmü Habibe Yüce | Danışman: Dr. Hasibe Büşra AYTEKİN

Klasik görüntü işleme yöntemleriyle yangın alevi segmentasyonu. Makine öğrenmesi veya derin öğrenme kullanılmamıştır.

---

## Sonuçlar

| Metrik | Ortalama | Medyan |
|---|---|---|
| Precision | %85.5 | %88.9 |
| Recall | %87.8 | %90.8 |
| F1-Score (Dice) | %85.5 | %88.0 |
| IoU (Jaccard) | %75.6 | %78.5 |

1278 test görseli üzerinde hesaplanmıştır. IoU ≥ %80 olan görsel sayısı: 551 / 1278.

---

## Yöntem

Pipeline 3 aşamadan oluşur:

1. **Ön işleme:** GaussianBlur → BilateralFilter → CLAHE (LAB uzayı)
2. **Renk maskesi (oylama ≥ 3/5):** HSV, YCrCb, RGB, LAB, Parlaklık kanalları
3. **Morfoloji:** Morph Open → Morph Close → Delik Doldurma → Dilatasyon → Kontur filtresi (≥ 200 px²)

### Parametre Analizi

İki farklı oylama eşiği karşılaştırılmıştır:
- **VOTE_THR = 2/5:** Daha geniş alan, daha fazla yanlış pozitif
- **VOTE_THR = 3/5 (seçilen):** Precision-recall dengesi daha iyi

---

## Veri Seti

1278 görsel (640×640 px), üç Roboflow kaynağından derlendi:

| Kaynak | Görsel |
|---|---|
| Fire Detection Dataset | 321 |
| Fire Segmentation Dataset | 753 |
| Fire and Smoke Dataset | 204 |

---

## Çalıştırma

```bash
pip install opencv-python numpy matplotlib scipy
python src/segmentation.py
```

Çıktılar `outputs/` klasörüne kaydedilir: segmentasyon sonuçları, pipeline görselleştirmesi ve parametre karşılaştırması.

```bash
python src/analysis_report.py  # ek rapor görselleri
```
