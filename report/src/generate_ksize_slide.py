"""
Slayt gorseli: Neden 9x9 Elips Cekirdek?
- Ust: Dikdortgen / Elips / Capraz cekirdek gorseli (9x9)
- Alt: K_SIZE bar grafigi (slayttaki degerler)
Cikti: outputs/report_ksize_analysis.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BG   = "#0f1923"
TC   = "#e8eaf6"
LC   = "#b0bec5"
KIRMIZI = "#e53935"
YESIL   = "#43a047"
GRI     = "#546e7a"

# ---- 9x9 cekirdek matrisleri ----
K = 13   # gorselde buyuk gorunsun diye 13x13 ciz
rect  = cv2.getStructuringElement(cv2.MORPH_RECT,    (K, K))
elips = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K))
capraz = cv2.getStructuringElement(cv2.MORPH_CROSS,  (K, K))

# Slayttaki K_SIZE analiz degerleri
k_degerler  = [3,   5,    9,    11,   13  ]
iou_degerler= [61.2,68.7, 73.8, 72.1, 69.4]
gozlemler   = [
    "İç boşluklar\nkapanmıyor",
    "Kısmi\niyileşme",
    "Peak —\nseçilen ✓",
    "Hafif aşırı\ndilatasyon",
    "Belirgin\nFP artışı",
]

# ============================================================
# LAYOUT: 2 satir
#   Ust (gs[0]): 3 cekirdek gorseli yan yana
#   Alt (gs[1]): bar grafik
# ============================================================
fig = plt.figure(figsize=(14, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 1, figure=fig,
                        height_ratios=[1, 1.4],
                        hspace=0.38,
                        left=0.06, right=0.97,
                        top=0.84, bottom=0.07)

# ---- UST: Cekirdek sekilleri ----
gs_ust = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0],
                                          wspace=0.18)

cekirdekler = [
    (rect,   "Dikdörtgen\n(MORPH_RECT)",    KIRMIZI, "Köşelere ağırlık\nköşeli artefakt"),
    (elips,  "Elips ✓\n(MORPH_ELLIPSE)",    YESIL,   "Dairesel ağırlık\nalev formuyla örtüşüyor"),
    (capraz, "Çapraz\n(MORPH_CROSS)",        GRI,     "Sadece yatay/dikey\nköşe bölgeler kaçırılıyor"),
]

for col, (kernel, baslik, renk, aciklama) in enumerate(cekirdekler):
    ax = fig.add_subplot(gs_ust[col])
    ax.set_facecolor("#0d1f2d")
    for sp in ax.spines.values():
        sp.set_edgecolor(renk)
        sp.set_linewidth(3.0 if renk == YESIL else 1.5)

    # Cekirdegi buyut ve renklendir
    gorsel = np.zeros((K, K, 3), dtype=np.uint8)
    gorsel[kernel == 1] = [200, 220, 255]    # aktif hucre: acik mavi
    gorsel[kernel == 0] = [15,  30,  45]     # pasif hucre: koyu

    # Izgara ciz
    img_zoom = cv2.resize(gorsel, (K*28, K*28), interpolation=cv2.INTER_NEAREST)
    for i in range(1, K):
        img_zoom[i*28-1:i*28+1, :] = [40, 55, 70]
        img_zoom[:, i*28-1:i*28+1] = [40, 55, 70]

    ax.imshow(img_zoom, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{baslik}\n{aciklama}",
                 color=renk, fontsize=9.5, fontweight="bold",
                 pad=7, linespacing=1.5)

# ---- ALT: Bar grafik ----
ax_bar = fig.add_subplot(gs[1])
ax_bar.set_facecolor("#0d1f2d")
for sp in ax_bar.spines.values():
    sp.set_edgecolor("#37474f"); sp.set_linewidth(1.2)

x = np.arange(len(k_degerler))
renkler = [YESIL if k == 9 else GRI for k in k_degerler]
alpha_  = [1.0   if k == 9 else 0.7  for k in k_degerler]
bars = ax_bar.bar(x, iou_degerler, width=0.55,
                  color=renkler, alpha=0.9,
                  edgecolor=["#ffffff" if k==9 else "#37474f" for k in k_degerler],
                  linewidth=[2.0 if k==9 else 0.8 for k in k_degerler])

# Deger etiketleri
for bar, iou, goz in zip(bars, iou_degerler, gozlemler):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4,
                f"%{iou}", ha="center", va="bottom",
                color=TC, fontsize=10.5, fontweight="bold")
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 3.5,
                goz, ha="center", va="top",
                color="#cfd8dc", fontsize=7.5, linespacing=1.4)

# Optimal isaretci
opt_idx = k_degerler.index(9)
ax_bar.annotate("← Optimal",
                xy=(opt_idx, iou_degerler[opt_idx] + 1),
                xytext=(opt_idx + 0.7, iou_degerler[opt_idx] + 2.5),
                color=YESIL, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=YESIL, lw=1.5))

# Eksen
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"{k}×{k}" + ("\n(Seçilen)" if k==9 else "")
                         for k in k_degerler],
                        color=TC, fontsize=10)
ax_bar.set_ylabel("Ortalama IoU (%)", color=LC, fontsize=10)
ax_bar.tick_params(colors=LC, labelsize=9)
ax_bar.set_ylim(55, 80)
ax_bar.yaxis.grid(True, color="#1e3448", linewidth=0.8, linestyle="--")
ax_bar.set_axisbelow(True)
ax_bar.set_title("K_SIZE Boyut Analizi  (50 görsel, VOTE_THR=3 sabit, MORPH_ELLIPSE)",
                 color=TC, fontsize=10.5, fontweight="bold", pad=8)
ax_bar.spines["bottom"].set_color(LC)
ax_bar.spines["left"].set_color(LC)

# Ana baslik
fig.suptitle("Neden 9×9 Elips Çekirdek?",
             color=TC, fontsize=14, fontweight="bold", y=0.96)

kayit = OUT_DIR / "report_ksize_analysis.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
