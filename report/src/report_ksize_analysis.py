"""
K_SIZE -- Ters-U eğrisi grafiği
Slayttaki tablodaki gercek verilerle: 3x3=73.8, 5x5=74.6, 7x7=75.1, 9x9=76.6, 11x11=75.0, 13x13=74.3
Çıktı: outputs/report_ksize_analysis.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "outputs" / "report_ksize_analysis.png"

BG   = "#0f1923"
TC   = "#e8eaf6"
LC   = "#b0bec5"
YESIL   = "#43a047"
KIRMIZI = "#e53935"
MAVI    = "#1e88e5"
SARI    = "#fdd835"

# Tablodaki gercek veriler
k_sizes  = [3,    5,    7,    9,    11,   13  ]
iou_vals = [73.8, 74.6, 75.1, 76.6, 75.0, 74.3]
etiketler = ["3×3", "5×5", "7×7", "9×9", "11×11", "13×13"]
x = np.array(k_sizes)

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
ax.set_facecolor("#0d1f2d")
for sp in ax.spines.values():
    sp.set_edgecolor("#37474f")
    sp.set_linewidth(1.2)

# Egri ciz
ax.plot(x, iou_vals, color=MAVI, linewidth=3, marker="o", markersize=9,
        zorder=3, label="Ortalama IoU (%)")

# Alti doldur (Ters-U görünümü)
ax.fill_between(x, 72.5, iou_vals, alpha=0.12, color=MAVI)

# Optimal nokta - 9x9 vurgula
ax.scatter([9], [76.6], s=180, color=YESIL, zorder=5, edgecolors="white", linewidths=1.5)

# Değer etiketleri
offsets = [(0, 8), (0, 8), (0, 8), (0, 10), (0, 8), (0, 8)]
colors  = [LC, LC, LC, YESIL, LC, LC]
sizes   = [9, 9, 9, 11, 9, 9]
bolds   = [False, False, False, True, False, False]
for i, (xi, yi) in enumerate(zip(k_sizes, iou_vals)):
    ox, oy = offsets[i]
    ax.annotate(f"{yi:.1f}%",
                xy=(xi, yi), xytext=(ox, oy),
                textcoords="offset points",
                color=colors[i], fontsize=sizes[i],
                fontweight="bold" if bolds[i] else "normal",
                ha="center", zorder=6)

# Optimal dikey cizgi
ax.axvline(x=9, color=YESIL, linewidth=1.5, linestyle="--", alpha=0.5, zorder=1)
ax.text(9.2, 73.2, "Peak\nK=9×9", color=YESIL, fontsize=9.5, fontweight="bold")

# Ok: sol taraf artis, sag taraf azalis
ax.annotate("", xy=(7, 75.5), xytext=(3.5, 74.0),
            arrowprops=dict(arrowstyle="->", color=YESIL, lw=1.8))
ax.text(4.2, 74.2, "Monoton\nartis", color=YESIL, fontsize=8.5)

ax.annotate("", xy=(11.5, 74.5), xytext=(10, 75.8),
            arrowprops=dict(arrowstyle="->", color=KIRMIZI, lw=1.8))
ax.text(10.3, 75.1, "Monoton\nazalis", color=KIRMIZI, fontsize=8.5)

# Eksenler
ax.set_xticks(k_sizes)
ax.set_xticklabels(etiketler, color=TC, fontsize=11)
ax.set_xlabel("Morfoloji Çekirdeği Boyutu (K_SIZE)", color=LC, fontsize=10, labelpad=8)
ax.set_ylabel("Ortalama IoU (%)", color=LC, fontsize=10)
ax.tick_params(colors=LC, labelsize=9)
ax.set_ylim(72.0, 78.5)
ax.yaxis.grid(True, color="#1e3448", linewidth=0.8, linestyle="--")
ax.set_axisbelow(True)

# Optimal kutu
ax.annotate("Optimal: 9×9  |  IoU = 76.6%",
            xy=(9, 76.6), xytext=(5.8, 77.8),
            fontsize=10, color=YESIL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d2b1a",
                      edgecolor=YESIL, linewidth=1.2),
            arrowprops=dict(arrowstyle="->", color=YESIL, lw=1.4))

ax.set_title(
    "K_SIZE Parametre Analizi — Ters-U Eğrisi\n"
    "VOTE_THR = 3 sabit  ·  50 görsel  ·  Çekirdek büyüdükçe boşluk kapanır, fazla büyüyünce FP artar",
    color=TC, fontsize=11, fontweight="bold", pad=10
)

fig.tight_layout()
fig.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {OUT_PATH}")
