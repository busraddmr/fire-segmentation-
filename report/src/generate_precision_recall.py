"""
Slayt gorseli: Precision vs Recall — VOTE_THR analizi
Cikti: outputs/report_precision_recall.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BG = "#0f1923"; TC = "#e8eaf6"; LC = "#b0bec5"
YESIL   = "#43a047"
KIRMIZI = "#e53935"
MAVI    = "#1e88e5"
SARI    = "#fdd835"
GRI     = "#546e7a"

# Hesaplanan degerler (1278 gorsel, tum dataset)
thr_vals  = [1,    2,    3,    4,    5   ]
precision = [65.3, 77.3, 83.6, 88.4, 94.9]
recall    = [96.9, 93.3, 90.1, 80.2, 16.0]
iou       = [63.9, 73.3, 76.6, 72.5, 15.9]
etiketler = ["1/5", "2/5", "3/5", "4/5", "5/5"]

fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor("#0d1f2d")
for sp in ax.spines.values():
    sp.set_edgecolor("#37474f"); sp.set_linewidth(1.2)

x = np.array(thr_vals)

# Egri ciz
ax.plot(x, precision, color=KIRMIZI, linewidth=2.5, marker="o", markersize=8,
        label="Precision ↑", zorder=3)
ax.plot(x, recall,    color=MAVI,    linewidth=2.5, marker="s", markersize=8,
        label="Recall ↓",    zorder=3)
ax.plot(x, iou,       color=YESIL,   linewidth=2.5, marker="^", markersize=8,
        label="IoU",          zorder=3)

# Deger etiketleri
offsets_prec = [(-18, 8), (-18, 8), (-18, 8), (8, 8), (8, 8)]
offsets_rec  = [(8, -18), (8, -18), (8, -18), (8, -18), (-18, -20)]
offsets_iou  = [(8, -18), (8, -18), (-20, 10), (8, -18), (8, 8)]

for i, (t, p, r, u) in enumerate(zip(thr_vals, precision, recall, iou)):
    ox_p, oy_p = offsets_prec[i]
    ax.annotate(f"{p:.1f}%", xy=(t, p), xytext=(ox_p, oy_p),
                textcoords="offset points", color=KIRMIZI,
                fontsize=8.5, fontweight="bold", zorder=5)
    ox_r, oy_r = offsets_rec[i]
    ax.annotate(f"{r:.1f}%", xy=(t, r), xytext=(ox_r, oy_r),
                textcoords="offset points", color=MAVI,
                fontsize=8.5, fontweight="bold", zorder=5)
    ox_u, oy_u = offsets_iou[i]
    ax.annotate(f"{u:.1f}%", xy=(t, u), xytext=(ox_u, oy_u),
                textcoords="offset points", color=YESIL,
                fontsize=8.5, fontweight="bold", zorder=5)

# THR=3 optimal isaretci
ax.axvline(x=3, color=YESIL, linewidth=1.5, linestyle="--", alpha=0.5, zorder=1)
ax.text(3.07, 20, "← Optimal\n    THR=3/5", color=YESIL,
        fontsize=9, fontweight="bold")

# Precision - Recall kesisim bolgesini dolgula
ax.fill_between(x, precision, recall,
                where=[p > r for p, r in zip(precision, recall)],
                alpha=0.07, color=KIRMIZI, label="_nolegend_")
ax.fill_between(x, precision, recall,
                where=[p <= r for p, r in zip(precision, recall)],
                alpha=0.07, color=MAVI, label="_nolegend_")

# Eksen
ax.set_xticks(thr_vals)
ax.set_xticklabels(etiketler, color=TC, fontsize=11)
ax.set_xlabel("VOTE_THR (kaç kanalın eşzamanlı pozitif olması gerektiği)",
              color=LC, fontsize=10, labelpad=8)
ax.set_ylabel("Metrik Değeri (%)", color=LC, fontsize=10)
ax.tick_params(colors=LC, labelsize=9)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, color="#1e3448", linewidth=0.8, linestyle="--")
ax.set_axisbelow(True)

# Ok isaretleri: precision yukari, recall asagi
ax.annotate("", xy=(4.7, 92), xytext=(4.7, 75),
            arrowprops=dict(arrowstyle="->", color=KIRMIZI, lw=1.8))
ax.text(4.75, 82, "Precision\nartar", color=KIRMIZI, fontsize=8, va="center")

ax.annotate("", xy=(4.7, 25), xytext=(4.7, 42),
            arrowprops=dict(arrowstyle="->", color=MAVI, lw=1.8))
ax.text(4.75, 33, "Recall\nazalır", color=MAVI, fontsize=8, va="center")

# Legend
legend = ax.legend(fontsize=10, facecolor="#152030", edgecolor="#37474f",
                   labelcolor=LC, loc="upper left",
                   bbox_to_anchor=(0.01, 0.99))

ax.set_title(
    "Precision ↑  vs  Recall ↓  —  VOTE_THR Arttıkça\n"
    "THR↑: Daha katı eşik → Yüksek Precision (az FP), Düşük Recall (çok FN)",
    color=TC, fontsize=11, fontweight="bold", pad=10
)

fig.tight_layout(rect=[0, 0, 1, 1])
kayit = OUT_DIR / "report_precision_recall.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
