"""
Slayt gorseli: VOTE_THR=2 vs VOTE_THR=3 maske karsilastirmasi
Cikti: outputs/report_vote_comparison.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from segmentation import (
    IMAGES_DIR, MASKS_DIR, OUT_DIR,
    iou_hesapla, segmente_et_vote
)

# En cok fark gosteren gorselleri bul (THR=2 gürültülü, THR=3 temiz)
print("En iyi ornekler seciliyor...")
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
farklar = []

for g in gorseller:
    img = cv2.imread(str(g))
    gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        continue
    p2 = segmente_et_vote(img, 2)
    p3 = segmente_et_vote(img, 3)
    if p2.shape != gt.shape:
        gt = cv2.resize(gt, (p2.shape[1], p2.shape[0]))
    if p3.shape != gt.shape:
        gt = cv2.resize(gt, (p3.shape[1], p3.shape[0]))
    iou2 = iou_hesapla(p2, gt)
    iou3 = iou_hesapla(p3, gt)
    # THR=3 daha iyi VE THR=2 gürültülü olan gorseller
    fark = iou3 - iou2
    farklar.append((fark, iou2, iou3, g))

farklar.sort(key=lambda x: x[0], reverse=True)
secilen = farklar[:4]  # THR=3'ün en çok kazandığı 4 görsel
print(f"Secilen ornekler: {[s[3].stem for s in secilen]}")

# ---- Gorsel olustur ----
BG = "#0f1923"; TC = "#e8eaf6"; LC = "#b0bec5"
RENK_THR2 = "#e53935"   # kirmizi - gürültülü
RENK_THR3 = "#43a047"   # yesil   - temiz
RENK_DIFF = "#ff8f00"   # turuncu - THR=2'nin ekstra FP'leri

fig = plt.figure(figsize=(18, 11), facecolor=BG)
gs  = gridspec.GridSpec(4, 5, figure=fig,
                        hspace=0.35, wspace=0.06,
                        left=0.02, right=0.98,
                        top=0.88, bottom=0.07)

SUTUN_BASLIKLAR = [
    ("Orijinal",         "#90a4ae"),
    ("Ground Truth",     "#90a4ae"),
    ("THR = 2/5\n(gürültülü)", RENK_THR2),
    ("THR = 3/5\n(optimal)",   RENK_THR3),
    ("Fark\nTHR=2 ekstra FP",  RENK_DIFF),
]

# Sutun basliklari (sadece ilk satirda)
for col, (baslik, renk) in enumerate(SUTUN_BASLIKLAR):
    ax = fig.add_subplot(gs[0, col])
    ax.set_visible(False)
    fig.text(
        (col + 0.5) / 5 * 0.96 + 0.02, 0.905,
        baslik, ha="center", va="center",
        color=renk, fontsize=10, fontweight="bold",
        transform=fig.transFigure
    )

for row, (fark, iou2, iou3, g) in enumerate(secilen):
    img_bgr = cv2.imread(str(g))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt      = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    p2      = segmente_et_vote(img_bgr, 2)
    p3      = segmente_et_vote(img_bgr, 3)
    if gt.shape != p2.shape:
        gt = cv2.resize(gt, (p2.shape[1], p2.shape[0]))

    # Fark maskesi: THR=2'nin gördüğü ama THR=3'ün görmediği (ekstra FP)
    ekstra = np.logical_and(p2 > 127, p3 <= 127).astype(np.uint8) * 255

    # Hata haritası yardımcısı
    def hata_overlay(img_rgb, pred, gt):
        pb = pred > 127; gb = gt > 127
        out = img_rgb.copy().astype(float)
        out[pb & gb]  = out[pb & gb]  * 0.3 + np.array([0,   200,  0 ]) * 0.7
        out[pb & ~gb] = out[pb & ~gb] * 0.3 + np.array([255, 80,   0 ]) * 0.7
        out[~pb & gb] = out[~pb & gb] * 0.3 + np.array([0,   80,  255]) * 0.7
        return np.clip(out, 0, 255).astype(np.uint8)

    # Fark gorseli: ekstra FP turuncu, geri kalan normal
    diff_vis = img_rgb.copy().astype(float)
    diff_vis[ekstra > 127] = (
        diff_vis[ekstra > 127] * 0.2 + np.array([255, 143, 0]) * 0.8
    )
    diff_vis = np.clip(diff_vis, 0, 255).astype(np.uint8)

    paneller = [
        (img_rgb,                    "#37474f", False),
        (gt,                         "#37474f", True),
        (hata_overlay(img_rgb, p2, gt), RENK_THR2, False),
        (hata_overlay(img_rgb, p3, gt), RENK_THR3, False),
        (diff_vis,                   RENK_DIFF, False),
    ]

    for col, (goruntu, cerceve, gri) in enumerate(paneller):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(goruntu, cmap="gray" if gri else None, aspect="auto")
        ax.set_facecolor("#152030")
        for sp in ax.spines.values():
            sp.set_edgecolor(cerceve)
            sp.set_linewidth(2.2 if col in (2, 3, 4) else 1.2)
        ax.set_xticks([]); ax.set_yticks([])

        # IoU etiketi
        if col == 2:
            ax.set_xlabel(f"IoU = {iou2*100:.1f}%", color=RENK_THR2,
                          fontsize=9, fontweight="bold", labelpad=3)
        elif col == 3:
            ax.set_xlabel(f"IoU = {iou3*100:.1f}%  (+{fark*100:.1f}%)",
                          color=RENK_THR3, fontsize=9, fontweight="bold", labelpad=3)
        elif col == 4:
            ekstra_pct = 100.0 * np.sum(ekstra > 127) / ekstra.size
            ax.set_xlabel(f"Ekstra FP: %{ekstra_pct:.1f} alan",
                          color=RENK_DIFF, fontsize=9, fontweight="bold", labelpad=3)
        elif col == 0:
            ax.set_xlabel(g.stem, color=LC, fontsize=7.5, labelpad=3)

# Legend
tp = mpatches.Patch(color=(0, 0.78, 0),   label="TP — Doğru Pozitif")
fp = mpatches.Patch(color=(1, 0.31, 0),   label="FP — Yanlış Pozitif")
fn = mpatches.Patch(color=(0, 0.31, 1),   label="FN — Yanlış Negatif")
ex = mpatches.Patch(color=(1, 0.56, 0),   label="THR=2 ekstra FP (THR=3'te yok)")
fig.legend(handles=[tp, fp, fn, ex], loc="lower center", ncol=4,
           fontsize=9, facecolor="#152030", edgecolor="#37474f",
           labelcolor=LC, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Oylama Eşiği Karşılaştırması  —  THR=2/5  vs  THR=3/5\n"
    "THR=2: Ortalama IoU=73.2%  |  THR=3: Ortalama IoU=75.6%  |  "
    "Görseller: THR=3'ün en çok kazandığı 4 örnek",
    color=TC, fontsize=11.5, fontweight="bold", y=0.965
)

kayit = OUT_DIR / "report_vote_comparison.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
