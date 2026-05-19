"""
Slayt gorseli: VOTE_THR=3 vs VOTE_THR=4 maske karsilastirmasi
Cikti: outputs/report_vote_comparison_34.png
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

# THR=3'un en cok kazandigi gorselleri bul (THR=4 piksel kaybediyor)
print("En iyi ornekler seciliyor...")
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
farklar = []

for g in gorseller:
    img = cv2.imread(str(g))
    gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        continue
    p3 = segmente_et_vote(img, 3)
    p4 = segmente_et_vote(img, 4)
    if p3.shape != gt.shape:
        gt = cv2.resize(gt, (p3.shape[1], p3.shape[0]))
    iou3 = iou_hesapla(p3, gt)
    iou4 = iou_hesapla(p4, gt)
    fark = iou3 - iou4
    farklar.append((fark, iou3, iou4, g))

farklar.sort(key=lambda x: x[0], reverse=True)
secilen = farklar[:4]
print(f"Secilen ornekler: {[s[3].stem for s in secilen]}")

# ---- Gorsel olustur ----
BG = "#0f1923"; TC = "#e8eaf6"; LC = "#b0bec5"
RENK_THR3 = "#43a047"   # yesil  - optimal
RENK_THR4 = "#e53935"   # kirmizi - cok katı, piksel kaybi
RENK_DIFF = "#1565c0"   # mavi - THR=4'un kaybettigi FN'ler

fig = plt.figure(figsize=(18, 11), facecolor=BG)
gs  = gridspec.GridSpec(4, 5, figure=fig,
                        hspace=0.35, wspace=0.06,
                        left=0.02, right=0.98,
                        top=0.88, bottom=0.07)

SUTUN_BASLIKLAR = [
    ("Orijinal",              "#90a4ae"),
    ("Ground Truth",          "#90a4ae"),
    ("THR = 3/5\n(optimal)",  RENK_THR3),
    ("THR = 4/5\n(çok katı)", RENK_THR4),
    ("Fark\nTHR=4 kaybı (FN)", RENK_DIFF),
]

for col, (baslik, renk) in enumerate(SUTUN_BASLIKLAR):
    ax = fig.add_subplot(gs[0, col])
    ax.set_visible(False)
    fig.text(
        (col + 0.5) / 5 * 0.96 + 0.02, 0.905,
        baslik, ha="center", va="center",
        color=renk, fontsize=10, fontweight="bold",
        transform=fig.transFigure
    )

for row, (fark, iou3, iou4, g) in enumerate(secilen):
    img_bgr = cv2.imread(str(g))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt      = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    p3      = segmente_et_vote(img_bgr, 3)
    p4      = segmente_et_vote(img_bgr, 4)
    if gt.shape != p3.shape:
        gt = cv2.resize(gt, (p3.shape[1], p3.shape[0]))

    # Fark maskesi: THR=3'un gördüğü ama THR=4'ün görmediği (kaybedilen FN)
    kayip = np.logical_and(p3 > 127, p4 <= 127).astype(np.uint8) * 255

    def hata_overlay(img_rgb, pred, gt):
        pb = pred > 127; gb = gt > 127
        out = img_rgb.copy().astype(float)
        out[pb & gb]  = out[pb & gb]  * 0.3 + np.array([0,   200,  0 ]) * 0.7
        out[pb & ~gb] = out[pb & ~gb] * 0.3 + np.array([255, 80,   0 ]) * 0.7
        out[~pb & gb] = out[~pb & gb] * 0.3 + np.array([0,   80,  255]) * 0.7
        return np.clip(out, 0, 255).astype(np.uint8)

    # Fark gorseli: kaybedilen alanlar mavi
    diff_vis = img_rgb.copy().astype(float)
    diff_vis[kayip > 127] = (
        diff_vis[kayip > 127] * 0.2 + np.array([21, 101, 192]) * 0.8
    )
    diff_vis = np.clip(diff_vis, 0, 255).astype(np.uint8)

    paneller = [
        (img_rgb,                       "#37474f", False),
        (gt,                            "#37474f", True),
        (hata_overlay(img_rgb, p3, gt), RENK_THR3, False),
        (hata_overlay(img_rgb, p4, gt), RENK_THR4, False),
        (diff_vis,                      RENK_DIFF,  False),
    ]

    for col, (goruntu, cerceve, gri) in enumerate(paneller):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(goruntu, cmap="gray" if gri else None, aspect="auto")
        ax.set_facecolor("#152030")
        for sp in ax.spines.values():
            sp.set_edgecolor(cerceve)
            sp.set_linewidth(2.2 if col in (2, 3, 4) else 1.2)
        ax.set_xticks([]); ax.set_yticks([])

        if col == 2:
            ax.set_xlabel(f"IoU = {iou3*100:.1f}%", color=RENK_THR3,
                          fontsize=9, fontweight="bold", labelpad=3)
        elif col == 3:
            ax.set_xlabel(f"IoU = {iou4*100:.1f}%  (−{fark*100:.1f}%)",
                          color=RENK_THR4, fontsize=9, fontweight="bold", labelpad=3)
        elif col == 4:
            kayip_pct = 100.0 * np.sum(kayip > 127) / kayip.size
            ax.set_xlabel(f"Kayıp FN: %{kayip_pct:.1f} alan",
                          color=RENK_DIFF, fontsize=9, fontweight="bold", labelpad=3)
        elif col == 0:
            ax.set_xlabel(g.stem, color=LC, fontsize=7.5, labelpad=3)

# Legend
tp = mpatches.Patch(color=(0, 0.78, 0),   label="TP — Doğru Pozitif")
fp = mpatches.Patch(color=(1, 0.31, 0),   label="FP — Yanlış Pozitif")
fn = mpatches.Patch(color=(0, 0.31, 1),   label="FN — Yanlış Negatif")
ex = mpatches.Patch(color=(0.08, 0.40, 0.75), label="THR=4 kaybı: THR=3'te var, THR=4'te yok (FN)")
fig.legend(handles=[tp, fp, fn, ex], loc="lower center", ncol=4,
           fontsize=9, facecolor="#152030", edgecolor="#37474f",
           labelcolor=LC, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Oylama Eşiği Karşılaştırması  —  THR=3/5  vs  THR=4/5\n"
    "THR=3: Ortalama IoU=75.6%  |  THR=4: Ortalama IoU=69.3%  |  "
    "Görseller: THR=4'ün en çok kaybettiği 4 örnek",
    color=TC, fontsize=11.5, fontweight="bold", y=0.965
)

kayit = OUT_DIR / "report_vote_comparison_34.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
