"""
Slayt gorseli: IoU & Dice — TP/FP/FN hata haritasi ornegi
Cikti: outputs/report_iou_examples.png
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
from segmentation import IMAGES_DIR, MASKS_DIR, OUT_DIR, iou_hesapla, segmente_et_vote

BG = "#0f1923"; TC = "#e8eaf6"; LC = "#b0bec5"
RENK_TP  = "#2e7d32"   # koyu yesil
RENK_FP  = "#e65100"   # turuncu
RENK_FN  = "#1565c0"   # mavi
RENK_TN  = "#152030"   # koyu

# TP/FP/FN dengeli olan gorsel bul
# Hedef: fp_oran ve fn_oran her ikisi de %10-35 arasinda
print("Gorsel seciliyor...")
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
en_iyi = None
en_iyi_skor = 999

# Hatalı annotasyonlu gorselleri atla
ATLA = {"fire_00248"}

for g in gorseller:
    if g.stem in ATLA:
        continue
    img = cv2.imread(str(g))
    gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        continue
    pred = segmente_et_vote(img, 3)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    pb = pred > 127; gb = gt > 127
    n_tp = int(np.sum(pb & gb))
    n_fp = int(np.sum(pb & ~gb))
    n_fn = int(np.sum(~pb & gb))
    total = n_tp + n_fp + n_fn
    if total < 500:
        continue
    fp_oran = n_fp / total
    fn_oran = n_fn / total
    # Her ikisi de %10-35 arasinda olsun, fark minimum olsun
    if 0.10 <= fp_oran <= 0.35 and 0.10 <= fn_oran <= 0.35:
        skor = abs(fp_oran - fn_oran)  # ne kadar dengeli
        if skor < en_iyi_skor:
            iou = iou_hesapla(pred, gt)
            en_iyi_skor = skor
            en_iyi = (g, img, gt, pred, iou)

secilen = en_iyi
if secilen is None:
    for g in gorseller[:50]:
        if g.stem in ATLA:
            continue
        img = cv2.imread(str(g))
        gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        pred = segmente_et_vote(img, 3)
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        iou = iou_hesapla(pred, gt)
        secilen = (g, img, gt, pred, iou)
        break

g, img_bgr, gt, pred, iou = secilen
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pb = pred > 127
gb = gt   > 127

tp_mask = pb & gb
fp_mask = pb & ~gb
fn_mask = ~pb & gb

n_tp = np.sum(tp_mask)
n_fp = np.sum(fp_mask)
n_fn = np.sum(fn_mask)
n_union = n_tp + n_fp + n_fn
dice = 2 * n_tp / (2 * n_tp + n_fp + n_fn + 1e-9)

print(f"Gorsel: {g.stem}  IoU={iou:.3f}  Dice={dice:.3f}")
print(f"TP={n_tp}  FP={n_fp}  FN={n_fn}")

# ---- Overlay gorseller ----
def hata_overlay(img_rgb, tp, fp, fn):
    out = img_rgb.copy().astype(float)
    out[tp] = out[tp] * 0.25 + np.array([46,  160, 50 ]) * 0.75   # yesil TP
    out[fp] = out[fp] * 0.25 + np.array([230, 81,  0  ]) * 0.75   # turuncu FP
    out[fn] = out[fn] * 0.25 + np.array([21,  101, 192]) * 0.75   # mavi FN
    return np.clip(out, 0, 255).astype(np.uint8)

overlay = hata_overlay(img_rgb, tp_mask, fp_mask, fn_mask)

# GT overlay (yesil)
gt_overlay = img_rgb.copy().astype(float)
gt_overlay[gb] = gt_overlay[gb] * 0.3 + np.array([46, 160, 50]) * 0.7
gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)

# Pred overlay (turuncu cerceve)
pred_overlay = img_rgb.copy().astype(float)
pred_overlay[pb] = pred_overlay[pb] * 0.3 + np.array([255, 160, 0]) * 0.7
pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)

# ============================================================
# LAYOUT: 1 satir x 4 panel + alt bilgi satiri
# ============================================================
fig = plt.figure(figsize=(16, 7), facecolor=BG)
gs = gridspec.GridSpec(2, 4, figure=fig,
                       height_ratios=[1, 0.12],
                       hspace=0.28, wspace=0.06,
                       left=0.02, right=0.98,
                       top=0.84, bottom=0.14)

paneller = [
    (img_rgb,      "Orijinal Görsel",        "#546e7a",  None),
    (gt_overlay,   "Ground Truth  (GT)",     "#42a5f5",  None),
    (pred_overlay, "Tahmin  (Pred)",         "#ffa726",  None),
    (overlay,      "TP / FP / FN Haritası",  "#e8eaf6",  True),
]

for col, (goruntu, baslik, border_renk, is_hata) in enumerate(paneller):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(goruntu, aspect="auto")
    ax.set_facecolor("#0d1f2d")
    lw = 3.0 if is_hata else 1.8
    for sp in ax.spines.values():
        sp.set_edgecolor(border_renk)
        sp.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(baslik, color=border_renk, fontsize=10.5,
                 fontweight="bold", pad=6)

# ---- TP/FP/FN piksel sayilari (hata haritasi panelinin altinda) ----
toplam = tp_mask.size
ax_bar = fig.add_subplot(gs[1, 3])
ax_bar.set_facecolor(BG)
ax_bar.set_visible(False)

# Metin etiketleri hata haritasi panelinin altina
for col, (label, val, renk) in enumerate([
    (f"TP = {n_tp:,}", n_tp, "#4caf50"),
    (f"FP = {n_fp:,}", n_fp, "#ff7043"),
    (f"FN = {n_fn:,}", n_fn, "#42a5f5"),
]):
    fig.text(
        (3 + (col + 0.5) / 3 * 1) / 4 * 0.96 + 0.02,
        0.105,
        label,
        ha="center", va="center",
        color=renk, fontsize=9, fontweight="bold",
        transform=fig.transFigure
    )

# ---- IoU ve Dice degerlerini goster ----
iou_str  = f"IoU  =  |Pred ∩ GT| / |Pred ∪ GT|  =  {n_tp:,} / {n_union:,}  =  {iou*100:.1f}%"
dice_str = f"Dice  =  2|Pred∩GT| / (|Pred|+|GT|)  =  2×{n_tp:,} / {2*n_tp+n_fp+n_fn:,}  =  {dice*100:.1f}%"

fig.text(0.5, 0.065, iou_str,
         ha="center", va="center",
         color="#e8eaf6", fontsize=9.5, fontweight="bold",
         fontfamily="monospace",
         transform=fig.transFigure)
fig.text(0.5, 0.040, dice_str,
         ha="center", va="center",
         color="#b0bec5", fontsize=9.5,
         fontfamily="monospace",
         transform=fig.transFigure)

# Legend
tp_p = mpatches.Patch(facecolor="#4caf50", label="TP — Tahmin=Alev & GT=Alev")
fp_p = mpatches.Patch(facecolor="#ff7043", label="FP — Tahmin=Alev & GT=Arka Plan")
fn_p = mpatches.Patch(facecolor="#42a5f5", label="FN — GT=Alev & Tahmin=Arka Plan")
fig.legend(handles=[tp_p, fp_p, fn_p], loc="lower center", ncol=3,
           fontsize=9, facecolor="#152030", edgecolor="#37474f",
           labelcolor=LC, bbox_to_anchor=(0.5, 0.005))

fig.suptitle(
    f"TP / FP / FN Hata Haritası  —  Örnek: {g.stem}",
    color=TC, fontsize=13, fontweight="bold", y=0.95
)

kayit = OUT_DIR / "report_iou_examples.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
