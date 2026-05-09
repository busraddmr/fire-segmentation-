"""
BIM320 - Goruntu Isleme Projesi
Rapor Gorselleri Uretici  |  analysis_report.py

KURAL: segmentation.py'ye DOKUNULMAZ. Sadece rapor gorselleri uretilir.

Ciktilar -> outputs/report_visuals/
  best_worst_summary.png     : En iyi 3 + en kotu 3 (4 panel)
  best_01_pipeline.png       : En iyi #1 icin 14 adim pipeline
  best_02_pipeline.png
  best_03_pipeline.png
  worst_01_pipeline.png
  worst_02_pipeline.png
  worst_03_pipeline.png
  kernel_size_analysis.png   : K_SIZE = 3,5,7,9,11,13 vs IoU/Dice grafigi
  iou_distribution.png       : IoU dagilim histogrami

Kullanim:
  python src/analysis_report.py
"""

import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segmentation import (
    IMAGES_DIR, MASKS_DIR, OUT_DIR,
    segmente_et, segmente_et_adim_adim,
    iou_hesapla, dice_hesapla,
    K_SIZE, VOTE_THR,
    on_isleme, tum_maskeler, morfo_ve_temizle,
)

REPORT_DIR = OUT_DIR / "report_visuals"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ADIM_LISTESI = [
    ("1. Orijinal",          "orijinal",    False),
    ("2. GaussianBlur",      "gaussian",    False),
    ("3. Bilateral Filter",  "bilateral",   False),
    ("4. CLAHE",             "clahe",       False),
    ("5. HSV Maskesi",       "m_hsv",       True),
    ("6. YCrCb Maskesi",     "m_ycc",       True),
    ("7. RGB Maskesi",       "m_rgb",       True),
    ("8. LAB Maskesi",       "m_lab",       True),
    ("9. Parlaklik Mask.",   "m_wht",       True),
    ("10. Oylama >=3/5",     "oylama",      True),
    ("11. Morph Open",       "morph_ac",    True),
    ("12. Morph Close",      "morph_kapat", True),
    ("13. Delik Doldurma",   "doldurma",    True),
    ("14. Final Tahmin",     "final",       True),
]


def segmente_et_ksize(img_bgr, k_size):
    """Sadece kernel analizi icin: K_SIZE degistirilebilir versiyon."""
    from scipy.ndimage import binary_fill_holes
    en       = on_isleme(img_bgr)
    tot      = tum_maskeler(en)
    combined = np.where(tot >= VOTE_THR * 255, 255, 0).astype(np.uint8)
    kk       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    op       = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kk, iterations=1)
    cl       = cv2.morphologyEx(op,       cv2.MORPH_CLOSE, kk, iterations=3)
    filled   = (binary_fill_holes(cl > 127) * 255).astype(np.uint8)
    kd       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated  = cv2.dilate(filled, kd, iterations=1)
    sonuc    = np.zeros_like(dilated)
    cnts, _  = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= 200:
            cv2.drawContours(sonuc, [c], -1, 255, -1)
    return sonuc


def hata_haritasi(pred, gt):
    hata = np.zeros((*pred.shape, 3), np.uint8)
    pb = pred > 127; gb = gt > 127
    hata[pb & gb]  = [0,   200,   0]
    hata[pb & ~gb] = [255, 100,   0]
    hata[~pb & gb] = [0,   100, 255]
    return hata


def pipeline_kaydet(img_bgr, g_stem, iou, dice, etiket, rank):
    adimlar = segmente_et_adim_adim(img_bgr)
    fig, axes = plt.subplots(2, 7, figsize=(21, 7))
    fig.suptitle(
        f"[{etiket} #{rank}] {g_stem}   IoU={iou:.3f}   Dice={dice:.3f}",
        fontsize=12, fontweight="bold"
    )
    for i, (baslik, anahtar, gri) in enumerate(ADIM_LISTESI):
        ax = axes.flatten()[i]
        ax.imshow(adimlar[anahtar], cmap="gray" if gri else None)
        ax.set_title(baslik, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    k = REPORT_DIR / f"{etiket.lower()}_{rank:02d}_pipeline.png"
    fig.savefig(str(k), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {k.name}")


def main():
    gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
    toplam    = len(gorseller)

    # ---- 1. TUM DATASET: IoU hesapla (tek gecis) ----
    print(f"Dataset taranıyor ({toplam} gorsel)...")
    sonuclar = []
    for i, g in enumerate(gorseller, 1):
        img = cv2.imread(str(g))
        gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        pred = segmente_et(img)
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
        sonuclar.append((iou_hesapla(pred, gt), dice_hesapla(pred, gt), g))
        if i % 200 == 0 or i == toplam:
            print(f"  [{i}/{toplam}]  ort. IoU: {np.mean([s[0] for s in sonuclar]):.4f}")

    sonuclar.sort(key=lambda x: x[0])
    iou_list  = [s[0] for s in sonuclar]
    dice_list = [s[1] for s in sonuclar]
    worst3    = sonuclar[:3]
    best3     = sonuclar[-3:][::-1]

    print(f"\nOrtalama IoU  : {np.mean(iou_list):.4f} ({np.mean(iou_list)*100:.1f}%)")
    print(f"Ortalama Dice : {np.mean(dice_list):.4f}")
    print(f"\nEn kotu 3: {[f'{s[2].stem} IoU={s[0]:.3f}' for s in worst3]}")
    print(f"En iyi  3: {[f'{s[2].stem} IoU={s[0]:.3f}' for s in best3]}")

    # ---- 2. BEST / WORST OZET (6 satir x 4 panel) ----
    print("\nBest/Worst ozet gorseli olusturuluyor...")
    fig, axes = plt.subplots(6, 4, figsize=(18, 26))
    fig.suptitle(
        "BIM320 - Alev Segmentasyonu | En Iyi 3 ve En Kotu 3\n"
        f"Ort. IoU={np.mean(iou_list):.3f} ({np.mean(iou_list)*100:.1f}%)  "
        f"Ort. Dice={np.mean(dice_list):.3f}  N={len(iou_list)}",
        fontsize=13, fontweight="bold"
    )
    for row_offset, (etiket, grup) in enumerate([("EN IYI", best3), ("EN KOTU", worst3)]):
        for j, (iou, dice, g) in enumerate(grup):
            row = row_offset * 3 + j
            img  = cv2.imread(str(g))
            gt   = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
            pred = segmente_et(img)
            if pred.shape != gt.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
            hata = hata_haritasi(pred, gt)
            axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[row, 0].set_title(f"[{etiket} #{j+1}] {g.stem}", fontsize=8)
            axes[row, 1].imshow(gt, cmap="gray"); axes[row, 1].set_title("Ground Truth", fontsize=8)
            axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].set_title("Tahmin", fontsize=8)
            axes[row, 3].imshow(hata)
            axes[row, 3].set_title(f"Hata  IoU={iou:.3f}  Dice={dice:.3f}", fontsize=8)
            for ax in axes[row]: ax.axis("off")

    tp = mpatches.Patch(color=(0,.78,0), label="TP"); fp = mpatches.Patch(color=(1,.39,0), label="FP"); fn = mpatches.Patch(color=(0,.39,1), label="FN")
    fig.legend(handles=[tp, fp, fn], loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.003))
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(str(REPORT_DIR / "best_worst_summary.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Kaydedildi: best_worst_summary.png")

    # ---- 3. PIPELINE GORSELLERI ----
    print("\nPipeline gorselleri (best 3 + worst 3)...")
    for iou, dice, g in best3:
        img = cv2.imread(str(g))
        pipeline_kaydet(img, g.stem, iou, dice, "BEST", best3.index((iou, dice, g)) + 1)
    for iou, dice, g in worst3:
        img = cv2.imread(str(g))
        pipeline_kaydet(img, g.stem, iou, dice, "WORST", worst3.index((iou, dice, g)) + 1)

    # ---- 4. KERNEL SIZE ANALIZI (50 ornek, hizli) ----
    print("\nKernel size analizi (K=3,5,7,9,11,13)...")
    k_degerler = [3, 5, 7, 9, 11, 13]
    ornek_list = [gorseller[i] for i in np.linspace(0, toplam-1, 50, dtype=int)]
    k_iou_ortalama = []
    k_dice_ortalama = []

    for k in k_degerler:
        kiou = []; kdice = []
        for g in ornek_list:
            img = cv2.imread(str(g))
            gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
            if img is None or gt is None: continue
            pred = segmente_et_ksize(img, k)
            if pred.shape != gt.shape: gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
            kiou.append(iou_hesapla(pred, gt)); kdice.append(dice_hesapla(pred, gt))
        k_iou_ortalama.append(np.mean(kiou)); k_dice_ortalama.append(np.mean(kdice))
        print(f"  K={k:2d} -> IoU={k_iou_ortalama[-1]:.4f}  Dice={k_dice_ortalama[-1]:.4f}")

    fig3, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    rk_iou  = ["#2ecc71" if k == K_SIZE else "#e74c3c" for k in k_degerler]
    rk_dice = ["#f39c12" if k == K_SIZE else "#3498db" for k in k_degerler]
    b1 = ax1.bar([x-.2 for x in range(len(k_degerler))], k_iou_ortalama,  width=.35, color=rk_iou,  alpha=.85, label="IoU")
    b2 = ax2.bar([x+.2 for x in range(len(k_degerler))], k_dice_ortalama, width=.35, color=rk_dice, alpha=.85, label="Dice")
    for bar, v in zip(b1, k_iou_ortalama):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003, f"{v:.3f}", ha="center", fontsize=8, color="#c0392b")
    for bar, v in zip(b2, k_dice_ortalama):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003, f"{v:.3f}", ha="center", fontsize=8, color="#2980b9")
    ax1.axvline(k_degerler.index(K_SIZE), color="green", linestyle="--", lw=1.5, label=f"Secilen K={K_SIZE}")
    ax1.set_xticks(range(len(k_degerler))); ax1.set_xticklabels([f"{k}×{k}" for k in k_degerler])
    ax1.set_xlabel("Kernel Boyutu (K_SIZE)"); ax1.set_ylabel("Ort. IoU", color="#c0392b"); ax2.set_ylabel("Ort. Dice", color="#2980b9")
    ax1.set_ylim(0, max(k_iou_ortalama)*1.15); ax2.set_ylim(0, max(k_dice_ortalama)*1.15)
    handles = [mpatches.Patch(color="#2ecc71", label=f"IoU optimal (K={K_SIZE})"),
               mpatches.Patch(color="#e74c3c", label="IoU diger"),
               mpatches.Patch(color="#f39c12", label=f"Dice optimal (K={K_SIZE})"),
               mpatches.Patch(color="#3498db", label="Dice diger")]
    ax1.legend(handles=handles, loc="lower right", fontsize=8)
    plt.title(f"BIM320 - Kernel Boyutu Parametresi Analizi\n(50 ornek, VOTE_THR={VOTE_THR} sabit)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(str(REPORT_DIR / "kernel_size_analysis.png"), dpi=120, bbox_inches="tight")
    plt.close(fig3)
    print("  Kaydedildi: kernel_size_analysis.png")

    # ---- 5. IOu DAGILIM HISTOGRAMI ----
    print("\nIoU dagilim histogrami olusturuluyor...")
    fig4, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches2 = ax.hist(iou_list, bins=40, edgecolor="white", lw=.5)
    for patch, left in zip(patches2, bins):
        patch.set_facecolor("#e74c3c" if left < .5 else "#f39c12" if left < .7 else "#2ecc71")
    ort = np.mean(iou_list); med = np.median(iou_list)
    ax.axvline(ort, color="#2c3e50", ls="--", lw=2, label=f"Ort. IoU = {ort:.3f}")
    ax.axvline(med, color="#8e44ad", ls=":",  lw=2, label=f"Medyan IoU = {med:.3f}")
    ax.axvline(.70, color="#27ae60", ls="-",  lw=1.5, alpha=.7,
               label=f"Esik 0.70 ({sum(1 for x in iou_list if x>=.70)} gorsel)")
    ax.set_xlabel("IoU Degeri"); ax.set_ylabel("Gorsel Sayisi")
    ax.set_title(f"BIM320 - IoU Dagilim Histogrami (N={len(iou_list)})\nKirmizi:<0.50 | Sari:0.50-0.70 | Yesil:>=0.70", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.set_xlim(0, 1)
    plt.tight_layout()
    fig4.savefig(str(REPORT_DIR / "iou_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close(fig4)
    print("  Kaydedildi: iou_distribution.png")

    # ---- OZET ----
    print(f"\n{'='*50}")
    print(f"  TAMAMLANDI -> {REPORT_DIR}")
    print(f"  best_worst_summary.png")
    print(f"  best_01~03_pipeline.png  |  worst_01~03_pipeline.png")
    print(f"  kernel_size_analysis.png")
    print(f"  iou_distribution.png")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
