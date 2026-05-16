"""
BIM320 - Goruntu Isleme Projesi
Veri: dataset/images + dataset/masks
Sonuc: Nihai temiz dataset
  - bitti (V3+V4) : IoU > 0.20 filtreli  (957 gorsel)
  - V1            : IoU > 0.70 filtreli  (370 gorsel)
  - Toplam        : 1327 gorsel
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import binary_fill_holes

# ===== YOLLAR =====
BASE_DIR   = Path(__file__).parent.parent / "dataset"
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR  = BASE_DIR / "masks"
OUT_DIR    = Path(__file__).parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ===== OPTIMAL PARAMETRELER =====
H_MAX   = 35
S_MIN   = 100
V_MIN   = 120
CR_MIN  = 150
CB_MAX  = 140
K_SIZE  = 9
MIN_AREA= 200
VOTE_THR= 3

# ===== ON ISLEME =====
def on_isleme(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)
    bil  = cv2.bilateralFilter(blur, d=9, sigmaColor=80, sigmaSpace=80)
    lab  = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ===== 5 KANAL RENK MASKELERI =====
def tum_maskeler(en):
    hsv = cv2.cvtColor(en, cv2.COLOR_BGR2HSV)
    m_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   S_MIN, V_MIN]), np.array([H_MAX, 255, 255])),
        cv2.inRange(hsv, np.array([180 - H_MAX, S_MIN, V_MIN]), np.array([180, 255, 255]))
    )
    ycc = cv2.cvtColor(en, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    m_ycc = np.zeros_like(Y, np.uint8)
    m_ycc[(Cr > CR_MIN) & (Cb < CB_MAX) & (Y > 50)] = 255
    rgb = cv2.cvtColor(en, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0].astype(float)
    G = rgb[:, :, 1].astype(float)
    B = rgb[:, :, 2].astype(float)
    m_rgb = np.zeros(en.shape[:2], np.uint8)
    m_rgb[(R > 150) & (R > G) & (G >= B) & (R - B > 40)] = 255
    lab2 = cv2.cvtColor(en, cv2.COLOR_BGR2LAB)
    L, A, Bch = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]
    m_lab = np.zeros_like(L, np.uint8)
    m_lab[(L > 80) & (A > 135) & (Bch > 130)] = 255
    m_wht = np.zeros(en.shape[:2], np.uint8)
    m_wht[(R > 200) & (G > 150) & (B < 180) & (R > B + 30)] = 255

    tot = (m_hsv.astype(np.uint16) + m_ycc.astype(np.uint16) +
           m_rgb.astype(np.uint16) + m_lab.astype(np.uint16) + m_wht.astype(np.uint16))
    return tot

# ===== MORFOLOJIK ISLEME =====
def morfo_ve_temizle(combined):
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_SIZE, K_SIZE))
    op     = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kk, iterations=1)
    cl     = cv2.morphologyEx(op,       cv2.MORPH_CLOSE, kk, iterations=3)
    filled = (binary_fill_holes(cl > 127) * 255).astype(np.uint8)
    kd     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(filled, kd, iterations=1)
    sonuc  = np.zeros_like(dilated)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            cv2.drawContours(sonuc, [c], -1, 255, -1)
    return sonuc

# ===== TAM PIPELINE =====
def segmente_et(img_bgr):
    en       = on_isleme(img_bgr)
    tot      = tum_maskeler(en)
    combined = np.where(tot >= VOTE_THR * 255, 255, 0).astype(np.uint8)
    return morfo_ve_temizle(combined)

# ===== ADIM ADIM PIPELINE (gorsellestirme icin) =====
def segmente_et_adim_adim(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)
    bil  = cv2.bilateralFilter(blur, d=9, sigmaColor=80, sigmaSpace=80)
    lab_tmp = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
    clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_tmp[:, :, 0] = clahe_obj.apply(lab_tmp[:, :, 0])
    en = cv2.cvtColor(lab_tmp, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(en, cv2.COLOR_BGR2HSV)
    m_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([H_MAX, 255, 255])),
        cv2.inRange(hsv, np.array([180 - H_MAX, S_MIN, V_MIN]), np.array([180, 255, 255]))
    )
    ycc = cv2.cvtColor(en, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    m_ycc = np.zeros_like(Y, np.uint8)
    m_ycc[(Cr > CR_MIN) & (Cb < CB_MAX) & (Y > 50)] = 255
    rgb = cv2.cvtColor(en, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0].astype(float); G = rgb[:, :, 1].astype(float); B = rgb[:, :, 2].astype(float)
    m_rgb = np.zeros(en.shape[:2], np.uint8)
    m_rgb[(R > 150) & (R > G) & (G >= B) & (R - B > 40)] = 255
    lab2 = cv2.cvtColor(en, cv2.COLOR_BGR2LAB)
    L, A, Bch = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]
    m_lab = np.zeros_like(L, np.uint8)
    m_lab[(L > 80) & (A > 135) & (Bch > 130)] = 255
    m_wht = np.zeros(en.shape[:2], np.uint8)
    m_wht[(R > 200) & (G > 150) & (B < 180) & (R > B + 30)] = 255

    tot = (m_hsv.astype(np.uint16) + m_ycc.astype(np.uint16) +
           m_rgb.astype(np.uint16) + m_lab.astype(np.uint16) + m_wht.astype(np.uint16))
    combined = np.where(tot >= VOTE_THR * 255, 255, 0).astype(np.uint8)

    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_SIZE, K_SIZE))
    op = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kk, iterations=1)
    cl = cv2.morphologyEx(op,       cv2.MORPH_CLOSE, kk, iterations=3)
    filled = (binary_fill_holes(cl > 127) * 255).astype(np.uint8)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(filled, kd, iterations=1)
    sonuc = np.zeros_like(dilated)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            cv2.drawContours(sonuc, [c], -1, 255, -1)

    return {
        "orijinal":    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "gaussian":    cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),
        "bilateral":   cv2.cvtColor(bil,  cv2.COLOR_BGR2RGB),
        "clahe":       cv2.cvtColor(en,   cv2.COLOR_BGR2RGB),
        "m_hsv":       m_hsv,
        "m_ycc":       m_ycc,
        "m_rgb":       m_rgb,
        "m_lab":       m_lab,
        "m_wht":       m_wht,
        "oylama":      combined,
        "morph_ac":    op,
        "morph_kapat": cl,
        "doldurma":    filled,
        "final":       sonuc,
    }

# ===== PARAMETRE KARSILASTIRMA ICIN YARDIMCI =====
def segmente_et_vote(img_bgr, vote_thr):
    en  = on_isleme(img_bgr)
    tot = tum_maskeler(en)
    combined = np.where(tot >= vote_thr * 255, 255, 0).astype(np.uint8)
    return morfo_ve_temizle(combined)

# ===== METRIKLER =====
def iou_hesapla(pred, gt):
    pb = pred > 127; gb = gt > 127
    return float(np.sum(pb & gb)) / (float(np.sum(pb | gb)) + 1e-7)

def dice_hesapla(pred, gt):
    pb = pred > 127; gb = gt > 127
    return 2.0 * float(np.sum(pb & gb)) / (float(np.sum(pb)) + float(np.sum(gb)) + 1e-7)

# ===== ANA DEGERLENDIRME =====
print("Goruntüler yükleniyor...")
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
print(f"Toplam gorsel: {len(gorseller)}")

iou_list  = []
dice_list = []

for g in gorseller:
    img = cv2.imread(str(g))
    gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        continue
    pred = segmente_et(img)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
    iou_list.append(iou_hesapla(pred, gt))
    dice_list.append(dice_hesapla(pred, gt))

ort_iou  = float(np.mean(iou_list))
ort_dice = float(np.mean(dice_list))
med_iou  = float(np.median(iou_list))
max_iou  = float(np.max(iou_list))

print(f"\n{'='*55}")
print(f"  Dataset        : V1(IoU>0.70) + bitti/V3+V4(IoU>0.20)")
print(f"  Gorsel sayisi  : {len(iou_list)}")
print(f"  Ortalama IoU   : {ort_iou:.4f}  ({ort_iou*100:.1f}%)")
print(f"  Medyan  IoU    : {med_iou:.4f}  ({med_iou*100:.1f}%)")
print(f"  Max     IoU    : {max_iou:.4f}  ({max_iou*100:.1f}%)")
print(f"  Ortalama Dice  : {ort_dice:.4f}  ({ort_dice*100:.1f}%)")
print(f"  IoU >= 0.80    : {sum(1 for x in iou_list if x>=0.80)} gorsel")
print(f"{'='*55}")

# ===== ORNEK GORSEL CIKTI (6 gorsel) =====
print("\nOrnek gorseller olusturuluyor...")
idx_liste = np.linspace(0, len(gorseller) - 1, 6, dtype=int)
fig, axes = plt.subplots(6, 4, figsize=(18, 24))
fig.suptitle(
    f"BIM320 - Alev Segmentasyonu  |  fire_merged (Nihai Dataset, N=1486)\n"
    f"Ortalama IoU = {ort_iou:.3f} ({ort_iou*100:.1f}%)  |  "
    f"Ortalama Dice = {ort_dice:.3f}  |  N={len(iou_list)}",
    fontsize=13, fontweight="bold"
)

for row, idx in enumerate(idx_liste):
    g        = gorseller[idx]
    img_bgr  = cv2.imread(str(g))
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt       = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    pred     = segmente_et(img_bgr)
    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

    pb = pred > 127; gb = gt > 127
    hata = np.zeros((*img_rgb.shape[:2], 3), np.uint8)
    hata[pb & gb]  = [0,   200,   0]
    hata[pb & ~gb] = [255, 100,   0]
    hata[~pb & gb] = [0,   100, 255]

    sc_iou  = iou_hesapla(pred, gt)
    sc_dice = dice_hesapla(pred, gt)

    axes[row, 0].imshow(img_rgb)
    axes[row, 0].set_title(f"Orijinal  [{g.stem[-8:]}]")
    axes[row, 1].imshow(gt,   cmap="gray")
    axes[row, 1].set_title("Ground Truth")
    axes[row, 2].imshow(pred, cmap="gray")
    axes[row, 2].set_title("Tahmin")
    axes[row, 3].imshow(hata)
    axes[row, 3].set_title(f"Hata Haritasi\nIoU={sc_iou:.3f}  Dice={sc_dice:.3f}")
    for ax in axes[row]:
        ax.axis("off")

tp_p = mpatches.Patch(color=(0, 0.78, 0),  label="TP - Dogru Pozitif")
fp_p = mpatches.Patch(color=(1, 0.39, 0),  label="FP - Yanlis Pozitif")
fn_p = mpatches.Patch(color=(0, 0.39, 1),  label="FN - Yanlis Negatif")
fig.legend(handles=[tp_p, fp_p, fn_p], loc="lower center",
           ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.02, 1, 1])
kayit = OUT_DIR / "segmentation_results.png"
plt.savefig(str(kayit), dpi=120, bbox_inches="tight")
plt.close()
print(f"Gorsel kaydedildi: {kayit}")

# ===== PIPELINE ADIM ADIM GORSELLESTIRME =====
print("\nPipeline adim adim gorsellestirmesi olusturuluyor...")
ornek_g   = gorseller[len(gorseller) // 3]
ornek_img = cv2.imread(str(ornek_g))
ornek_gt  = cv2.imread(str(MASKS_DIR / (ornek_g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
adimlar   = segmente_et_adim_adim(ornek_img)
if ornek_gt is not None and adimlar["final"].shape != ornek_gt.shape:
    ornek_gt = cv2.resize(ornek_gt, (adimlar["final"].shape[1], adimlar["final"].shape[0]))

# 2 satir x 7 sutun: 14 panel
adim_listesi = [
    ("1. Orijinal",          adimlar["orijinal"],    False),
    ("2. GaussianBlur",      adimlar["gaussian"],    False),
    ("3. Bilateral Filter",  adimlar["bilateral"],   False),
    ("4. CLAHE",             adimlar["clahe"],       False),
    ("5. HSV Maskesi",       adimlar["m_hsv"],       True),
    ("6. YCrCb Maskesi",     adimlar["m_ycc"],       True),
    ("7. RGB Maskesi",       adimlar["m_rgb"],       True),
    ("8. LAB Maskesi",       adimlar["m_lab"],       True),
    ("9. Parlaklik Maskesi", adimlar["m_wht"],       True),
    ("10. Oylama (>=3/5)",   adimlar["oylama"],      True),
    ("11. Morph Open",       adimlar["morph_ac"],    True),
    ("12. Morph Close",      adimlar["morph_kapat"], True),
    ("13. Delik Doldurma",   adimlar["doldurma"],    True),
    ("14. Final Tahmin",     adimlar["final"],       True),
]

fig2, axes2 = plt.subplots(2, 7, figsize=(21, 7))
fig2.suptitle(
    f"BIM320 - Alev Segmentasyonu Pipeline Adimlari  [{ornek_g.stem}]",
    fontsize=13, fontweight="bold"
)
axes2_flat = axes2.flatten()
for i, (baslik, goruntu, gri) in enumerate(adim_listesi):
    ax = axes2_flat[i]
    ax.imshow(goruntu, cmap="gray" if gri else None)
    ax.set_title(baslik, fontsize=8)
    ax.axis("off")

plt.tight_layout()
kayit2 = OUT_DIR / "pipeline_visualization.png"
plt.savefig(str(kayit2), dpi=120, bbox_inches="tight")
plt.close()
print(f"Pipeline gorseli kaydedildi: {kayit2}")

# ===== PARAMETRE ANALIZI: VOTE_THR = 2 vs 3 =====
print("\nParametre analizi olusturuluyor (VOTE_THR = 2 vs 3)...")
vote_degerler  = [2, 3]
ornek_idxler   = np.linspace(0, len(gorseller) - 1, 4, dtype=int)

fig3, axes3 = plt.subplots(len(ornek_idxler), 2 + len(vote_degerler),
                            figsize=(14, 4 * len(ornek_idxler)))
fig3.suptitle(
    "BIM320 - Parametre Analizi: Oylama Esigi  (VOTE_THR = 2 vs 3)\n"
    "Daha yuksek esik = daha az gurultu, daha dusuk esik = daha fazla kapsam",
    fontsize=12, fontweight="bold"
)

basliklar = ["Orijinal", "Ground Truth", "VOTE_THR = 2", "VOTE_THR = 3"]
for col, b in enumerate(basliklar):
    axes3[0, col].set_title(b, fontsize=10, fontweight="bold")

for row, idx in enumerate(ornek_idxler):
    g   = gorseller[idx]
    img = cv2.imread(str(g))
    gt  = cv2.imread(str(MASKS_DIR / (g.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    axes3[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes3[row, 0].axis("off")
    axes3[row, 1].imshow(gt, cmap="gray")
    axes3[row, 1].axis("off")
    for col, v in enumerate(vote_degerler, start=2):
        pred_v = segmente_et_vote(img, v)
        gt_r   = cv2.resize(gt, (pred_v.shape[1], pred_v.shape[0])) if pred_v.shape != gt.shape else gt
        sc     = iou_hesapla(pred_v, gt_r)
        axes3[row, col].imshow(pred_v, cmap="gray")
        axes3[row, col].set_xlabel(f"IoU = {sc:.3f}", fontsize=9)
        axes3[row, col].axis("off")

plt.tight_layout()
kayit3 = OUT_DIR / "parameter_analysis.png"
plt.savefig(str(kayit3), dpi=120, bbox_inches="tight")
plt.close()
print(f"Parametre analizi kaydedildi: {kayit3}")
print("TAMAMLANDI.")
