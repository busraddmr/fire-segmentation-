"""
Slayt gorseli: 5 Kanalli Renk Maskeleme
Cikti: outputs/report_masks.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
MASKS_DIR  = BASE_DIR / "dataset" / "masks"
OUT_DIR    = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Parametreler (segmentation.py ile ayni)
H_MAX  = 35;  S_MIN = 100;  V_MIN = 120
CR_MIN = 150; CB_MAX = 140
VOTE_THR = 3

# On isleme (segmentation.py ile ayni)
def on_isleme(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)
    bil  = cv2.bilateralFilter(blur, d=9, sigmaColor=80, sigmaSpace=80)
    lab  = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Gorsel sec - hem gunduz hem gece gormek icin ortadan biraz sonrasini al
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
idx = len(gorseller) // 2
img_bgr = cv2.imread(str(gorseller[idx]))
while img_bgr is None:
    idx += 1
    img_bgr = cv2.imread(str(gorseller[idx]))

img_name = gorseller[idx].stem
en = on_isleme(img_bgr)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 5 Maske hesapla
hsv = cv2.cvtColor(en, cv2.COLOR_BGR2HSV)
m_hsv = cv2.bitwise_or(
    cv2.inRange(hsv, np.array([0, S_MIN, V_MIN]), np.array([H_MAX, 255, 255])),
    cv2.inRange(hsv, np.array([180 - H_MAX, S_MIN, V_MIN]), np.array([180, 255, 255]))
)

ycc = cv2.cvtColor(en, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]
m_ycc = np.zeros_like(Y, np.uint8)
m_ycc[(Cr > CR_MIN) & (Cb < CB_MAX) & (Y > 50)] = 255

rgb = cv2.cvtColor(en, cv2.COLOR_BGR2RGB)
R = rgb[:,:,0].astype(float); G = rgb[:,:,1].astype(float); B = rgb[:,:,2].astype(float)
m_rgb = np.zeros(en.shape[:2], np.uint8)
m_rgb[(R > 150) & (R > G) & (G >= B) & (R - B > 40)] = 255

lab2 = cv2.cvtColor(en, cv2.COLOR_BGR2LAB)
L, A, Bch = lab2[:,:,0], lab2[:,:,1], lab2[:,:,2]
m_lab = np.zeros_like(L, np.uint8)
m_lab[(L > 80) & (A > 135) & (Bch > 130)] = 255

m_wht = np.zeros(en.shape[:2], np.uint8)
m_wht[(R > 200) & (G > 150) & (B < 180) & (R > B + 30)] = 255

tot = (m_hsv.astype(np.uint16) + m_ycc.astype(np.uint16) +
       m_rgb.astype(np.uint16) + m_lab.astype(np.uint16) + m_wht.astype(np.uint16))
m_vote = np.where(tot >= VOTE_THR * 255, 255, 0).astype(np.uint8)

# Renkli overlay: maskeyi orijinal uzerine bindir
def renkli_overlay(img_rgb, mask, renk_bgr):
    overlay = img_rgb.copy()
    overlay[mask > 127] = (overlay[mask > 127] * 0.35 + np.array(renk_bgr) * 0.65).astype(np.uint8)
    return overlay

# Oylama overlay: kac kanal oyladiysa o kadar parlak
vote_norm = (tot / (5 * 255.0) * 255).astype(np.uint8)
vote_color = np.zeros((*img_rgb.shape[:2], 3), np.uint8)
vote_color[:,:,0] = vote_norm  # kirmizi kanal
vote_color[:,:,1] = (vote_norm * 0.3).astype(np.uint8)

oy_overlay = img_rgb.copy()
oy_mask = tot > 0
oy_overlay[oy_mask] = (
    oy_overlay[oy_mask] * 0.4 +
    vote_color[oy_mask] * 0.6
).astype(np.uint8)

# Kanal renkleri (slide ile uyumlu)
KANALLAR = [
    (m_hsv,  "HSV Maskesi\nH∈[0,35]∪[145,180]  S≥100  V≥120", "#e65100", (230,80,0)),
    (m_ycc,  "YCrCb Maskesi\nCr>150  Cb<140  Y>50",            "#f57f17", (245,127,23)),
    (m_rgb,  "RGB Maskesi\nR>150  R>G≥B  R−B>40",              "#6a1b9a", (106,27,154)),
    (m_lab,  "LAB Maskesi\nL>80  A>135  B>130",                "#1b5e20", (27,94,32)),
    (m_wht,  "Parlaklik Maskesi\nR>200  G>150  B<180  R>B+30",  "#006064", (0,96,100)),
]

# ============================================================
# LAYOUT: 3 satir x 4 sutun
#   Satir 0: Orijinal (genis) | HSV | YCrCb | RGB
#   Satir 1: Orijinal (genis) | LAB | Parlaklik | Oylama>=3/5
#   Satir 2: tum maskeler overlay karsilastirmasi (genis tek panel)
# ============================================================
BG      = "#0f1923"
TC      = "#e8eaf6"   # baslik rengi
LC      = "#b0bec5"   # etiket rengi

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.38, wspace=0.07,
                        left=0.02, right=0.98,
                        top=0.84, bottom=0.05)

# Orijinal - tek kare (0,0)
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.imshow(img_rgb, aspect="auto")
ax_orig.set_facecolor("#152030")
for sp in ax_orig.spines.values():
    sp.set_edgecolor("#37474f"); sp.set_linewidth(1.5)
ax_orig.set_xticks([]); ax_orig.set_yticks([])
ax_orig.set_title(f"Orijinal Görsel\n{img_name}", color=TC,
                  fontsize=10, fontweight="bold", pad=7)

# 5 Maske: satir 0 saginda 3, satir 1 saginda 2
kanal_pozisyonlar = [
    (0, 1), (0, 2), (0, 3),   # HSV, YCrCb, RGB
    (1, 1), (1, 2),            # LAB, Parlaklik
]

for i, ((row, col), (mask, baslik, hex_renk, rgb_renk)) in enumerate(
        zip(kanal_pozisyonlar, KANALLAR)):
    ax = fig.add_subplot(gs[row, col])
    overlay = renkli_overlay(img_rgb, mask, rgb_renk)
    ax.imshow(overlay, aspect="auto")
    ax.set_facecolor("#152030")
    for sp in ax.spines.values():
        sp.set_edgecolor(hex_renk); sp.set_linewidth(2.8)
    ax.set_xticks([]); ax.set_yticks([])
    pct = 100.0 * np.sum(mask > 127) / mask.size
    ax.set_title(f"{baslik}\nKapsam: %{pct:.1f}", color=TC,
                 fontsize=8.5, fontweight="bold", pad=5, linespacing=1.45)

# Oylama sonucu (>=3/5) - (1,0) orijinalin altinda
ax_vote = fig.add_subplot(gs[1, 0])
ax_vote.imshow(oy_overlay, aspect="auto")
ax_vote.set_facecolor("#152030")
for sp in ax_vote.spines.values():
    sp.set_edgecolor("#ffd54f"); sp.set_linewidth(2.8)
ax_vote.set_xticks([]); ax_vote.set_yticks([])
pct_vote = 100.0 * np.sum(m_vote > 127) / m_vote.size
ax_vote.set_title(f"Oylama  (≥3/5)\nKapsam: %{pct_vote:.1f}",
                  color="#ffd54f", fontsize=8.5, fontweight="bold", pad=5, linespacing=1.45)

# Baslik
fig.suptitle(
    "5 Kanallı Renk Maskeleme — Her Kanalın Aynı Görüntüdeki Katkısı",
    color=TC, fontsize=13, fontweight="bold", y=0.96
)

# Alt aciklama
fig.text(0.5, 0.005,
         "Renkli alan = o kanalın tespit ettiği alev bölgesi  |  "
         "Her kanal farklı senaryoda güçlü/zayıf  →  Oylama zayıf noktaları telafi eder",
         ha="center", color=LC, fontsize=8.5, style="italic")

kayit = OUT_DIR / "report_masks.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
