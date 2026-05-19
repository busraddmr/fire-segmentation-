"""
Slayt gorseli: Adim 2 (Bilateral) + Adim 3 (CLAHE) on isleme adimlarinin karsilastirmasi
Cikti: outputs/report_preprocess.png
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
MASKS_DIR  = BASE_DIR / "dataset" / "masks"
OUT_DIR    = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Gece/karanlık sahneli iyi bir ornek sec
gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
# Ortalardan bir gorsel sec - genellikle daha cok cesitlilik var
idx = len(gorseller) // 2
img_bgr = cv2.imread(str(gorseller[idx]))
while img_bgr is None and idx < len(gorseller):
    idx += 1
    img_bgr = cv2.imread(str(gorseller[idx]))

img_name = gorseller[idx].stem

# ---- On isleme adimlari ----
step0_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)
step1_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

bil = cv2.bilateralFilter(blur, d=9, sigmaColor=80, sigmaSpace=80)
step2_rgb = cv2.cvtColor(bil, cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
final_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
step3_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

# ---- L kanalı karşılaştırması (CLAHE etkisini göstermek için) ----
lab_before = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
lab_after  = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2LAB)
L_before = lab_before[:, :, 0]
L_after  = lab_after[:, :, 0]

# ---- Fark haritasi ----
diff_rgb = cv2.absdiff(step2_rgb, step3_rgb)
diff_enhanced = np.clip(diff_rgb * 4, 0, 255).astype(np.uint8)

# ============================================================
# FIGUR DUZENI: 2 satir x 4 sutun
# Ust: Orijinal | Gaussian | Bilateral | CLAHE
# Alt: L kanali oncesi | L kanali sonrasi | Fark (x4) | histogram overlay
# ============================================================
fig = plt.figure(figsize=(16, 9), facecolor="#0f1923")
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.38, wspace=0.08,
                        left=0.03, right=0.97,
                        top=0.88, bottom=0.06)

TITLE_COLOR  = "#e8eaf6"
LABEL_COLOR  = "#b0bec5"
BOX_ACTIVE   = "#1a3a5c"  # vurgulanan adimin arka plani
BOX_NORMAL   = "#152030"
BORDER_ACTIVE = "#e53935"  # kirmizi cerceve
BORDER_NORMAL = "#37474f"

# Ust satir: 4 gorsel
ustler = [
    (step0_rgb, "Adım 1 · Orijinal",            False, BOX_NORMAL, BORDER_NORMAL),
    (step1_rgb, "Adım 1 · Gaussian Blur\nd=(5×5)  σ=1.0", False, BOX_NORMAL, BORDER_NORMAL),
    (step2_rgb, "Adım 2 · Bilateral Filter\nd=9  σColor=80  σSpace=80", False, BOX_ACTIVE, BORDER_ACTIVE),
    (step3_rgb, "Adım 3 · CLAHE\nclipLimit=2.5  tileGrid=(8×8)", False, BOX_ACTIVE, BORDER_ACTIVE),
]

for col, (img, title, gray, bg, border) in enumerate(ustler):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img, cmap="gray" if gray else None, aspect="auto")
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(2.5 if border == BORDER_ACTIVE else 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=TITLE_COLOR, fontsize=9.5, fontweight="bold",
                 pad=6, linespacing=1.5)

# Alt satir: L oncesi, L sonrasi, fark, histogram
alt_veriler = [
    (L_before, "L Kanalı · Bilateral Sonrası\n(CLAHE öncesi)", True,  BOX_NORMAL, BORDER_NORMAL),
    (L_after,  "L Kanalı · CLAHE Sonrası\n(kontrast artırılmış)", True,  BOX_ACTIVE, BORDER_ACTIVE),
    (diff_enhanced, "Fark Haritası (×4 güçlendirilmiş)\nCLAHE'nin etki bölgeleri", False, BOX_ACTIVE, BORDER_ACTIVE),
]

for col, (img, title, gray, bg, border) in enumerate(alt_veriler):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(img, cmap="gray" if gray else None, aspect="auto")
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor(border)
        spine.set_linewidth(2.5 if border == BORDER_ACTIVE else 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=TITLE_COLOR, fontsize=9.5, fontweight="bold",
                 pad=6, linespacing=1.5)

# Histogram karsilastirmasi (son sutun)
ax_hist = fig.add_subplot(gs[1, 3])
ax_hist.set_facecolor("#0d1f2d")
for spine in ax_hist.spines.values():
    spine.set_edgecolor(BORDER_ACTIVE)
    spine.set_linewidth(2.5)

hist_before, bins = np.histogram(L_before.ravel(), bins=64, range=(0, 256))
hist_after,  _    = np.histogram(L_after.ravel(),  bins=64, range=(0, 256))
centers = (bins[:-1] + bins[1:]) / 2

# Normalize et: her ikisi de 0-1 araligina
norm_before = hist_before / hist_before.max()
norm_after  = hist_after  / hist_after.max()

ax_hist.fill_between(centers, norm_before, alpha=0.35, color="#78909c")
ax_hist.plot(centers, norm_before, color="#90a4ae", linewidth=1.6,
             label="Bilateral sonrası", alpha=0.9)
ax_hist.fill_between(centers, norm_after, alpha=0.30, color="#ef5350")
ax_hist.plot(centers, norm_after, color="#ef5350", linewidth=2.2,
             label="CLAHE sonrası", alpha=1.0)

# Aciklama oku: CLAHE dagilimi yaydi
ax_hist.annotate("CLAHE dağılımı\nyaydı →",
                 xy=(185, norm_after[int(185*64/256)]),
                 xytext=(140, 0.72),
                 color="#ffcdd2", fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="#ef5350", lw=1.2))

ax_hist.set_title("L Kanalı Histogramı\n(normalize edilmiş)", color=TITLE_COLOR,
                  fontsize=9.5, fontweight="bold", pad=6, linespacing=1.5)
ax_hist.tick_params(colors=LABEL_COLOR, labelsize=7)
ax_hist.set_xlabel("Piksel değeri (0–255)", color=LABEL_COLOR, fontsize=7.5)
ax_hist.set_ylabel("Normalize yoğunluk", color=LABEL_COLOR, fontsize=7.5)
ax_hist.legend(fontsize=7.5, facecolor="#152030", edgecolor="#37474f",
               labelcolor=LABEL_COLOR, loc="upper left")
ax_hist.set_xlim(0, 255)
ax_hist.set_ylim(0, 1.12)
ax_hist.spines["bottom"].set_color(LABEL_COLOR)
ax_hist.spines["left"].set_color(LABEL_COLOR)

# Baslik
fig.suptitle(
    f"Ön İşleme Pipeline · Adım 2: Bilateral Filter  /  Adım 3: CLAHE\n"
    f"Görsel: {img_name}",
    color=TITLE_COLOR, fontsize=13, fontweight="bold", y=0.96
)

# Alt not
fig.text(0.5, 0.01,
         "Bilateral: Kenarları koruyarak gürültü azaltma  |  "
         "CLAHE: Yalnızca L kanalında yerel kontrast artırımı  (H & S kanalları sabit)",
         ha="center", color=LABEL_COLOR, fontsize=8.5, style="italic")

kayit = OUT_DIR / "report_preprocess.png"
plt.savefig(str(kayit), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Kaydedildi: {kayit}")
