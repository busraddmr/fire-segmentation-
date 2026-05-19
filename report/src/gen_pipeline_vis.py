"""
Pipeline adim adim gorsellestirme - README icin duzenlenmis versiyon
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from scipy.ndimage import binary_fill_holes

BASE_DIR   = Path(__file__).parent.parent / "dataset"
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR  = BASE_DIR / "masks"
OUT_DIR    = Path(__file__).parent.parent / "outputs"

H_MAX  = 35;  S_MIN  = 100; V_MIN  = 120
CR_MIN = 150; CB_MAX = 140; K_SIZE = 9
MIN_AREA = 200; VOTE_THR = 3

PHASE_COLORS = {
    "faz1": "#1a5276",   # koyu mavi
    "faz2": "#0e3d52",   # lacivert
    "faz3": "#922b21",   # kirmizi
    "faz4": "#1e8449",   # yesil
}

def on_isleme(img_bgr):
    blur = cv2.GaussianBlur(img_bgr, (5, 5), sigmaX=1.0)
    bil  = cv2.bilateralFilter(blur, d=9, sigmaColor=80, sigmaSpace=80)
    lab  = cv2.cvtColor(bil, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), blur, bil

def segmente_et_adim_adim(img_bgr):
    en, blur, bil = on_isleme(img_bgr)
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
        "orijinal":  cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        "gaussian":  cv2.cvtColor(blur, cv2.COLOR_BGR2RGB),
        "bilateral": cv2.cvtColor(bil,  cv2.COLOR_BGR2RGB),
        "clahe":     cv2.cvtColor(en,   cv2.COLOR_BGR2RGB),
        "m_hsv": m_hsv, "m_ycc": m_ycc, "m_rgb": m_rgb,
        "m_lab": m_lab, "m_wht": m_wht,
        "oylama": combined, "morph_ac": op, "morph_kapat": cl,
        "doldurma": filled, "final": sonuc,
    }

gorseller = sorted(IMAGES_DIR.glob("*.jpg"))
ornek_g   = gorseller[len(gorseller) // 3]
ornek_img = cv2.imread(str(ornek_g))
adimlar   = segmente_et_adim_adim(ornek_img)

# Faz tanimlari: (baslik, anahtar, gri_mi, adim_no)
fazlar = [
    {
        "renk": PHASE_COLORS["faz1"],
        "etiket": "FAZ 1 — ÖN İŞLEME",
        "adimlar": [
            ("1. Gauss\nFiltresi",    "gaussian",  False),
            ("2. Bilateral\nFiltre",  "bilateral", False),
            ("3. CLAHE",              "clahe",     False),
        ],
    },
    {
        "renk": PHASE_COLORS["faz2"],
        "etiket": "FAZ 2 — RENK MASKELEME",
        "adimlar": [
            ("4. HSV",      "m_hsv", True),
            ("5. YCrCb",    "m_ycc", True),
            ("6. RGB",      "m_rgb", True),
            ("7. LAB",      "m_lab", True),
            ("8. Parlaklik","m_wht", True),
        ],
    },
    {
        "renk": PHASE_COLORS["faz3"],
        "etiket": "FAZ 3 — OYLAMA",
        "adimlar": [
            ("9. Majority\nVoting\n(3/5)", "oylama", True),
        ],
    },
    {
        "renk": PHASE_COLORS["faz4"],
        "etiket": "FAZ 4 — MORFOLOJİ",
        "adimlar": [
            ("10. Morph\nOpen",       "morph_ac",    True),
            ("11. Morph\nClose",      "morph_kapat", True),
            ("12. Delik\nDoldurma",   "doldurma",    True),
            ("13. Dilatasyon\n+ Kontur", "final",    True),
        ],
    },
]

# Toplam sutun sayisi (orijinal + her faz adimi)
n_cols = 1 + sum(len(f["adimlar"]) for f in fazlar)
fig_w  = max(22, n_cols * 1.8)
fig = plt.figure(figsize=(fig_w, 6.5), facecolor="#111111")

# Ustbilgi
fig.text(0.5, 0.97,
         f"14-Adımlı Deterministik Pipeline  [{ornek_g.stem}]",
         ha="center", va="top", fontsize=14, fontweight="bold",
         color="white")
fig.text(0.01, 0.97, "GİRDİ: Ham RGB (640×640 px)",
         ha="left", va="top", fontsize=9, color="#aaaaaa")
fig.text(0.99, 0.97, "ÇIKTI: İkili Maske (0/255)",
         ha="right", va="top", fontsize=9, color="#aaaaaa")

TOP    = 0.88
BOTTOM = 0.08
LEFT   = 0.01
RIGHT  = 0.99
GAP    = 0.008   # adimlar arasi bosluk
HEADER = 0.10    # baslik yuksekligi (fig koordinatlarinda)

# Her adim icin genislik hesapla
col_items = [("orijinal", None, False, "Orijinal")] + [
    (a[1], f["renk"], a[2], a[0])
    for f in fazlar for a in f["adimlar"]
]
n_items = len(col_items)
total_w = RIGHT - LEFT - (n_items - 1) * GAP
col_w   = total_w / n_items

def col_x(i):
    return LEFT + i * (col_w + GAP)

# Faz arka plan bloklari
faz_col_start = 1  # orijinal haric
for f in fazlar:
    n = len(f["adimlar"])
    i0 = faz_col_start
    i1 = faz_col_start + n - 1
    x0 = col_x(i0) - GAP / 2
    x1 = col_x(i1) + col_w + GAP / 2
    rect = plt.Rectangle(
        (x0, BOTTOM - 0.01), x1 - x0, (TOP - BOTTOM + 0.01),
        transform=fig.transFigure, figure=fig,
        facecolor=f["renk"], alpha=0.25, zorder=0
    )
    fig.add_artist(rect)
    # Faz baslik bandı
    rect2 = plt.Rectangle(
        (x0, TOP - HEADER + 0.01), x1 - x0, HEADER - 0.01,
        transform=fig.transFigure, figure=fig,
        facecolor=f["renk"], alpha=0.85, zorder=1
    )
    fig.add_artist(rect2)
    fig.text((x0 + x1) / 2, TOP - HEADER / 2 + 0.01,
             f["etiket"], ha="center", va="center",
             fontsize=8.5, fontweight="bold", color="white", zorder=2)
    faz_col_start += n

# Gorsel axesleri
img_top    = TOP - HEADER + 0.005
img_bottom = BOTTOM

for i, (key, renk, gri, baslik) in enumerate(col_items):
    x = col_x(i)
    ax = fig.add_axes([x, img_bottom, col_w, img_top - img_bottom])
    goruntu = adimlar[key]
    ax.imshow(goruntu, cmap="gray" if gri else None)

    # Adim no badge
    step_no = i  # 0 = orijinal
    badge_color = "#555555" if step_no == 0 else renk
    ax.set_title(baslik, fontsize=7.5, color="white", pad=3,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_color, alpha=0.85))
    ax.axis("off")

    # Ok (son eleman haric)
    if i < n_items - 1:
        ax_next_x = col_x(i + 1)
        arrow_x   = x + col_w + 0.001
        arrow_xend = ax_next_x - 0.001
        fig.text(
            (arrow_x + arrow_xend) / 2,
            (img_top + img_bottom) / 2,
            "→", ha="center", va="center",
            fontsize=11, color="#cccccc", zorder=5
        )

plt.savefig(str(OUT_DIR / "pipeline_visualization.png"),
            dpi=130, bbox_inches="tight", facecolor="#111111")
plt.close()
print(f"Kaydedildi: {OUT_DIR / 'pipeline_visualization.png'}")
