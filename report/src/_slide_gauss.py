"""
Adım 1 slaytı için:
Layout 1-2-2:
  Üst (1): Ham görüntü — tam boy, sarı kutu zoom bölgesini gösterir
  Orta (2): Zoom — 3×3 | 5×5
  Alt  (2): Zoom — 7×7 | 9×9
Zoom bölgesi alev kenarında → blur farkı çıplak gözle görülür
"""
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

IMG = "dataset/images/fire_01541.jpg"
img_orig = cv2.imread(IMG)

# Hafif gürültü ekle — blur farkını belirginleştirir
np.random.seed(7)
noise = np.random.normal(0, 18, img_orig.shape).astype(np.int16)
img = np.clip(img_orig.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def to_rgb(b): return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

# Her kernel için uygula
blurs = {
    "Ham Görüntü": img,
    "3×3  σ=0.5\n(Yetersiz — gürültü kalıyor)":  cv2.GaussianBlur(img, (3,3), 0.5),
    "5×5  σ=1.0  ✓\n(Optimal — kenar korunur)":   cv2.GaussianBlur(img, (5,5), 1.0),
    "7×7  σ=1.5\n(Kenarlar bulanıklaşıyor)":       cv2.GaussianBlur(img, (7,7), 1.5),
    "9×9  σ=2.0\n(Ciddi kenar bozulması)":          cv2.GaussianBlur(img, (9,9), 2.0),
}

# Zoom bölgesi: alev kenarı (sarı-turuncu → gri geçiş)
ZY1, ZY2, ZX1, ZX2 = 95, 260, 220, 420

colors = ["#aaaaaa", "#e74c3c", "#2ecc71", "#e67e22", "#95a5a6"]

# --- Layout ---
fig = plt.figure(figsize=(12, 14))
fig.patch.set_facecolor("#111827")
gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.32, wspace=0.08,
                       height_ratios=[1.6, 1, 1])

# Üst: Ham görüntü tam boy (ortalanmış)
ax_top = fig.add_subplot(gs[0, 1:3])
ax_top.imshow(to_rgb(img))
ax_top.set_title("Ham Görüntü  (gürültülü)", fontsize=13,
                 color="#aaaaaa", fontweight="bold", pad=8)
ax_top.set_xlabel("Sarı kutu = zoom bölgesi", fontsize=10, color="#f1c40f", labelpad=5)
ax_top.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax_top.spines.values():
    sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(2.5)
rect = patches.Rectangle((ZX1, ZY1), ZX2-ZX1, ZY2-ZY1,
                          linewidth=2.5, edgecolor="#f1c40f",
                          facecolor="none", linestyle="--")
ax_top.add_patch(rect)

# Orta + Alt: zoom karşılaştırması (3×3, 5×5, 7×7, 9×9)
zoom_configs = [
    ("3×3  σ=0.5", "Yetersiz\ngürültü kalıyor",  "#e74c3c",  blurs["3×3  σ=0.5\n(Yetersiz — gürültü kalıyor)"]),
    ("5×5  σ=1.0  ✓", "Optimal\nkenar korunur",  "#2ecc71",  blurs["5×5  σ=1.0  ✓\n(Optimal — kenar korunur)"]),
    ("7×7  σ=1.5", "Kenarlar\nbulanıklaşıyor",   "#e67e22",  blurs["7×7  σ=1.5\n(Kenarlar bulanıklaşıyor)"]),
    ("9×9  σ=2.0", "Ciddi kenar\nbozulması",      "#95a5a6",  blurs["9×9  σ=2.0\n(Ciddi kenar bozulması)"]),
]

row_positions = [
    [gs[1, 0:2], gs[1, 2:4]],
    [gs[2, 0:2], gs[2, 2:4]],
]

for row_i, row in enumerate(row_positions):
    for col_i, pos in enumerate(row):
        cfg = zoom_configs[row_i * 2 + col_i]
        title, note, color, src = cfg
        zoom_patch = to_rgb(src)[ZY1:ZY2, ZX1:ZX2]
        ax = fig.add_subplot(pos)
        ax.imshow(zoom_patch, interpolation="bilinear")
        ax.set_title(title, fontsize=13, color=color, fontweight="bold", pad=7)
        ax.set_xlabel(note, fontsize=10.5, color="#cccccc", labelpad=6)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3.5)

fig.text(0.5, 0.97,
         "Gaussian Blur — Kernel Boyutu Karşılaştırması (Zoom)",
         ha="center", fontsize=15, color="white", fontweight="bold")

plt.savefig("outputs/slide_gauss.png", dpi=150, bbox_inches="tight",
            facecolor="#111827")
print("Kaydedildi: outputs/slide_gauss.png")
