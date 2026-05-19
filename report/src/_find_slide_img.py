"""
Slayt için uygun görsel bulma:
- Gece / düşük ışık (düşük ortalama parlaklık)
- Duman varlığı (gri ton piksel oranı yüksek)
- Alev renk spektrumu geniş
- Makul IoU (0.55–0.85 arası — ne çok iyi ne çok kötü)
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_DIR  = "dataset/images"
MASK_DIR = "dataset/masks"

files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

def smoke_ratio(img_bgr):
    """Gri/duman piksel oranı: düşük doygunluk + orta parlaklık."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(float)
    s, v = hsv[:,:,1], hsv[:,:,2]
    smoke = ((s < 50) & (v > 60) & (v < 200))
    return smoke.sum() / smoke.size

def night_score(img_bgr):
    """Düşük ortalama parlaklık → gece/karanlık sahne."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return 1.0 - gray.mean() / 255.0   # yüksek = daha karanlık

def hue_range(img_bgr, mask):
    """Alev bölgesindeki hue aralığı → renk spektrumu genişliği."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0][mask > 0]
    if len(h) == 0:
        return 0
    return int(h.max()) - int(h.min())

def iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt   = gt_mask > 0
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / union if union > 0 else 0.0

# Basit eşikli tahmin (pipeline yerine hızlı yaklaşım)
def quick_pred(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    return ((h <= 35) | (h >= 145)) & (s >= 100) & (v >= 120)

candidates = []
np.random.seed(42)
sample = np.random.choice(len(files), min(400, len(files)), replace=False)

for idx in sample:
    fname = files[idx]
    img_path  = os.path.join(IMG_DIR,  fname)
    mask_path = os.path.join(MASK_DIR, fname.replace(".jpg", ".png"))
    if not os.path.exists(mask_path):
        continue

    img  = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue

    sm   = smoke_ratio(img)
    ns   = night_score(img)
    hr   = hue_range(img, mask)
    pred = quick_pred(img).astype(np.uint8) * 255
    iou_v = iou(pred, mask)

    # Seçim kriterleri
    if 0.45 < iou_v < 0.88 and sm > 0.08 and ns > 0.45 and hr > 10:
        score = sm * 0.4 + ns * 0.4 + min(hr, 30) / 30.0 * 0.2
        candidates.append((score, fname, img, mask, sm, ns, hr, iou_v))

candidates.sort(reverse=True)
top = candidates[:9]
print(f"Toplam aday: {len(candidates)}, gösterilen: {len(top)}")

# 3×3 ızgara olarak kaydet
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, (score, fname, img, mask, sm, ns, hr, iou_v) in zip(axes.flat, top):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    ax.set_title(f"{fname}\nDuman:{sm:.2f} Gece:{ns:.2f} Hue:{hr} IoU:{iou_v:.2f}",
                 fontsize=7)
    ax.axis("off")

for ax in axes.flat[len(top):]:
    ax.axis("off")

plt.suptitle("Slayt İçin Aday Görseller (duman + gece + renk spektrumu)", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/_candidate_preview.png", dpi=100, bbox_inches="tight")
print("Kaydedildi: outputs/_candidate_preview.png")
for i, (score, fname, *_) in enumerate(top):
    print(f"  [{i+1}] {fname}  (skor={score:.3f})")
