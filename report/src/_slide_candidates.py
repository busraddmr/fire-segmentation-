import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

picks = [
    ("dataset/images/fire_00664.jpg", "1 · fire_00664\nTepe yangını — duman bulutu + yeşil ağaçlar\n✓ Renk spektrumu  ✓ Duman örtüşmesi\n✓ Yanlış pozitif (ağaç)  ✓ Perspektif/ölçek"),
    ("dataset/images/fire_01402.jpg", "2 · fire_01402\nAlev yoğun sise gömülmüş\n✓✓✓ Duman örtüşmesi  ✓ Düşük ışık\n✓ Renk spektrumu (sarı→kırmızı)"),
    ("dataset/images/fire_00683.jpg", "3 · fire_00683\nOrman yangını — ev + ağaçlar arka planda\n✓ Yanlış pozitif kaynağı (ev, ağaç)\n✓ Duman örtüşmesi  ✓ Perspektif"),
    ("dataset/images/fire_00994.jpg", "4 · fire_00994\nEndüstriyel baca — siyah duman kuyruğu\n✓ Siyah duman örtüşmesi  ✓ Renk spektrumu\n✓ Perspektif/ölçek (küçük uzak alev)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for ax, (path, label) in zip(axes.flat, picks):
    img = cv2.imread(path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(label, fontsize=9.5, loc="left", pad=8)
    ax.axis("off")

plt.suptitle("Slayt Adayları — Seçim için 1–4 arası belirt", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/_slide_candidates.png", dpi=120, bbox_inches="tight")
print("Kaydedildi: outputs/_slide_candidates.png")
