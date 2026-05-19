import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

imgs = [
    ("dataset/images/fire_00664.jpg", "Tepe Yangını\n(Duman + Ölçek)"),
    ("dataset/images/fire_01402.jpg", "Duman Örtüşmesi\n(Loş Sahne)"),
    ("dataset/images/fire_00683.jpg", "Orman Yangını\n(Yanlış Pozitif Kaynağı)"),
    ("dataset/images/fire_00994.jpg", "Endüstriyel Alev\n(Perspektif + Duman)"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.patch.set_facecolor("#1a1a1a")

for ax, (path, label) in zip(axes.flat, imgs):
    img = cv2.imread(path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(label, fontsize=11, color="white", labelpad=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
        spine.set_linewidth(1.5)

plt.subplots_adjust(wspace=0.06, hspace=0.12)
plt.savefig("outputs/slide_gorsel.png", dpi=150, bbox_inches="tight",
            facecolor="#1a1a1a")
print("Kaydedildi: outputs/slide_gorsel.png")
