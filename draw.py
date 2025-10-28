import matplotlib.pyplot as plt

# color maps for color and markerfacecolor using fancy colors,  makerfacecolor should be darker than color
color_maps = {
    'accuracy': ('#e76f51', '#f4a261'),
    'iou': ('#264653', '#2a9d8f'),
    'dice': ('#8ab17d', '#83c5be'),
    'cos': ('#833c0c', '#a0522d')
}

def draw_accuracy_vs_mask_ratio(accuracy, mask_ratio, tag="Accuracy", output_dir="output/plots"):
    plt.figure(figsize=(8, 5))
    plt.plot(mask_ratio, accuracy, marker='o', linewidth=2.5, markersize=9,
             color=color_maps[tag.lower()][0], markerfacecolor=color_maps[tag.lower()][1], markeredgecolor=color_maps[tag.lower()][0])

    plt.title(f"{tag} vs Mask Ratio", fontsize=14, fontweight='bold')
    plt.xlabel("Mask Ratio", fontsize=12)
    plt.ylabel(f"{tag} (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)

    for x, y in zip(mask_ratio, accuracy):
        # Shadow and text for readability
        plt.text(x + 0.003, y + 1.6, f"{y:.2f}%", ha='center', fontsize=11, color='white',
                 fontweight='bold', alpha=0.8)
        plt.text(x, y + 1.5, f"{y:.2f}%", ha='center', fontsize=11, color='#833c0c', fontweight='bold')

    plt.ylim(min(accuracy) - 5, max(accuracy) + 5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{tag}_vs_mask_ratio.png", dpi=300)

# Updated Data
mask_ratio = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
accuracy = [81.01, 76.25, 71.02, 66.11, 59.25, 45.20, 37.69]
iou = [100, 43.36, 32.42, 26.89, 22.25, 15.80, 8.66]
dice = [100, 59.60, 48.04, 41.49, 35.58, 26.63, 15.73]
cos = [100, 88.52, 82.08, 77.05, 71.96, 64.86, 55.59]
draw_accuracy_vs_mask_ratio(accuracy, mask_ratio, tag="Accuracy")
draw_accuracy_vs_mask_ratio(iou, mask_ratio, tag="IoU")
draw_accuracy_vs_mask_ratio(dice, mask_ratio, tag="Dice")
draw_accuracy_vs_mask_ratio(cos, mask_ratio, tag="Cos")
