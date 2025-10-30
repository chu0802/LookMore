import matplotlib.pyplot as plt
import random
# color maps for color and markerfacecolor using fancy colors,  makerfacecolor should be darker than color
color_maps = [
    ('#e76f51', '#f4a261'),
    ('#264653', '#2a9d8f'),
    ('#8ab17d', '#83c5be'),
    ('#833c0c', '#a0522d'),
    ('#6a4c93', '#9d7ed5'),
    ('#d62828', '#f77f00'),
    ('#007f5f', '#2b9348'),
    ('#f72585', '#b5179e'),
    ('#4361ee', '#4895ef'),
    ('#4cc9f0', '#3a86ff'),
    ('#9d0208', '#d00000'),
    ('#5f0f40', '#9a031e'),
]

def draw_accuracy_vs_mask_ratio(accuracy, mask_ratio, tag="Accuracy", output_dir="output/plots", save=True, per_line_tag=None):
    if not save:
        plt.figure(figsize=(8, 5))
    color_idx = random.randint(0, len(color_maps) - 1) 
    plt.plot(mask_ratio, accuracy, marker='o', linewidth=2.5, markersize=9,
             color=color_maps[color_idx][0], markerfacecolor=color_maps[color_idx][1], markeredgecolor=color_maps[color_idx][0])

    plt.title(f"{tag} vs Mask Ratio", fontsize=14, fontweight='bold')
    plt.xlabel("Mask Ratio", fontsize=12)
    plt.ylabel(f"{tag} (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # add a legend
    if per_line_tag is not None:
        plt.legend(per_line_tag, fontsize=11, loc='lower left')

    for x, y in zip(mask_ratio, accuracy):
        # Shadow and text for readability
        plt.text(x + 0.003, y + 1.6, f"{y:.2f}%", ha='center', fontsize=11, color='white',
                 fontweight='bold', alpha=0.8)
        plt.text(x, y + 1.5, f"{y:.2f}%", ha='center', fontsize=11, color='#833c0c', fontweight='bold')

    # plt.ylim(min(accuracy) - 5, max(accuracy) + 5)

    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/{tag}_vs_mask_ratio.png", dpi=300)

# Updated Data
mask_ratio = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
accuracy = [81.01, 76.25, 71.02, 66.11, 59.25, 45.20, 37.69]
iou = [100, 43.36, 32.42, 26.89, 22.25, 15.80, 8.66]
dice = [100, 59.60, 48.04, 41.49, 35.58, 26.63, 15.73]
cos = [100, 88.52, 82.08, 77.05, 71.96, 64.86, 55.59]

# finetuned results
f_mask_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
f_acc = [79.33, 77.73, 76.26, 74.31, 69.53, 45.54]
f_iou = [53.62, 47.73, 43.52, 39.37, 32.74, 14.24]
f_dice = [69.34, 64.02, 59.97, 55.74, 48.46, 24.39]
f_cos = [93.47, 91.83, 90.31, 88.55, 85.04, 64.14]

random.seed(1102)
draw_accuracy_vs_mask_ratio(accuracy, mask_ratio, tag="Accuracy", save=False)
draw_accuracy_vs_mask_ratio(f_acc, f_mask_ratio, tag="Accuracy", save=True, per_line_tag=["Pre-trained", "Fine-tuned"])

draw_accuracy_vs_mask_ratio(iou, mask_ratio, tag="IoU", save=False)
draw_accuracy_vs_mask_ratio(f_iou, f_mask_ratio, tag="IoU", save=True, per_line_tag=["Pre-trained", "Fine-tuned"])

draw_accuracy_vs_mask_ratio(dice, mask_ratio, tag="Dice", save=False)
draw_accuracy_vs_mask_ratio(f_dice, f_mask_ratio, tag="Dice", save=True, per_line_tag=["Pre-trained", "Fine-tuned"])

draw_accuracy_vs_mask_ratio(cos, mask_ratio, tag="Cos", save=False)
draw_accuracy_vs_mask_ratio(f_cos, f_mask_ratio, tag="Cos", save=True, per_line_tag=["Pre-trained", "Fine-tuned"])
