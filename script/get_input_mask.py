from src.utils import cumulative_mask_generator
import matplotlib.pyplot as plt
from src.eval.visualization import visualize_selector_map
import torch
from pathlib import Path

indices = list(range(50000))
size = 10
mask_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
output_dir = Path("output/vis/validation/cum_mask/input_mask/")

for mask_ratio in mask_ratios:
    sub_output_dir = output_dir / f"mask_ratio_{mask_ratio}"
    random_masks = (torch.rand(
        size, 
        1, 
        154, 
        154, 
    ) >= mask_ratio).float()

    cumulative_masks = cumulative_mask_generator(
        shape=(154, 154),
        mask_ratio=mask_ratio,
        indices=indices[:size],
        base_seed=1102,
        device="cuda"
    )

    # plot and save the cumulative_masks
    for i, (rand_mask, cum_mask) in enumerate(zip(random_masks, cumulative_masks)):
        breakpoint()
        visualize_selector_map(cum_mask.squeeze(), output_filename=f"cumulative_mask_{i}.png", is_flatten=False)
