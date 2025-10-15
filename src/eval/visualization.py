import torch
import matplotlib.pyplot as plt
import math
from pathlib import Path
from src.eval.mask_evaluation import maps_to_masks
from src.utils import Arg, argument_parser

def visualize_selector_map(selector_map, cmap="turbo", output_filename=None):
    if output_filename is None:
        output_filename = "output.png"
    grid_size = int(math.sqrt(selector_map.shape[0]))
    selector_map_grid = selector_map.reshape(grid_size, grid_size).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(selector_map_grid, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Selector Value")
    plt.title("Selector Map Visualization")
    plt.xlabel("Patch Column")
    plt.ylabel("Patch Row")
    plt.savefig(output_filename)
    plt.close()

def main(args):
    maps_filename = Path(f"selector_map_{args.maps_index}.pt")
    
    gt_maps = torch.load(args.maps_dir / "maps" / maps_filename)
    selector_maps = torch.load(args.maps_dir / f"masked_maps_{args.mask_ratio}" / maps_filename)
    
    selected_output_dir = args.output_dir / f"mask_ratio_{args.mask_ratio}" / maps_filename.stem
    gt_output_dir = args.output_dir / "mask_ratio_0" / maps_filename.stem
    
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    for i, (selector_map, gt_map) in enumerate(zip(selector_maps, gt_maps)):
        if i != -1 and i >= args.num:
            break
        selector_map = maps_to_masks(selector_map, args.k_ratio)
        gt_map = maps_to_masks(gt_map, args.k_ratio)
        visualize_selector_map(selector_map, output_filename=(selected_output_dir / f"selector_map_{i}.png").as_posix())
        visualize_selector_map(gt_map, output_filename=(gt_output_dir / f"selector_map_{i}.png").as_posix())
    


if __name__ == "__main__":
    main(argument_parser(
        Arg("--mask_ratio", type=float, default=0.2),
        Arg("--k_ratio", type=float, default=0.2),
        Arg("--maps_index", type=int, default=0),
        Arg("--maps_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output")),
        Arg("--ouptut_dir", type=Path, default=Path("output/vis")),
        Arg("--num", defualt=10, type=int),
    ))
