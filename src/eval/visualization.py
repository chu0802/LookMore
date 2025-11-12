import torch
import matplotlib.pyplot as plt
import math
from pathlib import Path
from src.eval.mask_evaluation import maps_to_masks
from src.utils import Arg, argument_parser

def visualize_selector_map(selector_map, cmap="turbo", output_filename=None, is_flatten=True):
    if output_filename is None:
        output_filename = "output.png"
    if is_flatten:
        grid_size = int(math.sqrt(selector_map.shape[0]))
        selector_map_grid = selector_map.reshape(grid_size, grid_size).cpu().numpy()
    else:
        selector_map_grid = selector_map.cpu().numpy()
    
    if isinstance(output_filename, Path):
        output_filename.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(selector_map_grid, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Selector Value")
    plt.title("Selector Map Visualization")
    plt.xlabel("Patch Column")
    plt.ylabel("Patch Row")
    plt.savefig(output_filename)
    plt.close()

def main(args):
    maps_filename = Path(f"selector_map_{args.maps_index:03d}.pt")
    
    gt_maps = torch.load(Path("/home/yuchuyu/project/lookwhere/output/validation") / "maps_masked_ratio_0.0" / maps_filename)
    selector_maps = torch.load(args.maps_dir / f"maps_masked_ratio_{args.mask_ratio:0.1f}" / maps_filename)
    
    if args.to_mask:
        base_output_dir = args.output_dir / f"masks"
    else:
        base_output_dir = args.output_dir / f"maps"
    selected_output_dir = base_output_dir / f"mask_ratio_{args.mask_ratio}" / maps_filename.stem
    gt_output_dir = base_output_dir / "mask_ratio_0.0" / maps_filename.stem
    
    selected_output_dir.mkdir(parents=True, exist_ok=True)
    gt_output_dir.mkdir(parents=True, exist_ok=True)

    for i, (selector_map, gt_map) in enumerate(zip(selector_maps, gt_maps)):
        if i != -1 and i >= args.num:
            break
        if args.to_mask:
            selector_map = maps_to_masks(selector_map, args.k_ratio)
            gt_map = maps_to_masks(gt_map, args.k_ratio)

        visualize_selector_map(selector_map, output_filename=(selected_output_dir / f"selector_map_{i}.png").as_posix())
        visualize_selector_map(gt_map, output_filename=(gt_output_dir / f"selector_map_{i}.png").as_posix())
    


if __name__ == "__main__":
    main(argument_parser(
        Arg("--mask_ratio", type=float, default=0.1),
        Arg("--k_ratio", type=float, default=0.2),
        Arg("--maps_index", type=int, default=0),
        Arg("--maps_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/validation/cumulative")),
        Arg("--output_dir", type=Path, default=Path("output/vis/validation/cum_mask/pretrained")),
        Arg("--num", default=10, type=int),
        Arg("--to_mask", action="store_true")
    ))
