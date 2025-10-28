import torch
from src.utils import Arg, argument_parser
from pathlib import Path
from functools import partial
from collections import defaultdict
from tqdm import tqdm

def iou(a, b, k_ratio=0.2, **kwargs):
    a_masks = maps_to_masks(a, k_ratio)
    b_masks = maps_to_masks(b, k_ratio)

    intersection = torch.sum(a_masks * b_masks, dim=-1)
    iou = intersection / (torch.sum(a_masks, dim=-1) + torch.sum(b_masks, dim=-1) - intersection + 1e-10)
    return iou.mean()

def dice(a, b, k_ratio=0.2, **kwargs):
    a_masks = maps_to_masks(a, k_ratio)
    b_masks = maps_to_masks(b, k_ratio)
    
    intersection = torch.sum(a_masks * b_masks, dim=-1)
    dice = 2 * intersection / (torch.sum(a_masks, dim=-1) + torch.sum(b_masks, dim=-1) + 1e-10)
    return dice.mean()

def cos(a, b, **kwargs):
    return (torch.sum(a*b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) + 1e-8)).mean()


def maps_to_masks(maps, k_ratio):
    k = int(maps.shape[-1] * k_ratio)
    topk_vals, _ = torch.topk(maps, k, dim=-1)
    thresh = topk_vals[..., -1, None]
    
    return (maps >= thresh).float()

def main(args):
    ground_truth_folder = args.output_dir / "maps_masked_ratio_0.0"
    predicted_folder = args.output_dir / f"maps_masked_ratio_{args.mask_ratio:0.1f}"


    total_scores = defaultdict(int)
    total_num = 0
    
    maps = sorted(predicted_folder.glob("selector_map_*.pt"))

    for filename in tqdm(maps, total=len(maps)):
        ground_truth_maps = torch.load(ground_truth_folder / filename.name)
        predicted_maps = torch.load(predicted_folder / filename.name)
        total_num += ground_truth_maps.shape[0]
        for eval_method in args.eval_methods:
            eval_func = partial(globals()[eval_method], k_ratio=args.k_ratio)            
            score = eval_func(ground_truth_maps, predicted_maps)
            total_scores[eval_method] += score.item() * ground_truth_maps.shape[0]

    
    for eval_method, score in total_scores.items():
        print(f"{eval_method}: {(100 * score / total_num):.2f}%")

if __name__ == "__main__":
    args = argument_parser(
        Arg("-o", "--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/validation")),
        Arg("-m", "--mask_ratio", type=float, default=0.1),
        Arg("-k", "--k_ratio", type=float, default=0.1),
        Arg("-e", "--eval_methods", type=str, nargs="+", default=["iou", "dice", "cos"]),
    )

    main(args)
