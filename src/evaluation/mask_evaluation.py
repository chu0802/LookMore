import torch
import argparse
from pathlib import Path

def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--k_ratio", type=float, default=0.1)
    
    return parser.parse_args()

def iou_calculation(ground_truth_map, predicted_map):
    intersection = torch.sum(ground_truth_map * predicted_map)
    union = torch.sum(ground_truth_map) + torch.sum(predicted_map) - intersection
    return intersection / union

def maps_to_masks(maps, k_ratio):
    k = int(maps.shape[-1] * k_ratio)
    topk_vals, _ = torch.topk(maps, k, dim=-1)
    thresh = topk_vals[..., -1, None]
    
    return (maps >= thresh).float()

def main(args):
    num_high_res_patches = (518 // 14)**2
    k = int(args.k_ratio * num_high_res_patches)

    output_folder = Path("/home/yuchuyu/project/lookwhere/output")
    ground_truth_folder = output_folder / "maps"
    
    predicted_folder = output_folder / f"masked_maps_{args.mask_ratio}"
    for filename in sorted(predicted_folder.glob("*.pt")):
        ground_truth_maps = torch.load(ground_truth_folder / filename.name)
        predicted_maps = torch.load(predicted_folder / filename.name)
        
        gt_masks = maps_to_masks(ground_truth_maps, args.k_ratio)
        pred_masks = maps_to_masks(predicted_maps, args.k_ratio)
        
        iou = iou_calculation(gt_masks, pred_masks)
        print(f"IOU for {filename.name}: {iou}")
        
        
    

if __name__ == "__main__":
    args = argument_parsing()
    main(args)