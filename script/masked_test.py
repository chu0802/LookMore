import torch
from torchvision import transforms
from src.lookwhere.modeling import LookWhereDownstream
from torch.utils.data import DataLoader
import argparse
from datasets import load_dataset
from pathlib import Path

import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--k_ratio", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--is_classification", action="store_true")
    parser.add_argument("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1102)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    args = parser.parse_args()
    return args

def trans(examples, high_res_img_size=518):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((high_res_img_size, high_res_img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]), 
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    ])
    examples["image"] = [transform(example) for example in examples["image"]]
    return examples

def main(args):
    seed_everything(args.seed)

    num_high_res_patches = (args.img_size // 14)**2
    k = int(args.k_ratio * num_high_res_patches)

    lw = LookWhereDownstream(
        pretrained_params_path=args.pretrained_params_path,
        high_res_size=args.img_size,
        num_classes=args.num_classes,
        k=k,
        is_cls=args.is_classification,
        device=args.device
    )
    
    ds = load_dataset("ILSVRC/imagenet-1k")
    ds.set_transform(trans)
    dataloader = DataLoader(ds["train"], batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    all_selector_map = []
    counter = 0
    
    output_dir = Path(f"/home/yuchuyu/project/lookwhere/output/masked_maps_{args.mask_ratio}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for batch in dataloader:
        images = batch["image"].to(args.device)
        
        # randomly generate a batch of masks with a specified percent
        bsize, height, width = images.shape[0], images.shape[2], images.shape[3]
        masks = (torch.rand(bsize, 1, height, width, device=args.device) >= args.mask_ratio).float()
        
        masked_images = images * masks
        
        with torch.no_grad():
            selector_map = lw.selector(masked_images)["selector_map"]  # (bs, num_high_res_patches)
            all_selector_map.append(selector_map.detach().to("cpu"))
        
        if len(all_selector_map) >= 10:
            print(f"Saving selector_map_{counter}.pt")
            all_selector_map = torch.cat(all_selector_map, dim=0)
            torch.save(all_selector_map, output_dir / f"selector_map_{counter}.pt")
            all_selector_map = []
            counter += 1
    
    all_selector_map = torch.cat(all_selector_map, dim=0)
    
    torch.save(all_selector_map, output_dir / f"selector_map_{counter}.pt")

if __name__ == "__main__":
    args = argument_parser()
    main(args)
