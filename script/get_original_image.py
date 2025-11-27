import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from src.utils import load_dataset_with_index, cumulative_mask_generator
from pathlib import Path
from PIL import Image
from src.utils import seed_everything
from tqdm import tqdm
from src.datasets.traffic_sign import TrafficSigns
from src.datasets.transforms import trans_for_save
from functools import partial

# def trans_for_save(examples, img_size=518):
#     """Transform that only resizes without normalization for saving"""
#     transform = transforms.Compose([
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.Resize((154, 154), interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.ToTensor(),
#     ])
#     examples["image"] = [transform(example) for example in examples["image"]]
#     return examples


transform_for_save = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((994, 994), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

def main(args):
    seed_everything(args.seed)
    
    # output_dir = Path(args.output_dir) / "vis" / args.mode / "original"
    output_dir = Path(args.output_dir) / "traffic_sign_no_finetune" / "original"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Traffic Signs {args.mode} dataset...")
    ds = TrafficSigns(train=False, transform=transform_for_save)
    
    for i in range(len(ds)):
        img = ds[i]["image"]
        img = transforms.ToPILImage()(img)
        img.save(output_dir / f"{i:05d}.png")

if __name__ == "__main__":
    args = argument_parser(
        Arg("--img_size", type=int, default=518),
        Arg("--seed", type=int, default=1102),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/LookMore/output")),
        Arg("--num", type=int, default=10, help="Number of images to save"),
        Arg("--mask_ratio", type=float, default=0.0)
    )
    main(args)
