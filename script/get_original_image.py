import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from src.utils import seed_everything
from tqdm import tqdm


def trans_for_save(examples, img_size=518):
    """Transform that only resizes without normalization for saving"""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    examples["image"] = [transform(example) for example in examples["image"]]
    return examples


def main(args):
    seed_everything(args.seed)
    
    output_dir = Path(args.output_dir) / "vis" / args.mode / "original"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ImageNet {args.mode} dataset...")
    ds = load_dataset("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(lambda x: trans_for_save(x, img_size=args.img_size))
    
    for i in range(args.num):
        img = ds[i]["image"]
        img.save(output_dir / f"{i:05d}.png")

if __name__ == "__main__":
    args = argument_parser(
        Arg("--img_size", type=int, default=518),
        Arg("--seed", type=int, default=1102),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/LookMore/output")),
        Arg("--num", type=int, default=10, help="Number of images to save"),
    )
    main(args)
