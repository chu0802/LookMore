import torch
from torchvision import transforms
from src.lookwhere.modeling import LookWhereDownstream
from src.lookwhere.transforms import trans
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

def main(args):
    num_high_res_patches = (args.img_size // 14)**2
    k = int(args.k_ratio * num_high_res_patches)

    lw = LookWhereDownstream(
        pretrained_params_path=args.pretrained_params_path,
        high_res_size=args.img_size,
        num_classes=args.num_classes,
        k=k,
        is_cls=False,
        device=args.device
    )
    
    ds = load_dataset("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(trans)
    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    all_selector_map = []
    counter = 0
    
    output_dir = Path("/home/yuchuyu/project/lookwhere/output/")
    if args.mode == "test":
        output_dir = output_dir / "test"
    output_dir /= "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for batch in tqdm(dataloader):
        images = batch["image"].to(args.device)
        
        with torch.no_grad():
            selector_map = lw.selector(images)["selector_map"]  # (bs, num_high_res_patches)
            all_selector_map.append(selector_map.detach().to("cpu"))
        
        if len(all_selector_map) >= 10:
            print(f"Saving selector_map_{counter}.pt")
            all_selector_map = torch.cat(all_selector_map, dim=0)
            torch.save(all_selector_map, output_dir / f"selector_map_{counter:03d}.pt")
            all_selector_map = []
            counter += 1
    
    all_selector_map = torch.cat(all_selector_map, dim=0)
    torch.save(all_selector_map, output_dir / f"selector_map_{counter:03d}.pt")

if __name__ == "__main__":
    args = argument_parser(
        Arg("--img_size", type=int, default=518),
        Arg("--k_ratio", type=float, default=0.1),
        Arg("--num_classes", type=int, default=0),
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--device", type=str, default="cuda"),
        Arg("--mode", type=str, default="test", choices=["train", "test"]),
    )
    main(args)
