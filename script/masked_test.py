import torch
from torchvision import transforms
from src.lookwhere.modeling import Selector
from src.lookwhere.transforms import trans
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from datasets import load_dataset
from pathlib import Path

from src.utils import seed_everything
from tqdm import tqdm

def main(args):
    seed_everything(args.seed)

    num_high_res_patches = (args.img_size // 14)**2
    k = int(args.k_ratio * num_high_res_patches)
    pretrained_params = torch.load(args.pretrained_params_path, map_location="cpu", weights_only=True)
    
    output_dir = args.output_dir
    if args.mode == "test":
        output_dir = output_dir / "test"
    
    if args.finetuned_params_path is not None:
        output_dir /= Path(args.finetuned_params_path).parent.stem

    output_dir /= f"masked_maps_{args.mask_ratio}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selector = Selector(
        pretrained_params=pretrained_params["selector"],
        lw_type="dinov2",
        hr_size=518,
        device="cpu",
    )

    if args.finetuned_params_path is not None:
        finetuned_parmas = torch.load(args.finetuned_params_path, map_location="cpu", weights_only=True)
        selector.load_state_dict(finetuned_parmas)

    selector.to(args.device)
    selector.eval()

    
    ds = load_dataset("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(trans)
    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    all_selector_map = []
    counter = 0

    for batch in tqdm(dataloader):
        images = batch["image"].to(args.device)

        bsize, height, width = images.shape[0], images.shape[2], images.shape[3]
        masks = (torch.rand(bsize, 1, height, width, device=args.device) >= args.mask_ratio).float()
        
        masked_images = images * masks
        
        with torch.no_grad():
            selector_map = selector(masked_images)["selector_map"]  # (bs, num_high_res_patches)
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
        Arg("--finetuned_params_path", type=str, default=None),
        Arg("--device", type=str, default="cuda"),
        Arg("--seed", type=int, default=1102),
        Arg("--mask_ratio", type=float, default=0.1),
        Arg("--mode", type=str, default="test", choices=["train", "test"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output"))
    )
    main(args)
