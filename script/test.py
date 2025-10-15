import torch
from torchvision import transforms
from src.lookwhere.modeling import LookWhereDownstream
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from datasets import load_dataset

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
    for batch in dataloader:
        images = batch["image"].to(args.device)
        labels = batch["label"].to(args.device)
        
        with torch.no_grad():
            selector_map = lw.selector(images)["selector_map"]  # (bs, num_high_res_patches)
            all_selector_map.append(selector_map.detach().to("cpu"))
        
        if len(all_selector_map) >= 10:
            print(f"Saving selector_map_{counter}.pt")
            all_selector_map = torch.cat(all_selector_map, dim=0)
            torch.save(all_selector_map, f"output/maps/selector_map_{counter}.pt")
            all_selector_map = []
            counter += 1
    
    all_selector_map = torch.cat(all_selector_map, dim=0)
    torch.save(all_selector_map, f"output/maps/selector_map_{counter}.pt")

if __name__ == "__main__":
    args = argument_parser(
        Arg("--img_size", type=int, default=518),
        Arg("--k_ratio", type=float, default=0.1),
        Arg("--num_classes", type=int, default=0),
        Arg("--is_classification", action="store_true"),
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--device", type=str, default="cuda"),
    )
    main(args)
