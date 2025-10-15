import torch
from src.utils import Arg, argument_parser
from pathlib import Path

def main(args):
    selector_maps = []
    for maps_pt in sorted(args.maps_dir.glob("*.pt")):
        selector_maps.append(torch.load(maps_pt))
    selector_maps = torch.cat(selector_maps, dim=0)

    torch.manual_seed(args.seed)
    indices = torch.randperm(selector_maps.shape[0])[:args.num_samples]
    
    subsampled_selector_maps = selector_maps[indices]
    torch.save(subsampled_selector_maps, args.output_dir / "selector_maps.pt")
    torch.save(indices, args.output_dir / "indices.pt")
    


if __name__ == "__main__":
    args = argument_parser(
        Arg("-n", "--num_samples", type=int, default=1000),
        Arg("-s", "--seed", type=int, default=1102),
        Arg("-m", "--maps_dir", type=Path, default="/home/yuchuyu/project/lookwhere/output/maps"),
        Arg("-o", "--output_dir", type=Path, default="/home/yuchuyu/project/lookwhere/output/subsample_maps")
    )
    args.output_dir = args.output_dir / f"{args.num_samples}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
