from src.eval.test import Tester
from src.eval.mask_evaluation import maps_to_masks
from src.utils import Arg, argument_parser
from src.utils import load_dataset_with_index
from src.lookwhere.modeling import load_model
from src.utils import seed_everything
import torch
from src.lookwhere.transforms import trans
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


class AttentionMaskTester(Tester):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def mask_generator(self, images):
        original_selector_dict = self.teacher_model(images)
        original_selector_maps = original_selector_dict["original_selector_map"]
        masks = torch.stack([
            maps_to_masks(map, k_ratio=self.mask_ratio, device=self.device).reshape(-1, 1)
            for map in original_selector_maps
        ])
        
        return masks
        
    
    @torch.no_grad()
    def test(self):
        for batch in tqdm(self.dataloader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            masks = self.mask_generator(images)
            
            selector_dict = self.lw.selector(images, masks_after_patch_embed=masks)
            outputs = self.lw.forward_with_selector_dict(images, selector_dict)
            
            selector_maps = selector_dict["selector_map"].detach().to("cpu")
            preds = outputs.argmax(dim=1)
            
            self.all_selector_maps.append(selector_maps)
            self.dump_selector_maps(threshold=10)
            self.all_preds.append(torch.stack((preds, labels), dim=1).detach().to("cpu"))

        self.dump_selector_maps()
        self.dump_accuracy()

def main(args):
    seed_everything(args.seed)

    lw = load_model(
        pretrained_params_path=args.pretrained_params_path, 
        head_params_path=args.head_params_path, 
        selector_params_path=args.selector_params_path,
        mask_prediction=True,
    )

    lw.eval()
    
    teacher_selector = load_model(pretrained_params_path=args.pretrained_params_path, head_params_path=args.head_params_path, device="cpu").selector
    teacher_selector.to(args.device)
    
    teacher_selector.eval()

    if args.mode == "train":
        subsampled_indices = torch.load(Path("/home/yuchuyu/project/lookwhere/output/subsample_maps/128000") / "indices.pt")
        raw_ds = load_dataset_with_index("ILSVRC/imagenet-1k", split="train")
        raw_ds.set_transform(trans)
        ds = raw_ds.select(subsampled_indices)
    else:
        ds = load_dataset_with_index("ILSVRC/imagenet-1k", split=args.mode)
        ds.set_transform(trans)

    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    tester = AttentionMaskTester(lw, dataloader, args.output_dir, teacher_model=teacher_selector, device=args.device, mask_ratio=args.mask_ratio)
    tester.test()

if __name__ == "__main__":
    args = argument_parser(
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--selector_params_path", type=str, default="/home/yuchuyu/project/lookwhere/mae_head/models/seed_1102_epoches_10_pretrained/selector_epoch_10.pt"),
        Arg("--head_params_path", type=str, default="models/imagenet_classifier_head_9.pt"),
        Arg("--device", type=str, default="cuda"),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/mae_prediction")),
        Arg("--tag", type=str, default=None),
        Arg("--mask_ratio", type=float, default=0.0),
        Arg("--seed", type=int, default=1102),
        Arg("--cumulative", action="store_true"),
    )
    base_output_dir = args.output_dir / args.mode
    if args.tag is not None:
        base_output_dir /= args.tag
    args.output_dir = base_output_dir / f"maps_masked_ratio_{args.mask_ratio:0.1f}"
    main(args)
