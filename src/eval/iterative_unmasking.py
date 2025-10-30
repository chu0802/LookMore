from src.eval.test import Tester
from src.eval.mask_evaluation import maps_to_masks
import torch
from torchvision import transforms
from src.lookwhere.modeling import load_model
from src.lookwhere.transforms import trans
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from src.utils import seed_everything


class IterativeUnmaskTester(Tester):
    def __init__(self, *args, iteration=2, k_ratio=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration = iteration
        self.k_ratio = k_ratio
    
    def test(self):
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            masks = (
                None
                if self.mask_ratio == 0 
                else (torch.rand(
                    images.shape[0], 
                    1, 
                    self.low_res_size, 
                    self.low_res_size, 
                    device=self.device,
                ) >= self.mask_ratio).float()
            )

            with torch.no_grad():
                for _ in range(self.iteration):
                    selector_dict = self.lw.selector(images, masks)
                    selector_maps = selector_dict["selector_map"].detach().to("cpu")
                    
                    selected_masks = maps_to_masks(selector_maps, self.k_ratio).to(self.device).reshape(-1, 1, 37, 37)

                    # up sampling the selected masks to match the image size 154 x 154
                    selected_masks = (torch.nn.functional.interpolate(
                        selected_masks, 
                        size=(self.low_res_size, self.low_res_size), 
                        mode="nearest",
                    ) > 0)
                    
                    # doing or operation to accumulate masks
                    masks = (masks.bool() | selected_masks).float()
                    if i == 0:
                        print(f"current masked ratio: {masks.reshape(len(masks), -1).mean(dim=1).mean().item():.4f}")
                
                outputs = self.lw.forward_with_selector_dict(images, selector_dict)
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
    )

    lw.eval()

    ds = load_dataset("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(trans)

    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    tester = IterativeUnmaskTester(lw, dataloader, args.output_dir, device=args.device, mask_ratio=args.mask_ratio, iteration=args.iteration)
    tester.test()

if __name__ == "__main__":
    args = argument_parser(
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--selector_params_path", type=str, default="/home/yuchuyu/project/lookwhere/models/seed_1102_mask_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_0.0_epoches_10/selector_final.pt"),
        Arg("--head_params_path", type=str, default="models/imagenet_classifier_head_9.pt"),
        Arg("--device", type=str, default="cuda"),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/iterative_unmask")),
        Arg("--tag", type=str, default=None),
        Arg("--mask_ratio", type=float, default=0.0),
        Arg("--iteration", type=int, default=2),
        Arg("--seed", type=int, default=1102),
    )
    base_output_dir = args.output_dir / args.mode
    if args.tag is not None:
        base_output_dir /= args.tag
    args.output_dir = base_output_dir / f"maps_masked_ratio_{args.mask_ratio:0.1f}"
    main(args)
