import torch
from torchvision import transforms
from src.lookwhere.modeling import load_model
from datasets.transforms import trans
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser, cumulative_mask_generator
from src.utils import load_dataset_with_index
from pathlib import Path
from tqdm import tqdm
from src.utils import seed_everything

class Tester:
    def __init__(self, lw, dataloader, output_dir, device="cuda", mask_ratio=0.0, base_seed=1102):
        self.lw = lw
        self.dataloader = dataloader
        self.lw.eval()
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.mask_ratio = mask_ratio
        self.low_res_size = 154 # pre-defined by LookWhere's Selector
        
        self.counter = 0
        self.all_selector_maps = []
        self.all_preds = []
        self.base_seed = base_seed
    
    def dump_selector_maps(self, threshold=0):
        if len(self.all_selector_maps) > threshold:
            all_selector_maps = torch.cat(self.all_selector_maps, dim=0)
            torch.save(all_selector_maps, self.output_dir / f"selector_map_{self.counter:03d}.pt")
            self.all_selector_maps = []
            self.counter += 1
    
    def masks_generator(self, indices, cumulative=False):
        if self.mask_ratio == 0:
            return None

        if not cumulative:
            return (torch.rand(
                indices.shape[0], 
                1, 
                self.low_res_size, 
                self.low_res_size, 
                device=self.device,
            ) >= self.mask_ratio).float()

        cumulative_mask = cumulative_mask_generator(
            shape=(self.low_res_size, self.low_res_size), 
            mask_ratio=self.mask_ratio, 
            indices=indices,
            base_seed=self.base_seed,
            device=self.device,
        )

        return cumulative_mask

    def test(self, cumulative=False):
        for batch in tqdm(self.dataloader):
            indices = batch["index"].to(self.device)
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            masks = self.masks_generator(indices, cumulative=cumulative)
            with torch.no_grad():
                selector_dict = self.lw.selector(images, masks)
                outputs = self.lw.forward_with_selector_dict(images, selector_dict)

                selector_maps = selector_dict["selector_map"].detach().to("cpu")
                preds = outputs.argmax(dim=1)

            self.all_selector_maps.append(selector_maps)
            self.dump_selector_maps(threshold=10)
            self.all_preds.append(torch.stack((preds, labels), dim=1).detach().to("cpu"))

        self.dump_selector_maps()
        self.dump_accuracy()

    def dump_accuracy(self):
        self.all_preds = torch.cat(self.all_preds, dim=0)
        torch.save(self.all_preds, self.output_dir / "preds_vs_labels.pt")

        accuracy = (self.all_preds[:, 0] == self.all_preds[:, 1]).float().mean().item()
        print(f"Accuracy: {accuracy*100:.2f}%")

def main(args):
    seed_everything(args.seed)

    lw = load_model(
        pretrained_params_path=args.pretrained_params_path, 
        head_params_path=args.head_params_path, 
        selector_params_path=args.selector_params_path,
    )

    lw.eval()

    ds = load_dataset_with_index("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(trans)

    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    tester = Tester(lw, dataloader, args.output_dir, device=args.device, mask_ratio=args.mask_ratio)
    tester.test(cumulative=args.cumulative)

if __name__ == "__main__":
    args = argument_parser(
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--selector_params_path", type=str, default=None),
        Arg("--head_params_path", type=str, default="models/imagenet_classifier_head_9.pt"),
        Arg("--device", type=str, default="cuda"),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output")),
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
