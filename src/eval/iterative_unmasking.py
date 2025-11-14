from src.eval.test import Tester
from src.eval.mask_evaluation import maps_to_masks
import torch
from torchvision import transforms
from src.lookwhere.modeling import load_model
from src.lookwhere.transforms import trans
from torch.utils.data import DataLoader
from src.utils import Arg, argument_parser
from src.utils import load_dataset_with_index
from pathlib import Path
from tqdm import tqdm
from src.utils import seed_everything


class IterativeUnmaskTester(Tester):
    def __init__(self, *args, attention_predictor=None, iteration=2, k_ratio=0.1, initial_mask_ratio=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_predictor = attention_predictor
        self.iteration = iteration
        self.k_ratio = k_ratio
        self.initial_mask_ratio = initial_mask_ratio
        
        self.counter = {i: 0 for i in range(self.iteration)}
        self.all_selector_maps = {i: [] for i in range(self.iteration)}
        self.mask_ratios = {i: 0 for i in range(self.iteration)}
        self.all_preds = {i: [] for i in range(self.iteration)}
        self.all_masks = {i: [] for i in range(self.iteration)}

        self.output_dir = {i: self.output_dir / f"iteration_{i}" for i in range(self.iteration)}
        for v in self.output_dir.values():
            v.mkdir(parents=True, exist_ok=True)
    
    def dump_masks(self):
        for k, v in self.all_masks.items():
            all_masks = torch.cat(v, dim=0)
            torch.save(all_masks, self.output_dir[k] / f"masks.pt")
    
    def dump_selector_maps(self, threshold=10):
        for k, v in self.all_selector_maps.items():
            if len(v) > threshold:
                all_selector_maps = torch.cat(v, dim=0)
                torch.save(all_selector_maps, self.output_dir[k] / f"selector_map_{self.counter[k]:03d}.pt")
                self.all_selector_maps[k] = []
                self.counter[k] += 1
    
    def dump_accuracy(self):
        for k, v in self.all_preds.items():
            all_preds = torch.cat(v, dim=0)
            torch.save(all_preds, self.output_dir[k] / f"preds_vs_labels.pt")
            
            accuracy = (all_preds[:,0] == all_preds[:,1]).float().mean().item()
            print(f"Iteration {k}, Accuracy: {accuracy*100:.2f}")
        
    
    def dump_mask_ratios(self):
        print("Mask Ratios per Iteration:")
        for k, v in self.mask_ratios.items():
            print(f"Iteration {k}: {v:.4f}")
    
    @torch.no_grad()
    def test(self):
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # build initial masks
            original_selector_dict = self.attention_predictor(images)
            original_selector_maps = original_selector_dict["original_selector_map"]
            masks = torch.stack([
                maps_to_masks(map, k_ratio=self.initial_mask_ratio, device="cuda").reshape(-1, 1)
                for map in original_selector_maps
            ])
            
            for iter in range(self.iteration):
                # compute selection maps and accuracy
                selector_dict = self.attention_predictor(images, masks_after_patch_embed=masks)
                outputs = self.lw.forward_with_selector_dict(images, selector_dict)
                
                selector_maps = selector_dict["selector_map"].detach().to("cpu")
                self.all_selector_maps[iter].append(selector_maps)
                
                preds = outputs.argmax(dim=1)
                self.all_preds[iter].append(torch.stack((preds, labels), dim=1).detach().to("cpu"))
                self.all_masks[iter].append(masks.squeeze().detach().to("cpu"))
                
                self.mask_ratios[iter] += (masks.squeeze().mean(dim=-1)).sum().item()

                # iterative unmasking
                next_tokens_maps = self.lw.selector(images, masks_after_patch_embed=masks)["original_selector_map"]
                next_tokens_masks = maps_to_masks(next_tokens_maps, k_ratio=0.1, device="cuda").reshape(-1, 121, 1)
                masks = (masks + next_tokens_masks).clamp(max=1.0)

                
            self.dump_selector_maps(threshold=10)

        self.mask_ratios = {k: v / len(self.dataloader.dataset) for k, v in self.mask_ratios.items()}
        self.dump_selector_maps()
        self.dump_accuracy()
        self.dump_mask_ratios()
        self.dump_masks()

def main(args):
    seed_everything(args.seed)

    lw = load_model(
        pretrained_params_path=args.pretrained_params_path, 
        head_params_path=args.head_params_path, 
        selector_params_path=args.selector_params_path,
    )
    
    attention_predictor = load_model(
        pretrained_params_path=args.pretrained_params_path, 
        head_params_path=args.head_params_path, 
    ).selector

    lw.eval()
    attention_predictor.eval()

    ds = load_dataset_with_index("ILSVRC/imagenet-1k", split=args.mode)
    ds.set_transform(trans)

    dataloader = DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    tester = IterativeUnmaskTester(lw, attention_predictor=attention_predictor, dataloader=dataloader, output_dir=args.output_dir, iteration=args.iteration)
    tester.test()

if __name__ == "__main__":
    args = argument_parser(
        Arg("--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("--selector_params_path", type=str, default="/home/yuchuyu/project/lookwhere/iterative_token_selection/models/seed_1102_epoches_10_pretrained/selector_epoch_10.pt"),
        Arg("--head_params_path", type=str, default="models/imagenet_classifier_head_9.pt"),
        Arg("--device", type=str, default="cuda"),
        Arg("--mode", type=str, default="validation", choices=["train", "test", "validation"]),
        Arg("--output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/iterative_unmask")),
        Arg("--tag", type=str, default=None),
        Arg("--iteration", type=int, default=10),
        Arg("--seed", type=int, default=1102),
    )
    base_output_dir = args.output_dir / args.mode
    if args.tag is not None:
        base_output_dir /= args.tag
    args.output_dir = base_output_dir
    main(args)
