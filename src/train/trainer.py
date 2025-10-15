from accelerate import Accelerator
from src.utils import Arg, argument_parser
from pathlib import Path
from src.lookwhere.modeling import Selector
from src.lookwhere.transforms import trans
from src.utils import seed_everything
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm
import wandb


class MaskedDataset(Dataset):
    def __init__(self, input_dataset, groundtruth_dataset, input_masked_ratio=0.1, output_masked_ratio=0.0):
        super().__init__()
        self.input = input_dataset
        self.groundtruth = groundtruth_dataset
        self.input_masked_ratio = input_masked_ratio
        self.output_masked_ratio = output_masked_ratio

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        image = self.input[idx]["image"]
        selector_map = self.groundtruth[idx]

        # randomly mask the input image
        mask = (torch.rand(1, image.shape[1], image.shape[2]) >= self.input_masked_ratio).to(image.dtype)
        
        masked_input = image * mask
        
        return masked_input, selector_map
        
        

def main(args):
    seed_everything(args.seed)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    pretrained_params = torch.load(args.pretrained_params_path, map_location="cpu", weights_only=True)
    selector = Selector(
        pretrained_params=pretrained_params["selector"],
        lw_type="dinov2",
        hr_size=518,
        device=device,
    )
    
    if accelerator.is_main_process:
        wandb.init(
            project="lookmore",
            config={
                "seed": args.seed,
                "input_masked_ratio": args.input_masked_ratio,
                "output_masked_ratio": args.output_masked_ratio,
                "num_epoches": args.num_epoches,
            }
        )
        wandb.watch(selector, log="all", log_freq=100)

    
    subsampled_indices = torch.load(args.dataset_dir / "indices.pt")
    subsampled_selector_maps = torch.load(args.dataset_dir / "selector_maps.pt")
    
    raw_ds = load_dataset("ILSVRC/imagenet-1k", split="train")
    raw_ds.set_transform(trans)
    sub_ds = raw_ds.select(subsampled_indices)
    
    dataset = MaskedDataset(
        input_dataset=sub_ds,
        groundtruth_dataset=subsampled_selector_maps,
        input_masked_ratio=args.input_masked_ratio,
        output_masked_ratio=args.output_masked_ratio,
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    
    optimizer = optim.AdamW(
        selector.parameters(),
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches * len(train_dataloader), eta_min=1e-6)

    selector, train_dataloader, optimizer, scheduler = accelerator.prepare(
        selector, train_dataloader, optimizer, scheduler
    )
    
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
    global_step = 0
    selector.train()

    disable_bar = not accelerator.is_main_process
    
    for epoch in tqdm(range(args.num_epoches), desc="Epochs", disable=disable_bar):
        batch_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epoches}",
            total=len(train_dataloader),
            leave=False,
            disable=disable_bar,
        )

        for batch in batch_bar:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = selector(inputs)
            loss = criterion(F.log_softmax(outputs["selector_map"], dim=1), F.log_softmax(targets, dim=1))
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            
            if accelerator.is_main_process:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_now,
                    },
                    step=global_step,
                )
            global_step += 1

    if accelerator.is_main_process:
        exp_name = f"seed_{args.seed}_mask_{args.input_masked_ratio}_{args.output_masked_ratio}_epoches_{args.num_epoches}"
        save_dir = args.final_output_dir / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(selector)
        torch.save(unwrapped.state_dict(), save_dir / "selector_final.pth")

        wandb.finish()

if __name__ == "__main__":
    args = argument_parser(
        Arg("-d", "--dataset_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/subsample_maps/128000")),
        Arg("-s", "--seed", type=int, default=1102),
        Arg("-p", "--pretrained_params_path", type=str, default="models/lookwhere_dinov2.pt"),
        Arg("-n", "--num_epoches", type=int, default=10),
        Arg("-i", "--input_masked_ratio", type=float, default=0.2),
        Arg("-o", "--output_masked_ratio", type=float, default=0.0),
        Arg("-f", "--final_output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/models")),
    )
    
    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
