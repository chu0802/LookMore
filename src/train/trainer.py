from accelerate import Accelerator
from src.utils import Arg, argument_parser
from pathlib import Path
from src.lookwhere.modeling import load_model
from datasets.transforms import trans
from src.utils import seed_everything, none_or_str
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_dataset_with_index
from tqdm.auto import tqdm
import wandb


class MaskedDataset(Dataset):
    def __init__(self, input_dataset, groundtruth_dataset, input_masked_ratio=[0.1], output_masked_ratio=0.0, low_res_size=154):
        super().__init__()
        self.input = input_dataset
        self.groundtruth = groundtruth_dataset
        self.input_masked_ratio = input_masked_ratio
        self.output_masked_ratio = output_masked_ratio
        self.low_res_size = low_res_size

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        image = self.input[idx]["image"]
        selector_map = self.groundtruth[idx]
        
        ratio = self.input_masked_ratio[torch.randint(len(self.input_masked_ratio), (1,))]
        
        # randomly mask the input image
        mask = (torch.rand(1, self.low_res_size, self.low_res_size) >= ratio).to(image.dtype)
            
        return image, mask, selector_map

class Trainer:
    def __init__(self, accelerator, model, dataloader, num_epoches=10, output_dir=Path("output/models"), device="cuda"):
        self.accelerator = accelerator
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        self.output_dir = output_dir

        self.num_epoches = num_epoches
        
        self.optimizer, self.scheduler = self.init_optimizer_scheduler()

        self.model, self.dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.dataloader, self.optimizer, self.scheduler
        )
        
        self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.global_steps = 0

    def init_optimizer_scheduler(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epoches * len(self.dataloader), eta_min=1e-6)
        return optimizer, scheduler

    def train(self):
        disable_bar = not self.accelerator.is_main_process
        
        for epoch in tqdm(range(self.num_epoches), desc="Epochs", disable=disable_bar):
            batch_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epoches}",
                total=len(self.dataloader),
                leave=False,
                disable=disable_bar,
            )

            for batch in batch_bar:
                self.optimizer.zero_grad()
                inputs, masks, targets = batch
                outputs = self.model(inputs, masks)
                loss = self.criterion(F.log_softmax(outputs["selector_map"], dim=1), F.log_softmax(targets, dim=1))
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                if self.accelerator.is_main_process:
                    lr_now = self.scheduler.get_last_lr()[0]
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr_now,
                        },
                        step=self.global_steps,
                    )
                self.global_steps += 1

        self.finish()
    
    def finish(self):
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped.state_dict(), self.output_dir / "selector_final.pt")

            wandb.finish()


def main(args):
    seed_everything(args.seed)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    lw = load_model(pretrained_params_path=args.pretrained_params_path, head_params_path=args.head_params_path, device="cpu")

    selector = lw.selector
    selector.to(device)
    
    if accelerator.is_main_process:
        wandb.init(
            project="lookmore",
            config={
                "seed": args.seed,
                "input_masked_ratio": "_".join([str(r) for r in args.input_masked_ratio]),
                "output_masked_ratio": args.output_masked_ratio,
                "num_epoches": args.num_epoches,
            }
        )
        wandb.watch(selector, log="all", log_freq=100)

    
    subsampled_indices = torch.load(args.dataset_dir / "indices.pt")
    subsampled_selector_maps = torch.load(args.dataset_dir / "selector_maps.pt")
    
    raw_ds = load_dataset_with_index("ILSVRC/imagenet-1k", split="train")
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
    )
    
    exp_name = f"seed_{args.seed}_mask_{'_'.join([str(r) for r in args.input_masked_ratio])}_{args.output_masked_ratio}_epoches_{args.num_epoches}"
    
    trainer = Trainer(
        accelerator=accelerator,
        model=selector,
        dataloader=train_dataloader,
        num_epoches=args.num_epoches,
        output_dir=args.final_output_dir / exp_name,
        device=device,
    )
    
    trainer.train()


if __name__ == "__main__":
    args = argument_parser(
        Arg("-d", "--dataset_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/subsample_maps/128000")),
        Arg("-s", "--seed", type=int, default=1102),
        Arg("-p", "--pretrained_params_path", type=none_or_str, default="models/lookwhere_dinov2.pt"),
        Arg("-e", "--head_params_path", type=none_or_str, default="models/imagenet_classifier_head_9.pt"),
        Arg("-n", "--num_epoches", type=int, default=10),
        Arg("-i", "--input_masked_ratio", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], nargs="+"),
        Arg("-o", "--output_masked_ratio", type=float, default=0.0),
        Arg("-f", "--final_output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/models")),
    )
    
    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
