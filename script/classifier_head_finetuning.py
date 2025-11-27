from accelerate import Accelerator
from src.utils import Arg, argument_parser
from pathlib import Path
from src.lookwhere.modeling import Selector
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
from src.lookwhere.modeling import load_model


def main(args):
    seed_everything(args.seed)
    
    accelerator = Accelerator()
    device = accelerator.device

    lw = load_model(pretrained_params_path=args.pretrained_params_path, device=device)
    
    # freeze the LookWhereDownstream model
    for param in lw.selector.parameters():
        param.requires_grad = False
    
    for param in lw.extractor.parameters():
        param.requires_grad = False
        
    
    if accelerator.is_main_process:
        wandb.init(
            project="lookwhere_classifier_head_finetuning",
            config={
                "seed": args.seed,
                "num_epoches": args.num_epoches,
            }
        )
        wandb.watch(lw, log="all", log_freq=100)
    
    dataset = load_dataset_with_index("ILSVRC/imagenet-1k", split=["train", "validation"])
    dataset.set_transform(trans)
    
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=1024,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )
    
    optimizer = optim.AdamW(
        lw.head.parameters(),
        lr=1e-2,
        weight_decay=0.05,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoches * len(train_dataloader), eta_min=1e-4)

    lw, train_dataloader, optimizer, scheduler = accelerator.prepare(
        lw, train_dataloader, optimizer, scheduler
    )
    
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
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = lw(images)

            loss = F.cross_entropy(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            batch_bar.set_postfix({"loss": loss.item()})
            if accelerator.is_main_process:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_now,
                    },
                    step=epoch * len(train_dataloader) + batch_bar.n,
                )
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(lw)
            torch.save(unwrapped.head.state_dict(), args.final_output_dir / f"imagenet_classifier_head_{epoch}.pt")

    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    args = argument_parser(
        Arg("-d", "--dataset_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/output/subsample_maps/128000")),
        Arg("-s", "--seed", type=int, default=1102),
        Arg("-p", "--pretrained_params_path", type=none_or_str, default="models/lookwhere_dinov2.pt"),
        Arg("-n", "--num_epoches", type=int, default=10),
        Arg("-i", "--input_masked_ratio", type=float, default=[0.2], nargs="+"),
        Arg("-o", "--output_masked_ratio", type=float, default=0.0),
        Arg("-f", "--final_output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/models/head")),
    )
    
    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
