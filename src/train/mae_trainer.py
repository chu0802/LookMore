from accelerate import Accelerator
from src.utils import Arg, argument_parser
from pathlib import Path
from src.lookwhere.modeling import load_model
from src.lookwhere.transforms import trans
from src.utils import seed_everything, none_or_str
from torch.utils.data import Dataset, DataLoader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_dataset_with_index
from tqdm.auto import tqdm
import wandb
from src.eval.mask_evaluation import maps_to_masks

class IterativeDataset(Dataset):
    def __init__(self, input_dataset):
        self.input = input_dataset
    
    def __len__(self):
        return len(self.input) * 9

    def __getitem__(self, idx):
        mask_ratio = (idx // len(self.input) + 1) / 10
        image_idx = idx % len(self.input)
        return self.input[image_idx]["image"], mask_ratio

class Trainer:
    def __init__(self, accelerator, model, teacher_model, dataloader, num_epoches=10, output_dir=Path("output/models"), device="cuda"):
        self.accelerator = accelerator
        self.model = model
        self.teacher_model = teacher_model
        self.dataloader = dataloader
        self.device = device
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
                images, mask_ratios = batch
                
                with torch.no_grad():
                    original_selector_maps = self.teacher_model(images)["original_selector_map"]
                    masks = torch.stack([
                        maps_to_masks(map, k_ratio=mask_ratio, device="cuda").reshape(-1, 1)
                        for map, mask_ratio in zip(original_selector_maps, mask_ratios)
                    ])

                student_output = self.model(images, masks_after_patch_embed=masks)
                student_pred_selector_maps = student_output["original_selector_map"]
                
                loss = self.criterion(
                    F.log_softmax(student_pred_selector_maps, dim=1),
                    F.log_softmax(original_selector_maps, dim=1)
                )
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
            
            self.save_model(epoch=epoch)

        self.finish()

    def save_model(self, epoch=None):
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            tag = f"epoch_{epoch+1}" if epoch is not None else "final"
            torch.save(unwrapped.state_dict(), self.output_dir / f"selector_{tag}.pt")

    def finish(self):
        if self.accelerator.is_main_process:
            wandb.finish()
    

def main(args):
    seed_everything(args.seed)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    lw = load_model(pretrained_params_path=args.pretrained_params_path, head_params_path=args.head_params_path, device="cpu", mask_prediction=True)

    selector = lw.selector

    selector.to(device)
    
    # copy & freeze a teacher model
    teacher_selector = load_model(pretrained_params_path=args.pretrained_params_path, head_params_path=args.head_params_path, device="cpu").selector
    teacher_selector.to(device)

    for param in teacher_selector.parameters():
        param.requires_grad = False
    
    teacher_selector.eval()
    
    if accelerator.is_main_process:
        wandb.init(
            project="mae_head_training",
            config={
                "seed": args.seed,
                "num_epoches": args.num_epoches,
            }
        )
        wandb.watch(selector, log="all", log_freq=100)
        
    subsampled_indices = torch.load(args.dataset_dir / "indices.pt")
    
    raw_ds = load_dataset_with_index("ILSVRC/imagenet-1k", split="train")
    raw_ds.set_transform(trans)
    sub_ds = raw_ds.select(subsampled_indices)
    
    dataset = IterativeDataset(sub_ds)
    train_dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    
    exp_name = f"seed_{args.seed}_epoches_{args.num_epoches}_pretrained"
    
    trainer = Trainer(
        accelerator=accelerator,
        model=selector,
        teacher_model=teacher_selector,
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
        Arg("-f", "--final_output_dir", type=Path, default=Path("/home/yuchuyu/project/lookwhere/mae_head/models")),
    )
    
    args.final_output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
