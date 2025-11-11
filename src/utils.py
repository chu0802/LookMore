import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import random
from datasets import load_dataset


@dataclass
class Arg:
    name: str
    abbr: Optional[str] = None
    type: Optional[object] = None
    help: Optional[str] = None
    default: Optional[object] = None
    required: bool = False
    action: Optional[str] = None
    choices: Optional[list] = None
    nargs: Optional[int] = None
    
    def __post_init__(self):
        # allowing swapping for abbr and name
        if self.abbr is not None and "--" not in self.name and "-" in self.name:
            self.name, self.abbr = self.abbr, self.name
        if "--" not in self.name:
            self.name = "--" + self.name
        if self.abbr is not None and "-" not in self.abbr:
            self.abbr = "-" + self.abbr

    def parse(self):
        kwargs = {k: v for k, v in vars(self).items() if k not in ["name", "abbr"]}
        if self.action is not None:
            kwargs.pop("type", None)
            kwargs.pop("choices", None)
            kwargs.pop("nargs", None)

        if self.abbr is None:
            return [self.name], kwargs
        return [self.name, self.abbr], kwargs

def argument_parser(*args):
    parser = argparse.ArgumentParser()
    for arg in args:
        if not isinstance(arg, Arg):
            arg = Arg(**arg)
        list_arguments, dict_arguments = arg.parse()
        parser.add_argument(*list_arguments, **dict_arguments)
    return parser.parse_args()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def none_or_str(x):
    return None if x == "None" else x


def cumulative_mask_generator(shape, mask_ratio, indices, base_seed=1102, device="cuda"):
    if isinstance(indices, int):
        indices = [indices]

    total_size = torch.prod(torch.tensor(shape)).item()
    masked_size = int(total_size * (1 - mask_ratio))
    
    masks = torch.zeros((len(indices), total_size), device=device)

    for i, idx in enumerate(indices):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        generator = torch.Generator(device=device)
        generator.manual_seed(int(base_seed + idx))

        perm = torch.randperm(total_size, generator=generator, device=device)
        masks[i, perm[:masked_size]] = 1

    return masks.reshape(len(indices), 1, *shape)

def load_dataset_with_index(name, split):
    ds = load_dataset(name, split=split)
    ds = ds.add_column("index", list(range(len(ds))))
    ds.set_format(type="torch", columns=ds.column_names)
    return ds
