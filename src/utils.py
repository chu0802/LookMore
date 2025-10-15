import argparse
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import random

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
