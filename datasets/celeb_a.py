import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from .datasets import register
import time
import copy

@register("celeb_a")
class Celeb_a(data.Dataset):
    def __init__(self, data_dir, split, tasks, **kwargs):
        super().__init__()
        # Reading from torch.save file
        self.tasks = tasks
        begin = time.time()
        data_path = os.path.join(data_dir, f'{split}_img_64_64.pth')
        self.data = torch.load(data_path)
        end = time.time()
        print(f"loading time {end-begin}\'s for the {split} part of celeb_a")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label, inp = self.data[index]
        inp = torch.from_numpy(inp)
        label_ = {task: torch.tensor(val, dtype=torch.long) for task, val in label.items() if task in self.tasks}
        out = copy.deepcopy(label_)
        out.update({"inp": inp.permute(2, 0, 1)})
        return out
