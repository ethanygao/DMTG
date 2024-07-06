import os
import time
import shutil
import math
from functools import partial

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
import utils.general_utils as gutils
import torch.distributed as dist
import datasets
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torchvision import transforms
import random
import time
import copy



def confusion_matrix(y_true, y_pred):
    confusion_ = []
    for i in np.unique(y_true):
        tmp = y_pred[y_true==i]
        if i == 0:
            confusion_.append([np.sum(tmp==i), np.sum(tmp!=i)])
        else:
            confusion_.append([np.sum(tmp!=i), np.sum(tmp==i)])
    return np.array(confusion_)


class Accumulator():
    def __init__(self, tasks):
        self.id_to_task = tasks
        self.total = 0.
        self.good = {task: 0. for task in tasks}
        self.gt_ = {task: None for task in tasks}
        self.pred_ = {task: None for task in tasks}

    def accumulate(self, kn_predictions, n_gt, grouping=None):
        if grouping is None:
            for task in self.id_to_task:
                gt = n_gt[task]
                pred = kn_predictions[task].argmax(-1)
                self.good[task] += torch.sum(pred == gt).item()
                self.gt_[task] = gt if self.gt_[task] is None else torch.cat([self.gt_[task], gt])
                self.pred_[task] = pred if self.pred_[task] is None else torch.cat([self.pred_[task], pred])
        else:
            n_tasks, k_groups = grouping.shape
            for i in range(n_tasks):
                task = self.id_to_task[i]
                gt = n_gt[task]
                for j in range(k_groups):
                    if grouping[i, j] > 0.:
                        pred = kn_predictions[f"group_{j}"][task].argmax(-1)
                        self.good[task] += torch.sum(pred == gt).item()
                        self.gt_[task] = gt if self.gt_[task] is None else torch.cat([self.gt_[task], gt])
                        self.pred_[task] = pred if self.pred_[task] is None else torch.cat([self.pred_[task], pred])


        self.total += n_gt[self.id_to_task[0]].shape[0]

    def items(self):
        res = {task: acc_/self.total for task, acc_ in self.good.items()}
        confusion_ = {task: confusion_matrix(gt.cpu().numpy(), pred.cpu().numpy()) for ((task, gt), (_, pred)) in zip(self.gt_.items(), self.pred_.items())}
        # res.update(confusion_)
        return res, confusion_


class IOUMeter():
    def __init__(self, n_classes):
        self.tp = [0] * n_classes
        self.fp = [0] * n_classes
        self.acc = [0] * n_classes
        self.n_classes = n_classes
        self.classes = set([])

    def accumulate(self, kn_predictions, n_gt, grouping):
        task = "segment_semantic"
        tasks = list(kn_predictions["group_0"].keys())
        i = tasks.index(task)
        gt = n_gt[task]
        non_zero_indices = torch.nonzero(gt, as_tuple=True)
        for j in range(grouping.shape[1]):
            if grouping[i, j] > 0.:
                pred = kn_predictions[f"group_{j}"][task].argmax(1).unsqueeze(1)
                # skipping 0's class
                for n in range(1, self.n_classes):
                    mask_gt = gt[non_zero_indices] == n
                    mask_pred = pred[non_zero_indices] == n
                    self.tp[n] += torch.sum(mask_gt, dtype=torch.long)
                    self.fp[n] += torch.sum(mask_pred, dtype=torch.long)
                    self.acc[n] += torch.sum((mask_gt.long() + mask_pred.long()) == 2, dtype=torch.long)
                self.classes = self.classes | set(torch.unique(gt).cpu().numpy())

    def items(self):
        res = {}
        avg = 0.
        for i in range(self.n_classes):
            if self.tp[i] == 0:
                pass
            else:
                iou = self.acc[i] / (self.tp[i] + self.fp[i] - self.acc[i])
                res[str(i)] = float(iou)
                avg += iou / self.n_classes
        res["mean iou"] = avg
        return res


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    # ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.2f}M'.format(tot / 1e6)
        else:
            return '{:.2f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            for key in self.next_data.keys():
                self.next_data[key] = self.next_data[key].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            for key in data.keys():
                data[key].record_stream(torch.cuda.current_stream())
        self.preload() 
        return data


def make_data_loader(loger, spec, tasks, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec, args={"tasks": tasks})
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank == 0:
            gutils.print_loger('{} dataset: size={}'.format(tag, len(dataset)), loger)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=(tag == 'train'))
        loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=4, pin_memory=True, sampler=sampler)
        return loader, sampler
    else:
        gutils.print_loger('{} dataset: size={}'.format(tag, len(dataset)), loger)
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            shuffle=(tag == 'train'), num_workers=2 * spec['n_gpus'], pin_memory=True)
        return loader

