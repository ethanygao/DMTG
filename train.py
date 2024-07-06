import os

import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim
import torch.multiprocessing as mp
import yaml
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import datasets
import models
from torch.distributed.elastic.multiprocessing.errors import record
import utils.model_utils as utils
import utils.general_utils as gutils

##########ENVIRONMENTAL SETTING##########
# This solves the fucking DistributedDataParallel deadlock!!
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TORCH_DISTRIBUTED_DEBUG "] = "DETAIL"
##########ENVIRONMENTAL SETTING##########


def reduce_on_dict(losses):
    global local_rank_
    for k1 in losses.keys():
        if isinstance(losses[k1], dict):
            for k2 in losses[k1].keys():
                loss = torch.tensor(losses[k1][k2], device=torch.device(local_rank_))
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                losses[k1][k2] = loss.item()
        else:
            loss = torch.tensor(losses[k1], device=torch.device(local_rank_))
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            losses[k1] = loss.item()

    return losses


def make_data_loaders():
    train_loader, train_sampler = utils.make_data_loader(None, config.train_dataset, config.run.tasks, tag='train')
    val_loader, val_sampler = utils.make_data_loader(None, config.val_dataset, config.run.tasks, tag='val')
    print("*" * 5 + f"Loader done at rank {local_rank_}" + "*" * 5)
    return train_loader, train_sampler, val_loader, val_sampler


def prepare_training():
    if os.path.isfile(config.run.resume):
        sv_file = torch.load(config.run.resume, map_location="cpu")
        model = models.make(sv_file["model"], args={"tasks": config.run.task}, load_sd=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        gutils.print_loger("*" * 5 + f"Resume syncBatchNorm at rank {local_rank_}" + "*" * 5)
        model = model.cuda()
        gutils.print_loger("*" * 5 + f"Resume model done at rank {local_rank_}" + "*" * 5)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True)
        gutils.print_loger("*" * 5 + f"Resume {sv_file['optimizer']['name']} optimizer at rank {local_rank_}" + "*" * 5)
        epoch_start = sv_file["epoch"] + 1
        if config.scheduler.name == "multi_step_lr":
            lr_scheduler = MultiStepLR(optimizer, **config.scheduler.args)
        elif config.scheduler.name == "cosine_annealing_Warm_Restarts":
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, **config.scheduler.args)
        elif config.scheduler.name == "reduce_on_plateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, **config.scheduler.args)
        else:
            lr_scheduler = None
        if lr_scheduler is not None and not isinstance(lr_scheduler, ReduceLROnPlateau):
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
        gutils.print_loger("*" * 5 + f"Resume scheduler done at rank {local_rank_}" + "*" * 5)
    else:
        model = models.make(config.meta_arch, args={"tasks": config.run.tasks})
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        gutils.print_loger("*" * 5 + f"SyncBatchNorm at rank {local_rank_}" + "*" * 5)
        model = model.cuda()
        gutils.print_loger("*" * 5 + f"Model at rank {local_rank_}" + "*" * 5)
        optimizer = utils.make_optimizer(
            model.parameters(), config.optimizer)
        gutils.print_loger("*" * 5 + f"{config.optimizer.name} Optimizer done at rank {local_rank_}" + "*" * 5)
        epoch_start = 1
        if config.scheduler.name == "multi_step_lr":
            lr_scheduler = MultiStepLR(optimizer, **config.scheduler.args)
        elif config.scheduler.name == "cosine_annealing_Warm_Restarts":
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, **config.scheduler.args)
        elif config.scheduler.name == "reduce_on_plateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, **config.scheduler.args)
        else:
            lr_scheduler = None
        gutils.print_loger("*" * 5 + f"Scheduler done at rank {local_rank_}" + "*" * 5)

    if local_rank_ == 0:
        gutils.print_loger('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    loss_fn = models.make(config.loss, args={"tasks": config.run.tasks})
    gutils.print_loger("*" * 5 + f"Loss done at rank {local_rank_}" + "*" * 5)
    return model, optimizer, epoch_start, lr_scheduler, loss_fn


def train(train_loader, model, optimizer, loss_fn, temperature_=1):
    global cur_epoch
    model.train()
    losses = {}
    tasks = model.module.tasks
    k_groups = model.module.k_groups
    for task in tasks:
        losses[task] = {}
        for i in range(k_groups):
            losses[task][f"group_{i}"] = utils.Averager()
    losses["entropy"] = utils.Averager()
    losses["total"] = utils.Averager()

    prefetcher = utils.data_prefetcher(train_loader)
    running_bar = tqdm(total=len(train_loader), leave=False, desc='train')
    batch = prefetcher.next()
    while batch is not None:
        kn_predictions_, grouping_, weight_ = model(batch["inp"], temperature_)
        losses_ = loss_fn(kn_predictions_, batch, grouping_, weight_)
        optimizer.zero_grad()
        losses_["total"].backward()
        optimizer.step()

        for k1, v1 in losses_.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    losses[k1][k2].add(v2.item())
            else:
                losses[k1].add(v1.item())

        batch = prefetcher.next()
        cur_epoch += 1
        running_bar.update(1)

    for k1, v1 in losses.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                losses[k1][k2] = v2.item()
        else:
            losses[k1] = v1.item()

    return losses


def evaluate(eval_loader, model, loss_fn):
    model.eval()
    hard = model.module.hard_
    model.module.hard_sample(True)
    tasks = model.module.tasks
    losses = {task: utils.Averager() for task in tasks}
    losses["total"] = utils.Averager()
    losses["entropy"] = utils.Averager()
    acc = utils.Accumulator(loss_fn.tasks) if config.run.is_classify else None

    with torch.no_grad():
        prefetcher = utils.data_prefetcher(eval_loader)
        running_bar = tqdm(total=len(eval_loader), leave=False, desc='eval')
        batch = prefetcher.next()
        while batch is not None:
            kn_predictions_, grouping_, weight_ = model(batch["inp"])
            losses_ = loss_fn(kn_predictions_, batch, grouping_, weight_)

            for i, (k1, v1) in enumerate(losses_.items()):
                if isinstance(v1, dict):
                    for j, (k2, v2) in enumerate(v1.items()):
                        if grouping_[i, j] > 0.:
                            losses[k1].add(v2.item())
                else:
                    losses[k1].add(v1.item())

            if acc:
                acc.accumulate(kn_predictions_, batch, grouping_)

            batch = prefetcher.next()
            running_bar.update(1)

    model.module.hard_sample(hard)
    for k1, v1 in losses.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                losses[k1] = v2.item()
        else:
            losses[k1] = v1.item()
    if acc:
        accuracy = acc.items()
        sum_error = 0
        for _, task_acc in accuracy.items():
            sum_error += 1 - task_acc
        losses["error"] = sum_error / len(accuracy)
    return losses


def main(cfg):
    gutils.seed_everything(cfg.run.seed)
    nproc = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])
    save_path = os.path.join(cfg.run.res_dir, cfg.run.exp_name)
    main_worker(local_rank=rank, nproc=nproc, cfg=cfg, save_path=save_path)


@record
def main_worker(local_rank, nproc, cfg, save_path):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', world_size=nproc, rank=local_rank)

    global config, cur_epoch, local_rank_, writer
    cur_epoch = 0
    local_rank_ = local_rank
    config = cfg
    _, writer = utils.set_save_path(save_path)
    gutils.mkdir(save_path)
    gutils.init_loger(save_path, 'train_grouping')
    if local_rank == 0:
        timer = utils.Timer()
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train_loader, train_sampler, val_loader, val_sampler = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler, loss_fn = prepare_training()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank_], output_device=local_rank_, find_unused_parameters=False)
    gutils.print_loger("*" * 5 + f"distributed warping done at rank {local_rank_}" + "*" * 5)

    epoch_max = config.run.epoch_max
    epoch_val = config.run.epoch_val
    epoch_save = config.run.epoch_save

    min_val_v = 1e18
    min_tr_v = 1e18

    for epoch in range(epoch_start, epoch_max + 1):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            t_epoch_start = timer.t()
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        tau = config.run.tau
        gutils.print_loger(f"setting tau as {tau} at epoch {epoch}")
        train_losses = train(train_loader, model, optimizer, loss_fn, tau)

        dist.barrier()
        reduce_on_dict(train_losses)
        train_loss = train_losses.pop("total")
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()
        if np.isnan(train_loss):
            gutils.print_loger(f"A learning rate of {optimizer.param_groups[0]['lr']} is too high so tha the training loss has become nan")
            exit(0)

        if local_rank == 0:
            log_info.append('train: loss={:.4f}'.format(train_loss))
            writer.add_scalars('losses', {'train': train_loss}, epoch)
            entropy = train_losses.pop("entropy")
            writer.add_scalar("entropy_loss", entropy, epoch)

            for task, losses in train_losses.items():
                writer.add_scalars(task, losses, epoch)

            model_ = model.module
            model_spec = config.meta_arch
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config.optimizer
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
            }

            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                           os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0 or epoch == 1):
            val_losses = evaluate(val_loader, model, loss_fn)

            dist.barrier()
            reduce_on_dict(val_losses)

            if local_rank == 0:
                val_loss = val_losses.pop("error", val_losses.pop("total"))
                writer.add_scalars("losses", {"val": val_loss}, epoch)
                log_info.append('val: loss={:.4f}'.format(val_loss))

                val_losses.pop("entropy")
                writer.add_scalars("tasks", val_losses, epoch)
                if val_loss < min_val_v and train_loss < min_tr_v:
                    min_val_v = val_loss
                    min_tr_v = train_loss
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        if local_rank == 0:
            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            gutils.print_loger(', '.join(log_info))
            writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learn2Group testing")
    parser.add_argument('--cfg', default="")
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = gutils.setup_cfg(args)
    os.environ['LOCAL_RANK'] = str(args.local_rank)

    ##########debug code###############
    # os.environ["MASTER_ADDR"]="localhost"
    # os.environ["MASTER_PORT"] = "23002"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["LOCAL_RANK"] = "0"
    ##########debug code###############

    main(cfg=cfg)
