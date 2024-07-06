import argparse
import os, sys
import time


import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts

import datasets
import models
from models.loss import Loss
import numpy as np
sys.path.append('.')
import utils.model_utils as utils
import utils.general_utils as gutils
import utils.metric_utils as metric_utils


def prepare_testing(cfg, loger):
    if os.path.isfile(cfg.run.load_ckpt_dir):
        sv_file = torch.load(cfg.run.load_ckpt_dir)
        if cfg.run.finetune:
            model = sv_file['model']
            model.set_current_group(None)
        else:
            model = models.make(sv_file['model'], args={"tasks": cfg.dataset.tasks_name}, load_sd=True)
    else:
        model = models.make(cfg.env.meta_arch, args={"tasks": cfg.dataset.tasks_name})
    model = model.cuda()
    gutils.print_loger('model: #params={}'.format(utils.compute_num_params(model, text=True)), loger)
    use_cuda = True if len(cfg.env.gpus) > 0 else False
    if cfg.run.finetune:
        loss_fn = models.make({"name": "group", "args": {"rotate": True}},
                              args={"tasks": cfg.dataset.tasks_name, "is_cuda": use_cuda})
    else:
        loss_fn = models.make(cfg.loss,
                              args={"tasks": cfg.dataset.tasks_name, "is_cuda": use_cuda})
    return model, None, None, None, loss_fn


def test(cfg, eval_loader, model, loss_fn, is_classification=False, is_iou=False):
    model.eval()
    model.hard_sample(True)
    losses = {task: utils.Averager() for task in model.tasks}
    losses["total"] = utils.Averager()
    losses["entropy"] = utils.Averager()
    acc = utils.Accumulator(loss_fn.tasks) if is_classification else None
    iou = utils.IOUMeter(18) if is_iou else None

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, leave=False, desc='eval')):
            for k, v in batch.items():
                batch[k] = v.cuda()

            if cfg.run.finetune:
                predictions_ = model(batch["inp"])
                losses_ = loss_fn(predictions_, batch)

                for task, loss in losses_.items():
                    if losses.get(task) is None:
                        losses[task] = utils.Averager()
                    losses[task].add(loss.item())

                if is_classification:
                    acc.accumulate(predictions_, batch)
            else:
                kn_predictions_, grouping_, weight_ = model(batch["inp"])
                losses_ = loss_fn(kn_predictions_, batch, grouping_, weight_)

                for i, (k1, v1) in enumerate(losses_.items()):
                    if isinstance(v1, dict):
                        for j, (k2, v2) in enumerate(v1.items()):
                            if grouping_[i, j] > 0.:
                                losses[k1].add(v2.item())
                    else:
                        losses[k1].add(v1.item())

                if is_classification:
                    acc.accumulate(kn_predictions_, batch, grouping_)

                if is_iou:
                    iou.accumulate(kn_predictions_, batch, grouping_)

    # model.hard_sample(hard)
    for k1, v1 in losses.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                losses[k1] = v2.item()
        else:
            losses[k1] = v1.item()
    res = losses
    if is_classification:
        accuracy, confusion_matrix = acc.items()
        res.update({f"{task}_acc": val for task, val in accuracy.items()})
        res.update({f"{task}_confusion": val for task, val in confusion_matrix.items()})
        acc_list = [val for _, val in accuracy.items()]
        res.update({"Total Error": metric_utils.accuracy_total_error(acc_list)})
        ref_acc = gutils.load_data(cfg.eval.ref_info)
        ref_acc_list = [val for _, val in ref_acc.items()]
        res.update({"Normalized Gain": metric_utils.normalized_gain(acc_list, ref_acc_list)})
    if is_iou:
        iou_metrics = iou.items()
        res.update(iou_metrics)
    return res


def main(cfg):
    save_path = os.path.join(cfg.run.res_dir, cfg.run.exp_name)
    gutils.mkdir(save_path)
    loger = gutils.init_loger(save_path, 'test_grouoing', comments='test')
    gutils.print_loger(f">> Run cmd: \n {sys.argv}", loger)
    gutils.print_loger(cfg, loger)

    val_loader = utils.make_data_loader(loger, {'name': cfg.dataset.name, 'args': cfg.dataset.args, 'batch_size': cfg.run.batch_size, 'n_gpus': len(cfg.env.gpus)}, cfg.dataset.tasks_name, tag='val')

    model, _, _, _, loss_fn = prepare_testing(cfg, loger)

    if len(cfg.env.gpus) > 1:
        model = nn.parallel.DataParallel(model)

    timer = utils.Timer()

    val_res = test(cfg, val_loader, model, loss_fn, cfg.run.is_classify, cfg.run.is_iou)

    t = timer.t()
    gutils.print_loger(f"inferring time: {t:.6f}", loger)
    gutils.print_loger(' ' .join([f"{metric}: {val:.6f}" if not isinstance(val, np.ndarray) else f"{metric}: {str(val)} \n" for metric, val in val_res.items()]), loger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learn2Group testing")
    parser.add_argument('--cfg', default="")
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = gutils.setup_cfg(args)
    main(cfg)
