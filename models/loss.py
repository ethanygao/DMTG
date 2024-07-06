import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from .models import register

@register("naive")
class Loss:
    def __init__(self, tasks, is_entropy=False, entropy_weight=1., **kwargs):
        super().__init__()
        self.tasks = tasks
        self.use_entropy = is_entropy
        self.loss_fn = {}
        self.entropy_weight = entropy_weight
        for task in tasks:
            self.loss_fn[task] = self.get_loss_fn(task, kwargs.get("rotate", False))

    def __call__(self, kn_predictions, n_gt, grouping, weight):
        losses = {}
        total_loss = 0.
        n_tasks, k_groups = grouping.shape[-2:]
        for i in range(n_tasks):
            for j in range(k_groups):
                ith_task = self.tasks[i]
                jth_group = f"group_{j}"
                pred = kn_predictions[jth_group][ith_task]
                if losses.get(ith_task) is None:
                    losses[ith_task] = {}
                if grouping.ndim >= 3:
                    tmp = self.loss_fn[ith_task](pred, n_gt[ith_task], reduction="none")
                    tmp = torch.mean(tmp * grouping[:, i, j])
                    total_loss += tmp
                else:
                    # 2023.10.21
                    if "mask" in n_gt.keys():
                        tmp = self.loss_fn[ith_task](pred, n_gt[ith_task], n_gt["mask"])
                    else:
                        tmp = self.loss_fn[ith_task](pred, n_gt[ith_task])
                    total_loss += tmp * grouping[i, j]
                losses[ith_task][jth_group] = tmp
        prob = F.softmax(weight, dim=-1)
        entropy_loss = torch.nanmean(torch.sum(-prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob), dim=-1))
        entropy_loss = entropy_loss if not torch.isnan(entropy_loss) else torch.tensor(0.)
        if self.use_entropy:
            total_loss += entropy_loss * self.entropy_weight
        losses["entropy"] = entropy_loss
        losses["total"] = total_loss
        return losses

    @staticmethod
    def get_loss_fn(task, is_rotate=False):
        if task == "segment_semantic":
            return segment_semantic_loss
        elif task == "normal":
            if is_rotate:
                return normal_loss
            else:
                normal_loss_simple
        elif task == "keypoints2d":
            return keypoints2d_loss
        elif task == "depth_zbuffer":
            if is_rotate:
                return depth_loss
            else:
                return depth_loss_simple
        elif task == "edge_texture":
            return edge2d_loss
        else:
            return F.cross_entropy


@register("group")
class GroupLoss(Loss):
    def __call__(self, predictions, n_gt):
        losses = {}
        total_loss = 0.

        for task, pred in predictions.items():
            if "mask" in n_gt.keys():
                tmp = self.loss_fn[task](pred, n_gt[task], n_gt["mask"])
            else:
                tmp = self.loss_fn[task](pred, n_gt[task])
            total_loss += tmp
            losses[task] = tmp

        losses["total"] = total_loss
        return losses


def segment_semantic_loss(output, target, mask):
    sl = torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1), ignore_index=0,
                                           reduction='mean')
    return sl


def normal_loss(output, target, mask):
    nl = rotate_loss(output, target, mask, normal_loss_base)
    return nl


def normal_loss_simple(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    nl = out.mean()
    return nl


def rotate_loss(output, target, mask, loss_name):
    target = target[:, :, 1:-1, 1:-1].float()
    mask = mask[:, :, 1:-1, 1:-1].float()
    output = output.float()
    val1 = loss = loss_name(output[:, :, 1:-1, 1:-1], target, mask)

    val2 = loss_name(output[:, :, 0:-2, 1:-1], target, mask)
    loss = torch.min(loss, val2)
    val3 = loss_name(output[:, :, 1:-1, 0:-2], target, mask)
    loss = torch.min(loss, val3)
    val4 = loss_name(output[:, :, 2:, 1:-1], target, mask)
    loss = torch.min(loss, val4)
    val5 = loss_name(output[:, :, 1:-1, 2:], target, mask)
    loss = torch.min(loss, val5)
    val6 = loss_name(output[:, :, 0:-2, 0:-2], target, mask)
    loss = torch.min(loss, val6)
    val7 = loss_name(output[:, :, 2:, 2:], target, mask)
    loss = torch.min(loss, val7)
    val8 = loss_name(output[:, :, 0:-2, 2:], target, mask)
    loss = torch.min(loss, val8)
    val9 = loss_name(output[:, :, 2:, 0:-2], target, mask)
    loss = torch.min(loss, val9)

    # lst = [val1,val2,val3,val4,val5,val6,val7,val8,val9]

    # print(loss.size())
    loss = loss.mean()
    # print(loss)
    return loss


def normal_loss_base(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    out = out.mean(dim=(1, 2, 3))
    return out


def normal2_loss(output, target, mask):
    diff = output.float() - target.float()
    out = torch.abs(diff)
    out = out * mask.float()
    nl3 = out.mean()
    return nl3


def depth_loss_simple(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    dl = out.mean()
    return dl


def depth_loss(output, target, mask):
    dl = rotate_loss(output, target, mask, depth_loss_base)
    return dl


def depth_loss_base(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask.float()
    out = out.mean(dim=(1, 2, 3))
    return out


def edge_loss_simple(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    el = out.mean()
    return el


def reshade_loss(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    rl = out.mean()
    return rl


def keypoints2d_loss(output, target, mask):
    kl = torch.nn.functional.l1_loss(output, target)
    return kl


def edge2d_loss(output, target, mask):
    tl = torch.nn.functional.l1_loss(output, target)
    return tl


def auto_loss(output, target, mask):
    al = torch.nn.functional.l1_loss(output, target)
    return al


def pc_loss(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    cl = out.mean()
    return cl


def edge_loss(output, target, mask):
    out = torch.nn.functional.l1_loss(output, target, reduction='none')
    out *= mask
    el = out.mean()
    return el
