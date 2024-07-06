import torch
from torch import nn
from torch.nn import functional as F
import math
from .models import register, make
import copy
import models.ozan_rep_fun as regulators
from collections import OrderedDict


class LearnToGroup(nn.Module):
    def __init__(self, backbone, k_group_bones, kn_decoders, sample="gumbel", is_hard=False, regularize_grad=None, eps=0.):
        super().__init__()
        self.backbone = backbone
        self.k_group_bones = k_group_bones if isinstance(k_group_bones, nn.ModuleDict) else nn.ModuleDict(k_group_bones)
        # self.k_group_decoders = k_group_decoders if isinstance(k_group_decoders, nn.ModuleDict) else nn.ModuleDict(k_group_decoders)
        if isinstance(kn_decoders, nn.ModuleDict):
            self.kn_decoders = kn_decoders
        else:
            self.kn_decoders = nn.ModuleDict()
            for g_key, decoders in kn_decoders.items():
                self.kn_decoders[g_key] = nn.ModuleDict(decoders)

        self.k_groups = len(k_group_bones)
        self.tasks = list(self.kn_decoders["group_0"].keys())
        self.n_tasks = len(self.tasks)
        self.param_sample = True

        self.sample = sample
        if sample != "diagonal":
            if sample == "diag_gumbel":
                group_matrix = torch.full((self.n_tasks, self.k_groups), 0.9)
                for i in range(self.n_tasks):
                    for j in range(self.k_groups):
                        if i != j:
                            group_matrix[i, j] = 0.1 / (self.k_groups - 1)
                group_matrix.requires_grad_()
                self.group_matrix = nn.Parameter(group_matrix)
            elif sample == "gaussian_gumbel":
                m = torch.distributions.MultivariateNormal(torch.ones(self.k_groups), torch.eye(self.k_groups)*0.04)
                group_matrix = m.rsample(torch.Size([self.n_tasks]))
                group_matrix.detach().requires_grad_()
                self.group_matrix = nn.Parameter(group_matrix)
            else:
                self.group_matrix = nn.Parameter(torch.ones(self.n_tasks, self.k_groups, requires_grad=True) +
                                                 torch.rand(self.n_tasks, self.k_groups, requires_grad=True) * eps)

        self.hard_ = is_hard
        self.group_regulator = self.set_regulator("group", regularize_grad) if regularize_grad is not None else nn.Identity()
        self.task_regulator = self.set_regulator("task", regularize_grad) if regularize_grad is not None else nn.Identity()

    def forward(self, x, temperature=1):
        x_base = self.backbone(x)

        feat_outputs = {}
        for g_key, group_bone in self.k_group_bones.items():
            feat_outputs[g_key] = group_bone(self.group_regulator(x_base))

        task_outputs = {g_key: {} for g_key in self.k_group_bones.keys()}
        for g_key, decoders in self.kn_decoders.items():
            for t_key, decoder in decoders.items():
                task_outputs[g_key][t_key] = decoder(self.task_regulator(feat_outputs[g_key]))

        if self.param_sample:
            grouping = self.sample_task_group(x, temperature)
        else:
            grouping = torch.ones_like(self.group_matrix, dtype=self.group_matrix.dtype, device=self.group_matrix.device)
        return task_outputs, grouping, self.group_matrix
        # return task_outputs, None

    def hard_sample(self, is_hard):
        self.hard_ = is_hard

    def set_regulator(self, prefix, reg_type):
        fun_name = f"{prefix}_{reg_type}"
        cls_name = "".join([s.capitalize() for s in fun_name.split("_")])
        regulator = regulators.__dict__[fun_name]
        regulators.__dict__[cls_name].n = self.k_groups if prefix == "group" else self.n_tasks
        return regulator

    def sample_task_group(self, x, temperature):
        if self.training:
            if self.sample == "gumbel" or self.sample == "diag_gumbel" or self.sample == "gaussian_gumbel":
                grouping = F.gumbel_softmax(self.group_matrix, tau=temperature, hard=self.hard_)
            elif self.sample == "batch_gumbel":
                grouping = F.gumbel_softmax(self.group_matrix.unsqueeze(0).repeat(x.shape[0], 1, 1), tau=temperature, hard=self.hard_)
            elif self.sample == "softmax":
                grouping = F.softmax(self.group_matrix, dim=-1)
            elif self.sample == "batch_relaxed":
                sampler = torch.distributions.RelaxedOneHotCategorical(torch.tensor([temperature], device=self.group_matrix.device), logits=self.group_matrix)
                # grouping = sampler.sample(torch.Size([x.shape[0]]))
                grouping = sampler.sample()
            elif self.sample == "diagonal":
                grouping = torch.eye(self.n_tasks, self.k_groups, device=x.device)
                self.group_matrix = grouping
            else:
                grouping = self.group_matrix
        else:
            max_indices = self.group_matrix.argmax(dim=-1)
            grouping = torch.zeros_like(self.group_matrix, device=self.group_matrix.device)
            grouping[torch.arange(self.n_tasks), max_indices] = 1.0
        return grouping

    def set_param_sample(self, flag):
        self.param_sample = flag


class FineTuningGroup(LearnToGroup):
    def forward(self, x):
        x_base = self.backbone(x)

        if self.current_group is None:
            feat_outputs = {}
            for g_key, group_bone in self.k_group_bones.items():
                feat_outputs[g_key] = group_bone(self.group_regulator(x_base))

            task_outputs = {}
            for g_key, decoders in self.kn_decoders.items():
                for t_key, decoder in decoders.items():
                    task_outputs[t_key] = decoder(self.task_regulator(feat_outputs[g_key]))
        else:
            feat_output = self.k_group_bones[f"group_{self.current_group}"](x_base)

            task_outputs = {}
            for t_key, decoder in self.kn_decoders[f"group_{self.current_group}"].items():
                task_outputs[t_key] = decoder(self.task_regulator(feat_output))

        return task_outputs

    def pruning(self):
        mask = torch.zeros_like(self.group_matrix, device=self.group_matrix.device)
        mask[torch.arange(self.n_tasks), self.group_matrix.argmax(dim=-1)] = 1
        groups = []
        for k in range(self.k_groups):
            drop_count = 0
            for n, task in enumerate(self.tasks):
                if mask[n, k] == 0.:
                    self.kn_decoders[f"group_{k}"].pop(task)
                    drop_count += 1

            if drop_count == self.n_tasks:
                self.kn_decoders.pop(f"group_{k}")
                self.k_group_bones.pop(f"group_{k}")
            else:
                groups.append(k)
        self.remain_group_id = groups
        self.current_group = groups[0]
        return groups

    def group_parameters(self):
        params = []
        for p in self.k_group_bones[f"group_{self.current_group}"].parameters():
            p.requires_grad_()
            params.append(p)

        for p in self.kn_decoders[f"group_{self.current_group}"].parameters():
            p.requires_grad_()
            params.append(p)

        return iter(params)

    def freezing(self):
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # close update for batchnorm in backbone
        self.backbone.eval()

    def get_groups(self):
        return self.remain_group_id

    def set_current_group(self, group_id):
        self.current_group = group_id


@register("learnToGroup")
def build_meta_arch(**kwargs):
    meta_arch_args = copy.deepcopy(kwargs)
    # loading parameters from trained mtl net
    if (resume := meta_arch_args.get("resume_from_mtl")) is not None:
        sv_file = torch.load(resume)
        mtl_net = make(sv_file["model"], args={"tasks": ["depth_zbuffer", "normal", "segment_semantic", "keypoints2d", "edge_texture"]},
                       load_sd=True)
        print(f"loading pretraining parameters from {meta_arch_args.pop('resume_from_mtl')}")

    if (spec := meta_arch_args.get("backbone")) is not None:
        backbone_ = make(spec) if resume is None else copy.deepcopy(mtl_net.backbone)
        meta_arch_args.pop("backbone")
    else:
        backbone_ = nn.Identity()
    group_bone = make(meta_arch_args.get("group_bone")) if resume is None else copy.deepcopy(mtl_net.k_group_bones["group_0"])
    meta_arch_args.pop("group_bone")
    k_groups = meta_arch_args.pop("k_groups")
    k_group_bones_ = {}

    for i in range(k_groups):
        k_group_bones_[f"group_{i}"] = copy.deepcopy(group_bone)

    kn_decoders_ = {}
    decoders = {}
    decoders_spec = meta_arch_args.pop("decoders")
    for task, channels_ in zip(meta_arch_args["tasks"], decoders_spec["args"]["channels_per_task"]):
        decoder_spec = copy.deepcopy(decoders_spec)
        decoder_spec["args"].pop("channels_per_task")
        decoder_spec["args"]["out_channel"] = channels_
        decoders[task] = make(decoder_spec) if resume is None else copy.deepcopy(mtl_net.kn_decoders["group_0"][task])

    for i in range(k_groups):
        kn_decoders_[f"group_{i}"] = copy.deepcopy(decoders)

    meta_arch_args.pop("tasks")
    return LearnToGroup(backbone=backbone_, k_group_bones=k_group_bones_, kn_decoders=kn_decoders_, **meta_arch_args)


@register("fineTuningGroup")
def build_tuning_arch(**kwargs):
    meta_arch_args = copy.deepcopy(kwargs)
    meta_arch_args.pop("resume_from_mtl", None)

    if (spec := meta_arch_args.get("backbone")) is not None:
        backbone_ = make(spec)
        meta_arch_args.pop("backbone")
    else:
        backbone_ = nn.Identity()
    if meta_arch_args["group_bone"].get("args", None):
        split = meta_arch_args["group_bone"]["args"].get("behind_first_k_layers", None)
    else:
        split = None

    if split is not None:
        group_bone = make(meta_arch_args.get("group_bone"), args={"relu": split>2})
    else:
        group_bone = make(meta_arch_args.get("group_bone"))
    meta_arch_args.pop("group_bone")
    k_groups = meta_arch_args.pop("k_groups")
    k_group_bones_ = {}

    for i in range(k_groups):
        k_group_bones_[f"group_{i}"] = copy.deepcopy(group_bone)

    kn_decoders_ = {}
    decoders = {}
    decoders_spec = meta_arch_args.pop("decoders")
    for task, channels_ in zip(meta_arch_args["tasks"], decoders_spec["args"]["channels_per_task"]):
        decoder_spec = copy.deepcopy(decoders_spec)
        decoder_spec["args"].pop("channels_per_task")
        decoder_spec["args"]["out_channel"] = channels_
        # decoders[task] = make(decoder_spec) if resume is None else copy.deepcopy(mtl_net.kn_decoders["group_0"][task])
        decoders[task] = make(decoder_spec)

    for i in range(k_groups):
        kn_decoders_[f"group_{i}"] = copy.deepcopy(decoders)

    meta_arch_args.pop("tasks")
    return FineTuningGroup(backbone=backbone_, k_group_bones=k_group_bones_, kn_decoders=kn_decoders_, **meta_arch_args)


@register("learnToGroupSTL")
def build_meta_arch_from_stl(**kwargs):
    meta_arch_args = copy.deepcopy(kwargs)

    if (spec := meta_arch_args.get("backbone")) is not None:
        backbone_ = make(spec)
        meta_arch_args.pop("backbone")
    else:
        backbone_ = nn.Identity()
    group_bone = make(meta_arch_args.get("group_bone"))
    meta_arch_args.pop("group_bone")
    k_groups = meta_arch_args.pop("k_groups")
    k_group_bones_ = {}

    for i in range(k_groups):
        k_group_bones_[f"group_{i}"] = copy.deepcopy(group_bone)

    kn_decoders_ = {}
    decoders = {}
    decoders_spec = meta_arch_args.pop("decoders")
    for task, channels_ in zip(meta_arch_args["tasks"], decoders_spec["args"]["channels_per_task"]):
        decoder_spec = copy.deepcopy(decoders_spec)
        decoder_spec["args"].pop("channels_per_task")
        decoder_spec["args"]["out_channel"] = channels_
        decoders[task] = make(decoder_spec)

    for i in range(k_groups):
        kn_decoders_[f"group_{i}"] = copy.deepcopy(decoders)

    def split_and_rename_weights(weights):
        group_weights = OrderedDict()
        head_weights = OrderedDict()
        for wk, wv in weights.items():
            if "backbone" in wk:
                postfix = wk[len("backbone."):]
                new_wk = f"0.{postfix}"
                group_weights[new_wk] = wv
            elif "k_group_bones.group_0" in wk:
                postfix = wk[len("k_group_bones.group_0."):]
                new_wk = f"1.{postfix}"
                group_weights[new_wk] = wv
            elif "kn_decoders.group_0" in wk:
                # parts = wk.split(".")
                # new_wk = ".".join(parts[3:])
                # head_weights[new_wk] = wv
                parts = wk.split(".")
                task = parts[2]
                new_wk = ".".join(parts[3:])
                if head_weights.get(task) is None:
                    head_weights[task] = OrderedDict()
                head_weights[task][new_wk] = wv
            else:
                pass
        return group_weights, head_weights

    if (resume := meta_arch_args.get("resume_from_stl")) is not None:
        if isinstance(resume, str):
            resume = [resume] * k_groups
        assert len(resume) == k_groups, "len(resume) != k_groups"

        if meta_arch_args.get("heads_to_same") is not None:
            heads_to_same = meta_arch_args["heads_to_same"]
        else:
            heads_to_same = False

        for i in range(len(resume)):
            sv_file = torch.load(resume[i], map_location="cpu")
            model_weights = sv_file["model"]["sd"]
            group_weights, head_weights = split_and_rename_weights(model_weights)
            # print(list(k_group_bones_[f"group_{i}"].state_dict().keys()))
            # print(list(group_weights.keys()))
            k_group_bones_[f"group_{i}"].load_state_dict(group_weights)
            for task in head_weights:
                if heads_to_same:
                    kn_decoders_["group_0"][task].load_state_dict(head_weights[task])
                else:
                    kn_decoders_[f"group_{i}"][task].load_state_dict(head_weights[task])

        if heads_to_same:
            decoders_state_dict = kn_decoders_["group_{0}"].state_dict()
            for i in range(1, k_groups):
                kn_decoders_[f"group_{i}"].load_state_dict(decoders_state_dict)

        meta_arch_args.pop("resume_from_stl")

    meta_arch_args.pop("tasks")
    return LearnToGroup(backbone=backbone_, k_group_bones=k_group_bones_, kn_decoders=kn_decoders_, **meta_arch_args)


@register("learnToGroupMTL")
def build_meta_arch_from_mtl(**kwargs):
    def convert_mtl_weights(weights, split_point):
        pre_id = None
        counter = 0
        res = OrderedDict({"backbone": OrderedDict(), "group_bone": OrderedDict()})
        reach = False

        if split_point == 6:
            for k, v in weights.items():
                nparts = k.split(".")
                module_name = nparts[0]
                if module_name == "backbone":
                    pparts = nparts[1:]
                    param_name = ".".join(pparts)
                    res["backbone"][param_name] = v
                elif module_name == "k_group_bones":
                    pparts = nparts[2:]
                    param_name = ".".join(pparts)
                    res["group_bone"][param_name] = v
                elif module_name == "kn_decoders":
                    pparts = nparts[3:]
                    param_name = ".".join(pparts)
                    task = nparts[2]
                    if res.get(task) is None:
                        res[task] = OrderedDict()
                    res[task][param_name] = v

        elif split_point < 6:
            for k, v in weights.items():
                nparts = k.split(".")
                module_name = nparts[0]
                if module_name == "backbone":
                    pparts = nparts[1:]
                    if "blocks" in k:
                        cur_id = int(pparts[1])
                        if pre_id is None:
                            pre_id = cur_id
                        if pre_id != cur_id:
                            counter += 1
                            pre_id = cur_id
                        if not reach and counter >= split_point - 2:
                            reach = True
                            counter = 0
                        if reach:
                            pparts[1] = str(counter)
                            param_name = ".".join(pparts)
                            res["group_bone"][param_name] = v
                            continue

                    param_name = ".".join(pparts)
                    res["backbone"][param_name] = v
                elif module_name == "k_group_bones":
                    pparts = nparts[2:]
                    if "blocks" in k:
                        cur_id = int(pparts[1])
                        if pre_id != cur_id:
                            counter += 1
                            pre_id = cur_id
                        pparts[1] = str(counter)
                    param_name = ".".join(pparts)
                    res["group_bone"][param_name] = v
                elif module_name == "kn_decoders":
                    pparts = nparts[3:]
                    param_name = ".".join(pparts)
                    task = nparts[2]
                    if res.get(task) is None:
                        res[task] = OrderedDict()
                    res[task][param_name] = v

        else:
            for k, v in weights.items():
                nparts = k.split(".")
                module_name = nparts[0]
                if module_name == "backbone":
                    pparts = nparts[1:]
                    if "blocks" in k:
                        cur_id = int(pparts[1])
                        if pre_id is None:
                            pre_id = cur_id
                        if pre_id != cur_id:
                            counter += 1
                            pre_id = cur_id
                    param_name = ".".join(pparts)
                    res["backbone"][param_name] = v
                elif module_name == "k_group_bones":
                    pparts = nparts[2:]
                    if "blocks" in k:
                        cur_id = int(pparts[1])
                        if pre_id != cur_id:
                            counter += 1
                            pre_id = cur_id
                        if not reach and counter >= split_point - 2:
                            reach = True
                            counter = 0
                        pparts[1] = str(counter)
                        if not reach:
                            param_name = ".".join(pparts)
                            res["backbone"][param_name] = v
                            continue

                    param_name = ".".join(pparts)
                    res["group_bone"][param_name] = v
                elif module_name == "kn_decoders":
                    pparts = nparts[3:]
                    param_name = ".".join(pparts)
                    task = nparts[2]
                    if res.get(task) is None:
                        res[task] = OrderedDict()
                    res[task][param_name] = v
        return res

    meta_arch_args = copy.deepcopy(kwargs)
    # loading parameters from trained mtl net
    if (resume := meta_arch_args.get("resume_from_mtl")) is not None:
        sv_file = torch.load(resume, map_location="cpu")
        pretrain_weights = sv_file["model"]["sd"]
        split = meta_arch_args["group_bone"]["args"]["behind_first_k_layers"]
        pretrain_weights = convert_mtl_weights(pretrain_weights, split)
        print(f"loading pretraining parameters from {meta_arch_args.pop('resume_from_mtl')}")

    if (spec := meta_arch_args.get("backbone")) is not None:
        if resume:
            spec["sd"] = pretrain_weights["backbone"]
        backbone_ = make(spec, load_sd=resume is not None)
        meta_arch_args.pop("backbone")
    else:
        backbone_ = nn.Identity()
    group_spec = meta_arch_args.pop("group_bone")
    relu = True
    if resume:
        group_spec["sd"] = pretrain_weights["group_bone"]
        relu = split > 2
    group_bone = make(group_spec, load_sd=resume is not None, args={"relu": relu})
    k_groups = meta_arch_args.pop("k_groups")
    k_group_bones_ = {}

    for i in range(k_groups):
        k_group_bones_[f"group_{i}"] = copy.deepcopy(group_bone)

    kn_decoders_ = {}
    decoders = {}
    decoders_spec = meta_arch_args.pop("decoders")
    for task, channels_ in zip(meta_arch_args["tasks"], decoders_spec["args"]["channels_per_task"]):
        decoder_spec = copy.deepcopy(decoders_spec)
        decoder_spec["args"].pop("channels_per_task")
        decoder_spec["args"]["out_channel"] = channels_
        if resume:
            decoder_spec["sd"] = pretrain_weights[task]
        decoders[task] = make(decoder_spec, load_sd=resume is not None)

    for i in range(k_groups):
        kn_decoders_[f"group_{i}"] = copy.deepcopy(decoders)

    meta_arch_args.pop("tasks")
    return LearnToGroup(backbone=backbone_, k_group_bones=k_group_bones_, kn_decoders=kn_decoders_, **meta_arch_args)




